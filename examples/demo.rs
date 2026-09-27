//! Runnable proof that the primitives actually work, not just that their
//! unit tests pass. Every check below prints what it expected and what it
//! measured, and panics if the two disagree — so `cargo run --example demo`
//! either prints a clean report or fails loudly.
//!
//! No fixture files: every frame is synthesised in code, so this runs
//! anywhere with no external assets.

use image::{Rgba, RgbaImage};
use syrup::geometry::{Rect, find_color_bar, measure_bar_fill};
use syrup::glyphs::{GlyphOptions, GlyphSet};
use syrup::motion::{MotionConfig, MotionDetector};
use syrup::template::{self, Template};
use syrup::threshold::Channel;

fn line(title: &str) {
    println!("\n== {title} ==");
}

/// A flat background with a horizontal HP-bar-style track: red fill on the
/// left, a slightly darker green groove on the right.
fn bar_frame(width: u32, height: u32, fill_fraction: f32) -> (RgbaImage, Rect) {
    let bar = Rect {
        x: 20,
        y: height / 2 - 6,
        w: width - 40,
        h: 12,
    };
    let filled_w = (bar.w as f32 * fill_fraction).round() as u32;
    let image = RgbaImage::from_fn(width, height, |x, y| {
        if y >= bar.y && y < bar.y + bar.h && x >= bar.x && x < bar.x + bar.w {
            if x < bar.x + filled_w {
                Rgba([210, 30, 30, 255]) // fill: red
            } else {
                Rgba([20, 70, 20, 255]) // groove: dark green
            }
        } else {
            Rgba([130, 130, 140, 255]) // panel background, clearly distinct from the groove
        }
    });
    (image, bar)
}

fn demo_bar_fill() {
    line("Geometry: reading a health-bar's fill percentage");
    for &fraction in &[0.15, 0.5, 0.83] {
        let (image, bar) = bar_frame(300, 40, fraction);
        let search = Rect {
            x: 0,
            y: 0,
            w: image.width(),
            h: image.height(),
        };
        let is_red = |p: &Rgba<u8>| p[0] > 150 && p[1] < 90 && p[2] < 90;
        let found = find_color_bar(&image, search, (0.0, 60.0), 0.4, 0.35)
            .expect("a red bar should be found in the synthetic frame");
        let measured = measure_bar_fill(&image, found, search, is_red)
            .expect("fill measurement should succeed against the learned groove colour");
        let expected_pct = fraction * 100.0;
        println!(
            "  drew {expected_pct:>5.1}% full -> found bar at {found:?}, measured {measured:>5.1}% full"
        );
        assert!(
            (measured - expected_pct).abs() < 4.0,
            "measured fill {measured} too far from the {expected_pct} actually drawn"
        );
        let _ = bar; // silence unused warning when assertions are compiled out
    }
    println!("  PASS: measured fill tracked the drawn fill within 4 percentage points.");
}

/// A synthetic textured background, sized like a real capture, with a
/// planted icon at a known location.
fn frame_with_icon(icon_size: u32, at: (u32, u32)) -> (RgbaImage, RgbaImage) {
    let frame = RgbaImage::from_fn(1024, 640, |x, y| {
        let v = ((x / 9 + y / 7) % 11) * 17 + ((x * 31 + y * 17) % 13) * 4;
        Rgba([v as u8, (v / 2 + 30) as u8, (240 - v.min(240)) as u8, 255])
    });
    let icon = RgbaImage::from_fn(icon_size, icon_size, |x, y| {
        let s = icon_size as f32;
        let (cx, cy) = (s * 0.4, s * 0.45);
        let d = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt();
        let v = if d < s * 0.3 {
            215
        } else if x > icon_size * 3 / 4 {
            15
        } else {
            85
        };
        Rgba([v, 255 - v / 2, v / 3, 255])
    });
    let mut frame = frame;
    image::imageops::replace(&mut frame, &icon, at.0 as i64, at.1 as i64);
    (frame, icon)
}

fn demo_template_matching() {
    line("Template matching: finding a planted icon in a 1024x640 frame");
    let planted_at = (611, 227);
    let (frame, icon) = frame_with_icon(40, planted_at);
    let template = Template::from_image(&icon, Channel::Luma).expect("icon has structure");
    let whole = Rect {
        x: 0,
        y: 0,
        w: frame.width(),
        h: frame.height(),
    };

    let started = std::time::Instant::now();
    let found = template::find_best(&frame, whole, &template).expect("icon should be found");
    let elapsed = started.elapsed();

    println!(
        "  planted icon at {planted_at:?}, found match at ({}, {}) size {}x{}, score {:.4}, {:?} to search",
        found.bounds.x, found.bounds.y, found.bounds.w, found.bounds.h, found.score, elapsed
    );
    assert_eq!(
        (found.bounds.x, found.bounds.y),
        planted_at,
        "match position should land exactly on the planted icon"
    );
    assert!(
        found.score > 0.99,
        "an exact copy should score almost perfectly"
    );

    // Prove the match is not a fluke: displace the icon slightly and check
    // the reported sub-pixel centre moves with it.
    let (cx1, cy1) = found.centre;
    let (frame2, _) = frame_with_icon(40, (planted_at.0 + 5, planted_at.1 + 3));
    let found2 = template::find_best(&frame2, whole, &template).expect("shifted icon found");
    let (cx2, cy2) = found2.centre;
    println!(
        "  shifted icon by (+5, +3) -> centre moved by ({:.1}, {:.1})",
        cx2 - cx1,
        cy2 - cy1
    );
    assert!((cx2 - cx1 - 5.0).abs() < 1.0 && (cy2 - cy1 - 3.0).abs() < 1.0);
    println!("  PASS: match position is exact, and tracks a real displacement precisely.");
}

fn demo_motion_detection() {
    line("Motion: a moving square is detected and its position tracked frame to frame");
    let (w, h) = (240u32, 160u32);
    let background = Rgba([20, 20, 20, 255]);
    let draw_square = |image: &mut RgbaImage, cx: i32, cy: i32| {
        for y in (cy - 8).max(0)..(cy + 8).min(h as i32) {
            for x in (cx - 8).max(0)..(cx + 8).min(w as i32) {
                image.put_pixel(x as u32, y as u32, Rgba([210, 140, 40, 255]));
            }
        }
    };
    let mut detector = MotionDetector::new(MotionConfig::default());
    let steps = 20;
    let mut last_x = None;
    let mut still_moving_reports = 0;
    for i in 0..steps {
        let cx = 20 + i * 8;
        let mut frame = RgbaImage::from_pixel(w, h, background);
        draw_square(&mut frame, cx, (h / 2) as i32);
        let detection = detector.detect(&frame);
        if let Some(blobs) = detection.value.as_ref()
            && !blobs.is_empty()
        {
            still_moving_reports += 1;
            last_x = Some(blobs[0].bounds.center().0);
        }
    }
    let final_x = last_x.expect("the moving square should have been detected at least once");
    println!(
        "  square walked from x=20 to x={}; last reported centre x={:.1}; motion reported on {}/{} frames after warm-up",
        20 + (steps - 1) * 8,
        final_x,
        still_moving_reports,
        steps - 1
    );
    assert!(
        still_moving_reports >= steps - 3,
        "motion should be reported on nearly every frame once the square starts moving"
    );
    assert!(
        (final_x - (20.0 + (steps - 1) as f32 * 8.0)).abs() < 20.0,
        "the reported position should track near the square's real final position"
    );
    println!("  PASS: motion detection followed the square's real, continuous position.");
}

fn demo_tracking_identity() {
    line("Tracking: two paths that cross do not swap identities");
    // Feed the tracker raw detections directly (bypassing pixel differencing)
    // so this isolates exactly what the Hungarian-assignment rewrite fixed:
    // greedy nearest-neighbour matching could let an early track steal the
    // wrong detection near a crossing, silently swapping which real-world
    // object each ID refers to.
    let mut tracker = syrup::tracking::ObjectTracker::new(60.0, 5);
    let steps = 21;
    let (mut a_id, mut b_id) = (None, None);
    for i in 0..steps {
        let t = i as f32 / (steps - 1) as f32;
        // A: left-to-right along y=40. B: right-to-left along y=42, just 2px
        // away — close enough that greedy matching by nearest-detection can
        // grab the wrong one right as the paths cross in the middle.
        let ax = 10.0 + t * 180.0;
        let bx = 190.0 - t * 180.0;
        let detections = [(ax, 40.0, 16.0, 16.0), (bx, 42.0, 16.0, 16.0)];
        let tracks = tracker.update(&detections);
        // Identify each real object by its detection's x position (nearest
        // track to that detection), not by array index, since the tracker
        // is free to report tracks in any order.
        let nearest = |x: f32, y: f32| {
            tracks
                .iter()
                .min_by(|p, q| {
                    let dp = (p.position.x - x).powi(2) + (p.position.y - y).powi(2);
                    let dq = (q.position.x - x).powi(2) + (q.position.y - y).powi(2);
                    dp.total_cmp(&dq)
                })
                .map(|t| t.id)
        };
        let (found_a, found_b) = (nearest(ax, 40.0), nearest(bx, 42.0));
        if i == 0 {
            a_id = found_a;
            b_id = found_b;
            println!("  frame 0: object A assigned id {a_id:?}, object B assigned id {b_id:?}");
        }
        if i == steps / 2 {
            println!(
                "  frame {i} (paths crossing, x_a={ax:.0}, x_b={bx:.0}): A is still id {found_a:?}, B is still id {found_b:?}"
            );
        }
        assert_eq!(found_a, a_id, "object A's identity swapped at step {i}");
        assert_eq!(found_b, b_id, "object B's identity swapped at step {i}");
    }
    println!(
        "  frame {}: object A ended as id {a_id:?}, object B ended as id {b_id:?}",
        steps - 1
    );
    println!("  PASS: neither object's identity swapped while their paths crossed.");
}

fn demo_glyph_reading() {
    line("Glyph reading: learn a pixel font from one crop, read a new value");
    // Render digits with the library's own debug bitmap font, so the whole
    // pipeline (draw -> learn -> read) runs without any external asset.
    let render = |text: &str| -> (RgbaImage, Rect) {
        let scale = 3;
        let w = syrup::draw::text_width(text, scale) + 16;
        let h = syrup::draw::text_height(scale) + 16;
        let mut image = RgbaImage::from_pixel(w, h, Rgba([15, 15, 15, 255]));
        syrup::draw::draw_text(text, 8, 8, scale, |x, y| {
            if x >= 0 && y >= 0 && (x as u32) < w && (y as u32) < h {
                image.put_pixel(x as u32, y as u32, Rgba([235, 235, 235, 255]));
            }
        });
        let region = Rect { x: 0, y: 0, w, h };
        (image, region)
    };

    let mut font = GlyphSet::new(GlyphOptions {
        channel: Channel::Luma,
        polarity: syrup::threshold::Polarity::LightText,
        ..GlyphOptions::default()
    });
    // Learn every digit and the slash from one rendered line — learn()
    // accumulates samples per character across calls, so a single crop
    // covering the whole alphabet the reader will need is enough.
    let (labelled_image, labelled_region) = render("0123456789/9876543210");
    font.learn(&labelled_image, labelled_region, "0123456789/9876543210")
        .expect("learning from a clean rendered crop should succeed");

    for value in ["42/100", "907/907", "1351/1351"] {
        let (image, region) = render(value);
        let reading = font.read(&image, region);
        match &reading.value {
            Some(text_reading) => {
                println!(
                    "  rendered {value:?} -> read {:?} (confidence {:.2})",
                    text_reading.text,
                    reading.confidence.value()
                );
                assert_eq!(
                    text_reading.text, value,
                    "read value must match what was rendered"
                );
            }
            None => panic!(
                "expected to read {value:?}, got failure: {:?}",
                reading.failure_reason
            ),
        }
    }
    println!("  PASS: every rendered value was read back exactly.");
}

fn main() {
    println!("syrup demo -- exercising the primitives against synthetic frames");
    demo_bar_fill();
    demo_template_matching();
    demo_motion_detection();
    demo_tracking_identity();
    demo_glyph_reading();
    println!("\nAll demos passed: this is real, running code, not a description of intent.");
}
