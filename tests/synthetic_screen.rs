//! Integration tests composing the primitives the way a real consumer does:
//! locate a status bar on a synthetic screen, measure its fill, and follow a
//! moving object across a frame sequence.

use image::{Rgba, RgbaImage};

use syrup::color::is_color_pixel;
use syrup::geometry::{Rect, find_color_bar, measure_bar_fill};
use syrup::motion::{MotionConfig, MotionDetector};

/// A synthetic 800x600 "application screen": dark background, a bottom
/// status band holding a red bar at a known fill ratio in a visible grey
/// track (a track indistinguishable from the background would be unreadable
/// to a person too), and a bright square that moves between frames.
fn screen(square_x: u32, bar_fill: f32) -> RgbaImage {
    let mut image = RgbaImage::from_pixel(800, 600, Rgba([28, 30, 34, 255]));

    // Status bar: track from x=60 to x=360, rows 570..580.
    let track_w = 300u32;
    let filled = (track_w as f32 * bar_fill) as u32;
    for y in 570..580 {
        for x in 60..60 + track_w {
            let pixel = if x < 60 + filled {
                Rgba([210, 40, 40, 255])
            } else {
                Rgba([70, 70, 78, 255])
            };
            image.put_pixel(x, y, pixel);
        }
    }

    // Moving square.
    for y in 200..240 {
        for x in square_x..square_x + 40 {
            image.put_pixel(x, y, Rgba([230, 230, 230, 255]));
        }
    }

    image
}

fn red(pixel: &Rgba<u8>) -> bool {
    is_color_pixel(pixel, (340.0, 30.0), 0.35, 0.30)
}

#[test]
fn bar_is_located_and_measured_on_a_synthetic_screen() {
    let image = screen(100, 0.4);
    let band = Rect {
        x: 0,
        y: 540,
        w: 800,
        h: 60,
    };

    let bar = find_color_bar(&image, band, (340.0, 30.0), 0.35, 0.30)
        .expect("the red bar should be found in the status band");
    assert!(bar.y >= 565 && bar.y <= 575, "bar found at y={}", bar.y);

    let percent =
        measure_bar_fill(&image, bar, band, red).expect("the bar's track should be measurable");
    assert!(
        (percent - 40.0).abs() <= 2.0,
        "a 40% bar should measure ~40%, got {percent}%"
    );
}

#[test]
fn a_moving_object_keeps_its_identity_across_the_sequence() {
    let mut detector = MotionDetector::new(MotionConfig::default());

    detector.detect(&screen(100, 0.4)); // warm-up
    let mut last_id = None;
    for step in 1..5u32 {
        let detection = detector.detect(&screen(100 + step * 15, 0.4));
        let blobs = detection.value.expect("motion should be detected");
        assert!(!blobs.is_empty(), "step {step} lost the moving square");
        // The largest blob is the square (or its enter+leave region).
        let main = blobs.iter().max_by_key(|b| b.bounds.area()).unwrap();
        if let Some(previous) = last_id {
            assert_eq!(
                main.id, previous,
                "the square's track id changed at step {step}"
            );
        }
        last_id = Some(main.id);
    }
}

#[test]
fn a_static_screen_reports_no_motion_while_the_bar_stays_measurable() {
    let mut detector = MotionDetector::new(MotionConfig::default());
    let image = screen(100, 0.75);

    detector.detect(&image);
    let motion = detector.detect(&image);
    assert!(motion.is_present());
    assert!(motion.value.unwrap().is_empty(), "nothing moved");

    let band = Rect {
        x: 0,
        y: 540,
        w: 800,
        h: 60,
    };
    let bar = find_color_bar(&image, band, (340.0, 30.0), 0.35, 0.30).unwrap();
    let percent = measure_bar_fill(&image, bar, band, red).unwrap();
    assert!((percent - 75.0).abs() <= 2.0, "got {percent}%");
}
