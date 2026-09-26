//! Benchmarks for the primitives that run per frame in real consumers:
//! bar-fill measurement, motion detection, uniform-panel search, tracking,
//! connected components, text evidence, and glyph reading.

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use image::{Rgba, RgbaImage};

use syrup::color::is_color_pixel;
use syrup::components::{self, Connectivity};
use syrup::draw;
use syrup::geometry::{self, Rect};
use syrup::glyphs::{GlyphOptions, GlyphSet};
use syrup::motion::{MotionConfig, MotionDetector};
use syrup::threshold::{self, Channel, Polarity};
use syrup::tracking::ObjectTracker;

/// A 1366x768 frame with a 60%-filled red bar in a dark groove near the
/// bottom — the shape of a status readout at a common screen resolution.
fn frame_with_bar() -> (RgbaImage, Rect) {
    let mut image = RgbaImage::from_pixel(1366, 768, Rgba([30, 30, 35, 255]));
    let (x0, y0, w, h) = (140u32, 730u32, 220u32, 10u32);
    let filled = (w as f32 * 0.6) as u32;
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let pixel = if x < x0 + filled {
                Rgba([220, 40, 40, 255])
            } else {
                Rgba([16, 16, 16, 255])
            };
            image.put_pixel(x, y, pixel);
        }
    }
    let fill = Rect {
        x: x0,
        y: y0,
        w: filled,
        h,
    };
    (image, fill)
}

fn bench_measure_bar_fill(c: &mut Criterion) {
    let (image, fill) = frame_with_bar();
    let search = Rect {
        x: 0,
        y: 700,
        w: 1366,
        h: 68,
    };
    c.bench_function("measure_bar_fill 1366x768", |b| {
        b.iter(|| {
            geometry::measure_bar_fill(black_box(&image), fill, search, |p| {
                is_color_pixel(p, (340.0, 30.0), 0.35, 0.30)
            })
        })
    });
}

fn bench_find_color_bar(c: &mut Criterion) {
    let (image, _) = frame_with_bar();
    let region = Rect {
        x: 0,
        y: 690,
        w: 1366,
        h: 78,
    };
    c.bench_function("find_color_bar in status band", |b| {
        b.iter(|| geometry::find_color_bar(black_box(&image), region, (340.0, 30.0), 0.35, 0.30))
    });
}

/// Motion detection over two full frames with a moving square, including
/// mask computation, blob extraction, and tracker update.
fn bench_motion_detect(c: &mut Criterion) {
    fn frame(at: u32) -> RgbaImage {
        let mut image = RgbaImage::from_pixel(1366, 768, Rgba([30, 30, 35, 255]));
        for y in 300..340 {
            for x in at..at + 40 {
                image.put_pixel(x, y, Rgba([220, 220, 220, 255]));
            }
        }
        image
    }
    let first = frame(200);
    let second = frame(240);

    // The detector is stateful (it keeps the previous frame), so each
    // iteration gets a fresh, pre-warmed detector from the setup closure.
    c.bench_function("motion detect 1366x768", |b| {
        b.iter_batched(
            || {
                let mut detector = MotionDetector::new(MotionConfig::default());
                detector.detect(&first);
                detector
            },
            |mut detector| detector.detect(black_box(&second)),
            BatchSize::PerIteration,
        )
    });
}

fn bench_uniform_panel(c: &mut Criterion) {
    let mut image = RgbaImage::from_pixel(1366, 768, Rgba([30, 30, 35, 255]));
    for y in 20..170 {
        for x in 20..220 {
            image.put_pixel(x, y, Rgba([70, 74, 80, 255]));
        }
    }
    let region = Rect {
        x: 0,
        y: 0,
        w: 455,
        h: 256,
    };
    c.bench_function("dominant color + uniform panel", |b| {
        b.iter(|| {
            let bucket = geometry::dominant_color_bucket(black_box(&image), region, 10)?;
            geometry::find_uniform_color_panel(black_box(&image), region, bucket, 10)
        })
    });
}

/// Motion detection on the worst case for frame differencing: the whole
/// view pans 3 px, so nearly every pixel of a textured scene changes and
/// no row can be skipped.
fn bench_motion_detect_busy(c: &mut Criterion) {
    fn frame(shift: u32) -> RgbaImage {
        RgbaImage::from_fn(1366, 768, |x, y| {
            let u = x + shift;
            // A texture with detail at several scales, like terrain.
            let v = ((u / 7 + y / 5) % 9) * 22 + ((u * 13 + y * 7) % 17) * 3;
            Rgba([v as u8, (v / 2 + 40) as u8, (255 - v) as u8, 255])
        })
    }
    let first = frame(0);
    let second = frame(3);
    c.bench_function("motion detect 1366x768 panning", |b| {
        b.iter_batched(
            || {
                let mut detector = MotionDetector::new(MotionConfig::default());
                detector.detect(&first);
                detector
            },
            |mut detector| detector.detect(black_box(&second)),
            BatchSize::PerIteration,
        )
    });
}

/// Draw `text` in the debug font, light on a dark bar.
fn text_crop(text: &str, scale: u32) -> (RgbaImage, Rect) {
    let (w, h) = (
        draw::text_width(text, scale) + 16,
        draw::text_height(scale) + 8,
    );
    let mut image = RgbaImage::from_pixel(w, h, Rgba([40, 90, 200, 255]));
    draw::draw_text(text, 8, 4, scale, |x, y| {
        image.put_pixel(x as u32, y as u32, Rgba([245, 245, 245, 255]))
    });
    (image, Rect { x: 0, y: 0, w, h })
}

fn bench_text_evidence(c: &mut Criterion) {
    let (image, region) = text_crop("1291/1351", 2);
    c.bench_function("text evidence 124x22 crop", |b| {
        b.iter(|| {
            threshold::text_evidence(
                black_box(&image),
                region,
                Channel::Min,
                Polarity::LightText,
                60,
            )
        })
    });
}

fn bench_glyph_read(c: &mut Criterion) {
    let mut glyphs = GlyphSet::new(GlyphOptions::default());
    // Learned from a whole line, as glyphs are measured against its height.
    let (image, region) = text_crop("0123456789/", 2);
    glyphs
        .learn(&image, region, "0123456789/")
        .expect("learnable");
    let (image, region) = text_crop("1291/1351", 2);
    c.bench_function("glyph read 9 chars", |b| {
        b.iter(|| glyphs.read(black_box(&image), region))
    });

    // The worst case: a smear of ink no split can explain, which makes the
    // reader search every way of cutting it into glyphs.
    let mut smear = RgbaImage::from_pixel(80, 22, Rgba([40, 90, 200, 255]));
    for y in 4..18 {
        for x in 20..60 {
            if (x * 7 + y * 3) % 5 != 0 {
                smear.put_pixel(x, y, Rgba([245, 245, 245, 255]));
            }
        }
    }
    let region = Rect {
        x: 0,
        y: 0,
        w: 80,
        h: 22,
    };
    c.bench_function("glyph read unreadable 40px smear", |b| {
        b.iter(|| glyphs.read(black_box(&smear), region))
    });
}

/// Labelling a full-frame mask holding 400 blobs.
fn bench_components(c: &mut Criterion) {
    let mut mask = image::GrayImage::new(1366, 768);
    for i in 0..400u32 {
        let (x0, y0) = ((i % 25) * 54 + 3, (i / 25) * 47 + 3);
        for y in y0..y0 + 30 {
            for x in x0..x0 + 40 {
                // Ring-shaped blobs, so labels merge across rows.
                if !(x0 + 10..x0 + 30).contains(&x) || !(y0 + 8..y0 + 22).contains(&y) {
                    mask.put_pixel(x, y, image::Luma([255]));
                }
            }
        }
    }
    c.bench_function("label components 1366x768, 400 blobs", |b| {
        b.iter(|| components::label_mask(black_box(&mask), Connectivity::Eight))
    });
}

/// Tracker update with 60 objects close enough to all compete for each
/// other's detections: one big assignment problem, the worst case.
fn bench_tracker_dense(c: &mut Criterion) {
    let row = |shift: f32| -> Vec<(f32, f32, f32, f32)> {
        (0..60)
            .map(|i| (i as f32 * 10.0 + shift, 0.0, 4.0, 4.0))
            .collect()
    };
    let (first, second) = (row(0.0), row(3.0));
    c.bench_function("tracker update 60 crowded objects", |b| {
        b.iter_batched(
            || {
                let mut tracker = ObjectTracker::new(40.0, 2);
                tracker.update(&first);
                tracker
            },
            |mut tracker| {
                tracker.update(black_box(&second));
                tracker
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(
    benches,
    bench_motion_detect_busy,
    bench_text_evidence,
    bench_glyph_read,
    bench_components,
    bench_tracker_dense,
    bench_measure_bar_fill,
    bench_find_color_bar,
    bench_motion_detect,
    bench_uniform_panel
);
criterion_main!(benches);
