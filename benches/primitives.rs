//! Benchmarks for the primitives that run per frame in real consumers:
//! bar-fill measurement, motion detection, and uniform-panel search.

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};
use image::{Rgba, RgbaImage};

use framelens::color::is_color_pixel;
use framelens::geometry::{self, Rect};
use framelens::motion::{MotionConfig, MotionDetector};

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

criterion_group!(
    benches,
    bench_measure_bar_fill,
    bench_find_color_bar,
    bench_motion_detect,
    bench_uniform_panel
);
criterion_main!(benches);
