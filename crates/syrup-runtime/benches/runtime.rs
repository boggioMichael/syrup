//! What using an operation costs: compiling it the first time, running it
//! on a frame, and running a session over a sequence.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use image::{Rgba, RgbaImage, imageops};
use qrcode::{Color, QrCode};
use syrup_runtime::providers::ModelSource;
use syrup_runtime::{Config, ImageInput, Mode, RunParams, Runtime, SessionOptions};

struct TempDir(PathBuf);

impl TempDir {
    fn new() -> TempDir {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "syrup-bench-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn runtime(cache: &TempDir) -> Runtime {
    Runtime::new(Config {
        cache_dir: cache.0.clone(),
        mode: Mode::Development,
        rustc: None,
        compile_timeout: Duration::from_secs(300),
        face_model: ModelSource::Bundled,
    })
}

fn fixture(name: &str) -> RgbaImage {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    image::open(path).unwrap().to_rgba8()
}

/// A 1280x720 screen: the astronaut photo, a status bar in a track, a grey
/// panel with text, and a QR code.
fn screen() -> RgbaImage {
    let mut frame = RgbaImage::from_pixel(1280, 720, Rgba([24, 26, 30, 255]));
    imageops::overlay(&mut frame, &fixture("astronaut.jpg"), 40, 40);
    imageops::overlay(&mut frame, &fixture("words.png"), 600, 60);
    for y in 600..700 {
        for x in 600..1240 {
            frame.put_pixel(x, y, Rgba([150, 150, 156, 255]));
        }
    }
    for y in 660..672 {
        for x in 640..1200 {
            let filled = x < 640 + 336;
            let pixel = if filled {
                Rgba([210, 40, 40, 255])
            } else {
                Rgba([70, 70, 78, 255])
            };
            frame.put_pixel(x, y, pixel);
        }
    }
    // A QR code with 6 px modules on a white quiet zone.
    let code = QrCode::new(b"syrup").unwrap();
    let width = code.width() as u32;
    for y in 276..300 + width * 6 + 24 {
        for x in 976..1000 + width * 6 + 24 {
            frame.put_pixel(x, y, Rgba([255, 255, 255, 255]));
        }
    }
    for (i, color) in code.to_colors().into_iter().enumerate() {
        if color == Color::Dark {
            let (mx, my) = (i as u32 % width, i as u32 / width);
            for d in 0..36 {
                frame.put_pixel(
                    1000 + mx * 6 + d % 6,
                    300 + my * 6 + d / 6,
                    Rgba([0, 0, 0, 255]),
                );
            }
        }
    }
    frame
}

fn compile(c: &mut Criterion) {
    let mut group = c.benchmark_group("compile");
    group.sample_size(10);
    for name in [
        "find_face",
        "find_red_bars_in_bottom_third",
        "measure_fill_of_largest_red_bar",
    ] {
        group.bench_function(format!("{name} cold"), |b| {
            b.iter_batched(
                TempDir::new,
                |cache| runtime(&cache).resolve(name).unwrap().prepare().unwrap(),
                BatchSize::PerIteration,
            )
        });
    }
    let cache = TempDir::new();
    let op = runtime(&cache)
        .resolve("find_2_largest_faces_in_top_half")
        .unwrap();
    group.bench_function("generate source", |b| b.iter(|| op.source()));
    group.finish();
}

fn frame(c: &mut Criterion) {
    let cache = TempDir::new();
    let runtime = runtime(&cache);
    let screen = screen();
    let input = ImageInput::from_rgba(&screen);
    let mut group = c.benchmark_group("frame 1280x720");
    for name in [
        "find_face",
        "find_red_bars",
        "find_red_regions",
        "find_qr_codes",
        "find_text_blocks",
        "find_panels_in_bottom_third",
        "measure_sharpness",
        "measure_fill_of_red_bars",
    ] {
        let op = runtime.resolve(name).unwrap();
        let found = op.run(&input, &RunParams::default()).unwrap();
        assert!(
            !found.items.is_empty(),
            "{name} finds nothing in the bench screen"
        );
        group.bench_function(name, |b| {
            b.iter(|| op.run(&input, &RunParams::default()).unwrap())
        });
    }
    group.finish();
}

fn session(c: &mut Criterion) {
    let cache = TempDir::new();
    let runtime = runtime(&cache);
    let screen = screen();
    // The screen panning left 4 px a frame.
    let frames: Vec<RgbaImage> = (0..16)
        .map(|i| imageops::crop_imm(&screen, 4 * i, 0, 1200, 720).to_image())
        .collect();
    let mut group = c.benchmark_group("session 1200x720 per frame");
    for name in ["track_moving_regions", "track_faces", "track_red_bars"] {
        let op = runtime.resolve(name).unwrap();
        group.bench_function(name, |b| {
            let mut session = op.session(SessionOptions::default()).unwrap();
            let mut i = 0;
            b.iter(|| {
                let frame = &frames[i % frames.len()];
                i += 1;
                session
                    .update(&ImageInput::from_rgba(frame), &RunParams::default())
                    .unwrap()
            })
        });
    }
    group.finish();
}

criterion_group!(benches, compile, frame, session);
criterion_main!(benches);
