#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use syrup_runtime::providers::ModelSource;
use syrup_runtime::{BoxF, Config, Mode};

pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new() -> TempDir {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "syrup-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn config(cache: &Path) -> Config {
    Config {
        cache_dir: cache.to_path_buf(),
        mode: Mode::Development,
        rustc: None,
        compile_timeout: Duration::from_secs(300),
        face_model: ModelSource::Bundled,
    }
}

pub fn fixture_path(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

pub fn fixture(name: &str) -> image::RgbImage {
    image::open(fixture_path(name)).unwrap().to_rgb8()
}

/// Where the astronaut's face is in astronaut.jpg, by eye.
pub const ASTRONAUT_FACE: BoxF = BoxF {
    x: 179.0,
    y: 65.0,
    w: 91.0,
    h: 109.0,
};

pub fn shifted(b: BoxF, dx: f32, dy: f32) -> BoxF {
    BoxF {
        x: b.x + dx,
        y: b.y + dy,
        ..b
    }
}

pub fn iou(a: &BoxF, b: &BoxF) -> f32 {
    let w = (a.x + a.w).min(b.x + b.w) - a.x.max(b.x);
    let h = (a.y + a.h).min(b.y + b.h) - a.y.max(b.y);
    let inter = w.max(0.0) * h.max(0.0);
    inter / (a.w * a.h + b.w * b.h - inter)
}
