mod common;

use std::process::Command;

use common::*;
use image::{Rgba, RgbaImage};
use syrup_runtime::frames::{FrameSource, ImageFiles, WindowCapture};
use syrup_runtime::{ErrorKind, RunParams, Runtime, SessionOptions};

/// Frames of a square moving 6 px right each time, saved as PNGs.
fn frames(dir: &TempDir, count: u32) -> Vec<std::path::PathBuf> {
    (0..count)
        .map(|i| {
            let mut frame = RgbaImage::from_pixel(200, 120, Rgba([20, 20, 24, 255]));
            for y in 50..70 {
                for x in 10 + 6 * i..30 + 6 * i {
                    frame.put_pixel(x, y, Rgba([230, 230, 230, 255]));
                }
            }
            let path = dir.0.join(format!("frame{i}.png"));
            frame.save(&path).unwrap();
            path
        })
        .collect()
}

#[test]
fn sessions_read_frames_from_a_source_until_it_ends() {
    let (cache, dir) = (TempDir::new(), TempDir::new());
    let runtime = Runtime::new(config(&cache.0));
    let mut session = runtime
        .resolve("track_moving_regions")
        .unwrap()
        .session(SessionOptions::default())
        .unwrap();
    let mut source = ImageFiles::new(frames(&dir, 4));
    let mut counts = vec![];
    while let Some(result) = session.next(&mut source, &RunParams::default()).unwrap() {
        counts.push(result.items.len());
    }
    assert_eq!(counts, [0, 2, 2, 2]);
    assert!(source.next_frame().unwrap().is_none());

    let mut missing = ImageFiles::new([dir.0.join("nope.png")]);
    let e = session
        .next(&mut missing, &RunParams::default())
        .unwrap_err();
    assert_eq!(e.kind, ErrorKind::BadImage);
}

#[test]
fn the_cli_watches_frames_one_json_line_each() {
    let (cache, dir) = (TempDir::new(), TempDir::new());
    let paths = frames(&dir, 3);
    let mut args = vec!["watch".to_string(), "track_moving_regions".to_string()];
    args.extend(paths.iter().map(|p| p.display().to_string()));
    let output = Command::new(env!("CARGO_BIN_EXE_syrup"))
        .args(&args)
        .env("SYRUP_CACHE_DIR", &cache.0)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines: Vec<serde_json::Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[2]["provenance"]["frame"], 3);
}

/// CI opens a window on each system and names it in SYRUP_TEST_WINDOW.
#[test]
fn windows_are_captured_into_sessions() {
    let Ok(title) = std::env::var("SYRUP_TEST_WINDOW") else {
        return;
    };
    let missing = WindowCapture::new("no window is called this 7f3a").err();
    assert_eq!(missing.map(|e| e.kind), Some(ErrorKind::BadParameter));
    let listed = WindowCapture::windows().unwrap();
    assert!(listed.iter().any(|t| t.contains(&title)), "{listed:?}");

    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let mut session = runtime
        .resolve("track_moving_regions")
        .unwrap()
        .session(SessionOptions::default())
        .unwrap();
    let mut window = WindowCapture::new(&title).unwrap();
    assert!(window.title().contains(&title));
    for frame in 1..=2 {
        let result = session
            .next(&mut window, &RunParams::default())
            .unwrap()
            .expect("the window is still open");
        assert_eq!(result.provenance.frame, Some(frame));
        assert!(result.provenance.image.0 > 0 && result.provenance.image.1 > 0);
    }
}
