mod common;

use common::*;
use image::{Rgb, RgbImage};
use qrcode::{Color, QrCode};
use syrup_runtime::{FindResult, ImageInput, RunParams, Runtime};

/// Draws `text` as a QR code with 6-pixel modules at (x, y).
fn paint(page: &mut RgbImage, text: &str, x: u32, y: u32) {
    let code = QrCode::new(text.as_bytes()).unwrap();
    let width = code.width() as u32;
    for (i, color) in code.to_colors().into_iter().enumerate() {
        if color == Color::Dark {
            let (mx, my) = (i as u32 % width, i as u32 / width);
            for dy in 0..6 {
                for dx in 0..6 {
                    page.put_pixel(x + mx * 6 + dx, y + my * 6 + dy, Rgb([0, 0, 0]));
                }
            }
        }
    }
}

fn page() -> RgbImage {
    let mut page = RgbImage::from_pixel(640, 480, Rgb([255, 255, 255]));
    paint(&mut page, "top", 250, 30);
    paint(&mut page, "left", 40, 300);
    paint(&mut page, "a longer right one", 420, 290);
    page
}

fn run(runtime: &Runtime, name: &str, image: &RgbImage) -> FindResult {
    runtime
        .resolve(name)
        .and_then(|op| op.run(&ImageInput::from_rgb(image), &RunParams::default()))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn qr_codes_are_found_read_and_composed() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let page = page();
    let texts = |result: FindResult| -> Vec<String> {
        result.items.into_iter().map(|f| f.text.unwrap()).collect()
    };

    assert_eq!(
        texts(run(&runtime, "find_qr_codes_left_to_right", &page)),
        ["left", "top", "a longer right one"]
    );
    let lower = run(
        &runtime,
        "find_qr_codes_in_bottom_half_left_to_right",
        &page,
    );
    assert_eq!(lower.items.len(), 2);
    let left = &lower.items[0];
    assert!((40.0..44.0).contains(&left.bbox.x) && (300.0..304.0).contains(&left.bbox.y));
    assert_eq!(left.keypoints[0].name, "top_left");
    assert_eq!(
        texts(run(&runtime, "find_largest_qr_code", &page)),
        ["a longer right one"]
    );
    assert!(
        run(&runtime, "find_qr_codes", &RgbImage::new(64, 64))
            .items
            .is_empty()
    );
}
