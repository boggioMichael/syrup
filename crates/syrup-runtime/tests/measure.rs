mod common;

use common::*;
use image::{Rgba, RgbaImage, imageops};
use syrup::color::is_color_pixel;
use syrup::geometry::{Rect, find_color_bar, measure_bar_fill};
use syrup::quality::assess_text_quality;
use syrup_runtime::{FindResult, ImageInput, RunParams, Runtime};

/// Two red bars in grey tracks: 40% full at the bottom, 75% near the top.
fn screen() -> RgbaImage {
    let mut image = RgbaImage::from_pixel(800, 600, Rgba([28, 30, 34, 255]));
    for (y0, fill) in [(570, 120), (40, 225)] {
        for y in y0..y0 + 10 {
            for x in 60..360 {
                let pixel = if x < 60 + fill {
                    Rgba([210, 40, 40, 255])
                } else {
                    Rgba([70, 70, 78, 255])
                };
                image.put_pixel(x, y, pixel);
            }
        }
    }
    image
}

fn run(runtime: &Runtime, name: &str, image: &RgbaImage) -> FindResult {
    runtime
        .resolve(name)
        .and_then(|op| op.run(&ImageInput::from_rgba(image), &RunParams::default()))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn bar_fill_is_what_the_core_measures() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let image = screen();
    let red = |p: &Rgba<u8>| is_color_pixel(p, (340.0, 20.0), 0.35, 0.30);
    let whole = Rect {
        x: 0,
        y: 0,
        w: 800,
        h: 600,
    };
    let lower = Rect {
        x: 0,
        y: 400,
        w: 800,
        h: 200,
    };
    let bar = find_color_bar(&image, lower, (340.0, 20.0), 0.35, 0.30).unwrap();
    let expected = measure_bar_fill(&image, bar, whole, red).unwrap() / 100.0;

    let found = run(&runtime, "measure_fill_of_red_bars_in_bottom_third", &image);
    assert_eq!(found.items.len(), 1);
    assert_eq!(found.items[0].value, Some(expected));
    assert!((expected - 0.40).abs() < 0.02, "{expected}");

    let both = run(&runtime, "measure_fill_of_red_bars_top_to_bottom", &image);
    let values: Vec<f32> = both.items.iter().map(|f| f.value.unwrap()).collect();
    assert_eq!(values.len(), 2);
    assert!((values[0] - 0.75).abs() < 0.02 && (values[1] - 0.40).abs() < 0.02);

    assert!(
        run(&runtime, "find_red_bars", &image).items[0]
            .value
            .is_none()
    );
}

#[test]
fn sharpness_tells_crisp_text_from_blurred() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let crisp = image::open(fixture_path("words.png")).unwrap().to_rgba8();
    let blurred = imageops::blur(&crisp, 2.0);
    let all = Rect {
        x: 0,
        y: 0,
        w: crisp.width(),
        h: crisp.height(),
    };

    let sharp = run(&runtime, "measure_sharpness", &crisp).items;
    assert_eq!(sharp.len(), 1);
    assert_eq!(
        (sharp[0].label, sharp[0].bbox.w),
        ("image", crisp.width() as f32)
    );
    assert_eq!(
        sharp[0].value,
        Some(assess_text_quality(&crisp, all).sharpness)
    );
    let soft = run(&runtime, "measure_sharpness", &blurred).items[0]
        .value
        .unwrap();
    assert!(
        sharp[0].value.unwrap() >= 0.35 && soft < 0.35,
        "crisp {:?}, blurred {soft}",
        sharp[0].value
    );

    let flat = RgbaImage::from_pixel(64, 64, Rgba([90, 90, 90, 255]));
    assert!(
        run(&runtime, "measure_sharpness", &flat).items.is_empty(),
        "no edges: nothing to measure, so nothing is reported"
    );
    let half = run(&runtime, "measure_sharpness_in_left_half", &crisp).items;
    assert_eq!(half[0].bbox.w, (crisp.width() as f32 / 2.0).round());
}

#[test]
fn fill_is_measured_within_the_searched_region() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    // A track across the whole width, red for its first 100 columns.
    let mut image = RgbaImage::from_pixel(400, 100, Rgba([28, 30, 34, 255]));
    for y in 50..60 {
        for x in 20..380 {
            let pixel = if x < 120 {
                Rgba([210, 40, 40, 255])
            } else {
                Rgba([70, 70, 78, 255])
            };
            image.put_pixel(x, y, pixel);
        }
    }
    let whole = run(&runtime, "measure_fill_of_red_bars", &image).items[0]
        .value
        .unwrap();
    let left = run(&runtime, "measure_fill_of_red_bars_in_left_half", &image).items[0]
        .value
        .unwrap();
    // 100 of 360 track columns in the image; 100 of the 180 in its left half.
    assert!((whole - 100.0 / 360.0).abs() < 0.02, "{whole}");
    assert!((left - 100.0 / 180.0).abs() < 0.02, "{left}");
}
