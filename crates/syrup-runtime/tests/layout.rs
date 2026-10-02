mod common;

use common::*;
use image::{Rgba, RgbaImage};
use syrup::geometry::{Rect, dominant_color_bucket, find_text_block, find_uniform_color_panel};
use syrup_runtime::{BoxF, FindResult, ImageInput, RunParams, Runtime};

fn run(runtime: &Runtime, name: &str, image: &RgbaImage) -> FindResult {
    runtime
        .resolve(name)
        .and_then(|op| op.run(&ImageInput::from_rgba(image), &RunParams::default()))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn boxed(r: Rect) -> BoxF {
    BoxF {
        x: r.x as f32,
        y: r.y as f32,
        w: r.w as f32,
        h: r.h as f32,
    }
}

#[test]
fn text_blocks_are_what_the_core_finds() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let page = image::open(fixture_path("words.png")).unwrap().to_rgba8();
    let (w, h) = page.dimensions();

    let whole = find_text_block(&page, Rect { x: 0, y: 0, w, h }).unwrap();
    let found = run(&runtime, "find_text_blocks", &page);
    assert_eq!(found.items.len(), 1);
    let clipped = Rect {
        w: whole.w.min(w - whole.x),
        h: whole.h.min(h - whole.y),
        ..whole
    };
    assert_eq!(found.items[0].bbox, boxed(clipped));
    assert!(found.items[0].confidence > 0.0);

    let half = Rect {
        x: 0,
        y: h / 2,
        w,
        h: h - h / 2,
    };
    let lower = find_text_block(&page, half).unwrap();
    let found = run(&runtime, "find_text_blocks_in_bottom_half", &page);
    // The core pads the block past the region's edge; results are clipped to it.
    assert_eq!(found.items[0].bbox.y, lower.y.max(half.y) as f32);
    assert_eq!(found.items[0].bbox.x, lower.x as f32);

    let blank = RgbaImage::from_pixel(80, 40, Rgba([20, 20, 20, 255]));
    assert!(run(&runtime, "find_text_blocks", &blank).items.is_empty());
}

#[test]
fn panels_are_what_the_core_finds() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    // A dark screen whose bottom half is mostly a grey panel with a red stripe.
    let mut screen = RgbaImage::from_pixel(400, 300, Rgba([20, 22, 26, 255]));
    for y in 170..290 {
        for x in 30..370 {
            let stripe = (200..210).contains(&y);
            let pixel = if stripe {
                Rgba([200, 40, 40, 255])
            } else {
                Rgba([150, 150, 156, 255])
            };
            screen.put_pixel(x, y, pixel);
        }
    }
    let lower = Rect {
        x: 0,
        y: 150,
        w: 400,
        h: 150,
    };
    let bucket = dominant_color_bucket(&screen, lower, 32).unwrap();
    let panel = find_uniform_color_panel(&screen, lower, bucket, 32).unwrap();

    let found = run(&runtime, "find_panels_in_bottom_half", &screen);
    assert_eq!(found.items.len(), 1);
    assert_eq!(found.items[0].bbox, boxed(panel));
    assert_eq!(
        (panel.x, panel.w),
        (30, 340),
        "the grey panel, not the background"
    );
    assert!(found.items[0].confidence > 0.9);

    let tiny = RgbaImage::from_pixel(10, 10, Rgba([0, 0, 0, 255]));
    assert!(run(&runtime, "find_panels", &tiny).items.is_empty());
}
