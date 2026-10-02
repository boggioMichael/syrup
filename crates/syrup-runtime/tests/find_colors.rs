mod common;

use common::*;
use image::{Rgba, RgbaImage};
use syrup::geometry::{Rect, find_color_bar, find_color_regions};
use syrup_runtime::{BoxF, FindResult, ImageInput, RunParams, Runtime};

/// The synthetic screen from the core's own integration tests: a red bar,
/// 40% full, in a grey track at the bottom of a dark 800x600 frame, plus
/// a bright square.
fn screen() -> RgbaImage {
    let mut image = RgbaImage::from_pixel(800, 600, Rgba([28, 30, 34, 255]));
    for y in 570..580 {
        for x in 60..360 {
            let filled = x < 60 + 120;
            image.put_pixel(
                x,
                y,
                if filled {
                    Rgba([210, 40, 40, 255])
                } else {
                    Rgba([70, 70, 78, 255])
                },
            );
        }
    }
    for y in 200..240 {
        for x in 100..140 {
            image.put_pixel(x, y, Rgba([230, 230, 230, 255]));
        }
    }
    image
}

fn boxed(r: Rect) -> BoxF {
    BoxF {
        x: r.x as f32,
        y: r.y as f32,
        w: r.w as f32,
        h: r.h as f32,
    }
}

fn run(runtime: &Runtime, name: &str, image: &RgbaImage) -> FindResult {
    runtime
        .resolve(name)
        .and_then(|op| op.run(&ImageInput::from_rgba(image), &RunParams::default()))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[test]
fn the_new_path_finds_what_the_core_primitives_find() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let image = screen();
    let whole = Rect {
        x: 0,
        y: 0,
        w: 800,
        h: 600,
    };
    let band = Rect {
        x: 0,
        y: 400,
        w: 800,
        h: 200,
    };

    let bar = find_color_bar(&image, band, (340.0, 20.0), 0.35, 0.30).unwrap();
    let found = run(&runtime, "find_red_bars_in_bottom_third", &image);
    assert_eq!(found.items.len(), 1);
    assert_eq!(found.items[0].bbox, boxed(bar));
    assert_eq!(found.items[0].confidence, 1.0, "the bar is solid red");

    let regions = find_color_regions(&image, whole, (340.0, 20.0), 0.35, 0.30);
    let found = run(&runtime, "find_red_regions", &image);
    assert_eq!(
        found.items.iter().map(|f| f.bbox).collect::<Vec<_>>(),
        regions.into_iter().map(boxed).collect::<Vec<_>>()
    );

    assert!(run(&runtime, "find_blue_regions", &image).items.is_empty());
    assert!(
        run(&runtime, "find_red_regions_in_top_half", &image)
            .items
            .is_empty()
    );
}

#[test]
fn regions_are_scored_by_how_much_of_them_has_the_colour() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    // A 40x40 green square with a 10x10 grey hole: 1500 of 1600 pixels.
    let mut image = RgbaImage::from_pixel(100, 100, Rgba([128, 128, 128, 255]));
    for y in 20..60 {
        for x in 20..60 {
            if !((35..45).contains(&x) && (35..45).contains(&y)) {
                image.put_pixel(x, y, Rgba([30, 200, 60, 255]));
            }
        }
    }
    let found = run(&runtime, "find_green_regions", &image);
    assert_eq!(found.items.len(), 1);
    assert_eq!(
        found.items[0].bbox,
        boxed(Rect {
            x: 20,
            y: 20,
            w: 40,
            h: 40
        })
    );
    assert_eq!(found.items[0].confidence, 1500.0 / 1600.0);
}
