mod common;

use common::*;
use syrup_runtime::{BoxF, FindResult, ImageInput, RunParams, Runtime};

fn texts(result: &FindResult) -> Vec<&str> {
    result
        .items
        .iter()
        .map(|f| f.text.as_deref().unwrap())
        .collect()
}

#[test]
fn words_are_read_and_placed_in_image_coordinates() {
    if !syrup::ocr::is_ocr_available() {
        eprintln!("skipped: Tesseract is not installed");
        return;
    }
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let image = image::open(fixture_path("words.png")).unwrap().to_rgb8();
    let run = |name: &str| {
        runtime
            .resolve(name)
            .and_then(|op| op.run(&ImageInput::from_rgb(&image), &RunParams::default()))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
    };

    let all = run("find_words");
    let mut read = texts(&all);
    read.sort();
    assert_eq!(read, ["HELLO", "SYRUP", "WORLD"]);
    let syrup = all
        .items
        .iter()
        .find(|f| f.text.as_deref() == Some("SYRUP"))
        .unwrap();
    // Where Pillow drew it.
    let drawn = BoxF {
        x: 40.0,
        y: 45.0,
        w: 173.0,
        h: 41.0,
    };
    assert!(iou(&syrup.bbox, &drawn) > 0.8, "{:?}", syrup.bbox);
    assert!(syrup.confidence > 0.9);

    let lower = run("find_words_in_bottom_half_left_to_right");
    assert_eq!(texts(&lower), ["HELLO", "WORLD"]);
    assert!(lower.items.iter().all(|f| f.bbox.y >= 150.0), "{lower:?}");

    assert_eq!(texts(&run("find_largest_word")), ["WORLD"]);
}
