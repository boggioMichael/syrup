mod common;

use common::*;
use image::{GrayImage, Rgb, RgbImage};
use syrup_runtime::{ArtifactStatus, BoxF, FindResult, ImageInput, PixelRect, RunParams, Runtime};

fn run(runtime: &Runtime, name: &str, image: &RgbImage, params: RunParams) -> FindResult {
    runtime
        .resolve(name)
        .and_then(|op| op.run(&ImageInput::from_rgb(image), &params))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn boxes(result: &FindResult) -> Vec<BoxF> {
    result.items.iter().map(|f| f.bbox).collect()
}

#[test]
fn finds_the_face_then_reuses_the_compiled_module() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let image = fixture("astronaut.jpg");
    let find_face = runtime.resolve("find_face").unwrap();

    let first = find_face
        .run(&ImageInput::from_rgb(&image), &RunParams::default())
        .unwrap();
    assert_eq!(first.provenance.artifact_status, ArtifactStatus::Compiled);
    assert_eq!(first.items.len(), 1, "{:?}", first.items);
    let face = &first.items[0];
    assert!(iou(&face.bbox, &ASTRONAUT_FACE) > 0.8, "{face:?}");
    assert!(face.confidence > 0.85, "{face:?}");
    let names: Vec<_> = face.keypoints.iter().map(|k| k.name).collect();
    assert_eq!(
        names,
        [
            "right_eye",
            "left_eye",
            "nose_tip",
            "right_mouth_corner",
            "left_mouth_corner"
        ]
    );
    assert!(
        face.keypoints[0].x < face.keypoints[1].x,
        "the subject's right eye is on the image's left"
    );
    let b = face.bbox;
    for k in &face.keypoints {
        assert!(
            (b.x..b.x + b.w).contains(&k.x) && (b.y..b.y + b.h).contains(&k.y),
            "{k:?} outside {b:?}"
        );
    }

    let again = find_face
        .run(&ImageInput::from_rgb(&image), &RunParams::default())
        .unwrap();
    assert_eq!(again.provenance.artifact_status, ArtifactStatus::InMemory);
    assert_eq!(again.items, first.items);

    let synonym = runtime.resolve("detect_faces").unwrap();
    assert_eq!(synonym.artifact_key(), find_face.artifact_key());
    assert_eq!(synonym.prepare().unwrap().status, ArtifactStatus::InMemory);

    let fresh = Runtime::new(config(&cache.0));
    let reloaded = run(&fresh, "locate_faces", &image, RunParams::default());
    assert_eq!(
        reloaded.provenance.artifact_status,
        ArtifactStatus::LoadedFromDisk
    );
    assert_eq!(reloaded.items, first.items);
    assert_eq!(
        reloaded.provenance.artifact.binary_sha256,
        first.provenance.artifact.binary_sha256
    );
}

#[test]
fn images_without_faces_give_empty_results() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    for name in ["coffee.jpg", "chelsea.jpg"] {
        let result = run(&runtime, "find_faces", &fixture(name), RunParams::default());
        assert!(result.items.is_empty(), "{name}: {:?}", result.items);
    }
    let blank = RgbImage::from_pixel(300, 200, Rgb([128, 128, 128]));
    assert!(
        run(&runtime, "find_faces", &blank, RunParams::default())
            .items
            .is_empty()
    );
}

#[test]
fn grey_and_rgba_inputs_follow_the_same_contract() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let rgb = fixture("astronaut.jpg");
    let op = runtime.resolve("find_face").unwrap();
    let expected = op
        .run(&ImageInput::from_rgb(&rgb), &RunParams::default())
        .unwrap()
        .items;

    let rgba = image::DynamicImage::ImageRgb8(rgb.clone()).to_rgba8();
    let from_rgba = op
        .run(&ImageInput::from_rgba(&rgba), &RunParams::default())
        .unwrap()
        .items;
    assert_eq!(from_rgba, expected, "alpha is ignored");

    let grey: GrayImage = image::DynamicImage::ImageRgb8(rgb).to_luma8();
    let from_grey = op
        .run(&ImageInput::from_gray(&grey), &RunParams::default())
        .unwrap()
        .items;
    assert_eq!(from_grey.len(), 1);
    assert!(iou(&from_grey[0].bbox, &ASTRONAUT_FACE) > 0.8);
}

/// Two astronauts on a 1024x1024 canvas, at (0, 0) and (512, 512).
fn two_astronauts() -> RgbImage {
    let one = fixture("astronaut.jpg");
    let mut canvas = RgbImage::from_pixel(1024, 1024, Rgb([90, 90, 90]));
    image::imageops::overlay(&mut canvas, &one, 0, 0);
    image::imageops::overlay(&mut canvas, &one, 512, 512);
    canvas
}

#[test]
fn compositions_nobody_wrote_restore_coordinates() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let image = two_astronauts();
    let top = ASTRONAUT_FACE;
    let bottom = shifted(ASTRONAUT_FACE, 512.0, 512.0);
    let matches = |found: Vec<BoxF>, expected: &[BoxF]| {
        assert_eq!(found.len(), expected.len(), "{found:?}");
        for (f, e) in found.iter().zip(expected) {
            assert!(iou(f, e) > 0.7, "{f:?} vs {e:?}");
        }
    };

    let all = run(&runtime, "find_faces", &image, RunParams::default());
    assert_eq!(all.items.len(), 2);
    matches(
        boxes(&run(
            &runtime,
            "find_faces_left_to_right",
            &image,
            RunParams::default(),
        )),
        &[top, bottom],
    );
    matches(
        boxes(&run(
            &runtime,
            "find_faces_bottom_to_top",
            &image,
            RunParams::default(),
        )),
        &[bottom, top],
    );
    matches(
        boxes(&run(
            &runtime,
            "find_faces_in_top_half",
            &image,
            RunParams::default(),
        )),
        &[top],
    );
    matches(
        boxes(&run(
            &runtime,
            "find_faces_in_bottom_half",
            &image,
            RunParams::default(),
        )),
        &[bottom],
    );
    matches(
        boxes(&run(
            &runtime,
            "find_rightmost_face",
            &image,
            RunParams::default(),
        )),
        &[bottom],
    );
    matches(
        boxes(&run(
            &runtime,
            "find_faces_in_top_right",
            &image,
            RunParams::default(),
        )),
        &[],
    );
    let region = RunParams {
        region: Some(PixelRect {
            x: 512,
            y: 512,
            w: 512,
            h: 512,
        }),
        ..RunParams::default()
    };
    matches(
        boxes(&run(&runtime, "find_faces_in_region", &image, region)),
        &[bottom],
    );
    let capped = RunParams {
        max_results: Some(1),
        ..RunParams::default()
    };
    assert_eq!(
        run(&runtime, "find_faces_by_size", &image, capped)
            .items
            .len(),
        1
    );
    // Each face covers about 0.95% of this image.
    assert_eq!(
        run(
            &runtime,
            "find_faces_larger_than_2pct",
            &image,
            RunParams::default()
        )
        .items
        .len(),
        0
    );
    assert_eq!(
        run(
            &runtime,
            "find_faces_smaller_than_2pct",
            &image,
            RunParams::default()
        )
        .items
        .len(),
        2
    );
}

#[test]
fn min_confidence_is_a_runtime_parameter() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let image = fixture("astronaut.jpg");
    let strict = RunParams {
        min_confidence: Some(0.99),
        ..RunParams::default()
    };
    let result = run(&runtime, "find_face", &image, strict);
    assert!(result.items.is_empty());
    assert_eq!(result.provenance.params.min_confidence, 0.99);
    assert_eq!(result.provenance.artifact_status, ArtifactStatus::Compiled);
    let default = run(&runtime, "find_face", &image, RunParams::default());
    assert_eq!(default.items.len(), 1);
    assert_eq!(
        default.provenance.artifact_status,
        ArtifactStatus::InMemory,
        "no rebuild for a threshold"
    );
}
