mod common;

use std::sync::Arc;

use common::*;
use syrup_runtime::abi::SyrupDetection;
use syrup_runtime::contract::ProviderInfo;
use syrup_runtime::custom::add_target;
use syrup_runtime::providers::{Detections, Provider, ViewRef};
use syrup_runtime::{ErrorKind, ImageInput, RunParams, Runtime};

/// Reports one 20x20 box at (10, 10) of whatever window it is shown.
struct Corner;

/// Reports a box with a negative width.
struct Inverted;

impl Provider for Inverted {
    fn info(&self) -> ProviderInfo {
        Corner.info()
    }

    fn detect(&self, view: &ViewRef<'_>) -> syrup_runtime::Result<Detections> {
        let mut found = Corner.detect(view)?;
        found.boxes[0].w = -20.0;
        Ok(found)
    }
}

impl Provider for Corner {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            capability: "corner",
            name: "test",
            model_sha256: None,
            runtime: "rust",
        }
    }

    fn detect(&self, _: &ViewRef<'_>) -> syrup_runtime::Result<Detections> {
        Ok(Detections {
            boxes: vec![SyrupDetection {
                x: 10.0,
                y: 10.0,
                w: 20.0,
                h: 20.0,
                score: 0.9,
                value: 0.0,
                n_keypoints: 0,
                keypoints: [0.0; 10],
                payload: 0,
            }],
            texts: vec![],
        })
    }
}

#[test]
fn added_targets_compose_like_built_in_ones() {
    add_target("corner", "corners", 0.5, Arc::new(Corner)).unwrap();
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let pixels = vec![0u8; 100 * 100 * 3];
    let image = ImageInput::new(&pixels, 100, 100, 3).unwrap();
    let run = |name: &str| {
        runtime
            .resolve(name)
            .and_then(|op| op.run(&image, &RunParams::default()))
            .unwrap_or_else(|e| panic!("{name}: {e}"))
    };

    let found = run("find_corners_in_bottom_right");
    assert_eq!((found.items[0].bbox.x, found.items[0].bbox.y), (60.0, 60.0));
    assert_eq!(found.items[0].label, "corner");
    assert_eq!(found.provenance.providers[0].name, "test");
    assert!(run("find_corners_smaller_than_1pct").items.is_empty());
}

#[test]
fn added_nouns_cannot_take_over_the_grammar() {
    for (singular, plural, kind) in [
        ("in_box", "in_boxes", ErrorKind::Conflicting),
        ("red_dot", "red_dots", ErrorKind::Conflicting),
        ("face", "faces", ErrorKind::Conflicting),
        ("Dot", "Dots", ErrorKind::Malformed),
        ("dot", "dot__s", ErrorKind::Malformed),
    ] {
        let e = add_target(singular, plural, 0.5, Arc::new(Corner)).unwrap_err();
        assert_eq!(e.kind, kind, "{singular}: {e}");
    }
    assert!(add_target("dot", "dots", 1.5, Arc::new(Corner)).is_err());
}

#[test]
fn detections_with_negative_sizes_fail_the_run() {
    add_target("flipped_dot", "flipped_dots", 0.5, Arc::new(Inverted)).unwrap();
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let pixels = vec![0u8; 100 * 100 * 3];
    let image = ImageInput::new(&pixels, 100, 100, 3).unwrap();
    let e = runtime
        .resolve("find_flipped_dots")
        .and_then(|op| op.run(&image, &RunParams::default()))
        .unwrap_err();
    assert_eq!(e.kind, ErrorKind::ProviderFailed, "{e}");
}
