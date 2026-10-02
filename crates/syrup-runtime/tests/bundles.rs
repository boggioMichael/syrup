mod common;

use std::fs;

use common::*;
use syrup_runtime::{ArtifactStatus, ErrorKind, ImageInput, Mode, RunParams, Runtime};

fn frozen(dir: &std::path::Path) -> Runtime {
    let mut config = config(dir);
    config.mode = Mode::Frozen;
    config.rustc = Some("/nonexistent/rustc".into());
    Runtime::new(config)
}

#[test]
fn bundles_run_where_nothing_can_compile() {
    let (cache, bundle) = (TempDir::new(), TempDir::new());
    let builder = Runtime::new(config(&cache.0));
    let index = builder
        .bundle(&bundle.0, &["find_largest_face", "find_red_bars"])
        .unwrap();
    assert_eq!(index.operations.len(), 2);
    builder.bundle(&bundle.0, &["find_words"]).unwrap();

    let runtime = frozen(&bundle.0);
    let image = fixture("astronaut.jpg");
    let faces = runtime
        .resolve("find_biggest_face")
        .unwrap()
        .run(&ImageInput::from_rgb(&image), &RunParams::default())
        .unwrap();
    assert_eq!(faces.items.len(), 1);
    assert_eq!(
        faces.provenance.artifact_status,
        ArtifactStatus::LoadedFromDisk
    );
    assert!(runtime.resolve("find_words").unwrap().prepare().is_ok());

    let e = runtime
        .resolve("find_faces")
        .unwrap()
        .prepare()
        .unwrap_err();
    assert_eq!(e.kind, ErrorKind::NotPrepared);
    assert!(e.reason.contains("not in the bundle"), "{e}");
}

#[test]
fn bundles_for_another_platform_say_so() {
    let (cache, bundle) = (TempDir::new(), TempDir::new());
    let builder = Runtime::new(config(&cache.0));
    builder.bundle(&bundle.0, &["find_face"]).unwrap();
    // What a bundle built on another platform looks like here: its
    // artifacts have other keys, since the target is part of the key.
    let key = builder
        .resolve("find_face")
        .unwrap()
        .artifact_key()
        .to_string();
    fs::rename(
        bundle.0.join("v1").join(&key),
        bundle.0.join("v1").join("0".repeat(64)),
    )
    .unwrap();
    let path = bundle.0.join("bundle.json");
    let mut index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    index["target"] = "riscv64gc-unknown-linux-gnu".into();
    fs::write(&path, index.to_string()).unwrap();

    let e = frozen(&bundle.0)
        .resolve("find_face")
        .unwrap()
        .prepare()
        .unwrap_err();
    assert_eq!(e.kind, ErrorKind::NotPrepared);
    assert!(e.reason.contains("built for riscv64gc"), "{e}");

    let e = builder.bundle(&bundle.0, &["find_faces"]).unwrap_err();
    assert!(e.reason.contains("riscv64gc"), "{e}");
}
