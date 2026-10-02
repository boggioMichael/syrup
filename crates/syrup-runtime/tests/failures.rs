mod common;

use std::fs;

use common::*;
use syrup_runtime::providers::ModelSource;
use syrup_runtime::{
    ArtifactStatus, ErrorKind, ImageInput, Mode, PixelRect, RunParams, Runtime, Stage, SyrupError,
};

fn fails<T>(result: Result<T, SyrupError>, stage: Stage, kind: ErrorKind) -> SyrupError {
    match result {
        Ok(_) => panic!("expected {stage}/{kind}"),
        Err(e) => {
            assert_eq!((e.stage, e.kind), (stage, kind), "{e}");
            assert!(
                e.operation.is_some() || stage == Stage::Input,
                "{e} names no operation"
            );
            e
        }
    }
}

#[test]
fn unresolvable_names_fail_before_any_image_is_involved() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    fails(
        runtime.resolve("find_best_face"),
        Stage::Resolve,
        ErrorKind::Ambiguous,
    );
    fails(
        runtime.resolve("find_smiling_faces"),
        Stage::Resolve,
        ErrorKind::Unsupported,
    );
    fails(
        runtime.resolve("find_cars"),
        Stage::Resolve,
        ErrorKind::Unsupported,
    );
    fails(
        runtime.resolve("find_faces_in_top_half_in_center"),
        Stage::Resolve,
        ErrorKind::Conflicting,
    );
    assert!(!cache.0.join("v1").exists(), "nothing was built");
}

#[test]
fn bad_inputs_are_input_errors() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let pixels = vec![0u8; 48];
    fails(
        ImageInput::new(&pixels, 4, 4, 2),
        Stage::Input,
        ErrorKind::BadImage,
    );
    fails(
        ImageInput::new(&pixels, 8, 8, 3),
        Stage::Input,
        ErrorKind::BadImage,
    );
    fails(
        ImageInput::new(&pixels, 0, 4, 3),
        Stage::Input,
        ErrorKind::BadImage,
    );

    let image = ImageInput::new(&pixels, 4, 4, 3).unwrap();
    let find_face = runtime.resolve("find_face").unwrap();
    let in_region = runtime.resolve("find_faces_in_region").unwrap();
    let region = Some(PixelRect {
        x: 0,
        y: 0,
        w: 2,
        h: 2,
    });
    for (op, params) in [
        (
            &find_face,
            RunParams {
                min_confidence: Some(1.5),
                ..RunParams::default()
            },
        ),
        (
            &find_face,
            RunParams {
                max_results: Some(0),
                ..RunParams::default()
            },
        ),
        (
            &find_face,
            RunParams {
                region,
                ..RunParams::default()
            },
        ),
        (&in_region, RunParams::default()),
        (
            &in_region,
            RunParams {
                region: Some(PixelRect {
                    x: 9,
                    y: 0,
                    w: 2,
                    h: 2,
                }),
                ..RunParams::default()
            },
        ),
    ] {
        fails(
            op.run(&image, &params),
            Stage::Input,
            ErrorKind::BadParameter,
        );
    }
}

#[test]
fn a_missing_compiler_is_reported_at_the_compile_stage() {
    let cache = TempDir::new();
    let mut config = config(&cache.0);
    config.rustc = Some("/nonexistent/rustc".into());
    let e = fails(
        Runtime::new(config).resolve("find_face").unwrap().prepare(),
        Stage::Compile,
        ErrorKind::MissingDependency,
    );
    assert!(e.hint.unwrap().contains("SYRUP_RUSTC"));
}

#[test]
fn frozen_mode_never_compiles() {
    let cache = TempDir::new();
    let mut frozen = config(&cache.0);
    frozen.mode = Mode::Frozen;
    frozen.rustc = Some("/nonexistent/rustc".into());
    let op = Runtime::new(frozen.clone()).resolve("find_face").unwrap();
    fails(op.prepare(), Stage::Load, ErrorKind::NotPrepared);

    Runtime::new(config(&cache.0))
        .resolve("find_face")
        .unwrap()
        .prepare()
        .unwrap();
    let prepared = Runtime::new(frozen)
        .resolve("find_faces")
        .unwrap()
        .prepare()
        .unwrap();
    assert_eq!(prepared.status, ArtifactStatus::LoadedFromDisk);
}

#[test]
fn damaged_artifacts_are_refused_and_rebuilt() {
    let cache = TempDir::new();
    let library = Runtime::new(config(&cache.0))
        .resolve("find_face")
        .unwrap()
        .prepare()
        .unwrap()
        .library;
    let mut bytes = fs::read(&library).unwrap();
    let middle = bytes.len() / 2;
    bytes[middle] ^= 0xff;
    fs::write(&library, bytes).unwrap();

    let mut frozen = config(&cache.0);
    frozen.mode = Mode::Frozen;
    fails(
        Runtime::new(frozen).resolve("find_face").unwrap().prepare(),
        Stage::Load,
        ErrorKind::Integrity,
    );

    let rebuilt = Runtime::new(config(&cache.0))
        .resolve("find_face")
        .unwrap()
        .prepare()
        .unwrap();
    assert_eq!(rebuilt.status, ArtifactStatus::Compiled);
    let moved_aside = fs::read_dir(cache.0.join("failed"))
        .unwrap()
        .flatten()
        .any(|e| e.file_name().to_string_lossy().contains(".damaged."));
    assert!(moved_aside);
}

#[test]
fn provider_failures_are_execution_errors() {
    let cache = TempDir::new();
    let image = fixture("astronaut.jpg");
    let input = ImageInput::from_rgb(&image);

    let mut missing = config(&cache.0);
    missing.face_model = ModelSource::File(cache.0.join("no-such-model.onnx"));
    let op = Runtime::new(missing).resolve("find_face").unwrap();
    fails(
        op.run(&input, &RunParams::default()),
        Stage::Execute,
        ErrorKind::MissingDependency,
    );

    let mut wrong = config(&cache.0);
    wrong.face_model = ModelSource::File(fixture_path("coffee.jpg"));
    let op = Runtime::new(wrong).resolve("find_face").unwrap();
    fails(
        op.run(&input, &RunParams::default()),
        Stage::Execute,
        ErrorKind::Integrity,
    );
}

#[test]
fn concurrent_first_use_compiles_once() {
    let cache = TempDir::new();
    let statuses: Vec<ArtifactStatus> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..6)
            .map(|_| {
                s.spawn(|| {
                    Runtime::new(config(&cache.0))
                        .resolve("find_faces_in_center")
                        .unwrap()
                        .prepare()
                        .unwrap()
                        .status
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let compiled = statuses
        .iter()
        .filter(|s| **s == ArtifactStatus::Compiled)
        .count();
    assert_eq!(compiled, 1, "{statuses:?}");
    assert_eq!(fs::read_dir(cache.0.join("v1")).unwrap().count(), 1);
}
