mod common;

use std::process::{Command, Output};

use common::*;
use serde_json::Value;

fn syrup(cache: &TempDir, args: &[&str], env: &[(&str, &str)]) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_syrup"));
    command.args(args).env("SYRUP_CACHE_DIR", &cache.0);
    for var in [
        "SYRUP_MODE",
        "SYRUP_RUSTC",
        "SYRUP_FACE_MODEL",
        "TESSERACT_BIN",
    ] {
        command.env_remove(var);
    }
    command.envs(env.iter().copied()).output().unwrap()
}

fn json(output: Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn a_new_process_reuses_the_compiled_module() {
    let cache = TempDir::new();
    let photo = fixture_path("astronaut.jpg");
    let args = ["run", "find_face", photo.to_str().unwrap()];
    let first = json(syrup(&cache, &args, &[]));
    let second = json(syrup(&cache, &args, &[]));
    assert_eq!(first["provenance"]["artifact_status"], "compiled");
    assert_eq!(second["provenance"]["artifact_status"], "loaded_from_disk");
    assert_eq!(first["items"], second["items"]);
    assert_eq!(
        first["provenance"]["artifact"]["binary_sha256"],
        second["provenance"]["artifact"]["binary_sha256"]
    );
    assert_eq!(first["items"].as_array().unwrap().len(), 1);
}

#[test]
fn prepared_modules_run_where_there_is_no_compiler() {
    let cache = TempDir::new();
    assert!(
        syrup(&cache, &["prepare", "find_largest_face"], &[])
            .status
            .success()
    );
    let photo = fixture_path("astronaut.jpg");
    let frozen = [
        ("SYRUP_MODE", "frozen"),
        ("SYRUP_RUSTC", "/nonexistent/rustc"),
    ];

    let result = json(syrup(
        &cache,
        &["run", "find_biggest_face", photo.to_str().unwrap()],
        &frozen,
    ));
    assert_eq!(result["provenance"]["artifact_status"], "loaded_from_disk");

    let refused = syrup(
        &cache,
        &["run", "find_faces_in_center", photo.to_str().unwrap()],
        &frozen,
    );
    assert_eq!(refused.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("[load/not_prepared]"));
}

#[test]
fn failures_exit_non_zero_with_the_stage() {
    let cache = TempDir::new();
    let photo = fixture_path("coffee.jpg");
    let out = syrup(
        &cache,
        &["run", "find_main_face", photo.to_str().unwrap()],
        &[],
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("[resolve/ambiguous]"));

    let empty = json(syrup(
        &cache,
        &["run", "find_faces", photo.to_str().unwrap()],
        &[],
    ));
    assert_eq!(empty["items"], Value::Array(vec![]));
}

#[test]
fn a_missing_ocr_engine_is_a_dependency_error() {
    let cache = TempDir::new();
    // Compile first, so the engine is the only thing missing at run time.
    assert!(
        syrup(&cache, &["prepare", "find_words"], &[])
            .status
            .success()
    );
    let page = fixture_path("words.png");
    let out = syrup(
        &cache,
        &["run", "find_words", page.to_str().unwrap()],
        &[("PATH", ""), ("TESSERACT_BIN", "/nonexistent/tesseract")],
    );
    assert_eq!(out.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("[execute/missing_dependency]"),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
