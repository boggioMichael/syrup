//! The intent mechanism end to end: a name becomes source, the source a
//! shared library, the library a running function — and it agrees with
//! the in-process implementation of the same plan.
//!
//! These tests build a crate with cargo, so they need a toolchain and take
//! a while the first time (the result is cached under
//! `$SYRUP_INTENT_CACHE`). Set `SYRUP_SKIP_NATIVE_TESTS=1` to skip them.

use std::sync::Mutex;

use syrup::geometry::Rect;
use syrup::intent::{self, IntentError, Match, Outcome};

/// Tests here change process-wide environment variables.
static ENV: Mutex<()> = Mutex::new(());

fn skipped() -> bool {
    std::env::var_os("SYRUP_SKIP_NATIVE_TESTS").is_some()
}

fn astronaut() -> image::RgbaImage {
    image::load_from_memory(include_bytes!("fixtures/astronaut_320.jpg"))
        .expect("fixture decodes")
        .to_rgba8()
}

#[test]
fn find_face_runs_as_a_shared_library_and_agrees_with_in_process() {
    if skipped() {
        return;
    }
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let image = astronaut();
    let region = Rect {
        x: 0,
        y: 0,
        w: 320,
        h: 320,
    };

    let native = intent::resolve_native("find_face").expect("find_face builds and loads");
    assert!(native.is_native());
    let path = native.library_path().expect("a loaded library has a path");
    assert!(path.is_file(), "{} should exist", path.display());
    assert!(
        path.file_name()
            .unwrap()
            .to_string_lossy()
            .contains("syrup_intent_find_face"),
        "{}",
        path.display()
    );

    let from_library = native.run(&image, Some(region));
    let in_process = intent::resolve("find_face")
        .unwrap()
        .run(&image, Some(region));
    assert_eq!(
        from_library.value, in_process.value,
        "the two vehicles must agree"
    );
    assert_eq!(from_library.confidence, in_process.confidence);
    assert_eq!(from_library.reliability, in_process.reliability);

    let Some(Outcome::Matches(faces)) = &from_library.value else {
        panic!("expected matches, got {from_library:?}");
    };
    assert_eq!(faces.len(), 1, "one face in the fixture: {faces:?}");
    let Match { bounds, score, .. } = &faces[0];
    assert!(
        bounds.x >= 100 && bounds.x <= 120 && bounds.y >= 30 && bounds.y <= 50,
        "{bounds:?}"
    );
    assert!(*score > 0.9, "score {score}");

    // The invented function is readable: its source sits next to the build.
    let source = native.source();
    assert!(source.contains("plans::find_faces("));
    assert!(source.contains("syrup::export_intent!(\"find_face\", run);"));

    // A region view that is null on the C side means "the whole frame".
    let whole = native.run(&image, None);
    assert_eq!(whole.value, from_library.value);
}

#[test]
fn a_build_that_cannot_find_the_library_fails_with_the_compiler_log() {
    if skipped() {
        return;
    }
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let previous = std::env::var_os("SYRUP_SOURCE_DIR");
    // SAFETY: the ENV lock serialises every test that touches the environment.
    unsafe { std::env::set_var("SYRUP_SOURCE_DIR", "/definitely/not/where/syrup/is") };
    let result = intent::resolve_native("measure_red_bar");
    // SAFETY: as above.
    unsafe {
        match previous {
            Some(value) => std::env::set_var("SYRUP_SOURCE_DIR", value),
            None => std::env::remove_var("SYRUP_SOURCE_DIR"),
        }
    }
    match result {
        Err(IntentError::BuildFailed { name, log }) => {
            assert_eq!(name, "measure_red_bar");
            assert!(
                log.contains("cargo build"),
                "the log should come from cargo:\n{log}"
            );
        }
        other => panic!("expected BuildFailed, got {other:?}"),
    }
}

#[test]
fn names_outside_the_vocabulary_never_reach_the_compiler() {
    // No toolchain needed: the refusal happens before anything is written.
    match intent::resolve_native("find_unicorn") {
        Err(IntentError::Unparsed { .. }) => {}
        other => panic!("expected Unparsed, got {other:?}"),
    }
    match intent::resolve_native("track_face") {
        Err(IntentError::Unsupported { .. }) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}
