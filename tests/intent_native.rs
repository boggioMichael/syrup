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

/// A plan with a picture and with state: the picture is embedded in the
/// library, the state lives inside it, and identities survive across calls.
#[test]
fn a_tracked_icon_keeps_its_picture_and_its_memory_inside_the_library() {
    if skipped() {
        return;
    }
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    // A distinctive 12x12 picture, planted twice in a frame.
    let icon = image::RgbaImage::from_fn(12, 12, |x, y| {
        let v = ((x * 37 + y * 91) % 200) as u8 + 30;
        image::Rgba([v, 255 - v, (x * y) as u8, 255])
    });
    let mut frame = image::RgbaImage::from_pixel(160, 90, image::Rgba([90, 90, 90, 255]));
    image::imageops::replace(&mut frame, &icon, 20, 30);
    image::imageops::replace(&mut frame, &icon, 120, 50);
    intent::register_template("nativetesticon", &icon);

    let native = intent::resolve_native("track_nativetesticon_icon")
        .expect("builds with the picture embedded");
    let first = native.run(&frame, None);
    let Some(Outcome::Matches(first)) = first.value else {
        panic!("expected matches, got {first:?}");
    };
    assert_eq!(first.len(), 2, "{first:?}");
    // A tracked match's score is the track's confidence, which starts low
    // and grows while the object keeps being seen.
    assert!(
        first
            .iter()
            .all(|m| m.id.is_some() && m.score > 0.0 && m.score < 1.0),
        "{first:?}"
    );
    let mut positions: Vec<(u32, u32)> = first.iter().map(|m| (m.bounds.x, m.bounds.y)).collect();
    positions.sort();
    assert_eq!(positions, vec![(20, 30), (120, 50)]);

    // Same frame again: the same two identities, remembered by the library.
    let second = native.run(&frame, None);
    let Some(Outcome::Matches(second)) = second.value else {
        panic!()
    };
    let ids = |m: &[Match]| {
        let mut ids: Vec<u64> = m.iter().filter_map(|m| m.id).collect();
        ids.sort();
        ids
    };
    assert_eq!(ids(&first), ids(&second));
    assert!(
        second[0].score >= first[0].score,
        "confidence grows: {first:?} -> {second:?}"
    );

    // The picture was written next to the generated source.
    let crate_dir = native.library_path().unwrap();
    let cache = crate_dir
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let embedded = std::fs::read_dir(cache)
        .unwrap()
        .flatten()
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("track_nativetesticon_icon-")
        })
        .any(|e| e.path().join("src").join("nativetesticon.png").is_file());
    assert!(
        embedded,
        "nativetesticon.png should sit in the generated crate under {}",
        cache.display()
    );
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
    match intent::resolve_native("find_icon") {
        Err(IntentError::Unsupported { .. }) => {}
        other => panic!("expected Unsupported, got {other:?}"),
    }
}
