mod common;

use common::*;
use image::{Rgba, RgbaImage, imageops};
use syrup_runtime::{ErrorKind, ImageInput, RunParams, Runtime, SessionOptions, Stage};

fn square_at(x: u32) -> RgbaImage {
    let mut frame = RgbaImage::from_pixel(200, 120, Rgba([20, 20, 24, 255]));
    for y in 50..70 {
        for x in x..x + 20 {
            frame.put_pixel(x, y, Rgba([230, 230, 230, 255]));
        }
    }
    frame
}

#[test]
fn moving_regions_keep_their_id() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let mut session = runtime
        .resolve("track_moving_regions")
        .unwrap()
        .session(SessionOptions::default())
        .unwrap();
    let mut seen = vec![];
    for step in 0..6 {
        let frame = square_at(10 + 6 * step);
        let result = session
            .update(&ImageInput::from_rgba(&frame), &RunParams::default())
            .unwrap();
        assert_eq!(result.provenance.frame, Some(step as u64 + 1));
        seen.push(result.items);
    }
    assert!(seen[0].is_empty(), "nothing has moved in the first frame");
    // Differencing sees the strip the square left and the strip it entered;
    // each keeps its id while the square moves.
    let ids = |items: &[syrup_runtime::Found]| {
        let mut ids: Vec<u64> = items.iter().map(|f| f.track.unwrap().id).collect();
        ids.sort();
        ids
    };
    for items in &seen[1..] {
        assert_eq!(items.len(), 2);
        assert_eq!(ids(items), ids(&seen[1]));
    }
    for found in &seen[5] {
        let track = found.track.unwrap();
        assert_eq!((track.velocity, track.age_frames), ((6.0, 0.0), 5));
    }
}

#[test]
fn faces_keep_their_id_as_they_move() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let photo = fixture("astronaut.jpg");
    let mut session = runtime
        .resolve("track_largest_face")
        .unwrap()
        .session(SessionOptions::default())
        .unwrap();
    let mut tracks = vec![];
    for step in 0..4 {
        // The camera pans right, so the face moves 8 px left per frame.
        let frame = imageops::crop_imm(&photo, 8 * step, 0, 400, 400).to_image();
        let result = session
            .update(&ImageInput::from_rgb(&frame), &RunParams::default())
            .unwrap();
        assert_eq!(result.items.len(), 1);
        tracks.push(result.items[0].track.unwrap());
    }
    assert!(tracks.iter().all(|t| t.id == tracks[0].id));
    let (vx, vy) = tracks[3].velocity;
    assert!((vx + 8.0).abs() < 2.0 && vy.abs() < 2.0, "{vx}, {vy}");
}

#[test]
fn tracking_needs_a_session_and_sessions_need_tracking() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let frame = square_at(10);
    let image = ImageInput::from_rgba(&frame);

    let e = runtime
        .resolve("track_faces")
        .unwrap()
        .run(&image, &RunParams::default())
        .unwrap_err();
    assert_eq!((e.stage, e.kind), (Stage::Input, ErrorKind::BadParameter));
    assert!(e.hint.unwrap().contains("session"));

    let e = runtime
        .resolve("find_faces")
        .unwrap()
        .session(SessionOptions::default())
        .err()
        .unwrap();
    assert_eq!(e.kind, ErrorKind::BadParameter);
    assert_eq!(
        runtime.resolve("find_moving_regions").err().unwrap().kind,
        ErrorKind::Unsupported
    );
    let track = runtime.resolve("track_faces").unwrap();
    let find = runtime.resolve("find_faces").unwrap();
    assert_eq!(
        track.artifact_key(),
        find.artifact_key(),
        "per frame, track_faces is find_faces"
    );
}

#[test]
fn a_failed_frame_leaves_the_session_as_it_was() {
    let cache = TempDir::new();
    let runtime = Runtime::new(config(&cache.0));
    let mut session = runtime
        .resolve("track_moving_regions_in_region")
        .unwrap()
        .session(SessionOptions::default())
        .unwrap();
    let region = |x, y| RunParams {
        region: Some(syrup_runtime::PixelRect {
            x,
            y,
            w: 200,
            h: 120,
        }),
        ..RunParams::default()
    };
    let first = square_at(10);
    session
        .update(&ImageInput::from_rgba(&first), &region(0, 0))
        .unwrap();
    // A region outside the image fails the frame...
    assert!(
        session
            .update(&ImageInput::from_rgba(&first), &region(500, 0))
            .is_err()
    );
    // ...and the next good frame still compares with the first one.
    let moved = square_at(40);
    let result = session
        .update(&ImageInput::from_rgba(&moved), &region(0, 0))
        .unwrap();
    assert_eq!(result.items.len(), 2);
    assert_eq!(result.provenance.frame, Some(2));
}
