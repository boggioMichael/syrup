//! Functions by name: declare them, call them, and — if you want a shared
//! library other processes can load — compile them.
//!
//!     cargo run --release --example intents
//!     SYRUP_INTENT_MODE=native cargo run --release --example intents   # first call compiles

use syrup::prelude::*;

// Basic: say what you need. Each line is a real function after this.
syrup::intent!(fn find_face(image: &RgbaImage) -> Detection<Vec<Match>>);
syrup::intent!(fn find_eyes(image: &RgbaImage) -> Detection<Vec<Match>>);
syrup::intent!(fn find_red_bar(image: &RgbaImage, region: Rect) -> Detection<Vec<Match>>);
syrup::intent!(fn measure_red_bar(image: &RgbaImage) -> Detection<f32>);
syrup::intent!(fn count_red_blobs_left(image: &RgbaImage) -> Detection<f32>);
// Understood, but this build cannot do it yet: every call explains why.
syrup::intent!(fn find_boss_icon(image: &RgbaImage) -> Detection<Vec<Match>>);
// Not in the vocabulary: every call says so, with the vocabulary.
syrup::intent!(fn find_unicorn(image: &RgbaImage) -> Detection<Vec<Match>>);

fn main() {
    let photo = image::load_from_memory(include_bytes!("../tests/fixtures/astronaut_320.jpg"))
        .expect("fixture")
        .to_rgba8();

    println!("== basic: declared functions ==");
    let faces = find_face(&photo);
    match &faces.value {
        Some(found) => {
            println!(
                "find_face      -> {} face(s), confidence {:.2}, {:?}",
                found.len(),
                faces.confidence.value(),
                faces.reliability
            );
            for face in found {
                println!(
                    "                  {:?} score {:.2}",
                    face.bounds, face.score
                );
            }
        }
        None => println!(
            "find_face      -> could not run: {}",
            faces.failure_reason.unwrap_or_default()
        ),
    }

    let hud = bar_frame();
    let whole = Rect {
        x: 0,
        y: 0,
        w: hud.width(),
        h: hud.height(),
    };
    let bars = find_red_bar(&hud, whole);
    println!(
        "find_red_bar   -> {:?}",
        bars.value
            .as_ref()
            .map(|b| b.iter().map(|m| m.bounds).collect::<Vec<_>>())
    );
    let fill = measure_red_bar(&hud);
    println!(
        "measure_red_bar-> {:?} ({})",
        fill.value,
        fill.failure_reason.unwrap_or_else(|| "measured".into())
    );

    let eyes = find_eyes(&photo);
    println!(
        "find_eyes      -> {:?}",
        eyes.value
            .as_ref()
            .map(|e| e.iter().map(|m| m.bounds).collect::<Vec<_>>())
    );
    println!(
        "count_red_blobs_left -> {:?}",
        count_red_blobs_left(&hud).value
    );
    let icon = find_boss_icon(&photo);
    println!(
        "find_boss_icon -> {}",
        icon.failure_reason.unwrap_or_default()
    );
    let unicorn = find_unicorn(&photo);
    println!(
        "find_unicorn   -> {}",
        unicorn.failure_reason.unwrap_or_default()
    );

    println!("\n== advanced: the same intent as a shared library ==");
    let resolved = intent::resolve("find_face").expect("find_face resolves");
    println!("plan: {:?}", resolved.plan());
    println!(
        "--- source the library wrote for it ---\n{}",
        resolved.source()
    );
    match resolved.compile() {
        Ok(path) => {
            println!("compiled and loaded: {}", path.display());
            let started = std::time::Instant::now();
            let native = resolved.run(&photo, None);
            println!(
                "native call -> {:?} in {:?}",
                native.value.map(|o| match o {
                    intent::Outcome::Matches(m) => m.len(),
                    _ => 0,
                }),
                started.elapsed()
            );
        }
        Err(IntentError::BuildFailed { log, .. }) => {
            println!("native build failed (is cargo on PATH?):\n{log}");
        }
        Err(error) => println!("native: {error}"),
    }
}

/// A synthetic HUD: a red bar 60% full in a dark panel.
fn bar_frame() -> RgbaImage {
    let mut image = RgbaImage::from_pixel(200, 40, image::Rgba([30, 30, 34, 255]));
    for y in 14..26 {
        for x in 20..128 {
            image.put_pixel(x, y, image::Rgba([215, 35, 35, 255]));
        }
        for x in 128..180 {
            image.put_pixel(x, y, image::Rgba([70, 70, 110, 255]));
        }
    }
    image
}
