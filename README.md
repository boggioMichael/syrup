# Syrup

A Rust computer-vision library where you name the function you need and
the library builds it.

```rust
use syrup::prelude::*;

syrup::intent!(fn find_face(image: &RgbaImage) -> Detection<Vec<Match>>);
syrup::intent!(fn measure_red_bar(image: &RgbaImage) -> Detection<f32>);

let frame: RgbaImage = image::open("frame.png")?.to_rgba8();
for face in find_face(&frame).value.unwrap_or_default() {
    println!("face at {:?}, score {:.2}", face.bounds, face.score);
}
println!("HP: {:?}%", measure_red_bar(&frame).value);
```

`find_face` does not exist until that line. The name is parsed against a
small vocabulary — a verb, optional qualifiers, a noun — and turned into a
plan over the library's primitives (here: the bundled Viola–Jones
frontal-face cascade). The plan runs in-process on the first call, or is
written out as Rust, compiled with cargo into a `.so`/`.dll`/`.dylib`, and
loaded back, so the function you invented is also a native library any
process can call (`intent::resolve("find_face")?.compile()`, or
`SYRUP_INTENT_MODE=native`). The generated source is kept next to the
build, so you can read what was invented, copy it, or edit it.

Syrup reads a screen the way a person does — "there is a bar here and it is
about 60% full", "that region moved left", "this text says 1291/1351" —
without pretending to more certainty than the pixels support. Every result
carries a confidence, a reliability grade, and a failure reason when nothing
was found.

## The convention

| Name          | Returns             | Meaning                                   |
|---------------|---------------------|-------------------------------------------|
| `find_*`      | `Detection<Vec<Match>>` | zero or more places, strongest first   |
| `track_*`     | `Detection<Vec<Match>>` | the same, with a stable `id` per object across calls |
| `count_*`     | `Detection<f32>`    | how many `find_*` would return            |
| `measure_*`   | `Detection<f32>`    | a percentage or distance                  |
| `read_*`      | `Detection<String>` | text                                      |

An empty `find_*` result (`value: Some(vec![])`) means the search ran and
found nothing. A missing one (`value: None` + `failure_reason`) means it
could not run. The two are never conflated, and a low-confidence answer is
reported as missing with its reason rather than as an answer.

It refuses instead of guessing. Resolution fails, typed and early, when the
name is outside the vocabulary (`Unparsed`, with the vocabulary as a hint),
when the meaning is clear but nothing here can carry it out yet
(`Unsupported`, saying what is missing — a picture of the icon, state for
tracking), when the declared return type does not fit the verb
(`WrongShape`), or when a native build or load fails (`BuildFailed` with
the compiler's log, `LoadFailed`). `syrup::intent::resolve("find_x")` gives
the typed error up front; a declared function that cannot be resolved
returns `Detection::missing` with the same reason on every call.

Vocabulary today — nouns: `face` (frontal, or `profile_face`), `eye`
(inside faces), `<colour>_bar`, `<colour>_blob`, `text`, `<name>_icon`
(a picture you register: `intent::register_template("boss", &image)`, or
`boss.png` in `$SYRUP_TEMPLATES`), `motion`. Verbs: `find`, `track`,
`count`, `measure` (bars), `read` (text). Qualifiers: colours `red orange
yellow green cyan blue purple magenta pink`; a position `top bottom left
right` (that half of the region, before or after the noun); `largest` /
`smallest` (just that one). So `count_largest_red_blobs_bottom`,
`track_boss_icon`, `find_eyes` and `measure_green_bar` are all functions
waiting to be declared. Everything below is what the plans are made of,
and is available directly.

## What it does

- **Geometry** — rectangle segmentation and grouping over pixel predicates,
  and horizontal-bar fill measurement that learns the bar's empty-track
  color from the frame instead of assuming it.
- **Color** — RGB→HSV conversion and the shared pixel predicates (hue-range
  match, "looks like UI text", opacity).
- **Components** — connected-component labelling of masks or pixel
  predicates: the exact regions, each with bounds, area and centroid, not
  just the rectangles around them.
- **Thresholds** — Otsu thresholds, integral images, channel views, and a
  "text evidence" map that lifts glyph strokes off bars and gradients by
  comparing each pixel with its own row's background.
- **Glyph reading** — reads counters and HUD values drawn in a fixed pixel
  font by matching each glyph against a font learned from a few labelled
  crops. It is fast, needs no external tools, and refuses to answer when
  any glyph is ambiguous: an unreadable value comes back as *unknown*,
  never as a wrong number.
- **Motion** — single-pass frame differencing plus a tracker that gives
  moving regions stable IDs, velocity, and occlusion grace. Detections are
  matched to tracks optimally, so nearby objects do not swap identities.
- **Template matching** — finds a known picture (an icon, a UI marker, a
  status glyph) anywhere in a frame by normalised cross-correlation, so
  brightness and contrast differences don't affect the score. Large
  searches use a coarse-to-fine image pyramid instead of scoring every
  position at full resolution; small searches are scored exhaustively.
  Returns a sub-pixel centre estimate.
- **Cascades** — Viola–Jones boosted cascades of Haar features, evaluated
  exactly as OpenCV evaluates its cascades, with OpenCV's frontal-face,
  profile-face and eye detectors bundled (245 KB in all, their licences
  alongside). Cross-checked against OpenCV on the same photograph: the
  same face, the same two eyes.
- **OCR** — text recognition via a Tesseract subprocess, with the crop
  prepared the way Tesseract reads best (dark text on light, levels
  stretched, enlarged, framed by a margin), per-word confidence and boxes,
  and a configurable page mode and character whitelist; on Windows, the
  OS's built-in OCR engine is also exposed, which is trained on screen
  content.
- **Quality** — a sharpness metric that predicts whether OCR on a region
  can succeed at all, so blurred input is reported as *blurred* rather than
  silently producing wrong text.
- **Capture** — live window capture by title on Windows (works while the
  window is occluded); portable stubs elsewhere.
- **Debug drawing** — rectangles and a dependency-free 5×7 bitmap font for
  annotating frames with what a detector saw.
- **Timing** — FPS and moving-average measurement.

## What it deliberately does not do

- No input synthesis, no window manipulation, no process inspection: the
  library **reads pixels and reports observations**, nothing else.
- No trained models and no model files: every primitive is deterministic
  and explainable, which keeps results reproducible in tests.
- No opinion about what an observation *means* — semantics belong to the
  application built on top.

## Example

```rust
use syrup::color::is_color_pixel;
use syrup::geometry::{Rect, find_color_bar, measure_bar_fill};

let image: image::RgbaImage = image::open("screen.png")?.to_rgba8();

// Look for a red horizontal bar in the bottom band of the screen…
let band = Rect { x: 0, y: image.height() * 9 / 10, w: image.width(), h: image.height() / 10 };
let red = |p: &image::Rgba<u8>| is_color_pixel(p, (340.0, 30.0), 0.35, 0.30);

if let Some(bar) = find_color_bar(&image, band, (340.0, 30.0), 0.35, 0.30) {
    // …and measure how full it is against its own track.
    if let Some(percent) = measure_bar_fill(&image, bar, band, red) {
        println!("bar at {bar:?} is {percent:.1}% full");
    }
}
# Ok::<(), image::ImageError>(())
```

Reading a value drawn in a known pixel font — learn the font once from a
crop whose text you know, then read new frames:

```rust
use syrup::glyphs::{GlyphOptions, GlyphSet};

let mut font = GlyphSet::new(GlyphOptions::default());
font.learn(&labelled_frame, value_region, "1291/1351")?;

let reading = font.read(&frame, value_region);
match reading.value {
    Some(text) => println!("{} ({})", text.text, reading.confidence),
    None => println!("unreadable: {}", reading.failure_reason.unwrap_or_default()),
}
```

Frames are plain `image::RgbaImage` buffers, so they can come from a
screenshot, frames extracted from a video, a synthetic fixture in a test,
or `syrup::capture` — every primitive behaves identically regardless of
the source.

## Architecture

```text
syrup::intent!(fn find_red_bar(image: &RgbaImage) -> Detection<Vec<Match>>)
    │
    │  intent      parse the name → Intent { Find, [Red], Bar }
    │              plan            → Plan::ColorBars { Red }
    │              run             → in-process, or
    │              compile         → source → cargo (cdylib) → libloading   ─┐
    ▼                                                                       │ C ABI
RgbaImage (any source)                                                      │ (abi)
    │                                                                       │
    ├─ geometry / color / threshold  locate regions by shape, colour and contrast
    ├─ components                    the exact connected regions
    ├─ motion / tracking             what moved, with stable identity
    ├─ template / cascade            where a known picture, or a face, is
    ├─ glyphs / ocr / quality        what text says, and whether it is readable at all
    │
    ▼
Detection<T> — value + confidence + reliability + failure reason
```

The primitives are the vocabulary; a plan is a sentence in it. The same
plan runs in-process or compiled, through the same functions
(`intent::plans`), so the two can never disagree — the native test checks
that they don't.

The `Detection<T>` vocabulary is the library's one contract: a detector
never returns a bare "not found" — it says *why* not, and never returns a
value without saying *how sure* it is.

## Testing

```sh
cargo test          # unit, integration and doc tests
cargo test -- --ignored   # also run OCR against a real Tesseract install
cargo clippy --all-targets -- -D warnings
cargo bench         # criterion benchmarks for the per-frame primitives
cargo run --release --example intents   # declare, run, and compile an intent
```

Tests run against synthetic, in-code fixtures plus one public-domain
photograph (`tests/fixtures`); no network access is required. The native
intent tests (`tests/intent_native.rs`) build a crate with cargo, which
takes about a minute the first time and is cached after; set
`SYRUP_SKIP_NATIVE_TESTS=1` to leave them out. OCR tests cover argument
construction, TSV parsing, box mapping, preprocessing and temp-file hygiene
without invoking Tesseract; one opt-in test runs the real engine end to
end. Nothing in the suite depends on a domain edition, so this repository
stands alone.

## Performance

Criterion benchmarks on one 1366×768 frame (`cargo bench`; one x86-64
core, default target features). "Before" is the code as it was extracted
from MapleSyrup.

| Benchmark                                   | Before  | Now     |
|---------------------------------------------|--------:|--------:|
| Motion detection, one moving object         | 6.25 ms | 0.66 ms |
| Motion detection, whole view panning        | 7.57 ms | 2.12 ms |
| Colour-bar search in a status band          | 1.18 ms | 0.52 ms |
| Dominant colour + uniform-panel search      | 3.37 ms | 1.42 ms |
| Bar-fill measurement                        | 14 µs   | 14 µs   |
| Connected components, 400 blobs             | —       | 0.72 ms |
| Text evidence, 124×22 crop                  | —       | 14 µs   |
| Glyph reading, 9-character value            | —       | 86 µs   |
| Glyph reading, unreadable 40 px smear       | —       | 0.70 ms |
| Tracker update, 60 crowded objects          | —       | 38 µs   |
| Template search, 32 px icon, whole frame    | —       | 15 ms   |
| Template search, 64 px icon, whole frame    | —       | 7.5 ms  |
| Template search, 16 px icon, 240×140 region | —       | 1.6 ms  |
| Face cascade, 320×320 photograph, 1 thread (OpenCV: 91 ms) | — | 130 ms |
| Face cascade, 320×320 photograph, 2 threads | —  | 90 ms   |

Motion is the slowest when the whole view changes, because no row can be
skipped; even then it stays well inside a 60 fps frame budget.

## Limitations

- The primitives are tuned for rendered UI content (flat colors, pixel
  fonts, hard edges), not for photographs or video of natural scenes.
- Tesseract must be installed separately for OCR (`TESSERACT_BIN`, `PATH`,
  or the standard Windows install locations); without it, OCR reports
  itself unavailable rather than failing. Even on clean text it misreads
  some pixel fonts — prefer the glyph reader wherever the font is fixed.
- The glyph reader reads one line per region, in fonts it was shown.
  Learn from whole lines as they appear on screen: glyphs are measured
  against their line's height, so a slash or a dot learned on its own will
  not match.
- The face detector is OpenCV's frontal cascade: roughly upright, roughly
  front-on faces of 20 px or more. Profiles, heavy tilt and tiny faces are
  not found. A learned detector behind the same plan is the natural next
  step.
- Compiling an intent to a shared library needs `cargo` on the machine at
  run time; running it in-process (the default) does not.
- Live capture is Windows-only. Other platforms consume file-based frames.

## Syrup and MapleSyrup

Syrup is the generic engine. A domain edition consumes it and adds the
knowledge Syrup deliberately lacks — what the pixels *mean* in one
particular application:

```text
        MapleSyrup            the MapleStory edition
             │                github.com/boggioMichael/ms
             │ submodule
             ▼
           Syrup              this repository
```

[MapleSyrup](https://github.com/boggioMichael/ms) is the first such
edition, and is where these primitives were developed against real
captures before being generalised. Syrup itself knows nothing about
MapleStory, or any other application — a second edition for a different
program would consume it exactly the same way.

## License

MIT
