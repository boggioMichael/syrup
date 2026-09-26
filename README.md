# Syrup

A small Rust toolkit for turning captured frames into structured,
confidence-scored visual observations.

Syrup gives you the pixel-level building blocks for reading a screen the
way a person does — "there is a bar here and it is about 60% full", "that
region moved left", "this text says 1291/1351" — without pretending to more
certainty than the pixels support. Every detector result carries a
confidence score, a reliability grade, and a failure reason when nothing was
found.

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
RgbaImage (any source)
    │
    ├─ geometry / color / threshold  locate regions by shape, colour and contrast
    ├─ components                    the exact connected regions
    ├─ motion / tracking             what moved, with stable identity
    ├─ glyphs / ocr / quality        what text says, and whether it is readable at all
    │
    ▼
Detection<T> — value + confidence + reliability + failure reason
```

The `Detection<T>` vocabulary is the library's one contract: a detector
never returns a bare "not found" — it says *why* not, and never returns a
value without saying *how sure* it is.

## Testing

```sh
cargo test          # unit, integration and doc tests
cargo test -- --ignored   # also run OCR against a real Tesseract install
cargo clippy --all-targets -- -D warnings
cargo bench         # criterion benchmarks for the per-frame primitives
```

All tests run against synthetic, in-code fixtures; no external tools,
assets, or network access are required. OCR tests cover argument
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
