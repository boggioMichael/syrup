# Changelog

## Unreleased

### Functions you name instead of write
- `syrup::intent!`: declare `fn find_face(image: &RgbaImage) -> Detection<Vec<Match>>`
  and the library implements it — in-process, or compiled to a `.so`/`.dll`
  and loaded back (`Resolved::compile`, `SYRUP_INTENT_MODE=native`).
- Vocabulary: `find/track/count/measure/read`; nouns `face`, `profile_face`,
  `eye`, `<colour>_bar`, `<colour>_blob`, `text`, `<name>_icon`, `motion`;
  qualifiers `top/bottom/left/right`, `largest/smallest`.
- Typed refusals: `Unparsed`, `Unsupported`, `WrongShape`, `BuildFailed`,
  `LoadFailed`. Never a guess.
- A C API (`syrup_resolve`, `syrup_run`, …) when built as a shared library,
  and `python/syrup` on top of it: `import syrup; syrup.find_face(image)`.

### Detectors
- Viola–Jones cascades evaluated as OpenCV evaluates them, with OpenCV's
  frontal-face, profile-face and eye cascades bundled; cross-checked
  against OpenCV on the same photograph; rows scanned on every core.
- Template matching by normalised cross-correlation, coarse to fine.
- Connected components by run labelling; motion detection with fixed-point
  luminance the compiler vectorises; Hungarian assignment in the tracker.
- Pixel-font glyph reading with `min_gap` for segmented (seven-segment) fonts.
- `face`: eyes and mouth of a found face, eye openness, mouth shape, and
  `mouth_patch`, a small normalised picture of the mouth for matching.
- `sequence`: dynamic time warping and a nearest-example matcher.

### Example project: a Minesweeper coach (`example/minesweeper-coach`)
- Watches the game on the screen (`syrup::capture::capture_screen`, new:
  every monitor, with the picture's place on the desktop), finds and reads
  the board with syrup's connected components at any cell size, and says
  what to do next through the system's voice, with the cells marked in a
  click-through window that keeps out of screen captures. It never clicks.
- The solver proves cells by the rules a person uses (each with its
  reason) and then by counting every way the numbers and the mines left can
  be satisfied, which also gives each hidden cell's exact chance of a mine.
  A board partly out of view is solved without taking the edge of the view
  for the board's edge.
- Measured: 91.3% / 80.8% / 42.0% of beginner / intermediate / expert
  games won by its advice, 0 wrong proofs; three screenshots of
  minesweeper.online read cell for cell. 34 tests, run on Linux and
  Windows.

### Example projects: lipreader (`example/`)
- `lipreader` (Python): many faces, IoU/Hungarian tracking with shot-cut
  detection, mouth localisation per face, speaking activity, batched visual
  speech recognition behind a model interface, CTC decoding with word timing
  and confidence, three modes (visual never opens the audio), an honest
  language registry (English runs; es/fr/pt/it/ar/de/el/ru/zh listed with
  licences as not runnable here; Hebrew has no public model), exports,
  an evaluation framework (WER, CER, attribution, track consistency, cuts,
  latency, RTF) and 25 tests on composited fixtures with exact truth.
- `inference-api`: jobs, streamed sessions, exports, deletion, token, rate
  limit, CORS; 9 tests.
- `chrome-extension`: Lip Read for YouTube (MV3, TypeScript), overlay with
  per-person boxes and subtitles, click-to-follow, side panel with seek and
  export; 25-check Playwright end-to-end test against the service.
- `web`: one page using the service; 7-check end-to-end test.
- `ios`: SwiftUI app sources (Vision, AVFoundation, Core ML, remote
  backend), written without a compiler and labelled so.
- `shared-types`, `ml` (model scripts behind licence gates, weights export,
  Core ML conversion), `docs` (architecture, models, benchmarks, API, setup,
  decisions).

### thelip.syrup
- `example/thelip`: The Lip in the browser, on a phone — the network in
  plain JavaScript in a Web Worker (int8 weights, incremental
  convolutions), camera or recorded clip, utterance detection, readings
  that form while you speak; deployed to GitHub Pages from `docs/`.

### The Lip
- `python/thelip`: lip reading from muted video with live subtitles —
  syrup locates the mouth, LipNet's published weights run in numpy (with a
  minimal HDF5 reader, no TensorFlow), the prefix is decoded again after
  every frame. 64/66 words on the GRID sample clips; `make_demo.sh` builds
  the demo video, `evaluate.py` the accuracy table, `proof.py` the proof
  video (muted, then with sound; a recorded fresh-clone session; the code).

### Fixes
- FPS readings 60x too high before the window filled; overflow panics in
  `draw_rect`; OCR word boxes left in enlarged-image coordinates; temp-file
  collisions across threads; luminance weights that undercounted 23 grey
  levels in motion detection.
