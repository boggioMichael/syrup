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
