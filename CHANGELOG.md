# Changelog

## Unreleased

### The vision engine of MapleSyrup
- `capture`: on Windows, frames come through Windows.Graphics.Capture —
  the compositor's own picture of the window, kept on the GPU, the client
  area only, cursor off and the capture border off where the system
  allows (Windows 11) — read back whole or a region at a time
  (`Window::capture_frame`, `Frame::read`, `Frame::read_all`). GDI
  (`PrintWindow`, then a copy of the screen, its objects kept between
  frames) takes over where that is not to be had; `Window::gpu_unavailable`
  says why, `Window::without_gpu` and `SYRUP_CAPTURE=cpu` ask for it.
  `CaptureError::Minimised`; the OCR engine Windows ships with
  (`ocr::engine`).
- `kernels`: the correlation sums of template matching at the CPU's
  vector width — AVX2, SSE2 or plain loops, chosen once at run time, the
  same integers on every path; `SYRUP_SIMD` forces a narrower one.
- `template`: `TemplateSet`, `find_set`, `foreground`, `Template::mirrored`
  and the colour check; `Prepared` and `find_set_in`, sharing a frame's
  pyramids between searches and threads; `find_set_near`, the full-
  resolution look where a tracked thing is expected; a level whose
  candidates would cover a small search region is scored everywhere;
  the coarse floor is 0.15 under the requested score (measured, from
  0.25).
- `bars::BarModel`, `geometry::NormRect`, `glyphs` cells summed in eight
  lanes, `tracking::ObjectTracker` derives `Clone`, `motion` tracks the
  largest `max_blobs` of a busy frame and says how many it left out.
- `bars`, from a lava map: a model learns the fill's own least saturation
  and brightness (`min_saturation`, `min_value`) and a pixel of its hue
  but duller or darker — the scene behind a translucent track — is not
  fill; the hue tolerance is 16° (`HUE_TOLERANCE`), whatever an old
  model says; `learn` with a colour expected takes the nearest hue with
  any presence in the box, however small, and gives up when none is
  within `EXPECTED_WITHIN` (25°) of it, rather than take the scenery for
  the bar; `find_bar` finds a bar as a band of rows whose runs of the
  colour start together and are about as long (not the biggest blob of
  the colour, nor a line of it). `geometry::measure_bar_fill` says
  nothing (None) when there is nothing past the fill to sample.
- `glyphs`, a number printed over a partly filled bar: the empty track
  beside the fill, lighter than the fill the digits sit on, is a block of
  ink in the evidence, not a glyph — a span holding a filled square half
  the line height on a side is dropped; and the band of the line is
  narrowed to the rows of the glyphs that survive, so a bar's end or the
  track standing taller than the writing no longer stretches it. A font
  learned on a full bar reads the partial ones, and the other way round.
- `glyphs`, one font at two sizes: a template remembers the height of the
  line it was learned from, an example joins the template of its own size
  (within 20%, `SAME_SIZE_WITHIN`) or starts one, and a glyph is matched
  against the templates of its own size when there are any (a size never
  learned is read with all of them). A pixel font drawn at two sizes is
  two fonts — the strokes do not scale — and averaging them broke both:
  a HUD's small EXP line could not be learned beside its HP line.
  `chars()` names each character once.
- CI publishes each check job's logs to `build-output/<branch>/<os>`, and
  `[vendor]` in a commit message publishes the vendored crates to
  `build-cache/vendor`.

### Compiled operations by name: `syrup-runtime` (by Eitan, from 0xGh0stAn0n/syrup)
- `crates/syrup-runtime`: a name such as `find_2_largest_faces_in_top_half`
  is parsed against a composable grammar (counts, selectors, regions,
  orders, area filters; ambiguous or unknown words refused with a reason)
  into a typed plan whose coordinate spaces are checked, generated as a
  dependency-free Rust module, compiled with `rustc` (no cargo, no network),
  checked by a source policy, validated against a reference interpreter on
  24 synthetic cases, published atomically into a content-addressed cache
  (SHA-256 checked before every load) and loaded. Frozen mode and bundles
  for machines without a compiler; `syrup explain | source | run | watch |
  windows | bundle | cache`.
- Targets: YuNet faces (2023mar, pinned by SHA-256, run by tract),
  Tesseract words with boxes, QR codes (rqrr), colour regions and bars
  (the per-pixel test generated per colour), text blocks, panels;
  `measure_sharpness`, `measure_fill`; `track_*` sessions with ids,
  including moving regions.
- `syrup-cv` for Python (PyO3, abi3 wheels): `from syrup.ops import
  find_largest_face`, `syrup.define`, `syrup.add_target` for detectors
  written in Python, YOLO and MediaPipe recipes, an optional planner
  constrained to Syrup's vocabulary, `syrup.capture.window(title)`.
  `syrup.find_face` and the other names on the package are the in-process
  intents, now `syrup.legacy`.
- Core: `ocr::recognize` says why recognition failed (`OcrError`) and gives
  word boxes in image coordinates (`ocr_region` unchanged);
  `ObjectTracker::assign` gives each detection's track id;
  `motion::extract_blobs` over a motion mask; window capture on macOS 14+,
  X11 and Wayland (screen-cast portal and PipeWire) besides Windows, as a
  `capture::Window` kept between frames, with
  `capture_window_by_title_info`, `list_windows` and `capture_screen`
  unchanged.
- CI: the Python package on three platforms, Wayland capture against
  PipeWire, the recipes on real photographs; wheels built and tested on
  install, published on a `v*` tag.

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
  Windows; on a Windows runner the coach watched a game shown on the
  screen through, all 18 positions, and said what the game called for.

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
