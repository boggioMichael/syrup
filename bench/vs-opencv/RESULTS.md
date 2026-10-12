# Syrup vs OpenCV — results (2026-10-12)

One working day on one machine: what was compared, how, what changed in
Syrup, and what came out on data that no optimisation decision saw —
wins and losses alike. The tasks, data split, metrics and thresholds were
fixed before any comparison ran (`PREREGISTRATION.md`); every later change
and its reason is in `AMENDMENTS.md`. Raw per-call timings and outputs,
the environment and the exact commands are in `results/` beside this file.

## Verdict

On this machine, on held-out data, with the protocol fixed in advance:

**Syrup is superior in the tasks and conditions tested for:**

- **Motion regions (F4).** The moved-pixel mask was identical to OpenCV's
  on all 267 held-out frame pairs. With one thread, Syrup is **3.22× faster
  at p50** (0.58 vs 1.86 ms, 95% CI 2.79–3.49) and **2.81× at p95** (0.88 vs
  2.46 ms, CI 2.69–3.08). It is faster on each of the three held-out videos:
  2.10–3.92× at p50 and 1.79–3.28× at p95. With two threads the figures are
  3.94× at p50 and 3.52× at p95. OpenCV ran in its best configuration found
  on dev: reused buffers and Grana's connected components. The regions
  themselves are grouped by different rules, so their F1 against OpenCV's
  connected components is 0.47; that difference is by design, not an error.
- **Haar face detection with OpenCV's own flat-window rule (F1,
  `min_deviation: Some(10.0)`, opt-in).** One thread: **1.19× faster at
  p50** (167 vs 200 ms, CI 1.14–1.28) and **1.29× at p95** (893 vs
  1150 ms, CI 1.13–1.36). Two threads: 1.30× at p50 and 1.32× at p95. Its
  boxes are identical to Syrup's default on all 15 photographs, and its F1
  against OpenCV's boxes is 0.985 (precision 0.990, recall 0.980, mean IoU
  0.946).

**Faster, but not as exact: template matching (F3).**

- Speed, one thread: **1.51× at p50** (2.52 vs 3.82 ms, CI 1.33–1.70) and
  **1.73× at p95** (11.2 vs 19.3 ms, CI 0.98–2.54, which touches parity).
- Accuracy: Syrup's coarse-to-fine search found the true position in
  98.8% of the 480 searches; OpenCV's exhaustive search found 100.0%.
- That **misses the pre-registered tolerance of 1 percentage point** (by
  0.2 points). The gap predates today: Syrup before had the same 98.8%.
- Verdict: a speed/accuracy trade-off, not an advantage.

**Near parity, slightly behind: Haar faces with Syrup's default scan (F1).**

- One thread: 0.87× at p50 (CI 0.73–1.06) and 0.90× at p95 (CI 0.78–1.00).
- Two threads: 0.97× at p50 and 0.89× at p95.
- The default evaluates windows OpenCV skips. That extra work also means
  it finds faces OpenCV misses: low-contrast faces with a window deviation
  of 10 grey levels or less, which a test now pins.

**Losses.**

- **YuNet (F2):** 5.3× slower at p50 with one thread and 7.6× with two,
  though the outputs are identical. The cause is Syrup's runtime (tract
  0.23): its generic depthwise convolution is 56% of the network's
  instructions.
- **Synthetic glyph reading (F5):**
  - Syrup read 48.2% of 400 lines right, made 0 wrong readings, and refused
    51.7%.
  - OpenCV's basic recipe read 93.5% right and 6.5% wrong.
  - Syrup could not learn the 11-pixel size of this (non-pixel) font, and
    was slower per call (0.11 vs 0.06 ms at p50).

**Syrup against itself, today.** Before and after gave the same outputs on
every input.

- **Haar faces, p95:** −31% with one thread and −53% with two.
- **Template matching, p95:** −88% (93.5 → 11.2 ms) and p99 1056 → 23 ms.
- **The lip-reading demo:** per-frame time is cut 41–48% per fixture, with
  every quality column unchanged. Its gap to OpenCV narrows from 2.1–3.9×
  to 1.1–2.0×.

This is not evidence of general superiority over OpenCV. It covers five
tasks, one 2-core VM, and processing time only, which is not product
response time. There is no evidence here about any game other than
MapleStory; GTA VI was not tested.


## Scope

**Machine.** 2 vCPU Intel Xeon @ 2.10 GHz (AVX2, AVX-512F) in a Firecracker
VM, 7.8 GiB, no GPU; Ubuntu 24.04.5, Linux 6.18. Shared host: run-to-run
noise is real, hence 5 rounds in rotated order and intervals below.

**Software.** Syrup before = `156475f` (the library as ms `8a14965` pins it,
`54dad1e`, plus the harness and `motion::regions`); Syrup after = `54b7c73`;
both built by `build.sh` with the same harness, rustc 1.97.0, release
profile, no LTO, default target CPU (SIMD chosen at run time). OpenCV 5.0.0
(`opencv-contrib-python` 5.0.0.93, pthreads, AVX2/AVX-512 dispatch), from
Python 3.13.15 / numpy 2.5.3; the Python call overhead was measured
(~0.8–1.0 µs per call) and is negligible next to every timing reported.
OpenCV runs in its optimised configuration: full-frame buffers reused
between calls, the fastest connected-components algorithm that gave the
same output on dev data (amendment 2).

**What is timed.** The library call only, on input already decoded into
each library's own format (Syrup: RGBA or grey `image` buffers; OpenCV:
BGR or grey numpy arrays). 3 untimed warm-up calls then 5 timed calls per
image; videos once per round. Initialisation (loading a model or cascade,
the first call) is reported separately. 1 thread (primary) and 2 threads.
Peak RSS includes the decoded inputs each harness holds (all frames of a
video for motion), so it compares harnesses more than libraries.

**Statistics.** p50/p95/p99 over all timed calls of a task and data source
(5 rounds pooled); the spread of each round's p95; 95% bootstrap intervals
that resample *inputs* — photographs, source images, or blocks of 10
consecutive frame pairs — never single adjacent frames.

| family | task | kind of comparison | dev (optimisation) | test (held out) | quality measure |
|---|---|---|---|---|---|
| faces | F1 Haar cascade, `frontalface_alt2`, scale 1.1, 3 neighbours, from 20 px | same model, same algorithm | 45 GRID video frames (3 videos) | 15 photographs: 13 from `opencv_extra` `cascadeandhog`, `lena`, the astronaut | precision/recall/F1 and IoU of boxes against OpenCV's |
| faces | F2 YuNet 2023mar, score 0.6, NMS 0.3, 640-px canvas | same model, different runtimes | the same 45 frames | the same 15 photographs | F1, IoU, landmark distance against OpenCV's |
| UI state | F3 best match of a template (zero-mean NCC) | same score; OpenCV searches exhaustively, Syrup coarse-to-fine where the template allows | 420 searches in MapleStory frames | 480 searches in 24 photographs (OpenCV samples, face photographs) | hit within 1 px of the true position; position and score against OpenCV's |
| motion | F4 moved-pixel mask and changed regions of consecutive frames | same mask formula; different grouping of regions | 100 pairs of a 1366×768 MapleStory recording | 267 pairs of `vtest`, `tree`, `Megamind` | mask IoU (expected 1), region F1 (different rules: not pass/fail) |
| text | F5 fixed-font line reading, learned from labelled lines | different solutions (Syrup `glyphs` vs Otsu + components + k-NN) | 120 synthetic lines | 400 synthetic lines (no labelled real non-game data offline) | exact-string accuracy, wrong reads, refusals |
| faces, applied | F1-pipeline: the `lipreader` evaluation, 7 GRID fixtures | the same application with each detector | (its fixtures overlap F1's dev videos: not held out) | — | every quality column must stay the same |

Not compared: tracking (Syrup's multi-object tracker of motion regions has
no OpenCV counterpart for the same task — OpenCV's trackers follow one
object from a given box), capture, Tesseract OCR, anything on a GPU.

## What changed in Syrup (before `156475f` → after `54b7c73`)

Every change keeps the output identical — tests compare the new code with
the old computation, and on all dev and test inputs the before and after
binaries returned the same boxes, matches, masks and readings (below).

| commit | change | why (dev profile) | equivalence test |
|---|---|---|---|
| `1ba36d2` | template: peaks picked in linear time (a kept peak marks its cell; 25 look-ups per peak), and the best one without a sort | 97% of a 16-px template search was a quadratic crowding check over ~100,000 peaks | the new `local_maxima` against the quadratic one on 300 random score maps full of ties and signed zeros, 6 limits × 3 floors; `find_best` = first of all matches |
| `031f5c4` | cascade: the scale pyramid resized by Syrup's own byte-exact triangle filter (one channel, row-major, AVX2 vertical pass); one bounds check per window instead of one per look-up | the `image` crate's resize was 39% of a detection's instructions; bounds checks a third of the evaluation's | byte equality with `image::imageops::resize(.., Triangle)` on 400 random sizes and down a cascade pyramid; rounding against `f32::round` |
| `031f5c4` | cascade: `CascadeOptions::min_deviation` (default `None`: unchanged) | OpenCV rejects windows with a grey deviation ≤ 10 before the first stage (`cascadedetect.cpp` 716–742); Syrup evaluates them | a low-contrast face found by the plain scan, missed by OpenCV 5.0 and by the gate; at full contrast the gate changes nothing |
| `703c80d` | template: exhaustive scoring with two template pixels per 16-bit multiply-add (AVX2), 8 positions a vector | the exhaustive path multiplied one template pixel at a time | AVX2 path against the row-by-row path, 10 template shapes and the largest sums (32-bit lanes past 2³¹) |
| `7a7fefc` | cascade: threads take whole scales (resize, integrals, scan), costliest first, results joined in scale order | two threads gave 1.2×: per-scale band spawns, the resize and integrals on one thread | the same windows with 1, 2, 3, 8 and 64 threads, with and without the gate |
| `54b7c73` | template: the exhaustive best match found while scoring, without the score map | the peak search over a full score map, when only the best is wanted | covered by the two template tests above |

Nine tests were added, and the whole suite passes (`cargo test
--release`): 196 unit tests and every integration test. The four
`intent_native` tests compile generated crates, so they need crates.io or
a vendored-crates cargo configuration; with the vendored one they passed
here.

## Results on the held-out test data

Generated by `report.py` from `results/test-t1`, `results/test-t2` and `results/lipreader`. `OpenCV ÷ this` above 1 means faster than OpenCV; intervals are 95% bootstrap over inputs. Rows per data source and, for templates, per template side.

### 1 thread

| task | data | implementation | p50 ms | p95 ms | p99 ms | OpenCV ÷ this, p50 [95% CI] | OpenCV ÷ this, p95 [95% CI] | p95 vs Syrup before [95% CI] |
|---|---|---|---|---|---|---|---|---|
| faces | all | syrup-before | 302 | 1864 | 2143 | 0.66 [0.55–0.83] | 0.62 [0.57–0.69] | – |
| faces | all | syrup-after | 230 | 1281 | 1604 | 0.87 [0.73–1.06] | 0.90 [0.78–1.00] | -31% [-33, -26] |
| faces | all | syrup-after-gate | 167 | 893 | 1059 | 1.19 [1.14–1.28] | 1.29 [1.13–1.36] | -52% [-57, -40] |
| faces | all | opencv | 200 | 1150 | 1227 | – | – | – |
| faces | opencv_extra | syrup-before | 306 | 1920 | 2161 | 0.68 [0.53–0.86] | 0.60 [0.57–0.69] | – |
| faces | opencv_extra | syrup-after | 244 | 1318 | 1652 | 0.86 [0.67–1.10] | 0.88 [0.78–1.00] | -31% [-31, -26] |
| faces | opencv_extra | syrup-after-gate | 175 | 909 | 1073 | 1.19 [1.14–1.32] | 1.27 [1.11–1.36] | -53% [-58, -39] |
| faces | opencv_extra | opencv | 209 | 1158 | 1230 | – | – | – |
| faces | opencv_samples | syrup-before | 325 | 369 | 384 | 0.61 | 0.59 | – |
| faces | opencv_samples | syrup-after | 228 | 260 | 277 | 0.87 | 0.84 | -29% |
| faces | opencv_samples | syrup-after-gate | 164 | 207 | 213 | 1.21 | 1.06 | -44% |
| faces | opencv_samples | opencv | 198 | 219 | 246 | – | – | – |
| faces | syrup | syrup-before | 96.6 | 110 | 120 | 0.81 | 0.82 | – |
| faces | syrup | syrup-after | 73.4 | 97.5 | 99.4 | 1.07 | 0.92 | -12% |
| faces | syrup | syrup-after-gate | 68.1 | 100 | 103 | 1.15 | 0.90 | -9% |
| faces | syrup | opencv | 78.5 | 90.2 | 92.2 | – | – | – |
| yunet | all | syrup | 102 | 168 | 215 | 0.19 [0.17–0.21] | 0.15 [0.13–0.18] | – |
| yunet | all | opencv | 19.0 | 25.7 | 28.1 | – | – | – |
| yunet | opencv_extra | syrup | 98.9 | 169 | 222 | 0.19 [0.17–0.22] | 0.15 [0.13–0.19] | – |
| yunet | opencv_extra | opencv | 18.8 | 25.7 | 27.9 | – | – | – |
| yunet | opencv_samples | syrup | 122 | 138 | 155 | 0.19 | 0.19 | – |
| yunet | opencv_samples | opencv | 23.4 | 26.9 | 28.5 | – | – | – |
| yunet | syrup | syrup | 111 | 162 | 172 | 0.21 | 0.15 | – |
| yunet | syrup | opencv | 23.1 | 24.2 | 24.7 | – | – | – |
| templates | all | syrup-before | 3.5 | 93.5 | 1056 | 1.08 [0.85–1.25] | 0.21 [0.02–0.42] | – |
| templates | all | syrup-after | 2.5 | 11.2 | 22.5 | 1.51 [1.33–1.70] | 1.73 [0.98–2.54] | -88% [-99, -79] |
| templates | all | opencv | 3.8 | 19.3 | 28.8 | – | – | – |
| templates | opencv_extra | syrup-before | 4.0 | 275 | 2090 | 0.92 [0.69–1.24] | 0.09 [0.02–0.49] | – |
| templates | opencv_extra | syrup-after | 2.8 | 12.0 | 22.5 | 1.32 [1.15–1.69] | 2.05 [1.07–3.32] | -96% [-99, -82] |
| templates | opencv_extra | opencv | 3.7 | 24.6 | 33.8 | – | – | – |
| templates | opencv_samples | syrup-before | 3.1 | 48.8 | 89.3 | 1.24 [1.02–1.51] | 0.20 [0.11–0.34] | – |
| templates | opencv_samples | syrup-after | 2.4 | 8.9 | 22.2 | 1.65 [1.47–1.98] | 1.07 [0.55–1.61] | -82% [-85, -75] |
| templates | opencv_samples | opencv | 3.9 | 9.5 | 13.0 | – | – | – |
| templates | side 16 | syrup-before | 21.3 | 899 | 3364 | 0.18 [0.14–0.23] | 0.02 [0.01–0.19] | – |
| templates | side 16 | syrup-after | 3.7 | 15.8 | 21.5 | 1.01 [0.91–1.13] | 1.22 [0.98–1.55] | -98% [-99, -85] |
| templates | side 16 | opencv | 3.8 | 19.3 | 28.7 | – | – | – |
| templates | side 32 | syrup-before | 3.1 | 44.4 | 981 | 1.26 [1.04–1.39] | 0.43 [0.14–1.62] | – |
| templates | side 32 | syrup-after | 2.5 | 9.6 | 22.5 | 1.57 [1.32–1.74] | 2.00 [0.85–2.88] | -78% [-84, -36] |
| templates | side 32 | opencv | 3.9 | 19.1 | 30.9 | – | – | – |
| templates | side 48 | syrup-before | 2.0 | 14.0 | 99.7 | 1.93 [1.72–2.23] | 1.34 [0.11–4.33] | – |
| templates | side 48 | syrup-after | 1.9 | 9.9 | 28.2 | 2.07 [1.79–2.34] | 1.90 [0.43–5.22] | -30% [-76, +7] |
| templates | side 48 | opencv | 3.9 | 18.8 | 29.0 | – | – | – |
| templates | side 64 | syrup-before | 2.1 | 5.9 | 7.8 | 1.82 [1.63–2.11] | 3.32 [1.57–4.19] | – |
| templates | side 64 | syrup-after | 2.0 | 4.7 | 5.6 | 1.92 [1.70–2.27] | 4.18 [2.12–5.26] | -21% [-36, -6] |
| templates | side 64 | opencv | 3.8 | 19.7 | 26.8 | – | – | – |
| motion | all | syrup-before | 0.589 | 0.912 | 1.1 | 3.16 [2.78–3.42] | 2.69 [2.43–2.91] | – |
| motion | all | syrup-after | 0.577 | 0.876 | 0.939 | 3.22 [2.79–3.49] | 2.81 [2.69–3.08] | -4% [-12, -2] |
| motion | all | opencv | 1.9 | 2.5 | 2.8 | – | – | – |
| motion | Megamind | syrup-before | 0.719 | 1.0 | 1.2 | 2.43 [2.25–2.64] | 2.29 [2.10–2.54] | – |
| motion | Megamind | syrup-after | 0.719 | 0.904 | 1.2 | 2.43 [2.25–2.65] | 2.58 [2.38–2.74] | -11% [-18, -3] |
| motion | Megamind | opencv | 1.7 | 2.3 | 2.7 | – | – | – |
| motion | tree | syrup-before | 0.188 | 0.290 | 0.375 | 2.03 [1.82–2.44] | 1.86 [1.59–2.13] | – |
| motion | tree | syrup-after | 0.181 | 0.301 | 0.444 | 2.10 [1.79–2.36] | 1.79 [1.67–2.24] | +4% [-8, +8] |
| motion | tree | opencv | 0.380 | 0.540 | 0.718 | – | – | – |
| motion | vtest | syrup-before | 0.587 | 0.843 | 0.899 | 3.78 [3.65–3.98] | 3.02 [2.85–3.20] | – |
| motion | vtest | syrup-after | 0.566 | 0.778 | 0.872 | 3.92 [3.74–4.12] | 3.28 [3.00–3.73] | -8% [-16, -3] |
| motion | vtest | opencv | 2.2 | 2.5 | 3.0 | – | – | – |
| glyphs | all | syrup-before | 0.100 | 0.436 | 0.669 | 0.62 [0.55–0.67] | 0.28 [0.25–0.33] | – |
| glyphs | all | syrup-after | 0.113 | 0.481 | 0.788 | 0.55 [0.50–0.59] | 0.26 [0.24–0.28] | +10% [+5, +17] |
| glyphs | all | opencv | 0.062 | 0.123 | 0.162 | – | – | – |
| glyphs | s050/bar | syrup-before | 0.237 | 0.578 | 0.916 | 0.23 [0.21–0.27] | 0.20 [0.17–0.24] | – |
| glyphs | s050/bar | syrup-after | 0.260 | 0.675 | 0.937 | 0.21 [0.20–0.25] | 0.17 [0.15–0.21] | +17% [+7, +22] |
| glyphs | s050/bar | opencv | 0.055 | 0.117 | 0.149 | – | – | – |
| glyphs | s050/flat | syrup-before | 0.139 | 0.479 | 0.643 | 0.39 [0.30–0.49] | 0.23 [0.21–0.29] | – |
| glyphs | s050/flat | syrup-after | 0.153 | 0.493 | 0.846 | 0.36 [0.28–0.43] | 0.22 [0.20–0.25] | +3% [-2, +21] |
| glyphs | s050/flat | opencv | 0.055 | 0.109 | 0.137 | – | – | – |
| glyphs | s050/noise | syrup-before | 0.161 | 0.488 | 0.690 | 0.34 [0.29–0.43] | 0.25 [0.20–0.30] | – |
| glyphs | s050/noise | syrup-after | 0.180 | 0.542 | 0.924 | 0.31 [0.25–0.38] | 0.22 [0.20–0.26] | +11% [-0, +23] |
| glyphs | s050/noise | opencv | 0.055 | 0.121 | 0.167 | – | – | – |
| glyphs | s075/bar | syrup-before | 0.067 | 0.184 | 0.600 | 0.98 [0.87–1.20] | 0.72 [0.52–0.86] | – |
| glyphs | s075/bar | syrup-after | 0.073 | 0.226 | 0.555 | 0.91 [0.84–1.06] | 0.58 [0.39–0.75] | +23% [+8, +37] |
| glyphs | s075/bar | opencv | 0.066 | 0.132 | 0.180 | – | – | – |
| glyphs | s075/flat | syrup-before | 0.055 | 0.181 | 0.224 | 1.14 [0.92–1.25] | 0.67 [0.60–0.80] | – |
| glyphs | s075/flat | syrup-after | 0.065 | 0.184 | 0.289 | 0.97 [0.88–1.07] | 0.66 [0.55–0.78] | +2% [-6, +16] |
| glyphs | s075/flat | opencv | 0.063 | 0.122 | 0.158 | – | – | – |
| glyphs | s075/noise | syrup-before | 0.070 | 0.150 | 0.197 | 0.97 [0.87–1.18] | 0.90 [0.77–1.09] | – |
| glyphs | s075/noise | syrup-after | 0.072 | 0.183 | 0.270 | 0.94 [0.86–1.10] | 0.74 [0.61–0.90] | +22% [+8, +34] |
| glyphs | s075/noise | opencv | 0.068 | 0.135 | 0.162 | – | – | – |

Process and initialisation, 1 thread (peak RSS includes the decoded inputs each harness holds; the Python side also numpy and the interpreter; 'first call' is the first warm-up call on the first input, model or cascade preparation included):

| task / implementation | CPU ÷ wall | peak RSS MiB | init: load ms | init: first call ms |
|---|---|---|---|---|
| faces/opencv | 0.98 | 135.4 | 6.11 | 568.1 |
| faces/syrup-after | 0.98 | 23.9 | 0.12 | 704.2 |
| faces/syrup-after-gate | 0.98 | 23.9 | 0.12 | 411.2 |
| faces/syrup-before | 0.99 | 40.5 | 0.12 | 970.3 |
| glyphs/opencv | 0.99 | 64.8 | 1.01 | 0.1 |
| glyphs/syrup-after | 0.97 | 4.4 | 1.72 | 0.3 |
| glyphs/syrup-before | 0.96 | 4.5 | 1.62 | 0.2 |
| motion/opencv | 0.99 | 206.9 | – | – |
| motion/syrup-after | 0.99 | 177.0 | – | – |
| motion/syrup-before | 0.97 | 177.1 | – | – |
| templates/opencv | 0.99 | 107.3 | – | 6.3 |
| templates/syrup-after | 0.99 | 73.3 | – | 4.2 |
| templates/syrup-before | 0.99 | 79.1 | – | 26.2 |
| yunet/opencv | 0.99 | 98.2 | 3.28 | 40.9 |
| yunet/syrup | 0.99 | 57.7 | 0.00 | 169.5 |

Quality, one thread. Every output was the same with two threads, except that OpenCV listed the same face boxes in a varying order (7 of 15 photographs; sorted, they are identical).

- F1 syrup-before vs OpenCV's boxes: precision 0.990, recall 0.980, F1 0.985, mean IoU 0.946 (101 vs 102 boxes)
- F1 syrup-after vs OpenCV's boxes: precision 0.990, recall 0.980, F1 0.985, mean IoU 0.946 (101 vs 102 boxes)
- F1 syrup-after-gate vs OpenCV's boxes: precision 0.990, recall 0.980, F1 0.985, mean IoU 0.946 (101 vs 102 boxes)
- F1 syrup-after vs syrup-before: identical boxes on 15 of 15 images, F1 1.000, box count change +0.0%
- F1 syrup-after-gate vs syrup-before: identical boxes on 15 of 15 images, F1 1.000, box count change +0.0%
- F2 Syrup YuNet vs OpenCV's: F1 1.000, mean IoU 1.0000, mean landmark difference 0.000 px, largest score difference 0.0000
- F3 opencv: hit rate 100.0% (480/480); clean 100.0%, opencv_extra 100.0%, opencv_samples 100.0%, perturbed 100.0%, side 16 100.0%, side 32 100.0%, side 48 100.0%, side 64 100.0%
- F3 syrup-after: hit rate 98.8% (474/480); clean 98.8%, opencv_extra 98.5%, opencv_samples 99.1%, perturbed 98.8%, side 16 100.0%, side 32 97.2%, side 48 97.9%, side 64 100.0%
- F3 syrup-before: hit rate 98.8% (474/480); clean 98.8%, opencv_extra 98.5%, opencv_samples 99.1%, perturbed 98.8%, side 16 100.0%, side 32 97.2%, side 48 97.9%, side 64 100.0%
- F3 syrup-after vs OpenCV: same position in 474 of 480; score difference mean 5.03e-04, max 5.58e-02
- F3 syrup-before vs OpenCV: same position in 474 of 480; score difference mean 5.03e-04, max 5.58e-02
- F3 syrup-after vs syrup-before: identical output in 480 of 480 searches
- F4 mask: pooled IoU 1.000000, minimum 1.000000, identical on 267 of 267 pairs
- F4 mask (before): pooled IoU 1.000000, minimum 1.000000, identical on 267 of 267 pairs
- F4 regions (different grouping rules) vs OpenCV's: F1 0.474 (6180 vs 3988)
- F4 syrup-after vs syrup-before: identical output on 258 of 258 pairs
- F5 opencv: 374/400 read right (93.5%), 26 wrong (6.5%), 0 refused (0.0%) — s050: 174/200 right, 26 wrong, 0 refused; s075: 200/200 right, 0 wrong, 0 refused
- F5 syrup-after: 193/400 read right (48.2%), 0 wrong (0.0%), 207 refused (51.7%) — s050: 36/200 right, 0 wrong, 164 refused; s075: 157/200 right, 0 wrong, 43 refused
- F5 syrup-before: 193/400 read right (48.2%), 0 wrong (0.0%), 207 refused (51.7%) — s050: 36/200 right, 0 wrong, 164 refused; s075: 157/200 right, 0 wrong, 43 refused
- F5 syrup-after could not learn: glyphs/test/train_s050.png: learned on trial, the font reads the region as "0123456719%. (not surely)", not the label
- F5 syrup-before could not learn: glyphs/test/train_s050.png: learned on trial, the font reads the region as "0123456719%. (not surely)", not the label

### 2 threads

| task | data | implementation | p50 ms | p95 ms | p99 ms | OpenCV ÷ this, p50 [95% CI] | OpenCV ÷ this, p95 [95% CI] | p95 vs Syrup before [95% CI] |
|---|---|---|---|---|---|---|---|---|
| faces | all | syrup-before | 208 | 1404 | 1562 | 0.53 [0.43–0.61] | 0.42 [0.42–0.53] | – |
| faces | all | syrup-after | 113 | 657 | 724 | 0.97 [0.80–1.13] | 0.89 [0.82–1.00] | -53% [-53, -44] |
| faces | all | syrup-after-gate | 84.7 | 443 | 518 | 1.30 [1.18–1.33] | 1.32 [1.18–1.45] | -68% [-68, -57] |
| faces | all | opencv | 110 | 587 | 662 | – | – | – |
| faces | opencv_extra | syrup-before | 213 | 1431 | 1584 | 0.53 [0.39–0.63] | 0.41 [0.41–0.53] | – |
| faces | opencv_extra | syrup-after | 115 | 664 | 725 | 0.98 [0.74–1.14] | 0.89 [0.82–1.00] | -54% [-54, -44] |
| faces | opencv_extra | syrup-after-gate | 91.0 | 463 | 519 | 1.24 [1.15–1.30] | 1.28 [1.18–1.45] | -68% [-68, -57] |
| faces | opencv_extra | opencv | 113 | 591 | 668 | – | – | – |
| faces | opencv_samples | syrup-before | 212 | 238 | 246 | 0.52 | 0.55 | – |
| faces | opencv_samples | syrup-after | 108 | 148 | 205 | 1.03 | 0.89 | -38% |
| faces | opencv_samples | syrup-after-gate | 82.1 | 92.2 | 94.1 | 1.35 | 1.43 | -61% |
| faces | opencv_samples | opencv | 111 | 132 | 155 | – | – | – |
| faces | syrup | syrup-before | 79.4 | 110 | 118 | 0.50 | 0.40 | – |
| faces | syrup | syrup-after | 37.1 | 42.5 | 54.4 | 1.07 | 1.03 | -61% |
| faces | syrup | syrup-after-gate | 33.7 | 45.7 | 64.3 | 1.18 | 0.95 | -59% |
| faces | syrup | opencv | 39.6 | 43.6 | 44.6 | – | – | – |
| yunet | all | syrup | 106 | 178 | 228 | 0.13 [0.12–0.14] | 0.12 [0.09–0.14] | – |
| yunet | all | opencv | 14.0 | 21.2 | 23.3 | – | – | – |
| yunet | opencv_extra | syrup | 98.5 | 183 | 236 | 0.14 [0.12–0.15] | 0.11 [0.09–0.14] | – |
| yunet | opencv_extra | opencv | 13.4 | 20.5 | 22.7 | – | – | – |
| yunet | opencv_samples | syrup | 120 | 163 | 170 | 0.15 | 0.14 | – |
| yunet | opencv_samples | opencv | 18.1 | 23.3 | 27.3 | – | – | – |
| yunet | syrup | syrup | 114 | 158 | 164 | 0.15 | 0.13 | – |
| yunet | syrup | opencv | 17.4 | 21.1 | 21.8 | – | – | – |
| templates | all | syrup-before | 3.5 | 94.3 | 1078 | 1.09 [0.89–1.25] | 0.18 [0.02–0.39] | – |
| templates | all | syrup-after | 2.6 | 11.7 | 22.9 | 1.46 [1.34–1.61] | 1.45 [0.90–2.34] | -88% [-98, -79] |
| templates | all | opencv | 3.8 | 17.0 | 28.1 | – | – | – |
| templates | opencv_extra | syrup-before | 4.1 | 306 | 2059 | 0.93 [0.73–1.21] | 0.08 [0.02–0.45] | – |
| templates | opencv_extra | syrup-after | 2.9 | 12.5 | 22.9 | 1.34 [1.22–1.70] | 1.95 [1.07–3.14] | -96% [-99, -82] |
| templates | opencv_extra | opencv | 3.8 | 24.4 | 32.7 | – | – | – |
| templates | opencv_samples | syrup-before | 3.1 | 48.7 | 93.5 | 1.25 [1.03–1.45] | 0.20 [0.10–0.34] | – |
| templates | opencv_samples | syrup-after | 2.4 | 10.1 | 22.8 | 1.58 [1.38–1.70] | 0.97 [0.49–1.49] | -79% [-85, -74] |
| templates | opencv_samples | opencv | 3.8 | 9.7 | 13.7 | – | – | – |
| templates | side 16 | syrup-before | 21.7 | 895 | 3407 | 0.17 [0.13–0.23] | 0.02 [0.01–0.17] | – |
| templates | side 16 | syrup-after | 3.8 | 16.5 | 22.6 | 1.00 [0.90–1.09] | 0.99 [0.91–1.37] | -98% [-99, -84] |
| templates | side 16 | opencv | 3.8 | 16.4 | 28.7 | – | – | – |
| templates | side 32 | syrup-before | 3.2 | 43.6 | 976 | 1.18 [1.01–1.31] | 0.40 [0.12–1.48] | – |
| templates | side 32 | syrup-after | 2.4 | 10.3 | 22.9 | 1.53 [1.29–1.65] | 1.71 [0.75–2.75] | -76% [-85, -36] |
| templates | side 32 | opencv | 3.7 | 17.6 | 30.2 | – | – | – |
| templates | side 48 | syrup-before | 2.0 | 14.8 | 102 | 2.01 [1.80–2.31] | 1.14 [0.11–4.36] | – |
| templates | side 48 | syrup-after | 1.9 | 8.6 | 26.9 | 2.12 [1.87–2.37] | 1.98 [0.45–4.58] | -42% [-76, +3] |
| templates | side 48 | opencv | 3.9 | 17.0 | 26.0 | – | – | – |
| templates | side 64 | syrup-before | 2.1 | 5.7 | 8.4 | 1.81 [1.68–2.19] | 3.16 [1.66–4.21] | – |
| templates | side 64 | syrup-after | 2.0 | 4.9 | 6.3 | 1.96 [1.82–2.31] | 3.69 [2.03–4.79] | -14% [-24, -0] |
| templates | side 64 | opencv | 3.8 | 18.0 | 27.2 | – | – | – |
| motion | all | syrup-before | 0.596 | 0.822 | 1.1 | 3.73 [3.51–4.11] | 3.62 [3.46–3.92] | – |
| motion | all | syrup-after | 0.566 | 0.847 | 0.994 | 3.94 [3.70–4.34] | 3.52 [3.25–3.89] | +3% [-2, +8] |
| motion | all | opencv | 2.2 | 3.0 | 3.6 | – | – | – |
| motion | Megamind | syrup-before | 0.697 | 0.900 | 1.4 | 3.47 [3.21–3.69] | 3.29 [2.96–3.73] | – |
| motion | Megamind | syrup-after | 0.697 | 0.925 | 1.1 | 3.47 [3.22–3.75] | 3.19 [3.05–3.50] | +3% [-5, +11] |
| motion | Megamind | opencv | 2.4 | 3.0 | 3.6 | – | – | – |
| motion | tree | syrup-before | 0.172 | 0.266 | 0.315 | 2.79 [2.29–3.22] | 2.67 [2.39–3.40] | – |
| motion | tree | syrup-after | 0.164 | 0.300 | 0.444 | 2.93 [2.25–3.41] | 2.37 [2.14–3.53] | +13% [-5, +21] |
| motion | tree | opencv | 0.480 | 0.710 | 0.896 | – | – | – |
| motion | vtest | syrup-before | 0.596 | 0.800 | 0.845 | 3.99 [3.79–4.36] | 3.95 [3.71–4.28] | – |
| motion | vtest | syrup-after | 0.553 | 0.698 | 0.744 | 4.30 [4.07–4.54] | 4.53 [4.29–4.83] | -13% [-15, -9] |
| motion | vtest | opencv | 2.4 | 3.2 | 3.8 | – | – | – |
| glyphs | all | syrup-before | 0.118 | 0.511 | 0.853 | 0.67 [0.61–0.73] | 0.29 [0.26–0.32] | – |
| glyphs | all | syrup-after | 0.108 | 0.501 | 0.834 | 0.73 [0.66–0.83] | 0.29 [0.26–0.32] | -2% [-10, +6] |
| glyphs | all | opencv | 0.079 | 0.147 | 0.205 | – | – | – |
| glyphs | s050/bar | syrup-before | 0.275 | 0.668 | 0.867 | 0.28 [0.25–0.31] | 0.23 [0.20–0.26] | – |
| glyphs | s050/bar | syrup-after | 0.263 | 0.737 | 1.2 | 0.29 [0.26–0.33] | 0.21 [0.16–0.24] | +10% [-10, +45] |
| glyphs | s050/bar | opencv | 0.076 | 0.152 | 0.226 | – | – | – |
| glyphs | s050/flat | syrup-before | 0.165 | 0.579 | 0.877 | 0.41 [0.35–0.50] | 0.24 [0.19–0.32] | – |
| glyphs | s050/flat | syrup-after | 0.171 | 0.521 | 0.837 | 0.40 [0.33–0.50] | 0.27 [0.22–0.32] | -10% [-25, +12] |
| glyphs | s050/flat | opencv | 0.068 | 0.141 | 0.215 | – | – | – |
| glyphs | s050/noise | syrup-before | 0.189 | 0.656 | 0.998 | 0.39 [0.32–0.48] | 0.23 [0.17–0.30] | – |
| glyphs | s050/noise | syrup-after | 0.178 | 0.570 | 0.836 | 0.41 [0.33–0.51] | 0.26 [0.23–0.32] | -13% [-33, +8] |
| glyphs | s050/noise | opencv | 0.073 | 0.149 | 0.201 | – | – | – |
| glyphs | s075/bar | syrup-before | 0.071 | 0.243 | 0.617 | 1.16 [1.06–1.27] | 0.62 [0.47–0.78] | – |
| glyphs | s075/bar | syrup-after | 0.063 | 0.178 | 0.544 | 1.31 [1.12–1.50] | 0.84 [0.57–1.02] | -27% [-32, -18] |
| glyphs | s075/bar | opencv | 0.083 | 0.150 | 0.193 | – | – | – |
| glyphs | s075/flat | syrup-before | 0.066 | 0.212 | 0.281 | 1.22 [1.10–1.31] | 0.67 [0.59–0.84] | – |
| glyphs | s075/flat | syrup-after | 0.058 | 0.197 | 0.246 | 1.40 [1.18–1.75] | 0.72 [0.65–0.97] | -7% [-21, +2] |
| glyphs | s075/flat | opencv | 0.080 | 0.142 | 0.181 | – | – | – |
| glyphs | s075/noise | syrup-before | 0.072 | 0.185 | 0.267 | 1.20 [1.10–1.27] | 0.81 [0.69–0.98] | – |
| glyphs | s075/noise | syrup-after | 0.068 | 0.153 | 0.225 | 1.27 [1.18–1.45] | 0.97 [0.86–1.17] | -17% [-24, -10] |
| glyphs | s075/noise | opencv | 0.086 | 0.149 | 0.202 | – | – | – |

Process and initialisation, 2 threads (peak RSS includes the decoded inputs each harness holds; the Python side also numpy and the interpreter; 'first call' is the first warm-up call on the first input, model or cascade preparation included):

| task / implementation | CPU ÷ wall | peak RSS MiB | init: load ms | init: first call ms |
|---|---|---|---|---|
| faces/opencv | 1.90 | 135.9 | 6.13 | 299.4 |
| faces/syrup-after | 1.96 | 41.8 | 0.13 | 376.9 |
| faces/syrup-after-gate | 1.95 | 39.1 | 0.12 | 230.2 |
| faces/syrup-before | 1.48 | 40.5 | 0.12 | 660.6 |
| glyphs/opencv | 1.96 | 65.0 | 1.20 | 0.1 |
| glyphs/syrup-after | 0.97 | 4.4 | 2.54 | 0.3 |
| glyphs/syrup-before | 0.97 | 4.4 | 1.76 | 0.2 |
| motion/opencv | 1.16 | 207.7 | – | – |
| motion/syrup-after | 0.99 | 177.0 | – | – |
| motion/syrup-before | 0.98 | 177.1 | – | – |
| templates/opencv | 0.99 | 107.4 | – | 7.4 |
| templates/syrup-after | 0.99 | 73.3 | – | 4.8 |
| templates/syrup-before | 0.99 | 79.2 | – | 25.2 |
| yunet/opencv | 1.82 | 98.3 | 3.61 | 40.9 |
| yunet/syrup | 0.99 | 58.2 | 0.00 | 164.4 |

### Lip-reading pipeline (F1-pipeline)

3 runs per detector, rotated order; ms per frame of the whole pipeline (detection, tracking, mouth crops, LipNet), median [range]; × = over OpenCV's median.

| fixture | opencv | syrup-after | syrup-before | quality columns as with syrup-before |
|---|---|---|---|---|
| five_faces | 83.2 [82.5–83.9] | 93.8 [92.1–94.0] (1.13×) | 174.9 [165.5–176.1] (2.10×) | syrup-after: yes |
| low_res | 18.3 [17.8–21.0] | 23.9 [22.7–25.8] (1.31×) | 40.3 [39.3–41.5] (2.21×) | syrup-after: yes |
| muted | 18.1 [17.8–19.5] | 34.1 [30.6–35.1] (1.89×) | 65.5 [60.5–67.1] (3.62×) | syrup-after: yes |
| one_speaker | 16.7 [16.2–17.1] | 33.8 [31.0–35.8] (2.02×) | 64.6 [64.2–70.1] (3.87×) | syrup-after: yes |
| rapid_cuts | 18.1 [18.0–18.3] | 32.2 [30.9–32.2] (1.78×) | 61.6 [60.6–61.9] (3.41×) | syrup-after: yes |
| two_alternating | 26.5 [25.8–26.9] | 39.9 [36.8–40.1] (1.50×) | 76.7 [75.0–77.0] (2.89×) | syrup-after: yes |
| two_simultaneous | 31.3 [30.6–33.2] | 44.5 [40.9–44.8] (1.42×) | 83.9 [81.7–85.9] (2.68×) | syrup-after: yes |

Quality columns (first run; every run gave the same: opencv yes, syrup-after yes, syrup-before yes)

| fixture | detector | WER | CER | attribution | track consistency | speaking P | speaking R |
|---|---|---|---|---|---|---|---|
| five_faces | opencv | 0.0 | 0.0 | 1.0 | 1.0 | 0.6613 | 0.9152 |
| five_faces | syrup-after | 0.0 | 0.0 | 1.0 | 1.0 | 0.6613 | 0.9152 |
| five_faces | syrup-before | 0.0 | 0.0 | 1.0 | 1.0 | 0.6613 | 0.9152 |
| low_res | opencv | 0.5 | 0.4444 | 1.0 | 1.0 | 0.7663 | 0.993 |
| low_res | syrup-after | 0.5 | 0.4722 | 1.0 | 1.0 | 0.7172 | 1.0 |
| low_res | syrup-before | 0.5 | 0.4722 | 1.0 | 1.0 | 0.7172 | 1.0 |
| muted | opencv | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| muted | syrup-after | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| muted | syrup-before | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| one_speaker | opencv | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| one_speaker | syrup-after | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| one_speaker | syrup-before | 0.0 | 0.0 | 1.0 | 1.0 | 0.7027 | 0.7429 |
| rapid_cuts | opencv | None | None | None | 1.0 | 1.0 | 0.575 |
| rapid_cuts | syrup-after | None | None | None | 1.0 | 1.0 | 0.475 |
| rapid_cuts | syrup-before | None | None | None | 1.0 | 1.0 | 0.475 |
| two_alternating | opencv | 0.0 | 0.0 | 1.0 | 1.0 | 0.7209 | 0.8732 |
| two_alternating | syrup-after | 0.0 | 0.0 | 1.0 | 1.0 | 0.7412 | 0.8873 |
| two_alternating | syrup-before | 0.0 | 0.0 | 1.0 | 1.0 | 0.7412 | 0.8873 |
| two_simultaneous | opencv | 0.0 | 0.0 | 1.0 | 1.0 | 0.6985 | 0.669 |
| two_simultaneous | syrup-after | 0.0 | 0.0 | 1.0 | 1.0 | 0.6985 | 0.669 |
| two_simultaneous | syrup-before | 0.0 | 0.0 | 1.0 | 1.0 | 0.6985 | 0.669 |


## Advantage, parity, loss, missing, unmeasured

| category | what | the numbers (test data, 1 thread unless said) |
|---|---|---|
| proven advantage | F4 motion regions | identical masks, 267/267 pairs; 3.22× p50, 2.81× p95, every video; 3.94× / 3.52× with 2 threads |
| proven advantage (opt-in rule) | F1 Haar faces with `min_deviation: Some(10.0)` — the rule OpenCV itself applies | 1.19× p50, 1.29× p95; 1.30× / 1.32× with 2 threads; F1 0.985 vs OpenCV, same boxes as Syrup's default |
| faster, below the accuracy bar | F3 template matching | 1.51× p50, 1.73× p95 (CI to 0.98); hit rate 98.8% vs 100% — pre-registered tolerance 1 point, missed by 0.2 |
| near parity, behind | F1 Haar faces, Syrup's default scan | 0.87× p50 (CI 0.73–1.06), 0.90× p95 (CI 0.78–1.00); finds low-contrast faces OpenCV drops |
| loss | F2 YuNet | 0.19× p50, 0.15× p95 (5.3× / 6.5× slower); 7.6× slower with 2 threads (tract runs on one) |
| loss | F5 synthetic glyph lines | 48.2% read right (0 wrong, 51.7% refused) vs 93.5% right, 6.5% wrong; 0.55× p50 |
| loss, narrowed | F1-pipeline, the lip-reading demo (2 threads, its default) | 1.13–2.02× OpenCV's ms/frame, from 2.10–3.87×; quality unchanged |
| missing in Syrup | an exact (exhaustive or FFT) template search for large regions; fast neural-network inference (generic depthwise convolution in tract); reading fonts that are not pixel fonts (it could not learn an 11-px Hershey line); single-object trackers (OpenCV's CSRT, KCF…: Syrup tracks motion regions, a different task, not compared); OpenCV's wider catalogue (calibration, features, optical flow, DNN zoo) — out of scope | |
| not measurable here | GPU paths (none on this machine); other CPUs, more cores, ARM; Windows (capture, its OCR); end-to-end product latency (no capture, network or voice in any number here — these are processing times); face *accuracy* against ground truth (F1 measures agreement with OpenCV; no labelled face set offline); real labelled non-game text; developer time; any game other than MapleStory — **GTA VI: no evidence, not tested** | |

Two more observations from the raw data:

- **Same output whatever the thread count.** Syrup gave identical outputs
  with one and two threads, and in every round. With two threads, OpenCV
  listed the same face boxes in a varying order on 7 of 15 photographs.
  Sorted, they are identical.
- **Small differences on unchanged code.** The library code behind F4 and
  F5 did not change, yet the after binary measured −4% (F4) and +10% (F5)
  at p95 with one thread. With two threads it was +3% and −2%. These are
  the size of difference this setup cannot attribute: binary layout and a
  shared host. Only effects well beyond them are claimed above.

## The lip-reading gap, isolated

**Reproduced.** The baseline measurements are in `results/baseline/`.
`example/docs/benchmarks.md` records 152/34/57/57/55/70/71 ms per frame with
Syrup and 74/15/17/16/16/24/28 with OpenCV. Here, before the changes,
Syrup measured 174.9/40.3/65.5/64.6/61.6/76.7/83.9 and OpenCV
83.2/18.3/18.1/16.7/18.1/26.5/31.3. That is 2.1–3.9× slower. The two
detectors give the same WER on every fixture. CER differs only on
`low_res` (Syrup 0.47, OpenCV 0.44). Speaking precision and recall differ
on `low_res`, `rapid_cuts` and `two_alternating`, in both directions.

**Isolated.** These measurements come from dev frames and the fixtures,
which are not held-out data, and use two threads as the demo does:

| cause | measured | status |
|---|---|---|
| The cascade implementation, same model | One thread: Syrup 147.5 ms vs OpenCV 74.0 (p50, dev frames). Two threads: 122 vs 38, because Syrup scaled 1.2× and OpenCV 1.95×. Profile of one frame: the `image` crate's resize was 39% of instructions and bounds checks about 19%. | **Fixed.** Now 100 vs 74 with one thread and 55 vs 38 with two, same output. |
| OpenCV skips windows with grey deviation ≤ 10 | 100 → 65 ms on the dev frames with the same rule. On held-out photographs, 1.19× faster than OpenCV with it. | Opt-in option. The demo does not use it. |
| The demo's configuration: Syrup's `find_face` searches from 20 px, its OpenCV detector from 40 px with 4 neighbours | `one_speaker` frames, two threads: Syrup 28.7 → 14.4 ms at OpenCV's settings; OpenCV 23.5 → 13.8 ms. | Not changed. It would change outputs, so it needs the quality check. |
| `find_eyes` runs the face cascade again inside every face crop before the eye cascade; the OpenCV detector runs only the eye cascade, in the face's upper 60% | `five_faces`, per frame: Syrup's eye stage 86.9 of 126.7 ms; OpenCV's 44.3 of 125.2 ms. | Not changed, for the same reason. |
| Python → native | Each frame is copied three times into RGBA bytes: 1.2–2.8 ms per frame, about 2% of Syrup's detection time. | Not changed. |

**After the changes.** The gap is 1.1–2.0×, with every quality column
identical to before. What remains is mostly configuration rather than
library speed:

- At OpenCV's settings, Syrup's face search costs what OpenCV's does:
  14.4 vs 13.8 ms.
- With OpenCV's window rule it costs less (10.3 ms on the same frames).
- The eye stage still searches each face twice.

## Pre-registered thresholds and goals

| | target | result |
|---|---|---|
| F1 threshold | F1 ≥ 0.95 against OpenCV on test; an optimisation may not lower it by > 0.01 or change dev box count by > 1% | 0.985; before = after = gate on all dev and test images — **met** |
| F1-pipeline threshold | no quality column may change | identical on all 7 fixtures, every run — **met** |
| F2 threshold | F1 ≥ 0.95 | 1.000, IoU 1.0000, landmarks within 0.000 px — **met** (speed: loss) |
| F3 threshold | hit rate within 1 point of OpenCV's on test | 98.8% vs 100.0% — **not met** (before and after alike: the coarse-to-fine search, not today's changes) |
| F4 threshold | mask IoU ≥ 0.99 | 1.000, identical masks — **met** |
| goal: p95 −20% in several tasks, quality unchanged | | F1 −31% (1 thread) / −53% (2 threads); F3 −88%; gate −52% / −68% — **met** for F1 and F3 |
| goal: lip-reading gap substantially narrower | | 2.1–3.9× → 1.1–2.0× — **met in part**: narrowed, not closed |
| goal: an advantage repeated across families on held-out data | | **met in part.** Two families show a clean advantage: motion, and faces with OpenCV's own rule. Templates are faster but fail the accuracy bar. YuNet and glyphs are losses. |

## Integration effort (facts; no developer-time claim)

| | Syrup | OpenCV 5.0 |
|---|---|---|
| getting it | Rust crate (git dependency), pure Rust: 4 direct dependencies on Linux (`image`, `libloading`, `x11rb`, `zbus`), 135 crates in all, no system libraries; clean release build of the library here 2 min 20 s; `libsyrup.so` 5.3 MB. YuNet needs `syrup-runtime` (adds tract: 232 crates, clean build about 5.5 min) | `pip install opencv-contrib-python==5.0.0.93`: one 89 MB extension, 100 MB installed, plus numpy; `import cv2` 176 ms here. OpenCV 5 wheels ship no Haar XML files and keep `CascadeClassifier` in the contrib build: the XMLs came from the 4.x repository. C++ use: not measured |
| from Python | the `syrup` package (ctypes) and the native library built with cargo (`SYRUP_LIBRARY`); every call copies the frame into RGBA | numpy arrays in and out, no copy |
| library calls per task in this harness | faces 3 (cascade, options, `detect`); templates 2 (`Template::from_image`, `find_best`); motion 1 (`motion::regions`); glyphs 3 (`GlyphSet::new`, `learn`, `read`, which refuses when unsure); YuNet 2 | faces 2; templates 2 (plus a reused score buffer); motion 5 calls and a numpy filter, plus three details to get Syrup's exact mask (fixed-point luma weights, a −0.5 offset, a constant border for the opening); glyphs an 18-line recipe (Otsu, components, grouping, resizing, k-NN) with no refusal; YuNet 3 |
| initialisation (test, 1 thread) | cascade parse 0.12 ms (lazy, bundled); YuNet prepares a tract model per input size: first call 170 ms | cascade XML 6.1 ms; YuNet load 3.3 ms, first call 41 ms |

## For Uri, or an investor

Syrup vs OpenCV, measured on 12 Oct 2026 on one 2-core x86 virtual
machine with no GPU, on held-out data that no optimisation decision saw:
15 photographs, 480 template searches in 24 photographs, 267 frame pairs
from 3 videos and 400 synthetic text lines, 5 rounds per configuration,
with confidence intervals. A day of profiling-led work made Syrup's face
detector 31% faster and its template matching 88% faster at the 95th
percentile, with identical outputs, and roughly halved the per-frame time
of its lip-reading demo (from 2.1–3.9× OpenCV's time to 1.1–2.0×, same
transcripts). Against OpenCV 5.0 in a tuned configuration, Syrup now wins
two comparisons outright. In motion-region extraction it produces the
same pixel mask 3.2× faster at the median, on each held-out video. In
Haar face detection with the window rule OpenCV itself uses it is 1.19×
faster at the median and 1.29× at the 95th percentile, finding the same
faces. It is 1.5× faster but slightly less exact at template matching
(98.8% vs 100% of searches correct), 10–13% slower at default face
detection, 5–8× slower running the YuNet neural face detector (its
pure-Rust inference runtime is the bottleneck), and behind on reading
synthetic text. These are processing times for five task types on one
machine. They are not product response times, not a general claim of
superiority over OpenCV, and say nothing about any game beyond
MapleStory.

## Next three actions

1. **YuNet inference, the largest loss (5–8×).** tract's generic
   depthwise convolution is 56% of the network's instructions. Give it
   SIMD kernels upstream, or put a faster backend (ONNX Runtime or
   OpenCV's DNN) behind a feature, then re-measure F2 on these
   photographs.
2. **Template matching, the accuracy bar.** Make the coarse-to-fine search
   exact where it diverges. Options are a full-resolution check of the
   runner-up candidates, or an exhaustive or FFT path for large searches;
   either would also shorten the small-template tail. Measure it on a
   fresh held-out set, because this one has now been seen.
3. **The lip-reading demo's configuration.** Expose the face-search
   options through the intents and the Python package: minimum face size,
   neighbours, eyes without re-detecting the face, and `min_deviation`.
   Run the demo at OpenCV's settings with every quality column checked.
   Then judge detection against a labelled face set, not only against
   OpenCV's boxes.

## Reproduce

```text
bench/vs-opencv/reproduce.sh WORK_DIR MS_REPO      # about two hours on 2 cores
```

`fetch.sh` gets OpenCV `1f2e793`, opencv_extra `22b4a7b` and LipNet
`bec592a`, and composes the lip-reading fixtures. `prepare.py` writes the
inputs and checks them against `results/inputs.json`: 1,615 files with
their SHA-256. `build.sh` builds Syrup before (`156475f`) and after
(`HEAD`) with the same harness. `run.py` measures and summarises, and
`report.py` turns the summaries into the tables above. Raw per-call
timings and outputs are in `results/test-t1/raw.tar.xz` and
`results/test-t2/raw.tar.xz`, with each run's environment and command
lines in `environment.json` and the lip-reading runs in
`results/lipreader/`. Requirements: git, ffmpeg, cargo (rustc 1.97),
python3 with numpy and `opencv-contrib-python` 5.0.0.93. The motion masks
used for F4 are regenerated, not kept: `run.py measure` dumps them.

