# Syrup vs OpenCV — pre-registration

Written and committed **before** any Syrup-vs-OpenCV comparison in this
directory was run. What is fixed here — tasks, parameters, data and the
dev/test split, metrics, thresholds, protocol — is not changed after seeing
results. Anything added later is marked as an amendment, with the reason, in
`AMENDMENTS.md`.

## Starting point (verified 2026-10-12)

| what | value |
|---|---|
| ms branch `claude/human` | `8a1496520e4367b2ad1f50bb6eeddca5644c62b6` |
| Syrup pinned by that ms (`Cargo.toml`) | `54dad1e2875d6698e922d5c26e5a8c32b9a1c028` (= `origin/claude/engine`) |
| Syrup `origin/main` | `265ade527a6c31bddf09668bd531f03a87645448` (older; not used) |
| this work | Syrup branch `claude/vs-opencv`, from `54dad1e` |

Reproduced before writing this (raw output in `results/baseline/`):

- ms `bench/phase6-cpu-linux.json`, the recording at 1366×768, 300 frames:
  recorded 5.16 ms mean / 8.95 ms p95; three re-runs at `8a14965` on this
  machine: 5.48 / 8.84, 6.76 / 10.82, 5.46 / 9.44. Same order; the current
  bench learns no glyph font from the fixtures (0 glyphs, then 16–18), so the
  workload is not identical. Perception only: no capture, network or voice.
  Not an OpenCV comparison.
- Syrup's lip-reading comparison (`example/docs/benchmarks.md`): the
  `lipreader` evaluation over the seven GRID fixtures, ms per frame with the
  Syrup detector 160/39/58/56/62/74/81 vs OpenCV's 86/18/19/20/18/29/32
  (recorded: 152/34/57/57/55/70/71 vs 74/15/17/16/16/24/28). Ratio 1.9–3.4×,
  same transcripts. Two configuration differences are visible before any
  profiling and are tested, not assumed, in task F1-pipeline: Syrup's
  `find_face` searches from 20 px (OpenCV's run: from 40 px, minNeighbors 4),
  and Syrup's `find_eyes` runs the face cascade again inside every face crop
  before the eye cascade.

## Machine and software

2 vCPU Intel Xeon @ 2.10 GHz (AVX2, AVX-512F), 7.8 GiB, no GPU; Ubuntu
24.04.5, Linux 6.18; rustc 1.97.0, release profile as in `Cargo.toml` (no LTO,
default target CPU; Syrup's kernels dispatch SIMD at run time); OpenCV 5.0.0
(`opencv-python` 5.0.0.93, pthreads, dispatched AVX2/AVX-512), Python 3.13.15,
numpy 2.5.3. The cascades are OpenCV's own XML from `opencv/opencv` 4.x
`1f2e793` (Syrup's bundled `.bin` files are converted from these by
`tools/cascade2bin.py`).

## Protocol (all tasks)

- **Threads.** Primary: one thread each (`cv2.setNumThreads(1)`; Syrup's
  thread options set to 1). Secondary: two threads each. Never mixed.
- **What is timed.** The library call only, on input already decoded and in
  the library's own format; any conversion either side needs (BGR→RGBA, a
  copy into the other library's type) is timed separately and reported.
- **Warm-up and initialisation.** The first 3 calls per input are warm-up and
  not counted; loading (cascade, model) and the first call are reported
  separately as initialisation cost.
- **Repetitions and order.** 5 rounds; in each round every implementation
  (Syrup before, Syrup after, OpenCV) runs over every input, in an order
  rotated round by round; each implementation runs in its own process.
- **Statistics.** p50/p95/p99 of per-call time per (task, data source,
  implementation); throughput (calls/s); CPU time / wall time and peak RSS
  per process. Uncertainty: the spread across the 5 rounds, and bootstrap
  confidence intervals that resample **inputs** (images, videos), never
  adjacent frames as if they were independent.
- **Reporting.** Per task and per data source, wins and losses alike; no
  overall average across tasks.

## Data, split by source

Dev data may be looked at while profiling and optimising. **Test data is
used only for the final measurements**: no parameter or code change is
judged on it.

| | dev (profiling, optimisation) | test (held out) |
|---|---|---|
| faces | frames of the GRID lip-reading fixtures (`one_speaker`, `two_simultaneous`, `five_faces`, every 5th frame) | the 13 photographs of `opencv_extra` `testdata/cv/cascadeandhog/images` (`22b4a7b`), OpenCV's `samples/data/lena.jpg`, Syrup's `tests/fixtures/astronaut_320.jpg` |
| template matching | MapleStory: `resources/maplestory.png` and 20 frames of `chaos-zakum-solo-lvl230.mp4` (ms) | OpenCV `samples/data` photographs (`baboon`, `fruits`, `building`, `home`, `box_in_scene`, `messi5`, `aero1`, `butterfly`, `starry_night`, `board`, `lena`) and the 13 face photographs |
| motion | `chaos-zakum-solo-lvl230.mp4` at 1366×768, 100 consecutive frame pairs | OpenCV `samples/data` `vtest.avi`, `tree.avi`, `Megamind.avi`, 100 consecutive frame pairs each, native size |
| glyphs | the ms HUD strips (`resources/*strip.png`, `resources/owner-classic`) | none available offline that is not MapleStory: synthetic only, labelled as such |

Every input is listed with its SHA-256 in `results/inputs.json`.

## Tasks, parameters and quality metrics

**F1 — Haar cascade face detection (same model, same algorithm).**
`haarcascade_frontalface_alt2`; scale factor 1.1, min neighbours 3, min size
20 (the window), no max size. Both get the identical grey image (OpenCV's
`cvtColor` RGB→grey, Rec. 601): Syrup through `Cascade::detect` on that
image, so the comparison is the detector alone. Output: boxes. Quality: agreement with OpenCV as the reference
implementation of the same model — precision, recall, F1 of Syrup's boxes
against OpenCV's at IoU ≥ 0.5, and the mean IoU of matched pairs.
*Threshold:* F1 ≥ 0.95 on test; an optimisation may not lower F1 by more than
0.01 or change the dev box count by more than 1%.

**F1-pipeline — the lip-reading path.** `lipreader evaluate` on the seven
fixtures with each detector: ms/frame and every quality column (WER, CER,
attribution, track consistency, cuts, speaking P/R). *Threshold:* no quality
column may change.

**F2 — YuNet face detection (same ONNX model, different runtimes).**
`face_detection_yunet_2023mar.onnx`; Syrup runtime (tract) vs
`cv2.FaceDetectorYN`, score threshold 0.6, NMS 0.3, input at the image's own
size. Quality: F1 at IoU ≥ 0.5 against OpenCV's boxes, mean landmark error in
pixels. *Threshold:* F1 ≥ 0.95. (Measured only if the runtime builds offline.)

**F3 — template matching (same algorithm: zero-mean normalised
cross-correlation).** Best match of a template in a grey image: Syrup
`template::find_best` vs `cv2.matchTemplate(TM_CCOEFF_NORMED)` + `minMaxLoc`.
Templates: crops chosen by a fixed seed (`20261012`), sides 16, 32, 48 and
64 px, 10 per image, kept only if their grey standard deviation is ≥ 10.
Search image: the original, and a perturbed copy (gain 0.85, offset +12,
Gaussian noise σ 4, seed fixed). Quality: share of searches whose best match
is within 1 px of where the template was cut; the score's absolute
difference from OpenCV's. *Threshold:* hit rate within 1 percentage point of
OpenCV's on test.

**F4 — motion regions (same algorithm family).** Consecutive frames; a pixel
moved when the Rec. 709 luma of the per-channel absolute difference is ≥ 28;
runs ≥ 4 px; regions ≥ 4 rows and ≥ 24 px. Syrup's detector path without the
tracker vs OpenCV: `absdiff` → `transform` (luma) → `threshold` →
`morphologyEx(OPEN, 1×4)` → `connectedComponentsWithStats` → the same
filters. Quality: IoU of the two moved-pixel masks (same formula: expected
1.0) and region F1 at IoU ≥ 0.5 (different grouping rules: expected < 1;
reported, not a pass/fail). *Threshold:* mask IoU ≥ 0.99.

**F5 — fixed-font glyph reading (different solutions to one task).** Syrup
`glyphs` learned from labelled crops vs the usual OpenCV recipe: Otsu,
connected components, one `cv2.ml.KNearest` per font trained on the same
crops. Quality: exact-string accuracy, wrong-read rate (a refusal is not a
wrong read), refusal rate. Reported with the data's limits; no
generalisation claim.

**Tracking** (Syrup's multi-object tracker of motion regions) has no OpenCV
counterpart for the same task (OpenCV's trackers follow one object from a
given box); it is listed as not compared, not as a win.

## Ambitious goals (not required results)

- p95 down ≥ 20% in several tasks, quality unchanged.
- The lip-reading path's gap substantially closed.
- An advantage that holds in several families on the held-out test data.

Any goal not met is reported as not met, with the reason.

## Rules for the optimisation work

- No change that lowers quality, skips required work, changes what an
  output means or drops failure cases. Any change of scheduling or reuse of
  earlier results would need its own latency and miss-rate measurement; none
  is planned.
- Every change has a test of output equivalence (or of the agreed tolerance)
  and of the failure cases it could create.
- What is optimised is chosen from the profile on dev data only.
