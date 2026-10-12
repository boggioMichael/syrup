# Amendments to the pre-registration

Each change to what PREREGISTRATION.md fixed, with when and why. None was
made after seeing a comparison result unless it says so.

1. **2026-10-12, before any comparison ran — `tree.avi` has 68 frames.** Its
   header says 444, but ffmpeg (and OpenCV) decode 68 (Cinepak). F4 uses its
   67 consecutive pairs instead of 100. A data availability fact, found while
   preparing the inputs.

2. **2026-10-12, after smoke runs on dev data only — how each side is run.**
   No test-data result had been seen. (a) The OpenCV side reuses its
   full-frame buffers between calls (`dst=`, `result=`, `labels=`, as a C++
   caller reuses its `Mat`s) and labels motion components with Grana's BBDT
   (`CCL_GRANA`): on the 100 dev pairs this gave output identical to the
   plain recipe and was the fastest of the algorithms tried (default,
   Spaghetti, Grana, Wu), so it is OpenCV's "reasonable optimised
   configuration" here. (b) F1 calls `detectMultiScale2` (to get each box's
   neighbour count, as Syrup reports it); minimum size 0 = the 20-pixel
   window on both sides, as fixed. (c) F4: the timed Syrup call runs before
   the untimed mask used for the comparison, so it does not find the frames
   already in cache. (d) Clarification of F3, not a change: both sides score
   positions by the same zero-mean normalised cross-correlation, but search
   differently — OpenCV scores every position (`matchTemplate`); Syrup's
   `find_best` scores every position only when the template has no usable
   coarse level or the search is small, and otherwise searches coarse to
   fine. This is why F3's quality columns (hit rate, position, score) are
   part of the comparison.

3. **2026-10-12, before any test-data run — additions decided on dev data.**
   (a) **F1, a fourth implementation: Syrup with
   `CascadeOptions::min_deviation = Some(10)`** ("syrup-after-gate"). OpenCV's
   `CascadeClassifier` rejects every window whose grey standard deviation is
   10 or less before its first stage (`HaarEvaluator::setWindow`,
   `opencv/opencv` 4.x `1f2e793`, `modules/objdetect/src/cascadedetect.cpp`
   lines 716–742; the behaviour checked on OpenCV 5.0 with a low-contrast
   face, see the test `the_flat_window_gate_drops_low_contrast_faces_as_opencv_does`).
   Syrup evaluates every window. The option makes Syrup's scan do what
   OpenCV's does; it is off by default, so "syrup-after" keeps Syrup's
   behaviour and its outputs must equal "syrup-before". On the 45 dev frames
   the gate changed no box. OpenCV also skips the next window after a
   first-stage rejection (`cascadedetect.cpp` line 1082); on dev that changed
   3 of 45 frames' boxes for 4% of the time and is not used.
   (b) **F5 made concrete.** No labelled non-MapleStory glyph data exists
   offline and the ms HUD crops are four unlabelled-format images, so F5 is
   synthetic only: OpenCV's Hershey simplex font, one pixel wide, not
   anti-aliased, white, at two sizes (`fontScale` 0.5 and 0.75: cap heights
   about 11 and 16 pixels), on a flat dark panel, a red gradient bar or a
   noisy panel (σ 8), in rotation. Both readers learn from the same crops:
   one line `0123456789/%.` per size. Cases: random HUD-like strings
   (`a/b`, `d.dd%`, a number), 200 per size on test (seed 20261012), 60 per
   size on dev (seed 20261013). Syrup: `GlyphSet` with default options,
   `learn` then `read` (a refusal is a refusal). OpenCV: grey, Otsu, 8-connected
   components of 2+ pixels, components whose columns overlap grouped, 16×16
   cells, one `KNearest` (k = 1) for the font, trained on both sizes.
   Scoring ignores whitespace (Syrup marks wide gaps with a space).
   On dev, Syrup could not learn the small size ("reads the region as
   0123456719%."), and refused 62 of 120 strings with no wrong reading;
   OpenCV read 114 correctly and 6 wrongly. The protocol is not changed
   for that.
   (c) **F2 (YuNet)** is measured if time allows, on the face photographs
   resized (`INTER_AREA`) so the longer side is 640 — the canvas Syrup's
   YuNet provider always runs at — so both networks see the same pixels at
   the same size; that resize is not timed.
