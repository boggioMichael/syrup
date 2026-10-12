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
