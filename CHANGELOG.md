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
- `face`: eyes and mouth of a found face, eye openness, mouth shape.
- `sequence`: dynamic time warping and a nearest-example matcher.

### Fixes
- FPS readings 60x too high before the window filled; overflow panics in
  `draw_rect`; OCR word boxes left in enlarged-image coordinates; temp-file
  collisions across threads; luminance weights that undercounted 23 grey
  levels in motion detection.
