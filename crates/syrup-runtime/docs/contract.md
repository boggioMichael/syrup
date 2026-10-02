# Contract

What every `find_*` operation honours, whichever name or language reached it.

## 1. How names reach Syrup

| | resolves | compiles |
|---|---|---|
| `from syrup.ops import find_face` / `syrup.ops.find_face` (PEP 562 module `__getattr__`) | on import or attribute access | on first call, or `op.prepare()` |
| `syrup.resolve("find_face")` | on the call | on first call, or `op.prepare()` |
| `syrup.define("find_header_faces", ...)`, for names outside the grammar | on the call | on first call, or `op.prepare()` |
| `syrup.add_target("licence_plate", detect)` adds a noun found by a Python function; names then use it like any other | on later resolves | on first call |
| `syrup.planner.plan("find_header_faces", "faces in the top fifth")` asks Claude for `define`'s arguments when the grammar does not cover a name | on the call | on first call |
| Rust: `Runtime::resolve` / `Runtime::define` | on the call | on first `run`, or `prepare()` |

`syrup.recipes` registers whole vocabularies with `add_target`:
`yolo()` (Ultralytics, COCO's classes by default), `mediapipe_objects()`
(EfficientDet-Lite0) and `mediapipe_poses()` (`pose`/`poses`). Labels
become nouns with plurals (`traffic light` -> `traffic_light`,
`traffic_lights`); labels that would change what a name means, such as
COCO's `orange`, are skipped and reported unless renamed. Provenance names
the model and its SHA-256.

The planner is optional (`pip install syrup-cv[planner]`). The model may
only answer with values from Syrup's own vocabulary, which the request's
JSON schema enumerates, or refuse as ambiguous or unsupported; it never
writes code. Its answer is a `define` like any other and is checked the
same way. Names the grammar resolves never reach it, and without a
description neither do names it found ambiguous.

Resolution never compiles, so a name Syrup cannot honour fails at import,
before any image is involved. A name that was never imported or defined is
an ordinary `NameError`; Python has no honest hook for that.

## 2. Grammar (v1)

```text
name      := find_name | measure_name | track_name
find_name := verb "_" body
track_name := "track_" body
body      := [count "_" selector "_" | selector "_" | "all_"] [colour "_"] target ("_" clause)*
measure_name := "measure_" quantity ["_of_" body | "_in_" region]
quantity  := sharpness | fill
verb      := find | detect | locate
target    := face | faces | human_face | human_faces | word | words
           | region | regions | blob | blobs | bar | bars
           | qr_code | qr_codes | text_block | text_blocks | panel | panels
           | moving_region | moving_regions | moving_blob | moving_blobs
colour    := red | orange | yellow | green | cyan | blue | purple | violet | magenta
selector  := largest | biggest | smallest | leftmost | rightmost | topmost | bottommost | most_confident
count     := 1..100, or one..ten
clause    := in_<region> | by_size | by_area | by_confidence | by_score
           | left_to_right | right_to_left | top_to_bottom | bottom_to_top
           | larger_than_<n>pct | smaller_than_<n>pct        (n = 1..100, % of image area)
region    := top_half | bottom_half | left_half | right_half
           | top_left | top_right | bottom_left | bottom_right
           | top_third | bottom_third | left_third | right_third | center | region
```

Faces come from YuNet, words from Tesseract, QR codes from rqrr (decoded
codes score 1, codes found but unreadable 0.5, below the default 0.6;
`text` is the content and the keypoints are the corners `top_left`,
`top_right`, `bottom_right`, `bottom_left` in the code's own orientation),
text blocks and panels from the core's layout helpers (`find_text_block`:
the padded bounds of text-like pixels; `find_uniform_color_panel`: the
largest rectangle of the searched region's dominant colour, quantised to
32 levels per channel; at most one of each per searched region, scored by
the share of their pixels that are text or the panel's colour), and nouns
added with
`add_target` from their own detector, which receives the searched pixels
and returns `(x, y, w, h, score[, text])` boxes in them; the compiled
module does the rest. Added nouns may not use words the grammar already
gives a meaning (verbs, selectors, colours, `in`, `by`, numbers, ...) or
existing nouns, so adding one never changes what an existing name means. Regions and bars need a
colour and are found by the generated module itself: pixels whose hue falls
in the colour's range (saturation ≥ 35%, value ≥ 30%, alpha ≥ 50%), in
horizontal runs of at least 3 pixels, grouped by the core's region grouping
into regions at least 3 rows tall. Bars use runs of 8 and gaps of up to 2
rows, like the core's bar finder, and must be at least 3 times wider than
tall.

| colour | hue (degrees) |
|---|---|
| red | 340–20 |
| orange | 20–45 |
| yellow | 45–70 |
| green | 70–165 |
| cyan | 165–195 |
| blue | 195–255 |
| purple, violet | 255–290 |
| magenta | 290–340 |

`measure_*` operations return the same items as the matching `find_*`
(or, without `_of_`, the image or region as one item labelled `image`),
each with a `value` in [0, 1]. The generated module selects the items; the
host measures them with the core's own functions, after ordering and
limits, so `measure_fill_of_largest_red_bar` measures the largest bar and
nothing else. An item that cannot be measured is left out, the same way a
detector leaves out what it does not accept.

| quantity | measures | from the core | cannot be measured when |
|---|---|---|---|
| `sharpness` | the share of strong horizontal luminance steps that are abrupt; below 0.35, text is too blurred to read reliably | `quality::assess_text_quality` | the item has fewer than 24 strong steps |
| `fill` | how full a bar is: filled columns over the bar's track | `geometry::measure_bar_fill` | no track can be found next to the bar |

`fill` is measured on bars only, and needs their colour
(`measure_fill_of_red_bars`).

`track_*` operations follow items across the frames of a session:

```python
session = syrup.ops.track_faces.session()      # max_distance=48, grace_frames=5
for frame in frames:
    for face in session(frame):
        print(face.track.id, face.track.velocity)
```

Each frame goes through the same compiled module as the matching `find_*`
(`track_faces` and `find_faces` share one artifact); the session then gives
each item the id the core's tracker matched it to: the nearest box centre
within `max_distance` pixels of where the object was heading. An object
unseen for up to `grace_frames` frames keeps its id when it comes back, but
frames report only what was seen in them. Calling a `track_*` operation
without a session, or making a session for any other operation, is an
`InputError`.

Moving regions are what changed since the session's previous frame, by the
core's frame differencing: pixels whose difference crosses the threshold,
grouped into regions of at least 24 pixels, scored by the share of the box
that changed. A moving object shows up as the area it left and the area it
entered, and nothing moves in a session's first frame. They exist only in
sessions (`find_moving_regions` is refused), and changing the session's
`region` starts the comparison over. A frame that fails leaves the session
as it was.

Frames can come from a source instead of the caller: image files
(`frames::ImageFiles`, `syrup watch <op> <files>...`), or a live window
(`frames::WindowCapture`, `syrup watch <op> --window TITLE`,
`syrup.capture.window(title)`), captured with the core's `capture` module
until it closes. The window is the first whose title contains the query,
ignoring case; on Wayland, where titles are hidden, the desktop's portal
asks the user to pick one and the choice is remembered for that query
(under `$XDG_STATE_HOME/syrup/screencast`). Capture fails as an
`input`-stage error: `bad_parameter` when no window matches,
`missing_dependency` where it is unavailable (no display, no portal, macOS
before 14), `permission_denied` when the user or the system refuses (on
macOS, the Screen Recording permission), and `io` for anything else.

Singular and plural mean the same: `find_face` returns every face, so
`faces = find_face(img)` reads right. One result is spelled with a selector:
`find_largest_face`. Names that differ only by synonym, number or clause
order resolve to the same intent and share one compiled artifact.

Regions are fractions `(fx, fy, fw, fh)` of the image, converted with
`x0 = round(fx·W)`, `x1 = round((fx + fw)·W)` (halves round up), so adjacent
regions tile the image exactly. Halves, quadrants and thirds are what their
names say; `center` is `(¼, ¼, ½, ½)`; `region` is passed per call as pixels
`(x, y, w, h)` and clipped to the image.

`in_<region>` means the detector only sees those pixels. Results are in
whole-image coordinates, clipped to the region; an object cut by the region
edge may be found partially or not at all.

## 3. Results

Every `find_*` returns a `FindResult`: an ordered, possibly empty sequence
of `Found` plus provenance. It is a sequence even when a selector limits it
to one item.

- `label`: the target, e.g. `"face"`.
- `box`: `(x, y, w, h)` floats in pixels of the image passed in. Origin at
  the top-left corner of the top-left pixel, x right, y down, covering
  `[x, x+w) × [y, y+h)`, clipped to the image and to the region.
- `confidence`: the provider's score in `[0, 1]`. Not a calibrated
  probability; only comparable within one provider and model, which
  provenance names.
- `keypoints`: named points, same coordinates, clipped. Faces carry
  `right_eye`, `left_eye`, `nose_tip`, `right_mouth_corner`,
  `left_mouth_corner` (the subject's right and left).
- `value`: the measured quantity for `measure_*` operations, otherwise
  empty.
- `track`: for `track_*` sessions, `id` (stable while the object stays in
  view), `age_frames` and `velocity` (box centre movement since the
  previous frame, in pixels); otherwise empty.
- `text`: what a word says; empty for targets that do not read text. A
  word's confidence is Tesseract's, divided by 100. A region's or bar's is
  the share of its box's pixels that have the colour.

Default order is confidence, highest first. Every order ends with the same
tie-breakers (confidence ↓, then y, x, h, w ↑), so it is total.

| clause (all results) | selector (one result) | order |
|---|---|---|
| none, `by_confidence`, `by_score` | `most_confident` | confidence ↓ |
| `by_size`, `by_area` | `largest`, `biggest` | area ↓ |
| | `smallest` | area ↑ |
| `left_to_right` | `leftmost` | x ↑ |
| `right_to_left` | `rightmost` | right edge ↓ |
| `top_to_bottom` | `topmost` | y ↑ |
| `bottom_to_top` | `bottommost` | bottom edge ↓ |

A count before a selector (`find_3_largest_faces`) keeps that many instead
of one. Area filters run before
ordering and limits.

Per-call parameters never trigger a rebuild:

| | default | |
|---|---|---|
| `min_confidence` | faces 0.6, words 0.5, QR codes 0.6, regions, bars and moving regions 0, added targets as added (0.5 by default) | in `[0, 1]`; the face provider never reports below 0.1 |
| `max_results` | none | ≥ 1, applied after the operation's own limit |
| `region` | | required by `_in_region` operations, refused by all others |

**Empty is not failure.** An empty result means the operation ran and the
detector accepted nothing under the recorded configuration; it does not
prove the image has no faces. Anything that stops an operation from running
completely raises. Syrup never returns a success-shaped result for a failed
run, and never swaps in another operation, provider or model.

## 4. Inputs

8-bit images with 1, 3 (RGB) or 4 (RGBA, alpha ignored) channels, row-major,
1 to 16384 pixels per side. Python accepts NumPy `uint8` arrays shaped
`(H, W)`, `(H, W, 1)`, `(H, W, 3)` or `(H, W, 4)`; PIL images (`L`, `RGB`,
`RGBA`; palette images are converted to RGB); image file paths; and
`syrup.Image`. Channel order is RGB. OpenCV's BGR must be converted by the
caller, because Syrup cannot tell the difference. Anything else is an
`InputError`, not a coercion.

## 5. Failures

Every failure is a `SyrupError` with `stage`, `kind`, `operation`, `reason`
and `hint`. Python raises `DependencyError` for `missing_dependency`, and
otherwise the class for the stage.

| stage | class | kinds |
|---|---|---|
| `input` | `InputError` | `bad_image`, `bad_parameter`, `permission_denied`, `missing_dependency`, `io` |
| `resolve` | `IntentError` | `malformed`, `ambiguous`, `unsupported`, `conflicting` |
| `plan` | `PlanError` | `invalid_plan` |
| `generate`, `compile` | `BuildError` | `policy`, `compiler_failed`, `timeout`, `missing_dependency` |
| `load` | `LoadError` | `not_prepared`, `integrity`, `abi_mismatch`, `dlopen` |
| `validate` | `ValidationError` | `mismatch` |
| `execute` | `ExecutionError` | `provider_failed`, `module_failed`, `panic`, `contract_violation`, `missing_dependency`, `integrity` |

Refused rather than guessed:

| request | kind | why |
|---|---|---|
| `find_best_face`, `find_main_face`, `find_first_face` | ambiguous | best, main or first by what? |
| `find_large_faces` | ambiguous | large compared to what? Use `larger_than_<n>pct` |
| `find_highest_face` | ambiguous | by position or by confidence? |
| `find_two_faces`, `find_largest_faces` | ambiguous | a count needs an ordering; a plural selector needs a count |
| `find_smiling_faces`, `find_cat_faces` | unsupported | no capability judges that |
| `find_regions` | ambiguous | regions of which colour? |
| `find_white_regions`, `find_red_faces` | unsupported | white has no hue; faces are not found by colour |
| `find_cars`, `find_people` | unsupported | no capability finds that |
| `measure_fill` | ambiguous | fill of what? |
| `measure_fill_of_faces`, `measure_weight` | unsupported | fill is measured on bars; weight not at all |
| `measure_sharpness_by_size` | conflicting | one image or region has nothing to order |
| `find_faces_in_top_half_in_left_half` | conflicting | one region per operation |
| any unknown word | malformed or unsupported | unknown words are never dropped |

## 6. Artifacts

An artifact is identified by its plan, code generator version, ABI version,
target triple and compiler flags; never by the operation name. The
compiler's version is recorded but not part of the key, since modules only
speak C ABI.

Artifacts are built in a private directory, validated, and published with
one rename; concurrent first use builds once; published artifacts never
change, and the library's SHA-256 is checked before every load. Damaged
artifacts are moved aside and rebuilt in development mode, and refused in
frozen mode (`SYRUP_MODE=frozen`), which never generates or compiles.

A bundle (`syrup bundle <dir> <operation>...`, `syrup.bundle`) is a cache
directory holding only the named operations, plus `bundle.json` recording
the target, ABI and code generator. Frozen mode pointed at a bundle loads
from it; an operation missing from it, or a bundle for another target or
Syrup version, is `not_prepared` with that reason.

An artifact is published only after:

1. the plan type-checks, with results in input-image coordinates;
2. the source passes a policy check (no filesystem, network, process,
   environment or foreign-code access; only the three ABI exports);
3. `rustc` builds it with warnings denied;
4. the library exports the expected ABI version and plan hash;
5. on 24 synthetic cases its calls and results match the interpreter's
   exactly.

## 7. Provenance

Each result records the requested name, canonical intent, plan hash,
artifact key, library SHA-256, `rustc` version, target, whether this call
compiled the artifact or reused it from memory or disk, each provider and
model hash, the effective parameters, the image shape, and timings.
