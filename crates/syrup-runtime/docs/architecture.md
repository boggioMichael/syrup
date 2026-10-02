# Architecture

The contract this design serves is in [contract.md](contract.md).

## Starting point

The runtime was written by Eitan in `0xGh0stAn0n/syrup`, starting from
`main` (`265ade5`): a ~3,300-line crate of deterministic pixel primitives
(geometry, colour, motion, tracking, Tesseract OCR, sharpness, Windows
capture, debug drawing) whose shared result type is `Detection<T>`.

In this repository it sits on top of `claude/intents-2`, whose core adds
more primitives (cascades, templates, glyphs, components, thresholds, face
parts, sequences) and its own in-process mechanism, `syrup::intent`: a
closed vocabulary whose plans call handwritten `plans::*` functions, run
in-process, with an optional cargo-built shared library that calls the
same functions. The two coexist:

| | `syrup::intent` (core) | `syrup-runtime` |
|---|---|---|
| reached as | `syrup::intent!(fn find_face(..))`, Python `syrup.find_face` | `Runtime::resolve`, Python `syrup.ops.find_face` |
| names | `verb_[qualifiers_]noun`, closed | composable grammar: counts, selectors, regions, orders, area filters; refuses ambiguity |
| what runs | the core's functions, in-process | a generated module: region, detector call, coordinates, filters, order, limits |
| native | optional, `cargo build` of a crate calling the core | always: `rustc` (0.2 s on Linux to 0.6 s on Windows), validated, cached by plan |
| faces | Viola–Jones cascades (frontal, profile, eyes) | YuNet (CNN) through tract |
| also | icons by template, glyph text, count/read verbs | QR codes, words with boxes, panels, sessions, Python detectors as targets |

## What stays

The core crate stays at the repository root, and the runtime reaches it
only through its public API: the colour predicates and region grouping
(which the generated colour kernels are validated against), bar fill,
sharpness, layout helpers, motion masks, the tracker, OCR and capture. Those
are capabilities the runtime composes; the core's own entry points, including
the original `ocr::ocr_region` and `capture::capture_window_by_title_info`,
are unchanged.

## What is added

```text
from syrup.ops import find_face        Runtime::resolve("find_face")
            └──────────────┬──────────────────┘
                           ▼
 crates/syrup-runtime
   intent.rs    name → canonical Intent, or a refusal that says why
   plan.rs      Intent → typed Plan; every value carries its coordinate space
   interp.rs    reference semantics, used only to validate modules
   codegen.rs   Plan → dependency-free Rust source (one template per step)
   compiler.rs  rustc directly: no cargo, no build scripts, no network
   cache.rs     content-addressed store, locked builds, atomic publish
   loader.rs    load by absolute path, check ABI version and plan hash
   validate.rs  module vs interpreter on synthetic images
   providers/   YuNet face detection via tract, behind the C ABI
 crates/syrup-python + python/   one generic binding, errors, results
```

## Decisions

1. **The generated module is the operation.** Region selection, the detector
   call, coordinate restoration, clipping, filtering, ordering and limiting
   are emitted from the plan. Only learned or external detectors (YuNet,
   Tesseract), the core's region grouping and the core's measurements
   (sharpness, bar fill) live in the host. For colour
   targets the per-pixel test itself is generated, specialised to the
   colour, and validated against the core's `is_color_pixel`.
2. **Generated code has no dependencies.** The ONNX runtime and the model
   are compiled into the host once, so a module builds with `rustc` in about
   half a second and never downloads anything.
3. **Names are parsed, not looked up.** Clauses compose, so
   `find_2_largest_faces_in_top_half` works without anyone writing it. Names
   with the same meaning share a plan, so they share one artifact.
4. **Validation is differential.** Before an artifact is published it runs on
   synthetic images against a mock detector and must match the interpreter.
   The mock reads the window it was given from the view pointer, so a crop
   at the wrong offset fails validation rather than production.
5. **Python resolution is real.** `syrup.ops` uses module `__getattr__`
   (PEP 562), so `from syrup.ops import find_face` resolves at import and
   compiles on first call. A bare name that was never imported cannot be
   intercepted honestly, so it is not attempted. Names outside the grammar
   go through `syrup.define`.
6. **State lives in sessions, not modules.** A generated module looks at one
   frame. `track_*` sessions run it per frame and keep what spans frames in
   the host: the core's tracker, and for moving regions the previous frame,
   served to the module as a capability like any detector.
7. **New detectors plug in without Rust.** `syrup.add_target(noun, detect)`
   registers a Python function (any ML library) as a provider; names using
   the noun compile to modules that call it through the same ABI.
8. **The model is a pinned dependency of the runtime.** YuNet 2023mar (MIT,
   232 KB, SHA-256 checked) runs through `tract-onnx`: pure Rust, no OpenCV.
   The core crate stays model-free.

## Delivery

1. Mechanism and the `find_face` proof: runtime, YuNet provider, Python
   bridge, contract, tests.
2. Migrate the existing core: OCR gets explicit errors and image-coordinate
   word boxes; existing primitives become catalog capabilities; old entry
   points stay as marked adapters.
3. Measurements, sessions for motion and tracking, prepared bundles, wheels
   for Linux, macOS and Windows.
4. An optional language-model planner (`syrup.planner`) that may only emit
   `define` specs from Syrup's vocabulary, checked like hand-written ones.
5. Later: more providers.
