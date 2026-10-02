//! syrup: turn captured frames into structured, confidence-scored
//! visual observations.
//!
//! The library operates on plain [`image::RgbaImage`] buffers, so frames can
//! come from anywhere — a screenshot on disk, frames extracted from a video,
//! a synthetic test fixture, or a live window capture — and
//! every primitive behaves identically regardless of the source.
//!
//! ```text
//!   RgbaImage  ──▶  primitives (geometry, color, motion, OCR, quality)
//!                        │
//!                        ▼
//!                  Detection<T>: value + confidence + provenance
//! ```
//!
//! Most of the time you do not call the primitives directly: you declare
//! the function you need and the [`intent`] module builds it from them —
//! `syrup::intent!(fn find_face(image: &RgbaImage) -> Detection<Vec<Match>>)`
//! — in-process, or compiled to a shared library through the [`abi`].
//!
//! What each module owns:
//!
//! - [`intent`]: functions by name — parse the name, plan a composition of
//!   primitives, run it, or write it out and compile it.
//! - [`abi`]: the C contract between the library and the implementations
//!   it loads.
//! - [`capi`]: the library's own C API, for Python and other hosts.
//! - [`detection`]: the result vocabulary — [`detection::Detection`],
//!   [`detection::Confidence`], [`detection::Reliability`].
//! - [`geometry`]: rectangles, pixel-run segmentation, region grouping, and
//!   horizontal-bar fill measurement.
//! - [`color`]: RGB→HSV conversion and the pixel predicates detectors share.
//! - [`components`]: connected-component labelling of masks and predicates.
//! - [`threshold`]: Otsu thresholds, integral images, channel views, and a
//!   local-contrast "text evidence" map.
//! - [`motion`]: frame differencing and a moving-blob detector with stable
//!   IDs across frames (see [`tracking`]).
//! - [`tracking`]: a minimal centroid multi-object tracker.
//! - [`glyphs`]: reading pixel-font text (counters, HUD values) by template
//!   matching against glyphs learned from labelled examples.
//! - [`template`]: finding a known picture (an icon, a marker) anywhere in a
//!   frame by normalised cross-correlation, coarse to fine.
//! - [`cascade`]: boosted cascades of Haar features (Viola–Jones), with the
//!   bundled frontal-face, profile-face and eye detectors.
//! - [`face`]: a found face's parts — eyes, mouth — and how open they are.
//! - [`sequence`]: dynamic time warping and a nearest-example matcher, for
//!   things that unfold over frames (a mouthed word, a gesture).
//! - [`ocr`]: text recognition via a Tesseract subprocess, or the OCR engine
//!   built into Windows.
//! - [`quality`]: is a region sharp enough for OCR to stand a chance?
//! - [`capture`]: live window capture by title on Windows, macOS and Linux
//!   (X11 and Wayland), and the whole screen on Windows.
//! - [`draw`]: debug-overlay primitives — rectangles and a small bitmap font.
//! - [`timing`]: FPS and moving-average measurement.
//!
//! The library reports what it measured and how sure it is; deciding what an
//! observation *means* — and what to do about it — belongs to the consumer.

pub mod abi;
pub mod capi;
pub mod capture;
pub mod cascade;
pub mod color;
pub mod components;
pub mod detection;
pub mod draw;
pub mod face;
pub mod geometry;
pub mod glyphs;
pub mod intent;
pub mod motion;
pub mod ocr;
pub mod quality;
mod scan;
pub mod sequence;
pub mod template;
pub mod threshold;
pub mod timing;
pub mod tracking;

pub use detection::{Confidence, Detection, Reliability, Timestamp};
pub use geometry::Rect;
/// The image crate this library is built on, so callers and generated
/// code use the same version without naming it.
pub use image;

/// Everything a program that declares intents needs, in one import:
/// `use syrup::prelude::*;`.
pub mod prelude {
    pub use crate::detection::{Confidence, Detection, Reliability};
    pub use crate::geometry::Rect;
    pub use crate::image::RgbaImage;
    pub use crate::intent::{self, IntentError, Match};
}
