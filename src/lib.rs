//! syrup: turn captured frames into structured, confidence-scored
//! visual observations.
//!
//! The library operates on plain [`image::RgbaImage`] buffers, so frames can
//! come from anywhere — a screenshot on disk, frames extracted from a video,
//! a synthetic test fixture, or a live window capture (Windows only) — and
//! every primitive behaves identically regardless of the source.
//!
//! ```text
//!   RgbaImage  ──▶  primitives (geometry, color, motion, OCR, quality)
//!                        │
//!                        ▼
//!                  Detection<T>: value + confidence + provenance
//! ```
//!
//! What each module owns:
//!
//! - [`detection`]: the result vocabulary — [`detection::Detection`],
//!   [`detection::Confidence`], [`detection::Reliability`].
//! - [`geometry`]: rectangles, pixel-run segmentation, region grouping, and
//!   horizontal-bar fill measurement.
//! - [`color`]: RGB→HSV conversion and the pixel predicates detectors share.
//! - [`motion`]: frame differencing and a moving-blob detector with stable
//!   IDs across frames (see [`tracking`]).
//! - [`tracking`]: a minimal centroid multi-object tracker.
//! - [`ocr`]: text recognition via a Tesseract subprocess, or the OCR engine
//!   built into Windows.
//! - [`quality`]: is a region sharp enough for OCR to stand a chance?
//! - [`capture`]: window capture by title (Windows; stubs elsewhere).
//! - [`draw`]: debug-overlay primitives — rectangles and a small bitmap font.
//! - [`timing`]: FPS and moving-average measurement.
//!
//! The library reports what it measured and how sure it is; deciding what an
//! observation *means* — and what to do about it — belongs to the consumer.

pub mod capture;
pub mod color;
pub mod detection;
pub mod draw;
pub mod geometry;
pub mod motion;
pub mod ocr;
pub mod quality;
pub mod timing;
pub mod tracking;

pub use detection::{Confidence, Detection, Reliability, Timestamp};
pub use geometry::Rect;
