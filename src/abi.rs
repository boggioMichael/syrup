//! The C ABI between the library and the implementations it loads at run
//! time — the functions it invents for an intent and compiles to a shared
//! library (see [`crate::intent`]), and anything else built to the same
//! contract, from any language.
//!
//! A loaded library exports four symbols:
//!
//! ```text
//!   u32   syrup_abi_version()
//!   char* syrup_intent_name()
//!   i32   syrup_intent_run(const FrameView*, const RegionView*, ResultView*)
//!   void  syrup_intent_free(ResultView*)
//! ```
//!
//! The host fills a [`FrameView`] over its own pixels (no copy), the
//! library fills a [`ResultView`] it allocated, the host copies what it
//! needs out and hands the view back to `syrup_intent_free`. Memory never
//! crosses an allocator boundary. Rust implementations get all four
//! exports from [`crate::export_intent!`].

use std::ffi::{CStr, CString, c_char};

use image::RgbaImage;

use crate::detection::{Confidence, Detection, Reliability};
use crate::geometry::Rect;
use crate::intent::{Match, Outcome};

/// Bumped whenever the layout of any view or the meaning of a field
/// changes. The host refuses libraries built against another version.
pub const ABI_VERSION: u32 = 1;

pub const SYMBOL_VERSION: &[u8] = b"syrup_abi_version\0";
pub const SYMBOL_NAME: &[u8] = b"syrup_intent_name\0";
pub const SYMBOL_RUN: &[u8] = b"syrup_intent_run\0";
pub const SYMBOL_FREE: &[u8] = b"syrup_intent_free\0";

/// A borrowed RGBA8 frame: `height` rows of `width` pixels, rows `stride`
/// bytes apart.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FrameView {
    pub data: *const u8,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
}

impl FrameView {
    pub fn of(image: &RgbaImage) -> Self {
        Self {
            data: image.as_raw().as_ptr(),
            width: image.width(),
            height: image.height(),
            stride: image.width() * 4,
        }
    }

    /// Copy the viewed pixels into an owned image.
    ///
    /// # Safety
    /// `data` must point to `height` rows of `stride` bytes, each holding
    /// `width * 4` valid bytes.
    pub unsafe fn to_image(&self) -> Option<RgbaImage> {
        if self.data.is_null()
            || self.width == 0
            || self.height == 0
            || self.stride < self.width * 4
        {
            return None;
        }
        let row = self.width as usize * 4;
        let mut pixels = Vec::with_capacity(row * self.height as usize);
        for y in 0..self.height as usize {
            // SAFETY: the caller guarantees the row lies within `data`.
            let src =
                unsafe { std::slice::from_raw_parts(self.data.add(y * self.stride as usize), row) };
            pixels.extend_from_slice(src);
        }
        RgbaImage::from_raw(self.width, self.height, pixels)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RegionView {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl From<Rect> for RegionView {
    fn from(r: Rect) -> Self {
        Self {
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
        }
    }
}

impl From<RegionView> for Rect {
    fn from(r: RegionView) -> Self {
        Rect {
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MatchView {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
    pub score: f32,
    pub cx: f32,
    pub cy: f32,
}

/// `ResultView::status`.
pub const STATUS_FOUND: u32 = 0;
pub const STATUS_MISSING: u32 = 1;

/// `ResultView::kind`: which of the payload fields is meaningful.
pub const KIND_MATCHES: u32 = 0;
pub const KIND_SCALAR: u32 = 1;
pub const KIND_TEXT: u32 = 2;

/// A detection result crossing the boundary. Allocated and freed by the
/// library that produced it.
#[repr(C)]
#[derive(Debug)]
pub struct ResultView {
    pub status: u32,
    pub kind: u32,
    pub confidence: f32,
    /// [`Reliability`] as an integer; see [`reliability_code`].
    pub reliability: u32,
    pub matches: *mut MatchView,
    pub match_count: usize,
    pub scalar: f32,
    /// NUL-terminated UTF-8, or null.
    pub text: *mut c_char,
    /// NUL-terminated UTF-8 failure reason, or null.
    pub reason: *mut c_char,
}

impl ResultView {
    /// An empty view for the library to fill.
    pub fn empty() -> Self {
        Self {
            status: STATUS_MISSING,
            kind: KIND_MATCHES,
            confidence: 0.0,
            reliability: reliability_code(Reliability::Unreliable),
            matches: std::ptr::null_mut(),
            match_count: 0,
            scalar: 0.0,
            text: std::ptr::null_mut(),
            reason: std::ptr::null_mut(),
        }
    }

    /// Move a detection into a view. The view owns the allocations until
    /// [`ResultView::release`].
    pub fn from_detection(detection: Detection<Outcome>) -> Self {
        let mut view = Self::empty();
        view.confidence = detection.confidence.value();
        view.reliability = reliability_code(detection.reliability);
        view.reason = detection
            .failure_reason
            .and_then(|r| CString::new(r).ok())
            .map(CString::into_raw)
            .unwrap_or(std::ptr::null_mut());
        match detection.value {
            None => view.status = STATUS_MISSING,
            Some(outcome) => {
                view.status = STATUS_FOUND;
                match outcome {
                    Outcome::Matches(matches) => {
                        view.kind = KIND_MATCHES;
                        let mut boxed: Box<[MatchView]> = matches
                            .into_iter()
                            .map(|m| MatchView {
                                x: m.bounds.x,
                                y: m.bounds.y,
                                w: m.bounds.w,
                                h: m.bounds.h,
                                score: m.score,
                                cx: m.centre.0,
                                cy: m.centre.1,
                            })
                            .collect();
                        view.match_count = boxed.len();
                        view.matches = if boxed.is_empty() {
                            std::ptr::null_mut()
                        } else {
                            let ptr = boxed.as_mut_ptr();
                            std::mem::forget(boxed);
                            ptr
                        };
                    }
                    Outcome::Scalar(value) => {
                        view.kind = KIND_SCALAR;
                        view.scalar = value;
                    }
                    Outcome::Text(text) => {
                        view.kind = KIND_TEXT;
                        view.text = CString::new(text.replace('\0', ""))
                            .map(CString::into_raw)
                            .unwrap_or(std::ptr::null_mut());
                    }
                }
            }
        }
        view
    }

    /// Copy a view the library filled into an owned detection.
    ///
    /// # Safety
    /// The view must have been produced by [`ResultView::from_detection`]
    /// (or follow its layout exactly) and not yet been released.
    pub unsafe fn to_detection(&self, source: &'static str) -> Detection<Outcome> {
        let reason = if self.reason.is_null() {
            None
        } else {
            // SAFETY: the producer wrote a NUL-terminated string.
            Some(
                unsafe { CStr::from_ptr(self.reason) }
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        let mut detection = if self.status == STATUS_FOUND {
            let outcome = match self.kind {
                KIND_SCALAR => Outcome::Scalar(self.scalar),
                KIND_TEXT => Outcome::Text(if self.text.is_null() {
                    String::new()
                } else {
                    // SAFETY: as above.
                    unsafe { CStr::from_ptr(self.text) }
                        .to_string_lossy()
                        .into_owned()
                }),
                _ => {
                    let matches = if self.matches.is_null() || self.match_count == 0 {
                        &[][..]
                    } else {
                        // SAFETY: the producer allocated `match_count` views.
                        unsafe { std::slice::from_raw_parts(self.matches, self.match_count) }
                    };
                    Outcome::Matches(
                        matches
                            .iter()
                            .map(|m| Match {
                                bounds: Rect {
                                    x: m.x,
                                    y: m.y,
                                    w: m.w,
                                    h: m.h,
                                },
                                score: m.score,
                                centre: (m.cx, m.cy),
                            })
                            .collect(),
                    )
                }
            };
            Detection::found(
                outcome,
                Confidence::new(self.confidence),
                source,
                reliability_from_code(self.reliability),
            )
        } else {
            Detection::missing(
                source,
                reason.clone().unwrap_or_else(|| "no reason given".into()),
            )
        };
        detection.failure_reason = reason;
        detection
    }

    /// Free what [`ResultView::from_detection`] allocated. Only the
    /// producing library may call this (through `syrup_intent_free`).
    ///
    /// # Safety
    /// The view must have been produced by [`ResultView::from_detection`]
    /// in this same library and not been released before.
    pub unsafe fn release(&mut self) {
        if !self.matches.is_null() && self.match_count > 0 {
            // SAFETY: allocated as a boxed slice of this length in from_detection.
            drop(unsafe {
                Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                    self.matches,
                    self.match_count,
                ))
            });
        }
        if !self.text.is_null() {
            // SAFETY: allocated by CString::into_raw in from_detection.
            drop(unsafe { CString::from_raw(self.text) });
        }
        if !self.reason.is_null() {
            // SAFETY: as above.
            drop(unsafe { CString::from_raw(self.reason) });
        }
        *self = Self::empty();
    }
}

pub fn reliability_code(reliability: Reliability) -> u32 {
    match reliability {
        Reliability::Corroborated => 0,
        Reliability::Heuristic => 1,
        Reliability::Predicted => 2,
        Reliability::Unreliable => 3,
    }
}

pub fn reliability_from_code(code: u32) -> Reliability {
    match code {
        0 => Reliability::Corroborated,
        1 => Reliability::Heuristic,
        2 => Reliability::Predicted,
        _ => Reliability::Unreliable,
    }
}

/// Export a Rust implementation of an intent under the C ABI.
///
/// `$run` is a `fn(&RgbaImage, Rect) -> Detection<Outcome>`. Panics inside
/// it are caught and reported as a failed call rather than unwinding
/// across the boundary.
///
/// ```ignore
/// fn run(image: &RgbaImage, region: Rect) -> Detection<Outcome> { /* … */ }
/// syrup::export_intent!("find_face", run);
/// ```
#[macro_export]
macro_rules! export_intent {
    ($name:literal, $run:path) => {
        #[unsafe(no_mangle)]
        pub extern "C" fn syrup_abi_version() -> u32 {
            $crate::abi::ABI_VERSION
        }

        #[unsafe(no_mangle)]
        pub extern "C" fn syrup_intent_name() -> *const ::std::ffi::c_char {
            concat!($name, "\0").as_ptr() as *const ::std::ffi::c_char
        }

        /// # Safety
        /// `frame` and `out` must be valid; `region` may be null.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn syrup_intent_run(
            frame: *const $crate::abi::FrameView,
            region: *const $crate::abi::RegionView,
            out: *mut $crate::abi::ResultView,
        ) -> i32 {
            if frame.is_null() || out.is_null() {
                return 1;
            }
            let result = ::std::panic::catch_unwind(|| {
                // SAFETY: the caller passes a valid frame view.
                let image = unsafe { (*frame).to_image() };
                let Some(image) = image else {
                    return $crate::Detection::missing("intent", "invalid frame view");
                };
                let region = if region.is_null() {
                    $crate::Rect {
                        x: 0,
                        y: 0,
                        w: image.width(),
                        h: image.height(),
                    }
                } else {
                    // SAFETY: non-null region views are valid.
                    unsafe { *region }.into()
                };
                $run(&image, region)
            });
            match result {
                Ok(detection) => {
                    // SAFETY: `out` is valid per the contract.
                    unsafe { out.write($crate::abi::ResultView::from_detection(detection)) };
                    0
                }
                Err(_) => 2,
            }
        }

        /// # Safety
        /// `out` must have been filled by `syrup_intent_run` from this library.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn syrup_intent_free(out: *mut $crate::abi::ResultView) {
            if !out.is_null() {
                // SAFETY: per the contract, produced here and not yet freed.
                unsafe { (*out).release() };
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detections_survive_the_round_trip() {
        let matches = vec![
            Match {
                bounds: Rect {
                    x: 1,
                    y: 2,
                    w: 3,
                    h: 4,
                },
                score: 0.5,
                centre: (2.5, 4.0),
            },
            Match {
                bounds: Rect {
                    x: 9,
                    y: 9,
                    w: 1,
                    h: 1,
                },
                score: 1.0,
                centre: (9.5, 9.5),
            },
        ];
        let detection = Detection::found(
            Outcome::Matches(matches.clone()),
            Confidence::new(0.75),
            "test",
            Reliability::Corroborated,
        );
        let mut view = ResultView::from_detection(detection);
        // SAFETY: produced just above.
        let back = unsafe { view.to_detection("test") };
        assert_eq!(back.value, Some(Outcome::Matches(matches)));
        assert_eq!(back.confidence.value(), 0.75);
        assert_eq!(back.reliability, Reliability::Corroborated);
        assert_eq!(back.failure_reason, None);
        // SAFETY: produced by from_detection, released once.
        unsafe { view.release() };
        assert!(view.matches.is_null());
    }

    #[test]
    fn scalars_text_and_failures_cross_too() {
        let mut view = ResultView::from_detection(Detection::found(
            Outcome::Scalar(42.5),
            Confidence::CERTAIN,
            "t",
            Reliability::Heuristic,
        ));
        // SAFETY: produced just above.
        assert_eq!(
            unsafe { view.to_detection("t") }.value,
            Some(Outcome::Scalar(42.5))
        );
        unsafe { view.release() };

        let mut view = ResultView::from_detection(Detection::found(
            Outcome::Text("29:56".into()),
            Confidence::new(0.9),
            "t",
            Reliability::Heuristic,
        ));
        assert_eq!(
            unsafe { view.to_detection("t") }.value,
            Some(Outcome::Text("29:56".into()))
        );
        unsafe { view.release() };

        let mut view = ResultView::from_detection(Detection::missing("t", "nothing there"));
        let back = unsafe { view.to_detection("t") };
        assert_eq!(back.value, None);
        assert_eq!(back.failure_reason.as_deref(), Some("nothing there"));
        unsafe { view.release() };
    }

    #[test]
    fn frame_views_copy_rows_by_stride() {
        let image = RgbaImage::from_fn(3, 2, |x, y| image::Rgba([x as u8, y as u8, 7, 255]));
        let view = FrameView::of(&image);
        // SAFETY: the view borrows `image`, which is alive.
        let copy = unsafe { view.to_image() }.expect("valid view");
        assert_eq!(copy, image);
        let bad = FrameView {
            data: std::ptr::null(),
            width: 1,
            height: 1,
            stride: 4,
        };
        assert!(unsafe { bad.to_image() }.is_none());
    }
}
