//! The library's own C API, for hosts that are not Rust: resolve an intent
//! by name, run it on a frame, read the result. This is what the Python
//! package in `python/` calls through `ctypes`; any language with a C FFI
//! can do the same. Frames and results use the views in [`crate::abi`].
//!
//! ```text
//!   handle = syrup_resolve("find_face", &error)   // NULL + error text on refusal
//!   syrup_run(handle, &frame, NULL, &result)      // 0 on success
//!   … read result.matches[0..match_count] …
//!   syrup_result_free(&result)
//!   syrup_release(handle)
//! ```
//!
//! Every string the library hands out is freed with [`syrup_string_free`],
//! every result with [`syrup_result_free`], every handle with
//! [`syrup_release`]; nothing crosses an allocator boundary. The names stay
//! clear of the `syrup_intent_*` exports a compiled intent carries, since
//! an intent links this library in.
//!
//! Only built when the crate is compiled as a shared library (`cdylib`);
//! Rust callers use [`crate::intent`] directly.

use std::ffi::{CStr, CString, c_char};

use crate::abi::{FrameView, RegionView, ResultView};
use crate::intent::{self, Resolved};

/// An intent the host resolved: an opaque handle over [`Resolved`].
pub struct SyrupIntent(Resolved);

fn leak(text: String) -> *mut c_char {
    CString::new(text.replace('\0', ""))
        .map(CString::into_raw)
        .unwrap_or(std::ptr::null_mut())
}

/// # Safety
/// `text` must be a NUL-terminated string.
unsafe fn read(text: *const c_char) -> Option<String> {
    if text.is_null() {
        return None;
    }
    // SAFETY: the caller passes a valid C string.
    Some(
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// The library version, as a static string (do not free).
#[unsafe(no_mangle)]
pub extern "C" fn syrup_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Understand `name`. Returns a handle, or NULL with `*error` set to a
/// message the caller frees with [`syrup_string_free`] (when `error` is
/// not NULL).
///
/// # Safety
/// `name` must be a NUL-terminated string; `error` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrup_resolve(
    name: *const c_char,
    error: *mut *mut c_char,
) -> *mut SyrupIntent {
    // SAFETY: per the contract.
    let Some(name) = (unsafe { read(name) }) else {
        if !error.is_null() {
            // SAFETY: `error` is writable per the contract.
            unsafe { *error = leak("name is NULL".into()) };
        }
        return std::ptr::null_mut();
    };
    match intent::resolve(&name) {
        Ok(resolved) => Box::into_raw(Box::new(SyrupIntent(resolved))),
        Err(failure) => {
            if !error.is_null() {
                // SAFETY: as above.
                unsafe { *error = leak(failure.to_string()) };
            }
            std::ptr::null_mut()
        }
    }
}

/// Run the intent on `frame` (all of it when `region` is NULL), filling
/// `out`. Returns 0 on success, 1 on bad arguments. `out` is released
/// with [`syrup_result_free`].
///
/// # Safety
/// `handle` must come from [`syrup_resolve`]; `frame` and `out` must be
/// valid; `region` NULL or valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrup_run(
    handle: *const SyrupIntent,
    frame: *const FrameView,
    region: *const RegionView,
    out: *mut ResultView,
) -> i32 {
    if handle.is_null() || frame.is_null() || out.is_null() {
        return 1;
    }
    // SAFETY: per the contract.
    let (resolved, image) = unsafe { (&(*handle).0, (*frame).to_image()) };
    let Some(image) = image else {
        return 1;
    };
    // SAFETY: a non-null region is valid per the contract.
    let region = (!region.is_null()).then(|| unsafe { *region }.into());
    let detection = resolved.run(&image, region);
    // SAFETY: `out` is writable per the contract.
    unsafe { out.write(ResultView::from_detection(detection)) };
    0
}

/// The Rust source the library generated for the intent, to free with
/// [`syrup_string_free`].
///
/// # Safety
/// `handle` must come from [`syrup_resolve`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrup_source(handle: *const SyrupIntent) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: per the contract.
    leak(unsafe { &(*handle).0 }.source())
}

/// Compile the intent to its own shared library and use it from now on.
/// Returns the library's path (free with [`syrup_string_free`]), or NULL
/// with `*error` set.
///
/// # Safety
/// `handle` must come from [`syrup_resolve`]; `error` NULL or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrup_compile(
    handle: *const SyrupIntent,
    error: *mut *mut c_char,
) -> *mut c_char {
    if handle.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: per the contract.
    match unsafe { &(*handle).0 }.compile() {
        Ok(path) => leak(path.to_string_lossy().into_owned()),
        Err(failure) => {
            if !error.is_null() {
                // SAFETY: `error` is writable per the contract.
                unsafe { *error = leak(failure.to_string()) };
            }
            std::ptr::null_mut()
        }
    }
}

/// Make `frame` the picture behind `find_<name>_icon` from now on.
/// Returns 0 on success, 1 on bad arguments.
///
/// # Safety
/// `name` must be a NUL-terminated string and `frame` a valid view.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrup_register_template(
    name: *const c_char,
    frame: *const FrameView,
) -> i32 {
    if frame.is_null() {
        return 1;
    }
    // SAFETY: per the contract.
    let (name, image) = unsafe { (read(name), (*frame).to_image()) };
    match (name, image) {
        (Some(name), Some(image)) => {
            intent::register_template(&name, &image);
            0
        }
        _ => 1,
    }
}

/// Release a result filled by [`syrup_run`].
///
/// # Safety
/// `out` must have been filled by `syrup_run` and not released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrup_result_free(out: *mut ResultView) {
    if !out.is_null() {
        // SAFETY: per the contract.
        unsafe { (*out).release() };
    }
}

/// Release a handle from [`syrup_resolve`].
///
/// # Safety
/// `handle` must come from `syrup_resolve` and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrup_release(handle: *mut SyrupIntent) {
    if !handle.is_null() {
        // SAFETY: per the contract.
        drop(unsafe { Box::from_raw(handle) });
    }
}

/// Release a string the library handed out.
///
/// # Safety
/// `text` must come from this library and not have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrup_string_free(text: *mut c_char) {
    if !text.is_null() {
        // SAFETY: per the contract.
        drop(unsafe { CString::from_raw(text) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::{KIND_MATCHES, STATUS_FOUND};

    #[test]
    fn a_host_can_resolve_run_and_free() {
        let image = image::load_from_memory(include_bytes!("../tests/fixtures/astronaut_320.jpg"))
            .unwrap()
            .to_rgba8();
        let name = CString::new("find_face").unwrap();
        let mut error: *mut c_char = std::ptr::null_mut();
        // SAFETY: valid arguments throughout.
        unsafe {
            let handle = syrup_resolve(name.as_ptr(), &mut error);
            assert!(!handle.is_null());
            assert!(error.is_null());
            let frame = FrameView::of(&image);
            let mut out = ResultView::empty();
            assert_eq!(syrup_run(handle, &frame, std::ptr::null(), &mut out), 0);
            assert_eq!(out.status, STATUS_FOUND);
            assert_eq!(out.kind, KIND_MATCHES);
            assert_eq!(out.match_count, 1);
            let face = *out.matches;
            assert!(face.x > 100 && face.x < 120);
            syrup_result_free(&mut out);
            let source = syrup_source(handle);
            assert!(
                CStr::from_ptr(source)
                    .to_str()
                    .unwrap()
                    .contains("plans::find_faces")
            );
            syrup_string_free(source);
            syrup_release(handle);
        }
    }

    #[test]
    fn refusals_come_back_as_text() {
        let name = CString::new("find_unicorn").unwrap();
        let mut error: *mut c_char = std::ptr::null_mut();
        // SAFETY: valid arguments.
        unsafe {
            let handle = syrup_resolve(name.as_ptr(), &mut error);
            assert!(handle.is_null());
            let text = CStr::from_ptr(error).to_str().unwrap().to_string();
            assert!(text.contains("not an intent I understand"), "{text}");
            syrup_string_free(error);
            assert_eq!(
                syrup_run(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut()
                ),
                1
            );
        }
    }
}
