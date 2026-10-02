// Syrup operation ABI v3. Included by the host and pasted verbatim into every
// generated module, so both sides always agree. Bump SYRUP_ABI_VERSION when a
// type's layout or a function's meaning changes; new capability ids are not
// breaking, since hosts refuse ids they do not know.
//
// Views are borrowed for one call. Detections returned by `detect` belong to
// the host and stay valid only until the module's next call into the host.

pub const SYRUP_ABI_VERSION: u32 = 3;

pub const SYRUP_OK: i32 = 0;
pub const SYRUP_ERR_PROVIDER: i32 = 1;
pub const SYRUP_ERR_INPUT: i32 = 2;
pub const SYRUP_ERR_ABI: i32 = 3;
pub const SYRUP_ERR_PANIC: i32 = 4;
pub const SYRUP_ERR_EMIT: i32 = 5;

pub const SYRUP_CAP_FACE: u32 = 1;
pub const SYRUP_CAP_TEXT: u32 = 2;
pub const SYRUP_CAP_MOTION: u32 = 3;
pub const SYRUP_CAP_QR: u32 = 4;
pub const SYRUP_CAP_TEXT_BLOCKS: u32 = 5;
pub const SYRUP_CAP_PANELS: u32 = 6;

pub const SYRUP_MEASURE_SHARPNESS: u32 = 1;
pub const SYRUP_MEASURE_FILL: u32 = 2;

pub const SYRUP_MAX_KEYPOINTS: usize = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SyrupImageView {
    pub data: *const u8,
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub channels: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SyrupDetection {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub score: f32,
    // Set by `measure`; meaningless before.
    pub value: f32,
    pub n_keypoints: u32,
    pub keypoints: [f32; 2 * SYRUP_MAX_KEYPOINTS],
    // Opaque to modules, passed through untouched.
    pub payload: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SyrupParams {
    pub min_confidence: f32,
    // 0 means no cap.
    pub max_results: u32,
    // The host clips the region to the image before the call.
    pub has_region: u32,
    pub region_x: u32,
    pub region_y: u32,
    pub region_w: u32,
    pub region_h: u32,
}

pub type SyrupDetectFn = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    capability: u32,
    view: *const SyrupImageView,
    out_ptr: *mut *const SyrupDetection,
    out_len: *mut usize,
) -> i32;

/// A horizontal run of pixels on row `y`, from `x0` to `x1` inclusive.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyrupRun {
    pub y: u32,
    pub x0: u32,
    pub x1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyrupRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// Merge runs (in row order) into rectangles with the core's region
/// grouping. Rectangles are valid until the module's next host call.
pub type SyrupGroupFn = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    runs: *const SyrupRun,
    n_runs: usize,
    min_height: u32,
    max_gap: u32,
    out_ptr: *mut *const SyrupRect,
    out_len: *mut usize,
) -> i32;

/// What to measure. Fill counts pixels of the hue range (degrees, wrapping
/// when `hue_lo` is larger) with the given saturation and value.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyrupMeasure {
    pub kind: u32,
    pub hue_lo: u32,
    pub hue_hi: u32,
    pub min_saturation_pct: u32,
    pub min_value_pct: u32,
}

/// Writes one value per box (in `view`'s pixels) to `values`; NaN where the
/// quantity cannot be measured.
pub type SyrupMeasureFn = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    what: *const SyrupMeasure,
    view: *const SyrupImageView,
    boxes: *const SyrupDetection,
    n_boxes: usize,
    values: *mut f32,
) -> i32;

pub type SyrupEmitFn =
    unsafe extern "C" fn(ctx: *mut core::ffi::c_void, detection: *const SyrupDetection) -> i32;

#[repr(C)]
pub struct SyrupHost {
    pub abi_version: u32,
    pub struct_size: u32,
    pub ctx: *mut core::ffi::c_void,
    pub detect: SyrupDetectFn,
    pub emit: SyrupEmitFn,
    pub group: SyrupGroupFn,
    pub measure: SyrupMeasureFn,
}

pub type SyrupRunFn = unsafe extern "C" fn(
    host: *const SyrupHost,
    input: *const SyrupImageView,
    params: *const SyrupParams,
) -> i32;
