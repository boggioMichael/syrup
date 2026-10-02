//! The host side of a real run: serves `detect` from the providers, checks
//! what they return, and collects what the module emits.

use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};

use syrup::color::is_color_pixel;
use syrup::geometry::{Rect, measure_bar_fill};
use syrup::quality::{Legibility, assess_text_quality};

use crate::abi::*;
use crate::catalog::Capability;
use crate::contract::ImageInput;
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::interp::group_runs;
use crate::providers::{Detections, Provider, Providers, ViewRef};

pub struct ExecHost<'a> {
    image: ImageInput<'a>,
    providers: &'a Providers,
    // A session's motion provider, which only sessions have.
    motion: Option<&'a dyn Provider>,
    scratch: Vec<SyrupDetection>,
    rects: Vec<SyrupRect>,
    pub out: Vec<SyrupDetection>,
    /// Text read by providers; a detection's payload is its index + 1.
    pub texts: Vec<String>,
    pub error: Option<SyrupError>,
}

impl<'a> ExecHost<'a> {
    pub fn new(
        image: ImageInput<'a>,
        providers: &'a Providers,
        motion: Option<&'a dyn Provider>,
    ) -> Self {
        ExecHost {
            image,
            providers,
            motion,
            scratch: vec![],
            rects: vec![],
            out: vec![],
            texts: vec![],
            error: None,
        }
    }

    /// `self` must stay put while the table is in use.
    pub fn table(&mut self) -> SyrupHost {
        SyrupHost {
            abi_version: SYRUP_ABI_VERSION,
            struct_size: size_of::<SyrupHost>() as u32,
            ctx: (self as *mut Self).cast(),
            detect: host_detect,
            emit: host_emit,
            group: host_group,
            measure: host_measure,
        }
    }

    fn view(&self, view: &SyrupImageView) -> Result<ViewRef<'a>> {
        let data = self.image.data();
        let (stride, channels) = (self.image.stride(), self.image.channels() as usize);
        let offset = (view.data as usize).wrapping_sub(data.as_ptr() as usize);
        let (x, y) = ((offset % stride) / channels, offset / stride);
        let inside = view.stride == stride
            && view.channels as usize == channels
            && offset < data.len()
            && (offset % stride).is_multiple_of(channels)
            && view.width > 0
            && view.height > 0
            && x + view.width as usize <= self.image.width() as usize
            && y + view.height as usize <= self.image.height() as usize;
        if !inside {
            return Err(SyrupError::new(
                Stage::Execute,
                ErrorKind::ModuleFailed,
                format!("the module passed a view outside the input image ({view:?})"),
            ));
        }
        let len = (view.height as usize - 1) * stride + view.width as usize * channels;
        Ok(ViewRef {
            data: &data[offset..offset + len],
            width: view.width,
            height: view.height,
            stride,
            channels: view.channels,
        })
    }

    fn detect(&mut self, capability: u32, view: &SyrupImageView) -> Result<()> {
        let capability = Capability::from_abi_id(capability).ok_or_else(|| {
            SyrupError::new(
                Stage::Execute,
                ErrorKind::ModuleFailed,
                format!("the module asked for unknown capability {capability}"),
            )
        })?;
        let view = self.view(view)?;
        let Detections { mut boxes, texts } = match (capability, self.motion) {
            (Capability::Motion, Some(motion)) => motion.detect(&view)?,
            _ => self.providers.get(capability)?.detect(&view)?,
        };
        let malformed = |what: String| {
            SyrupError::new(
                Stage::Execute,
                ErrorKind::ProviderFailed,
                format!("the {} provider returned {what}", capability.as_str()),
            )
        };
        if let Some(bad) = boxes.iter().find(|d| !well_formed(d)) {
            return Err(malformed(format!("a malformed detection: {bad:?}")));
        }
        if !texts.is_empty() && texts.len() != boxes.len() {
            return Err(malformed(format!(
                "{} texts for {} boxes",
                texts.len(),
                boxes.len()
            )));
        }
        for d in &mut boxes {
            d.payload = 0;
        }
        for (d, text) in boxes.iter_mut().zip(texts) {
            self.texts.push(text);
            d.payload = self.texts.len() as u64;
        }
        self.scratch = boxes;
        Ok(())
    }
}

/// The core's measurements, on the pixels of `view`; `boxes` are in its
/// coordinates.
pub fn measure(
    view: &ViewRef<'_>,
    what: &SyrupMeasure,
    boxes: &[SyrupDetection],
) -> Result<Vec<f32>> {
    let rect = |d: &SyrupDetection| {
        let x0 = (d.x.floor().max(0.0) as u32).min(view.width);
        let y0 = (d.y.floor().max(0.0) as u32).min(view.height);
        let x1 = ((d.x + d.w).ceil().max(0.0) as u32).min(view.width);
        let y1 = ((d.y + d.h).ceil().max(0.0) as u32).min(view.height);
        Rect {
            x: x0,
            y: y0,
            w: x1.saturating_sub(x0),
            h: y1.saturating_sub(y0),
        }
    };
    let rects: Vec<Rect> = boxes.iter().map(rect).collect();
    // Only the boxes' rows are read (fill searches along them, across the
    // whole view), so only those are converted.
    let top = rects.iter().map(|r| r.y).min().unwrap_or(0);
    let bottom = rects.iter().map(|r| r.y + r.h).max().unwrap_or(0);
    let pixels = view.rows_to_rgba(top, bottom.max(top));
    let local = |r: &Rect| Rect { y: r.y - top, ..*r };
    let whole = Rect {
        x: 0,
        y: 0,
        w: view.width,
        h: bottom.saturating_sub(top),
    };
    match what.kind {
        SYRUP_MEASURE_SHARPNESS => Ok(rects
            .iter()
            .map(|r| {
                let quality = assess_text_quality(&pixels, local(r));
                match quality.legibility {
                    Legibility::NoText => f32::NAN,
                    _ => quality.sharpness,
                }
            })
            .collect()),
        SYRUP_MEASURE_FILL => {
            let hue = (what.hue_lo as f32, what.hue_hi as f32);
            let (sat, val) = (
                what.min_saturation_pct as f32 / 100.0,
                what.min_value_pct as f32 / 100.0,
            );
            Ok(rects
                .iter()
                .map(|r| {
                    measure_bar_fill(&pixels, local(r), whole, |p| {
                        is_color_pixel(p, hue, sat, val)
                    })
                    .map_or(f32::NAN, |pct| pct / 100.0)
                })
                .collect())
        }
        kind => Err(SyrupError::new(
            Stage::Execute,
            ErrorKind::ModuleFailed,
            format!("the module asked for unknown measurement {kind}"),
        )),
    }
}

pub fn well_formed(d: &SyrupDetection) -> bool {
    [d.x, d.y, d.w, d.h]
        .iter()
        .chain(&d.keypoints)
        .all(|v| v.is_finite())
        && d.w >= 0.0
        && d.h >= 0.0
        && (0.0..=1.0).contains(&d.score)
        && d.n_keypoints as usize <= SYRUP_MAX_KEYPOINTS
}

/// Runs a host callback for the module: errors and panics become a status
/// the module sees and an error the run reports. The first error wins.
fn guarded(host: &mut ExecHost, what: &str, call: impl FnOnce(&mut ExecHost) -> Result<()>) -> i32 {
    let error = match catch_unwind(AssertUnwindSafe(|| call(&mut *host))) {
        Ok(Ok(())) => return SYRUP_OK,
        Ok(Err(e)) => e,
        Err(_) => SyrupError::new(Stage::Execute, ErrorKind::Panic, format!("{what} panicked")),
    };
    host.error.get_or_insert(error);
    SYRUP_ERR_PROVIDER
}

/// `n` items at `ptr`, which may be null when `n` is 0.
///
/// # Safety
/// When `n > 0`, `ptr` must point at `n` valid items for the call.
unsafe fn items<'a, T>(ptr: *const T, n: usize) -> &'a [T] {
    match n {
        0 => &[],
        // SAFETY: forwarded to the caller.
        n => unsafe { std::slice::from_raw_parts(ptr, n) },
    }
}

// SAFETY, for the callbacks below: `ctx` is the ExecHost that built the
// table, and every pointer comes from the module, valid for the call as the
// ABI requires.

unsafe extern "C" fn host_detect(
    ctx: *mut c_void,
    capability: u32,
    view: *const SyrupImageView,
    out_ptr: *mut *const SyrupDetection,
    out_len: *mut usize,
) -> i32 {
    let host = unsafe { &mut *(ctx as *mut ExecHost) };
    let view = unsafe { &*view };
    let status = guarded(host, "a provider", |host| host.detect(capability, view));
    if status == SYRUP_OK {
        unsafe {
            *out_ptr = host.scratch.as_ptr();
            *out_len = host.scratch.len();
        }
    }
    status
}

unsafe extern "C" fn host_measure(
    ctx: *mut c_void,
    what: *const SyrupMeasure,
    view: *const SyrupImageView,
    boxes: *const SyrupDetection,
    n_boxes: usize,
    values: *mut f32,
) -> i32 {
    let host = unsafe { &mut *(ctx as *mut ExecHost) };
    let (what, view, boxes) = unsafe { (&*what, &*view, items(boxes, n_boxes)) };
    guarded(host, "a measurement", |host| {
        let measured = measure(&host.view(view)?, what, boxes)?;
        for (i, value) in measured.into_iter().enumerate() {
            unsafe { *values.add(i) = value };
        }
        Ok(())
    })
}

unsafe extern "C" fn host_emit(ctx: *mut c_void, detection: *const SyrupDetection) -> i32 {
    unsafe { (*(ctx as *mut ExecHost)).out.push(*detection) };
    SYRUP_OK
}

unsafe extern "C" fn host_group(
    ctx: *mut c_void,
    runs: *const SyrupRun,
    n_runs: usize,
    min_height: u32,
    max_gap: u32,
    out_ptr: *mut *const SyrupRect,
    out_len: *mut usize,
) -> i32 {
    let host = unsafe { &mut *(ctx as *mut ExecHost) };
    let runs = unsafe { items(runs, n_runs) };
    let status = guarded(host, "region grouping", |host| {
        host.rects = group_runs(runs, min_height, max_gap);
        Ok(())
    });
    if status == SYRUP_OK {
        unsafe {
            *out_ptr = host.rects.as_ptr();
            *out_len = host.rects.len();
        }
    }
    status
}
