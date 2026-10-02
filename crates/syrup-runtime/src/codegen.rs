//! Plan → Rust source for a dependency-free `cdylib`. Each step kind has a
//! small template; `run` in the generated file is the composed program.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::intent::{Intent, OrderKey, RegionSpec};
use crate::plan::{Plan, Step};

pub const CODEGEN_VERSION: u32 = 3;

pub const EXPORTS: [&str; 3] = ["syrup_op_abi_version", "syrup_op_plan_hash", "syrup_op_run"];

const PRELUDE: &str = include_str!("abi_prelude.rs");

const ENTRY: &str = r#"
#[unsafe(no_mangle)]
pub extern "C" fn syrup_op_abi_version() -> u32 {
    SYRUP_ABI_VERSION
}

#[unsafe(no_mangle)]
pub extern "C" fn syrup_op_plan_hash() -> *const u8 {
    PLAN_HASH.as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn syrup_op_run(
    host: *const SyrupHost,
    input: *const SyrupImageView,
    params: *const SyrupParams,
) -> i32 {
    if host.is_null() || input.is_null() || params.is_null() {
        return SYRUP_ERR_INPUT;
    }
    // SAFETY: the host passes pointers that stay valid for the whole call.
    let (host, input, params) = unsafe { (&*host, &*input, &*params) };
    if host.abi_version != SYRUP_ABI_VERSION
        || (host.struct_size as usize) < core::mem::size_of::<SyrupHost>()
    {
        return SYRUP_ERR_ABI;
    }
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(host, input, params))) {
        Ok(Ok(())) => SYRUP_OK,
        Ok(Err(status)) => status,
        Err(_) => SYRUP_ERR_PANIC,
    }
}
"#;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Helper {
    Window,
    SelectFixed,
    SelectCaller,
    Detect,
    Color,
    Whole,
    Measure,
    Restore,
    TieBreak,
    Emit,
}

impl Helper {
    fn source(self) -> &'static str {
        match self {
            Helper::Window => {
                r#"
#[derive(Clone, Copy)]
struct Window {
    view: SyrupImageView,
    x: u32,
    y: u32,
    // The whole input image, so pixel reads can be bounds-checked.
    base: *const u8,
    len: usize,
}

fn input_window(input: &SyrupImageView) -> Window {
    let len = match input.height {
        0 => 0,
        h => (h as usize - 1) * input.stride + input.width as usize * input.channels as usize,
    };
    Window {
        view: *input,
        x: 0,
        y: 0,
        base: input.data,
        len,
    }
}

fn sub_window(parent: &Window, x: u32, y: u32, w: u32, h: u32) -> Window {
    let offset = y as usize * parent.view.stride + x as usize * parent.view.channels as usize;
    Window {
        view: SyrupImageView {
            data: parent.view.data.wrapping_add(offset),
            width: w,
            height: h,
            stride: parent.view.stride,
            channels: parent.view.channels,
        },
        x: parent.x + x,
        y: parent.y + y,
        base: parent.base,
        len: parent.len,
    }
}
"#
            }
            Helper::SelectFixed => {
                r#"
fn round_frac(num: u64, den: u64, extent: u32) -> u32 {
    ((2 * num * extent as u64 + den) / (2 * den)) as u32
}

fn select_fixed(parent: &Window, x0: (u64, u64), y0: (u64, u64), x1: (u64, u64), y1: (u64, u64)) -> Window {
    let (w, h) = (parent.view.width, parent.view.height);
    let left = round_frac(x0.0, x0.1, w);
    let top = round_frac(y0.0, y0.1, h);
    let right = round_frac(x1.0, x1.1, w);
    let bottom = round_frac(y1.0, y1.1, h);
    sub_window(parent, left, top, right - left, bottom - top)
}
"#
            }
            Helper::SelectCaller => {
                r#"
fn select_caller(parent: &Window, params: &SyrupParams) -> Result<Window, i32> {
    if params.has_region != 1 {
        return Err(SYRUP_ERR_INPUT);
    }
    let (w, h) = (parent.view.width, parent.view.height);
    let left = params.region_x.min(w);
    let top = params.region_y.min(h);
    let right = (params.region_x as u64 + params.region_w as u64).min(w as u64) as u32;
    let bottom = (params.region_y as u64 + params.region_h as u64).min(h as u64) as u32;
    Ok(sub_window(parent, left, top, right.saturating_sub(left), bottom.saturating_sub(top)))
}
"#
            }
            Helper::Detect => {
                r#"
fn detect(host: &SyrupHost, capability: u32, window: &Window) -> Result<Vec<SyrupDetection>, i32> {
    if window.view.width == 0 || window.view.height == 0 {
        return Ok(Vec::new());
    }
    let mut ptr: *const SyrupDetection = core::ptr::null();
    let mut len = 0usize;
    // SAFETY: the view borrows the input image, which outlives the call.
    let status = unsafe { (host.detect)(host.ctx, capability, &window.view, &mut ptr, &mut len) };
    if status != SYRUP_OK {
        return Err(status);
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(SYRUP_ERR_PROVIDER);
    }
    // SAFETY: the host keeps `len` detections at `ptr` until our next call into it.
    Ok(unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec())
}
"#
            }
            Helper::Color => {
                r#"
// The same arithmetic as syrup::color::hsv_from_rgb, stopping early: the
// hue (a division) is only worked out for pixels saturated and bright
// enough to need it.
fn hue_if(r: u8, g: u8, b: u8, min_s: f32, min_v: f32) -> Option<f32> {
    let rf = r as f32 / 255.0;
    let gf = g as f32 / 255.0;
    let bf = b as f32 / 255.0;
    let max = rf.max(gf).max(bf);
    if max.clamp(0.0, 1.0) < min_v {
        return None;
    }
    let min = rf.min(gf).min(bf);
    let delta = max - min;
    let s = if max == 0.0 { 0.0 } else { delta / max };
    if s.clamp(0.0, 1.0) < min_s {
        return None;
    }
    let h = if delta == 0.0 {
        0.0
    } else if max == rf {
        60.0 * ((gf - bf) / delta % 6.0)
    } else if max == gf {
        60.0 * ((bf - rf) / delta + 2.0)
    } else {
        60.0 * ((rf - gf) / delta + 4.0)
    };
    Some(if h < 0.0 { h + 360.0 } else { h })
}

fn row(window: &Window, y: u32) -> Result<&[u8], i32> {
    let v = &window.view;
    let start = (v.data as usize).wrapping_sub(window.base as usize) + y as usize * v.stride;
    let len = v.width as usize * v.channels as usize;
    if start.checked_add(len).is_none_or(|end| end > window.len) {
        return Err(SYRUP_ERR_INPUT);
    }
    // SAFETY: the row lies inside the input image, which outlives the call.
    Ok(unsafe { core::slice::from_raw_parts(window.base.add(start), len) })
}

fn group(host: &SyrupHost, runs: &[SyrupRun], min_height: u32, max_gap: u32) -> Result<Vec<SyrupRect>, i32> {
    let mut ptr: *const SyrupRect = core::ptr::null();
    let mut len = 0usize;
    // SAFETY: the host reads `runs` during the call and owns what it returns.
    let status = unsafe {
        (host.group)(host.ctx, runs.as_ptr(), runs.len(), min_height, max_gap, &mut ptr, &mut len)
    };
    if status != SYRUP_OK {
        return Err(status);
    }
    if len == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(SYRUP_ERR_PROVIDER);
    }
    // SAFETY: valid until our next call into the host.
    Ok(unsafe { core::slice::from_raw_parts(ptr, len) }.to_vec())
}

fn find_color(
    host: &SyrupHost,
    window: &Window,
    matches: fn(u8, u8, u8, u8) -> bool,
    min_run: u32,
    min_height: u32,
    max_gap: u32,
) -> Result<Vec<SyrupDetection>, i32> {
    let (width, height) = (window.view.width, window.view.height);
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }
    let channels = window.view.channels as usize;
    let pixel = |p: &[u8]| match channels {
        1 => matches(p[0], p[0], p[0], 255),
        3 => matches(p[0], p[1], p[2], 255),
        _ => matches(p[0], p[1], p[2], p[3]),
    };
    let mut runs = Vec::new();
    for y in 0..height {
        let mut start = None;
        for (x, p) in row(window, y)?.chunks_exact(channels).enumerate() {
            let x = x as u32;
            match (pixel(p), start) {
                (true, None) => start = Some(x),
                (false, Some(s)) => {
                    if x - s >= min_run {
                        runs.push(SyrupRun { y, x0: s, x1: x - 1 });
                    }
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            if width - s >= min_run {
                runs.push(SyrupRun { y, x0: s, x1: width - 1 });
            }
        }
    }
    let mut out = Vec::new();
    for r in group(host, &runs, min_height, max_gap)? {
        let inside = r.w > 0
            && r.h > 0
            && r.x.checked_add(r.w).is_some_and(|e| e <= width)
            && r.y.checked_add(r.h).is_some_and(|e| e <= height);
        if !inside {
            return Err(SYRUP_ERR_PROVIDER);
        }
        let mut hits = 0u32;
        for y in r.y..r.y + r.h {
            let line = &row(window, y)?[r.x as usize * channels..(r.x + r.w) as usize * channels];
            for p in line.chunks_exact(channels) {
                hits += pixel(p) as u32;
            }
        }
        out.push(SyrupDetection {
            x: r.x as f32,
            y: r.y as f32,
            w: r.w as f32,
            h: r.h as f32,
            score: hits as f32 / (r.w * r.h) as f32,
            value: 0.0,
            n_keypoints: 0,
            keypoints: [0.0; 2 * SYRUP_MAX_KEYPOINTS],
            payload: 0,
        });
    }
    Ok(out)
}
"#
            }
            Helper::Whole => {
                r#"
fn whole(window: &Window) -> Vec<SyrupDetection> {
    if window.view.width == 0 || window.view.height == 0 {
        return Vec::new();
    }
    vec![SyrupDetection {
        x: 0.0,
        y: 0.0,
        w: window.view.width as f32,
        h: window.view.height as f32,
        score: 1.0,
        value: 0.0,
        n_keypoints: 0,
        keypoints: [0.0; 2 * SYRUP_MAX_KEYPOINTS],
        payload: 0,
    }]
}
"#
            }
            Helper::Measure => {
                r#"
fn measure(
    host: &SyrupHost,
    what: &SyrupMeasure,
    window: &Window,
    mut boxes: Vec<SyrupDetection>,
) -> Result<Vec<SyrupDetection>, i32> {
    if boxes.is_empty() {
        return Ok(boxes);
    }
    // The host measures in the window's pixels.
    let local: Vec<SyrupDetection> = boxes
        .iter()
        .map(|d| SyrupDetection {
            x: d.x - window.x as f32,
            y: d.y - window.y as f32,
            ..*d
        })
        .collect();
    let mut values = vec![f32::NAN; boxes.len()];
    // SAFETY: the host reads the boxes and writes one value per box during the call.
    let status = unsafe {
        (host.measure)(host.ctx, what, &window.view, local.as_ptr(), local.len(), values.as_mut_ptr())
    };
    if status != SYRUP_OK {
        return Err(status);
    }
    for (d, value) in boxes.iter_mut().zip(values) {
        d.value = value;
    }
    boxes.retain(|d| d.value.is_finite());
    Ok(boxes)
}
"#
            }
            Helper::Restore => {
                r#"
fn restore(boxes: Vec<SyrupDetection>, window: &Window) -> Vec<SyrupDetection> {
    let left = window.x as f32;
    let top = window.y as f32;
    let right = (window.x + window.view.width) as f32;
    let bottom = (window.y + window.view.height) as f32;
    let mut out = Vec::with_capacity(boxes.len());
    for d in boxes {
        let x0 = (d.x + left).max(left).min(right);
        let y0 = (d.y + top).max(top).min(bottom);
        let x1 = (d.x + left + d.w).max(left).min(right);
        let y1 = (d.y + top + d.h).max(top).min(bottom);
        let (w, h) = (x1 - x0, y1 - y0);
        if !(w > 0.0 && h > 0.0) {
            continue;
        }
        let n = (d.n_keypoints as usize).min(SYRUP_MAX_KEYPOINTS);
        let mut keypoints = [0.0f32; 2 * SYRUP_MAX_KEYPOINTS];
        for k in 0..n {
            keypoints[2 * k] = (d.keypoints[2 * k] + left).max(left).min(right);
            keypoints[2 * k + 1] = (d.keypoints[2 * k + 1] + top).max(top).min(bottom);
        }
        out.push(SyrupDetection {
            x: x0,
            y: y0,
            w,
            h,
            score: d.score,
            value: d.value,
            n_keypoints: n as u32,
            keypoints,
            payload: d.payload,
        });
    }
    out
}
"#
            }
            Helper::TieBreak => {
                r#"
fn tie_break(a: &SyrupDetection, b: &SyrupDetection) -> core::cmp::Ordering {
    b.score
        .total_cmp(&a.score)
        .then(a.y.total_cmp(&b.y))
        .then(a.x.total_cmp(&b.x))
        .then(a.h.total_cmp(&b.h))
        .then(a.w.total_cmp(&b.w))
}

// A stable merge sort: the standard library's sort is far more code for
// rustc to optimise, and these lists are short.
fn sort(v: &mut Vec<SyrupDetection>, cmp: impl Fn(&SyrupDetection, &SyrupDetection) -> core::cmp::Ordering) {
    let n = v.len();
    let mut from = v.clone();
    let mut to = v.clone();
    let mut width = 1;
    while width < n {
        let mut start = 0;
        while start < n {
            let (mid, end) = ((start + width).min(n), (start + 2 * width).min(n));
            let (mut i, mut j, mut k) = (start, mid, start);
            while i < mid && j < end {
                // Take from the right run only when strictly smaller, so equal items keep their order.
                if cmp(&from[j], &from[i]) == core::cmp::Ordering::Less {
                    to[k] = from[j];
                    j += 1;
                } else {
                    to[k] = from[i];
                    i += 1;
                }
                k += 1;
            }
            to[k..k + mid - i].copy_from_slice(&from[i..mid]);
            k += mid - i;
            to[k..k + end - j].copy_from_slice(&from[j..end]);
            start = end;
        }
        core::mem::swap(&mut from, &mut to);
        width *= 2;
    }
    *v = from;
}
"#
            }
            Helper::Emit => {
                r#"
fn emit(host: &SyrupHost, d: &SyrupDetection) -> Result<(), i32> {
    // SAFETY: the host copies the detection before returning.
    match unsafe { (host.emit)(host.ctx, d) } {
        SYRUP_OK => Ok(()),
        status => Err(status),
    }
}
"#
            }
        }
    }
}

fn frac(r: crate::intent::Ratio) -> String {
    format!("({}, {})", r.num, r.den)
}

fn primary_order(key: OrderKey) -> Option<&'static str> {
    Some(match key {
        OrderKey::ConfidenceDesc => return None,
        OrderKey::AreaDesc => "(b.w * b.h).total_cmp(&(a.w * a.h))",
        OrderKey::AreaAsc => "(a.w * a.h).total_cmp(&(b.w * b.h))",
        OrderKey::LeftToRight => "a.x.total_cmp(&b.x)",
        OrderKey::RightToLeft => "(b.x + b.w).total_cmp(&(a.x + a.w))",
        OrderKey::TopToBottom => "a.y.total_cmp(&b.y)",
        OrderKey::BottomToTop => "(b.y + b.h).total_cmp(&(a.y + a.h))",
    })
}

/// The module's source. The plan must have passed `Plan::check`.
pub fn generate(plan: &Plan, intent: &Intent, plan_hash: &str) -> String {
    let mut helpers = BTreeSet::new();
    let mut predicates = String::new();
    let mut body = String::new();
    let b = &mut body;
    if plan
        .steps
        .iter()
        .any(|s| matches!(s, Step::FilterArea { .. }))
    {
        let _ = writeln!(
            b,
            "    let total = input.width as f64 * input.height as f64;"
        );
    }
    for (i, step) in plan.steps.iter().enumerate() {
        match *step {
            Step::Input => {
                helpers.insert(Helper::Window);
                let _ = writeln!(b, "    let v{i} = input_window(input);");
            }
            Step::SelectRegion {
                view,
                region: RegionSpec::Fixed { rect },
            } => {
                helpers.extend([Helper::Window, Helper::SelectFixed]);
                let _ = writeln!(
                    b,
                    "    let v{i} = select_fixed(&v{view}, {}, {}, {}, {});",
                    frac(rect.x),
                    frac(rect.y),
                    frac(rect.right()),
                    frac(rect.bottom())
                );
            }
            Step::SelectRegion {
                view,
                region: RegionSpec::Caller,
            } => {
                helpers.extend([Helper::Window, Helper::SelectCaller]);
                let _ = writeln!(b, "    let v{i} = select_caller(&v{view}, params)?;");
            }
            Step::Detect { view, capability } => {
                helpers.insert(Helper::Detect);
                let _ = writeln!(
                    b,
                    "    let v{i} = detect(host, {}, &v{view})?;",
                    capability.abi_id()
                );
            }
            Step::FindColor {
                view,
                hue,
                min_saturation_pct,
                min_value_pct,
                min_run,
                min_height,
                max_gap,
            } => {
                helpers.insert(Helper::Color);
                let (lo, hi) = (hue.0 as f32, hue.1 as f32);
                let hue_test = if lo <= hi {
                    format!("h >= {lo:?} && h <= {hi:?}")
                } else {
                    format!("h >= {lo:?} || h <= {hi:?}")
                };
                let (sat, val) = (
                    min_saturation_pct as f32 / 100.0,
                    min_value_pct as f32 / 100.0,
                );
                let _ = write!(
                    predicates,
                    "\nfn matches_v{i}(r: u8, g: u8, b: u8, a: u8) -> bool {{\n    \
                     a as f32 / 255.0 >= 0.5 && hue_if(r, g, b, {sat:?}, {val:?}).is_some_and(|h| {hue_test})\n}}\n"
                );
                let _ = writeln!(
                    b,
                    "    let v{i} = find_color(host, &v{view}, matches_v{i}, {min_run}, {min_height}, {max_gap})?;"
                );
            }
            Step::Whole { view } => {
                helpers.insert(Helper::Whole);
                let _ = writeln!(b, "    let v{i} = whole(&v{view});");
            }
            Step::Measure { boxes, view, what } => {
                helpers.insert(Helper::Measure);
                let m = what.abi();
                let _ = writeln!(
                    b,
                    "    let v{i} = measure(host, &SyrupMeasure {{ kind: {}, hue_lo: {}, hue_hi: {}, min_saturation_pct: {}, min_value_pct: {} }}, &v{view}, v{boxes})?;",
                    m.kind, m.hue_lo, m.hue_hi, m.min_saturation_pct, m.min_value_pct
                );
            }
            Step::FilterAspect { boxes, min } => {
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                let _ = writeln!(b, "    v{i}.retain(|d| d.w >= {min}_f32 * d.h);");
            }
            Step::Restore { boxes, view } => {
                helpers.insert(Helper::Restore);
                let _ = writeln!(b, "    let v{i} = restore(v{boxes}, &v{view});");
            }
            Step::FilterConfidence { boxes } => {
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                let _ = writeln!(b, "    v{i}.retain(|d| d.score >= params.min_confidence);");
            }
            Step::FilterArea {
                boxes,
                min_pct,
                max_pct,
            } => {
                let mut tests = vec![];
                if let Some(p) = min_pct {
                    tests.push(format!("area >= {p}_f64 * total"));
                }
                if let Some(p) = max_pct {
                    tests.push(format!("area < {p}_f64 * total"));
                }
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                let _ = writeln!(b, "    v{i}.retain(|d| {{");
                let _ = writeln!(b, "        let area = d.w as f64 * d.h as f64 * 100.0;");
                let _ = writeln!(b, "        {}", tests.join(" && "));
                let _ = writeln!(b, "    }});");
            }
            Step::Order { boxes, key } => {
                helpers.insert(Helper::TieBreak);
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                match primary_order(key) {
                    Some(primary) => {
                        let _ = writeln!(
                            b,
                            "    sort(&mut v{i}, |a, b| {primary}.then_with(|| tie_break(a, b)));"
                        );
                    }
                    None => {
                        let _ = writeln!(b, "    sort(&mut v{i}, tie_break);");
                    }
                }
            }
            Step::Limit { boxes, n } => {
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                let _ = writeln!(b, "    v{i}.truncate({n});");
            }
            Step::LimitParam { boxes } => {
                let _ = writeln!(b, "    let mut v{i} = v{boxes};");
                let _ = writeln!(b, "    if params.max_results > 0 {{");
                let _ = writeln!(b, "        v{i}.truncate(params.max_results as usize);");
                let _ = writeln!(b, "    }}");
            }
            Step::Emit { boxes } => {
                helpers.insert(Helper::Emit);
                let _ = writeln!(b, "    for d in &v{boxes} {{");
                let _ = writeln!(b, "        emit(host, d)?;");
                let _ = writeln!(b, "    }}");
            }
        }
    }

    let mut src = String::new();
    let _ = writeln!(
        src,
        "// Generated by syrup-runtime {} (codegen {CODEGEN_VERSION}).",
        env!("CARGO_PKG_VERSION")
    );
    let _ = writeln!(src, "// intent: {intent}");
    let _ = writeln!(src, "// plan: {plan_hash}");
    for line in plan.describe().lines() {
        let _ = writeln!(src, "//   {line}");
    }
    src.push_str("\n#![allow(dead_code, unused_variables)]\n#![deny(unsafe_op_in_unsafe_fn)]\n\n");
    let _ = writeln!(src, "pub mod abi {{\n{PRELUDE}}}\n\nuse abi::*;\n");
    let _ = writeln!(src, "const PLAN_HASH: &[u8] = b\"{plan_hash}\\0\";");
    src.push_str(ENTRY);
    let _ = write!(
        src,
        "\nfn run(host: &SyrupHost, input: &SyrupImageView, params: &SyrupParams) -> Result<(), i32> {{\n{body}    Ok(())\n}}\n"
    );
    for helper in helpers {
        src.push_str(helper.source());
    }
    src.push_str(&predicates);
    src
}

const FORBIDDEN: &[&str] = &[
    "std::fs",
    "std::net",
    "std::process",
    "std::env",
    "std::os",
    "std::thread",
    "std::io",
    "core::arch",
    "asm!",
    "include!",
    "include_str!",
    "include_bytes!",
    "env!",
    "extern crate",
    "#[link",
    "extern \"C\" {",
    "extern \"system\"",
    "static mut",
    "export_name",
    "link_section",
    "#[no_mangle]",
];

/// Generated code may only compute over what the host hands it.
pub fn check_policy(source: &str) -> Result<()> {
    let policy = |reason: String| SyrupError::new(Stage::Generate, ErrorKind::Policy, reason);
    if let Some(token) = FORBIDDEN.iter().find(|t| source.contains(*t)) {
        return Err(policy(format!("generated source uses `{token}`")));
    }
    let exports: Vec<&str> = source
        .match_indices("#[unsafe(no_mangle)]")
        .filter_map(|(at, _)| {
            let rest = &source[at..];
            let name = &rest[rest.find("fn ")? + 3..];
            name.split('(').next()
        })
        .collect();
    if exports != EXPORTS {
        return Err(policy(format!(
            "generated source exports {exports:?}, expected {EXPORTS:?}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::parse;

    fn source(name: &str) -> String {
        let intent = parse(name).unwrap();
        let plan = Plan::build(&intent).unwrap();
        generate(&plan, &intent, &plan.hash())
    }

    #[test]
    fn generated_sources_pass_the_policy() {
        for name in [
            "find_red_bars_in_bottom_third",
            "find_face",
            "find_2_largest_faces_in_top_half_larger_than_2pct",
            "find_faces_in_region_left_to_right",
            "measure_fill_of_red_bars_in_bottom_third",
            "measure_sharpness_of_words_in_region",
            "measure_sharpness",
            "track_qr_codes",
        ] {
            check_policy(&source(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        }
    }

    #[test]
    fn only_used_templates_are_emitted() {
        let plain = source("find_face");
        assert!(!plain.contains("fn select_fixed") && !plain.contains("fn select_caller"));
        let region = source("find_faces_in_top_half");
        assert!(
            region.contains("select_fixed(&v0, (0, 1), (0, 1), (1, 1), (1, 2))"),
            "{region}"
        );
    }

    #[test]
    fn the_policy_rejects_escapes() {
        let ok = source("find_face");
        for extra in [
            "\nfn x() { std::fs::remove_file(\"a\"); }",
            "\nunsafe extern \"C\" { fn system(); }",
            "\n#[unsafe(no_mangle)]\npub extern \"C\" fn other() {}",
        ] {
            assert_eq!(
                check_policy(&format!("{ok}{extra}")).unwrap_err().kind,
                ErrorKind::Policy,
                "{extra}"
            );
        }
    }
}
