//! The inner loops, at the width of the CPU's vectors.
//!
//! Template matching spends its time summing a window of pixels, their
//! squares and their products with the template, over and over. These
//! are integer multiply-adds, and a CPU does 16 or 32 of them in one
//! instruction — when the code is compiled for it. A build that must run
//! on any x86-64 is compiled for the first one (SSE2), so the wider
//! instruction sets are chosen here at run time, once, from what the CPU
//! reports: AVX2 where it is there, SSE2 otherwise, and plain loops on
//! other architectures (where the compiler vectorises them for NEON by
//! itself).
//!
//! Every path computes exactly the same integers, so a search gives the
//! same answer whichever runs; the tests hold them to that.
//! `SYRUP_SIMD=scalar` (or `sse2`) in the environment forces a narrower
//! path, for checking and for comparing.

use std::sync::OnceLock;

/// Which instruction set the kernels run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// 256-bit vectors: 16 pixels per step.
    Avx2,
    /// 128-bit vectors, the x86-64 baseline: 16 pixels per step, in halves.
    Sse2,
    /// Plain loops, as the compiler vectorises them for the target.
    Scalar,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Avx2 => "avx2",
            Backend::Sse2 => "sse2",
            Backend::Scalar => "scalar",
        }
    }
}

/// The backend in use: the widest the CPU has, unless `SYRUP_SIMD` asks
/// for a narrower one. Decided once.
pub fn backend() -> Backend {
    static CHOSEN: OnceLock<Backend> = OnceLock::new();
    *CHOSEN.get_or_init(|| {
        let asked = std::env::var("SYRUP_SIMD")
            .map(|v| v.trim().to_ascii_lowercase())
            .unwrap_or_default();
        let widest = widest();
        match asked.as_str() {
            "scalar" => Backend::Scalar,
            "sse2" if widest != Backend::Scalar => Backend::Sse2,
            _ => widest,
        }
    })
}

#[cfg(target_arch = "x86_64")]
fn widest() -> Backend {
    if is_x86_feature_detected!("avx2") {
        Backend::Avx2
    } else {
        Backend::Sse2
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn widest() -> Backend {
    Backend::Scalar
}

/// Dot-product instructions beyond AVX2, which some CPUs have: four byte
/// products summed into each 32-bit lane in one instruction (VNNI), on
/// 512-bit vectors (AVX-512) or 256-bit ones (AVX-VNNI). The exhaustive
/// correlation runs on them when they are there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vnni {
    /// AVX-512 VNNI: 16 positions a vector.
    Zmm,
    /// AVX-VNNI: 8 positions a vector.
    Ymm,
}

impl Vnni {
    pub fn name(self) -> &'static str {
        match self {
            Vnni::Zmm => "avx512-vnni",
            Vnni::Ymm => "avx-vnni",
        }
    }
}

/// The dot-product instructions in use, if any: the widest the CPU has,
/// unless `SYRUP_SIMD` asks for a narrower backend (`avx2`, `sse2`,
/// `scalar`) or `avx-vnni`. Decided once.
pub fn vnni() -> Option<Vnni> {
    static CHOSEN: OnceLock<Option<Vnni>> = OnceLock::new();
    *CHOSEN.get_or_init(|| {
        if backend() != Backend::Avx2 {
            return None;
        }
        let asked = std::env::var("SYRUP_SIMD")
            .map(|v| v.trim().to_ascii_lowercase())
            .unwrap_or_default();
        let (zmm, ymm) = vnni_support();
        match asked.as_str() {
            "avx2" => None,
            "avx-vnni" | "avxvnni" => ymm.then_some(Vnni::Ymm),
            _ if zmm => Some(Vnni::Zmm),
            _ => ymm.then_some(Vnni::Ymm),
        }
    })
}

/// Whether the CPU has AVX-512 VNNI (with the byte instructions) and
/// AVX-VNNI.
#[cfg(target_arch = "x86_64")]
fn vnni_support() -> (bool, bool) {
    (
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vnni"),
        is_x86_feature_detected!("avxvnni"),
    )
}

#[cfg(not(target_arch = "x86_64"))]
fn vnni_support() -> (bool, bool) {
    (false, false)
}

/// Whether the scores are computed on 512-bit vectors too: with AVX-512
/// in use for the dot products, and its double-word and 64-bit integer
/// conversions (DQ) and 256-bit forms (VL) there as well. Decided once.
fn zmm_scores() -> bool {
    static CHOSEN: OnceLock<bool> = OnceLock::new();
    *CHOSEN.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            vnni() == Some(Vnni::Zmm)
                && is_x86_feature_detected!("avx512dq")
                && is_x86_feature_detected!("avx512vl")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Largest window, in pixels, whose sums the vector paths accumulate in
/// 32-bit lanes without overflow: each lane gathers two products of at
/// most 255² per 16-pixel step, over `area / 16` steps.
const MAX_VECTOR_AREA: usize = 200_000;

/// Over the `w`×`h` window of a plane (`stride` bytes per row) whose top
/// left is `(x, y)`: the sum of its pixels, the sum of their squares, and
/// the sum of their products with `weights` (a `w`×`h` plane of its own).
/// The three sums a normalised correlation is made of.
///
/// # Panics
///
/// When the window or the weights reach past their slices.
pub fn window_stats(
    pixels: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    weights: &[u8],
    w: usize,
    h: usize,
) -> (u64, u64, u64) {
    debug_assert!(weights.len() >= w * h);
    debug_assert!(pixels.len() >= (y + h - 1) * stride + x + w);
    if w * h > MAX_VECTOR_AREA {
        return window_stats_scalar(pixels, stride, x, y, weights, w, h);
    }
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `backend()` reported AVX2 on this CPU.
        Backend::Avx2 => unsafe { x86::window_stats_avx2(pixels, stride, x, y, weights, w, h) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: SSE2 is part of x86-64.
        Backend::Sse2 => unsafe { x86::window_stats_sse2(pixels, stride, x, y, weights, w, h) },
        _ => window_stats_scalar(pixels, stride, x, y, weights, w, h),
    }
}

/// [`window_stats`] with the weights also given widened to 16 bits
/// (`wide`, the same `w`×`h` values): the vector path then loads them as
/// they are multiplied, where it would widen them for every window. For
/// the windows scored one at a time — a refinement's climb, a match's
/// sub-pixel centre.
#[allow(clippy::too_many_arguments)]
pub(crate) fn window_stats_wide(
    pixels: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    weights: &[u8],
    wide: &[i16],
    w: usize,
    h: usize,
) -> (u64, u64, u64) {
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `backend()` reported AVX2 on this CPU; the kernel checks
        // the sizes.
        Backend::Avx2 if w >= 32 && w * h <= MAX_VECTOR_AREA => unsafe {
            x86::window_stats_wide_avx2(pixels, stride, x, y, wide, w, h)
        },
        _ => window_stats(pixels, stride, x, y, weights, w, h),
    }
}

/// Pairs of neighbouring template pixels, row by row, as [`block_stats`]
/// takes them: `t[2k] | t[2k + 1] << 16`, `w.div_ceil(2)` to a row, the last
/// pixel of an odd row paired with zero.
pub(crate) fn weight_pairs(weights: &[u8], w: usize, h: usize) -> Vec<u32> {
    let half = w.div_ceil(2);
    let mut pairs = Vec::with_capacity(h * half);
    for line in weights[..w * h].chunks_exact(w.max(1)) {
        for k in 0..half {
            let a = u32::from(line[2 * k]);
            let b = line.get(2 * k + 1).map_or(0, |&b| u32::from(b));
            pairs.push(a | (b << 16));
        }
    }
    pairs
}

/// Largest template area whose sums of products fit in 32 bits.
const MAX_AREA_32: usize = (u32::MAX / (255 * 255)) as usize;

/// [`window_stats`] at every position of a block: for the `bw`×`bh`
/// positions from `(x0, y0)`, row by row, `out[j·bw + i]` is the window
/// whose top left is `(x0 + i, y0 + j)`. The same three sums, computed
/// together: on AVX2 the products of 8 positions a vector, each row of
/// pixels loaded once for every position row it reaches, and the pixel
/// sums slid along columns, where a window at a time sums every pixel
/// once per position. `pairs` is [`weight_pairs`] of `weights`.
///
/// # Panics
///
/// When a window or the weights reach past their slices.
#[allow(clippy::too_many_arguments)]
pub(crate) fn block_stats(
    pixels: &[u8],
    stride: usize,
    (x0, y0): (usize, usize),
    (bw, bh): (usize, usize),
    weights: &[u8],
    pairs: &[u32],
    (w, h): (usize, usize),
    out: &mut Vec<(u64, u64, u64)>,
) {
    out.clear();
    if bw == 0 || bh == 0 {
        return;
    }
    assert!(w > 0 && h > 0 && x0 + bw - 1 + w <= stride);
    assert!(pixels.len() >= (y0 + bh - 1 + h - 1) * stride + x0 + bw - 1 + w);
    assert!(weights.len() >= w * h && pairs.len() >= w.div_ceil(2) * h);
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `backend()` reported AVX2 on this CPU; the sizes are
        // checked above.
        Backend::Avx2 if w <= MAX_AREA_32 => unsafe {
            x86::block_stats_avx2(pixels, stride, (x0, y0), (bw, bh), pairs, (w, h), out)
        },
        _ => {
            for y in y0..y0 + bh {
                for x in x0..x0 + bw {
                    out.push(window_stats(pixels, stride, x, y, weights, w, h));
                }
            }
        }
    }
}

/// Over the `w`×`h` region of an RGBA image (`stride` bytes a row) whose
/// top left is `(x, y)`: the sums of the red, green and blue of its opaque
/// pixels (alpha at least 128), and how many there are. Exact integers, so
/// a mean taken from them is the same whatever order they were added in.
pub(crate) fn opaque_sums(
    raw: &[u8],
    stride: usize,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
) -> ([u64; 3], u64) {
    if w == 0 || h == 0 {
        return ([0; 3], 0);
    }
    assert!((x + w) * 4 <= stride && raw.len() >= (y + h - 1) * stride + (x + w) * 4);
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `backend()` reported AVX2 on this CPU; the region is
        // checked above.
        Backend::Avx2 => unsafe { x86::opaque_sums_avx2(raw, stride, (x, y), (w, h)) },
        _ => opaque_sums_scalar(raw, stride, (x, y), (w, h)),
    }
}

fn opaque_sums_scalar(
    raw: &[u8],
    stride: usize,
    (x, y): (usize, usize),
    (w, h): (usize, usize),
) -> ([u64; 3], u64) {
    let (mut sums, mut count) = ([0u64; 3], 0u64);
    for row in y..y + h {
        let start = row * stride + x * 4;
        for p in raw[start..start + w * 4].as_chunks::<4>().0 {
            if p[3] >= 128 {
                for (sum, &v) in sums.iter_mut().zip(&p[..3]) {
                    *sum += u64::from(v);
                }
                count += 1;
            }
        }
    }
    (sums, count)
}

fn window_stats_scalar(
    pixels: &[u8],
    stride: usize,
    x: usize,
    y: usize,
    weights: &[u8],
    w: usize,
    h: usize,
) -> (u64, u64, u64) {
    let (mut sum, mut squares, mut products) = (0u64, 0u64, 0u64);
    for v in 0..h {
        let row = &pixels[(y + v) * stride + x..][..w];
        let line = &weights[v * w..][..w];
        let (mut s, mut q, mut p) = (0u32, 0u32, 0u32);
        for (&pixel, &weight) in row.iter().zip(line) {
            let pixel = u32::from(pixel);
            s += pixel;
            q += pixel * pixel;
            p += pixel * u32::from(weight);
        }
        sum += u64::from(s);
        squares += u64::from(q);
        products += u64::from(p);
    }
    (sum, squares, products)
}

/// One row of a template correlated with one row of the plane at every
/// position: `totals[i] += Σ weights[u] · source[u + i]` for every `i`
/// with `u + i` within `source`. The long loop of scoring every position
/// at once: each weight is multiplied into a whole row of positions.
pub(crate) fn correlate_row(totals: &mut [u32], source: &[u8], weights: &[u8]) {
    debug_assert!(source.len() + 1 >= totals.len() + weights.len());
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `backend()` reported AVX2 on this CPU.
        Backend::Avx2 => unsafe { x86::correlate_row_avx2(totals, source, weights) },
        _ => correlate_row_scalar(totals, source, weights),
    }
}

#[inline(always)]
fn correlate_row_scalar(totals: &mut [u32], source: &[u8], weights: &[u8]) {
    let columns = totals.len();
    for (u, &weight) in weights.iter().enumerate() {
        if weight == 0 {
            continue;
        }
        let weight = u16::from(weight);
        for (total, &pixel) in totals.iter_mut().zip(&source[u..u + columns]) {
            *total += u32::from(weight * u16::from(pixel));
        }
    }
}

/// The products of a `w`×`h` template with the window of a plane
/// (`width`×`height`, row-major) at every position, row by row:
/// `out[y · columns + x] = Σ weights[v · w + u] · plane[(y + v) · width + x + u]`
/// over the `columns = width - w + 1` by `height - h + 1` positions. The
/// sums must fit in 32 bits (a template of at most `u32::MAX / 255²`
/// pixels).
pub(crate) fn correlate_everywhere(
    plane: &[u8],
    width: usize,
    height: usize,
    weights: &[u8],
    w: usize,
    h: usize,
) -> Vec<u32> {
    assert!(w > 0 && h > 0 && w <= width && h <= height);
    assert!(plane.len() >= width * height && weights.len() >= w * h);
    let (columns, rows) = (width - w + 1, height - h + 1);
    let mut out = vec![0u32; columns * rows];
    #[cfg(target_arch = "x86_64")]
    if let Some(vnni) = vnni() {
        // SAFETY: `vnni()` reported the instructions on this CPU; the sizes
        // are checked above.
        unsafe {
            x86::correlate_everywhere_vnni(vnni, &mut out, plane, width, height, weights, w, h)
        };
        return out;
    }
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `backend()` reported AVX2 on this CPU; the sizes are
        // checked above.
        Backend::Avx2 => unsafe {
            x86::correlate_everywhere_avx2(&mut out, plane, width, height, weights, w, h)
        },
        _ => correlate_everywhere_rows(&mut out, plane, width, weights, w, h),
    }
    out
}

/// The normalised correlation of a template with a window, from the
/// window's sum of products with the template, pixel sum and sum of
/// squares, and the template's `n` pixels, their sum and their spread
/// (`n Σt² - (Σt)²`).
///
/// Everything up to the final ratio is exact integer arithmetic: the
/// variance-like terms are differences of large, nearly equal numbers,
/// which floating point would round away on flat windows.
#[inline]
pub(crate) fn ncc(n: i64, t_sum: i64, t_spread: i64, products: u64, sum: u64, squares: u64) -> f32 {
    // Products reach 255² per pixel, so n * products stays within i64 for
    // any template up to ~11 million pixels, far beyond a screen.
    let (products, sum, squares) = (products as i64, sum as i64, squares as i64);
    let spread = n * squares - sum * sum;
    if 4 * spread <= n * n {
        // A flat window correlates with nothing.
        return 0.0;
    }
    let covariance = n * products - sum * t_sum;
    let denominator = (spread as f32 * t_spread as f32).sqrt();
    (covariance as f32 / denominator).clamp(-1.0, 1.0)
}

/// [`ncc`] at every position of a plane (`width`×`height`), given the
/// products of a `w`×`h` template at every position (`products`, as
/// [`correlate_everywhere`] gives them), into `out` in the same order.
/// The windows' sums come from column totals over the template's height,
/// slid down a row at a time, and each row's from a running total along
/// it. The template must have at most `u32::MAX / 255²` pixels, so every
/// window's sums fit 32 bits.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scores_everywhere(
    plane: &[u8],
    width: usize,
    height: usize,
    (w, h): (usize, usize),
    products: &[u32],
    (t_sum, t_spread): (i64, i64),
    out: &mut Vec<f32>,
) {
    let n = w * h;
    assert!(w > 0 && h > 0 && w <= width && h <= height && n <= MAX_AREA_32);
    assert!(plane.len() >= width * height);
    let (columns, rows) = (width - w + 1, height - h + 1);
    assert!(products.len() >= columns * rows);
    out.clear();
    out.reserve(columns * rows);
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx2 {
        // SAFETY: `backend()` reported AVX2 on this CPU; sizes checked.
        unsafe {
            x86::scores_everywhere_avx2(
                plane,
                width,
                height,
                (w, h),
                products,
                (t_sum, t_spread),
                out,
            )
        };
        return;
    }
    scores_everywhere_body(
        plane,
        width,
        (w, h),
        products,
        (t_sum, t_spread),
        out,
        |row, sums, squares, out| {
            out.extend(
                row.iter().zip(sums).zip(squares).map(|((&p, &s), &q)| {
                    ncc(n as i64, t_sum, t_spread, p.into(), s.into(), q.into())
                }),
            )
        },
        |values, totals| {
            let mut total = 0u32;
            for (t, &v) in totals[1..].iter_mut().zip(values) {
                total = total.wrapping_add(v);
                *t = total;
            }
        },
    );
}

/// The loop of [`scores_everywhere`], with its two steps given: scoring a
/// row of positions from their products and sums (`score_row`), and
/// running totals (`running`: `totals[i + 1] = totals[i] + values[i]`,
/// wrapping, `totals[0]` left at 0).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn scores_everywhere_body(
    plane: &[u8],
    width: usize,
    (w, h): (usize, usize),
    products: &[u32],
    _template: (i64, i64),
    out: &mut Vec<f32>,
    mut score_row: impl FnMut(&[u32], &[u32], &[u32], &mut Vec<f32>),
    mut running: impl FnMut(&[u32], &mut [u32]),
) {
    let height = plane.len() / width;
    let (columns, rows) = (width - w + 1, height - h + 1);
    let (mut sums, mut squares) = (vec![0u32; width], vec![0u32; width]);
    for line in plane[..width * h].chunks_exact(width) {
        for ((s, q), &p) in sums.iter_mut().zip(squares.iter_mut()).zip(line) {
            *s += u32::from(p);
            *q += u32::from(p) * u32::from(p);
        }
    }
    // Running totals of the column totals along the row (wrapping: a
    // window's total, the difference of two, fits 32 bits).
    let (mut run_sums, mut run_squares) = (vec![0u32; width + 1], vec![0u32; width + 1]);
    let (mut window_sums, mut window_squares) = (vec![0u32; columns], vec![0u32; columns]);
    for y in 0..rows {
        if y > 0 {
            let (leaving, entering) = (
                &plane[(y - 1) * width..y * width],
                &plane[(y - 1 + h) * width..(y + h) * width],
            );
            for (((s, q), &old), &new) in sums
                .iter_mut()
                .zip(squares.iter_mut())
                .zip(leaving)
                .zip(entering)
            {
                let (old, new) = (u32::from(old), u32::from(new));
                *s = *s + new - old;
                *q = *q + new * new - old * old;
            }
        }
        running(&sums, &mut run_sums);
        running(&squares, &mut run_squares);
        for ((ws, wq), ((s0, s1), (q0, q1))) in
            window_sums.iter_mut().zip(window_squares.iter_mut()).zip(
                run_sums
                    .iter()
                    .zip(&run_sums[w..])
                    .zip(run_squares.iter().zip(&run_squares[w..])),
            )
        {
            *ws = s1.wrapping_sub(*s0);
            *wq = q1.wrapping_sub(*q0);
        }
        score_row(
            &products[y * columns..(y + 1) * columns],
            &window_sums,
            &window_squares,
            out,
        );
    }
}

/// [`crate::color::is_color_pixel`] for each pixel of `pixels` (RGBA, 4
/// bytes a pixel), as bits: bit `i % 64` of `out[i / 64]` (the words
/// replace what `out` held). On AVX2, 8 pixels at a time with the very same
/// single-precision arithmetic, so the same answers.
pub(crate) fn color_mask(
    pixels: &[u8],
    hue_range: (f32, f32),
    min_saturation: f32,
    min_value: f32,
    out: &mut Vec<u64>,
) {
    let count = pixels.len() / 4;
    out.clear();
    out.resize(count.div_ceil(64), 0);
    let mut from = 0;
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx2 {
        // SAFETY: `backend()` reported AVX2 on this CPU.
        from = unsafe { x86::color_mask_avx2(pixels, hue_range, min_saturation, min_value, out) };
    }
    for (i, p) in pixels.as_chunks::<4>().0.iter().enumerate().skip(from) {
        if crate::color::is_color_pixel(&image::Rgba(*p), hue_range, min_saturation, min_value) {
            out[i / 64] |= 1 << (i % 64);
        }
    }
}

/// The positions of `values` that are at least `floor`, in order, into
/// `out` (replacing what it held): 8 compared at once on AVX2.
pub(crate) fn at_least(values: &[f32], floor: f32, out: &mut Vec<usize>) {
    out.clear();
    let mut from = 0;
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx2 {
        // SAFETY: `backend()` reported AVX2 on this CPU.
        from = unsafe { x86::at_least_avx2(values, floor, out) };
    }
    out.extend(
        values
            .iter()
            .enumerate()
            .skip(from)
            .filter(|&(_, &v)| v >= floor)
            .map(|(i, _)| i),
    );
}

/// Template pixels in fours, row by row, as the dot-product instructions
/// take them: `t[4k] | t[4k+1] << 8 | t[4k+2] << 16 | t[4k+3] << 24`,
/// `w.div_ceil(4)` to a row, zero past its end; and `128 Σ t`, wrapping.
fn weight_quads(weights: &[u8], w: usize, h: usize) -> (Vec<u32>, u32) {
    let quads = w.div_ceil(4);
    let mut out = Vec::with_capacity(quads * h);
    let mut sum = 0u32;
    for line in weights[..w * h].chunks_exact(w) {
        for k in 0..quads {
            let mut quad = 0u32;
            for b in 0..4 {
                let t = line.get(4 * k + b).map_or(0, |&t| u32::from(t));
                quad |= t << (8 * b);
                sum = sum.wrapping_add(t);
            }
            out.push(quad);
        }
    }
    (out, sum.wrapping_mul(128))
}

/// One row of a plane halved across, the first pass of the pyramid's
/// 1-3-3-1 halving: `out[i] = row[2i-1] + 3·row[2i] + 3·row[2i+1] + row[2i+2]`,
/// the row's first and last pixels standing in for those past its ends.
/// `out` is half the row, rounded down; each value is at most 8 · 255.
pub(crate) fn halve_across(row: &[u8], out: &mut [u16]) {
    assert_eq!(out.len(), row.len() / 2);
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `backend()` reported AVX2 on this CPU.
        Backend::Avx2 => unsafe { x86::halve_across_avx2(row, out) },
        _ => halve_across_scalar(row, out, 0, out.len()),
    }
}

/// [`halve_across`] for the outputs `from..to`.
#[inline(always)]
fn halve_across_scalar(row: &[u8], out: &mut [u16], from: usize, to: usize) {
    if from >= to {
        return;
    }
    let last = row.len() - 1;
    for (i, value) in out[..to].iter_mut().enumerate().skip(from) {
        let left = u16::from(row[(2 * i).saturating_sub(1)]);
        let right = u16::from(row[(2 * i + 2).min(last)]);
        *value = left + 3 * (u16::from(row[2 * i]) + u16::from(row[2 * i + 1])) + right;
    }
}

/// Four consecutive rows halved across, combined down — the second pass of
/// the halving — and appended to `out`:
/// `(a[i] + 3·b[i] + 3·c[i] + d[i] + 32) / 64`, rounded down. The total
/// stays within 16 bits (64 · 255 + 32).
pub(crate) fn combine_down(a: &[u16], b: &[u16], c: &[u16], d: &[u16], out: &mut Vec<u8>) {
    let n = a.len();
    assert!(b.len() == n && c.len() == n && d.len() == n);
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `backend()` reported AVX2 on this CPU; the lengths agree.
        Backend::Avx2 => unsafe { x86::combine_down_avx2(a, b, c, d, out) },
        _ => combine_down_scalar(a, b, c, d, out),
    }
}

#[inline(always)]
fn combine_down_scalar(a: &[u16], b: &[u16], c: &[u16], d: &[u16], out: &mut Vec<u8>) {
    out.extend(
        a.iter()
            .zip(b)
            .zip(c.iter().zip(d))
            .map(|((&a, &b), (&c, &d))| ((a + 3 * b + 3 * c + d + 32) >> 6) as u8),
    );
}

/// Row by row: each template row multiplied into a whole row of positions
/// (`correlate_row`).
fn correlate_everywhere_rows(
    out: &mut [u32],
    plane: &[u8],
    width: usize,
    weights: &[u8],
    w: usize,
    h: usize,
) {
    let columns = width - w + 1;
    for (y, accumulator) in out.chunks_exact_mut(columns).enumerate() {
        for v in 0..h {
            correlate_row(
                accumulator,
                &plane[(y + v) * width..(y + v + 1) * width],
                &weights[v * w..(v + 1) * w],
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// 32 pixels a step: the pixel sum through the byte-sum instruction
    /// (`sad` against zero), and each half widened to 16-bit lanes for
    /// the squares and the products, summed in pairs into 32-bit lanes
    /// (`madd`). A row that is not a whole number of steps ends with one
    /// more over its last 32 pixels, the ones already counted masked out;
    /// a row shorter than a step goes 16 at a time the same way, and one
    /// shorter than 16 pixel by pixel.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn window_stats_avx2(
        pixels: &[u8],
        stride: usize,
        x: usize,
        y: usize,
        weights: &[u8],
        w: usize,
        h: usize,
    ) -> (u64, u64, u64) {
        let zero = _mm256_setzero_si256();
        let mut sums = _mm256_setzero_si256();
        let mut squares = _mm256_setzero_si256();
        let mut products = _mm256_setzero_si256();
        let (mut s_tail, mut q_tail, mut p_tail) = (0u32, 0u32, 0u32);
        let steps = w / 32;
        let (tail, mask) = tail_mask_32(w);
        let (steps16, mask16) = (w % 32 / 16, tail_mask(w % 32).1);
        // Checked once here, so the rows below can be taken without a
        // check each.
        assert!(h > 0 && w > 0);
        assert!(pixels.len() >= (y + h - 1) * stride + x + w);
        assert!(weights.len() >= w * h);
        for v in 0..h {
            // SAFETY: within the slices, by the checks above.
            let row =
                unsafe { std::slice::from_raw_parts(pixels.as_ptr().add((y + v) * stride + x), w) };
            let line = unsafe { std::slice::from_raw_parts(weights.as_ptr().add(v * w), w) };
            let mut step = |p: __m256i, t: __m256i| {
                sums = _mm256_add_epi64(sums, _mm256_sad_epu8(p, zero));
                let (p_lo, p_hi) = (
                    _mm256_cvtepu8_epi16(_mm256_castsi256_si128(p)),
                    _mm256_cvtepu8_epi16(_mm256_extracti128_si256(p, 1)),
                );
                let (t_lo, t_hi) = (
                    _mm256_cvtepu8_epi16(_mm256_castsi256_si128(t)),
                    _mm256_cvtepu8_epi16(_mm256_extracti128_si256(t, 1)),
                );
                squares = _mm256_add_epi32(squares, _mm256_madd_epi16(p_lo, p_lo));
                squares = _mm256_add_epi32(squares, _mm256_madd_epi16(p_hi, p_hi));
                products = _mm256_add_epi32(products, _mm256_madd_epi16(p_lo, t_lo));
                products = _mm256_add_epi32(products, _mm256_madd_epi16(p_hi, t_hi));
            };
            for i in 0..steps {
                // SAFETY: both slices hold at least `steps * 32` bytes.
                let p = unsafe { _mm256_loadu_si256(row.as_ptr().add(i * 32).cast()) };
                let t = unsafe { _mm256_loadu_si256(line.as_ptr().add(i * 32).cast()) };
                step(p, t);
            }
            if tail == 0 {
                continue;
            }
            if w >= 32 {
                // SAFETY: the row holds `w >= 32` bytes.
                let p = unsafe { _mm256_loadu_si256(row.as_ptr().add(w - 32).cast()) };
                let t = unsafe { _mm256_loadu_si256(line.as_ptr().add(w - 32).cast()) };
                step(_mm256_and_si256(p, mask), _mm256_and_si256(t, mask));
            } else if w >= 16 {
                // Under a step: 16 at a time, in the low half of the
                // vectors (the high half zero).
                let load16 = |slice: &[u8], at: usize| {
                    // SAFETY: the slice holds at least `at + 16` bytes.
                    _mm256_castsi128_si256(unsafe {
                        _mm_loadu_si128(slice.as_ptr().add(at).cast())
                    })
                };
                for i in 0..steps16 {
                    step(load16(row, i * 16), load16(line, i * 16));
                }
                if !w.is_multiple_of(16) {
                    let mask = _mm256_castsi128_si256(mask16);
                    step(
                        _mm256_and_si256(load16(row, w - 16), mask),
                        _mm256_and_si256(load16(line, w - 16), mask),
                    );
                }
            } else {
                for (&pixel, &weight) in row.iter().zip(line) {
                    let pixel = u32::from(pixel);
                    s_tail += pixel;
                    q_tail += pixel * pixel;
                    p_tail += pixel * u32::from(weight);
                }
            }
        }
        let sum = sum_u64x4(sums) + u64::from(s_tail);
        let squares = sum_i32x8(squares) + u64::from(q_tail);
        let products = sum_i32x8(products) + u64::from(p_tail);
        (sum, squares, products)
    }

    /// `window_stats_avx2` for rows of at least 32 pixels, the weights
    /// already 16-bit: per 32 pixels two widening loads, the byte sum, and
    /// four multiply-adds, two of them straight from the weights. A row
    /// that is not a whole number of steps ends with one more over its
    /// last 32 pixels, the ones already counted masked out.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn window_stats_wide_avx2(
        pixels: &[u8],
        stride: usize,
        x: usize,
        y: usize,
        wide: &[i16],
        w: usize,
        h: usize,
    ) -> (u64, u64, u64) {
        assert!(w >= 32 && h > 0);
        assert!(pixels.len() >= (y + h - 1) * stride + x + w);
        assert!(wide.len() >= w * h);
        let zero = _mm256_setzero_si256();
        let mut sums = _mm256_setzero_si256();
        let mut squares = _mm256_setzero_si256();
        let mut products = _mm256_setzero_si256();
        let steps = w / 32;
        let (tail, mask) = tail_mask_32(w);
        let base = pixels.as_ptr();
        let weights = wide.as_ptr();
        let mut step = |p: __m256i, t: *const i16| {
            sums = _mm256_add_epi64(sums, _mm256_sad_epu8(p, zero));
            let lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(p));
            let hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(p));
            squares = _mm256_add_epi32(squares, _mm256_madd_epi16(lo, lo));
            squares = _mm256_add_epi32(squares, _mm256_madd_epi16(hi, hi));
            // SAFETY: `t` points at 32 weights within `wide` (below).
            let (t_lo, t_hi) = unsafe {
                (
                    _mm256_loadu_si256(t.cast()),
                    _mm256_loadu_si256(t.add(16).cast()),
                )
            };
            products = _mm256_add_epi32(products, _mm256_madd_epi16(lo, t_lo));
            products = _mm256_add_epi32(products, _mm256_madd_epi16(hi, t_hi));
        };
        for v in 0..h {
            // SAFETY: the window's row and the weights' row, by the checks
            // above: every load below ends within them.
            unsafe {
                let row = base.add((y + v) * stride + x);
                let line = weights.add(v * w);
                for i in 0..steps {
                    step(_mm256_loadu_si256(row.add(i * 32).cast()), line.add(i * 32));
                }
                if tail > 0 {
                    let p = _mm256_loadu_si256(row.add(w - 32).cast());
                    // Masked pixels are zero: they add nothing to any sum.
                    step(_mm256_and_si256(p, mask), line.add(w - 32));
                }
            }
        }
        (sum_u64x4(sums), sum_i32x8(squares), sum_i32x8(products))
    }

    /// For a row of `w` pixels: how many are left after the whole steps of
    /// 32, and a mask keeping just those, as the last `tail` bytes of 32.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn tail_mask_32(w: usize) -> (usize, __m256i) {
        let tail = w % 32;
        let mut bytes = [0u8; 32];
        for b in &mut bytes[32 - tail..] {
            *b = 0xFF;
        }
        // SAFETY: `bytes` is 32 bytes, and the load is unaligned.
        (tail, unsafe { _mm256_loadu_si256(bytes.as_ptr().cast()) })
    }

    /// For a row of `w` pixels: how many are left after the whole steps of
    /// 16, and a mask keeping just those, as the last `tail` bytes of 16.
    #[inline]
    #[target_feature(enable = "sse2")]
    fn tail_mask(w: usize) -> (usize, __m128i) {
        let tail = w % 16;
        let mut bytes = [0u8; 16];
        for b in &mut bytes[16 - tail..] {
            *b = 0xFF;
        }
        // SAFETY: `bytes` is 16 bytes, and the load is unaligned.
        (tail, unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) })
    }

    /// The same with 128-bit vectors: 16 pixels a step, widened in halves.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn window_stats_sse2(
        pixels: &[u8],
        stride: usize,
        x: usize,
        y: usize,
        weights: &[u8],
        w: usize,
        h: usize,
    ) -> (u64, u64, u64) {
        let zero = _mm_setzero_si128();
        let mut sums = _mm_setzero_si128();
        let mut squares = _mm_setzero_si128();
        let mut products = _mm_setzero_si128();
        let (mut s_tail, mut q_tail, mut p_tail) = (0u32, 0u32, 0u32);
        let steps = w / 16;
        let (tail, mask) = tail_mask(w);
        for v in 0..h {
            let row = &pixels[(y + v) * stride + x..][..w];
            let line = &weights[v * w..][..w];
            let mut step = |p: __m128i, t: __m128i| {
                let (p_lo, p_hi) = (_mm_unpacklo_epi8(p, zero), _mm_unpackhi_epi8(p, zero));
                let (t_lo, t_hi) = (_mm_unpacklo_epi8(t, zero), _mm_unpackhi_epi8(t, zero));
                sums = _mm_add_epi64(sums, _mm_sad_epu8(p, zero));
                squares = _mm_add_epi32(squares, _mm_madd_epi16(p_lo, p_lo));
                squares = _mm_add_epi32(squares, _mm_madd_epi16(p_hi, p_hi));
                products = _mm_add_epi32(products, _mm_madd_epi16(p_lo, t_lo));
                products = _mm_add_epi32(products, _mm_madd_epi16(p_hi, t_hi));
            };
            for i in 0..steps {
                // SAFETY: both slices hold at least `steps * 16` bytes.
                let p = unsafe { _mm_loadu_si128(row.as_ptr().add(i * 16).cast()) };
                let t = unsafe { _mm_loadu_si128(line.as_ptr().add(i * 16).cast()) };
                step(p, t);
            }
            if tail > 0 && w >= 16 {
                // SAFETY: the row holds `w >= 16` bytes.
                let p = unsafe { _mm_loadu_si128(row.as_ptr().add(w - 16).cast()) };
                let t = unsafe { _mm_loadu_si128(line.as_ptr().add(w - 16).cast()) };
                step(_mm_and_si128(p, mask), _mm_and_si128(t, mask));
            } else {
                for (&pixel, &weight) in row[steps * 16..].iter().zip(&line[steps * 16..]) {
                    let pixel = u32::from(pixel);
                    s_tail += pixel;
                    q_tail += pixel * pixel;
                    p_tail += pixel * u32::from(weight);
                }
            }
        }
        let sum = sum_u64x2(sums) + u64::from(s_tail);
        let squares = sum_i32x4(squares) + u64::from(q_tail);
        let products = sum_i32x4(products) + u64::from(p_tail);
        (sum, squares, products)
    }

    /// Every position's products, 8 positions a vector: each pixel is
    /// stored beside its right-hand neighbour as the two 16-bit halves of a
    /// 32-bit lane, so that one multiply-add of 16-bit pairs (`madd`) with
    /// two neighbouring template weights adds two taps of the correlation
    /// for 8 positions at once — a load, a multiply-add and an add per 16
    /// products, the accumulators staying in registers over the whole
    /// template. The paired copy of the plane is made once per search.
    /// Integer sums, so the order they are added in changes nothing; the
    /// 32-bit lanes wrap exactly as the `u32` totals do.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn correlate_everywhere_avx2(
        out: &mut [u32],
        plane: &[u8],
        width: usize,
        height: usize,
        weights: &[u8],
        w: usize,
        h: usize,
    ) {
        let (columns, rows) = (width - w + 1, height - h + 1);
        assert!(out.len() >= columns * rows);
        assert!(plane.len() >= width * height && weights.len() >= w * h);
        // The pairs go in a buffer kept between calls (every one written
        // before it is read); the last pixel of a row pairs with zero.
        PAIRS_SCRATCH.with_borrow_mut(|pairs| {
            if pairs.len() < width * height {
                pairs.resize(width * height, 0);
            }
            for (paired, row) in pairs.chunks_exact_mut(width).zip(plane.chunks_exact(width)) {
                pair_up(row, 0, paired);
            }
            // SAFETY: AVX2, and `pairs` holds the plane's pairs.
            unsafe {
                correlate_pairs_avx2(
                    out,
                    plane,
                    &pairs[..width * height],
                    width,
                    height,
                    weights,
                    w,
                    h,
                )
            }
        });
    }

    thread_local! {
        static PAIRS_SCRATCH: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// `correlate_everywhere_avx2` from the plane's pixel pairs.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    unsafe fn correlate_pairs_avx2(
        out: &mut [u32],
        plane: &[u8],
        pairs: &[u32],
        width: usize,
        height: usize,
        weights: &[u8],
        w: usize,
        h: usize,
    ) {
        let (columns, rows) = (width - w + 1, height - h + 1);
        assert!(pairs.len() >= width * height);
        // Neighbouring weights in pairs, row by row; an odd row's last
        // weight is paired with zero.
        let half = w.div_ceil(2);
        let mut weight_pairs = Vec::with_capacity(h * half);
        for line in weights[..w * h].chunks_exact(w) {
            for k in 0..half {
                let a = u32::from(line[2 * k]);
                let b = line.get(2 * k + 1).map_or(0, |&b| u32::from(b));
                weight_pairs.push(_mm256_set1_epi32((a | (b << 16)) as i32));
            }
        }
        // The furthest a vector load reaches: position `x + 7` of a block
        // and pair `2 (half - 1)`, i.e. `columns - 1 + 2 half - 2 <= width - 1`.
        debug_assert!(columns + 2 * half - 2 <= width);
        let base = pairs.as_ptr();
        for (y, out_row) in out[..columns * rows].chunks_exact_mut(columns).enumerate() {
            let mut x = 0;
            while x + 32 <= columns {
                let mut acc = [_mm256_setzero_si256(); 4];
                for v in 0..h {
                    // SAFETY: rows `y + v < height`, and every load below
                    // ends within the row (the bound above).
                    let row = unsafe { base.add((y + v) * width + x) };
                    for (k, &pair) in weight_pairs[v * half..(v + 1) * half].iter().enumerate() {
                        let at = unsafe { row.add(2 * k) };
                        for (lane, sum) in acc.iter_mut().enumerate() {
                            let pixels = unsafe { _mm256_loadu_si256(at.add(8 * lane).cast()) };
                            *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(pixels, pair));
                        }
                    }
                }
                for (lane, sum) in acc.iter().enumerate() {
                    // SAFETY: `x + 32 <= columns`, the row's length.
                    unsafe {
                        _mm256_storeu_si256(out_row.as_mut_ptr().add(x + 8 * lane).cast(), *sum)
                    };
                }
                x += 32;
            }
            while x + 8 <= columns {
                let mut sum = _mm256_setzero_si256();
                for v in 0..h {
                    // SAFETY: as above.
                    let row = unsafe { base.add((y + v) * width + x) };
                    for (k, &pair) in weight_pairs[v * half..(v + 1) * half].iter().enumerate() {
                        let pixels = unsafe { _mm256_loadu_si256(row.add(2 * k).cast()) };
                        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(pixels, pair));
                    }
                }
                // SAFETY: `x + 8 <= columns`.
                unsafe { _mm256_storeu_si256(out_row.as_mut_ptr().add(x).cast(), sum) };
                x += 8;
            }
            // The last few positions one at a time.
            for (x, total) in out_row.iter_mut().enumerate().skip(x) {
                let mut sum = 0u32;
                for v in 0..h {
                    let row = &plane[(y + v) * width + x..(y + v) * width + x + w];
                    let line = &weights[v * w..(v + 1) * w];
                    for (&p, &t) in row.iter().zip(line) {
                        sum = sum.wrapping_add(u32::from(p) * u32::from(t));
                    }
                }
                *total = sum;
            }
        }
    }

    /// Buffers `block_stats_avx2` keeps between calls, one set a thread:
    /// a block's pixel pairs are tens of kilobytes, and allocating and
    /// clearing them for every refinement cost more than filling them.
    #[derive(Default)]
    struct BlockScratch {
        pairs: Vec<u32>,
        sums: Vec<u32>,
        squares: Vec<u32>,
    }

    thread_local! {
        static BLOCK_SCRATCH: std::cell::RefCell<BlockScratch> = Default::default();
    }

    /// One pass of `products_rows` for `ROWS` position rows, as a pointer.
    type ProductsRows =
        unsafe fn(&[u32], usize, usize, &[u32], usize, std::ops::Range<usize>, &mut [[u32; 8]; 8]);

    /// The block's sums of pixels and of their squares from column totals
    /// over the template's height, slid down a row at a time, each
    /// window's slid along its row; the products 8 positions a vector, as
    /// in `correlate_everywhere_avx2`, from a copy of the block's pixels in
    /// neighbouring pairs, with up to 8 position rows at once so that each
    /// load of a pixel pair serves every position row it reaches.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn block_stats_avx2(
        pixels: &[u8],
        stride: usize,
        (x0, y0): (usize, usize),
        (bw, bh): (usize, usize),
        pairs: &[u32],
        (w, h): (usize, usize),
        out: &mut Vec<(u64, u64, u64)>,
    ) {
        let (span, rows) = (bw - 1 + w, bh - 1 + h);
        assert!(x0 + span <= stride && pixels.len() >= (y0 + rows - 1) * stride + x0 + span);
        let half = w.div_ceil(2);
        assert!(pairs.len() >= half * h && w <= super::MAX_AREA_32);
        out.clear();
        out.resize(bw * bh, (0, 0, 0));
        let row_of = |r: usize| &pixels[(y0 + r) * stride + x0..][..span];
        BLOCK_SCRATCH.with_borrow_mut(|scratch| {
            // Sums and squares: each block row's at every position along it
            // (the first window's 32 pixels a step, then slid a pixel at a
            // time), and those summed down each column of positions over
            // the template's height, then slid down a row at a time.
            // (Every value is written before it is read: the buffers are
            // only grown, never cleared.)
            for buffer in [&mut scratch.sums, &mut scratch.squares] {
                if buffer.len() < rows * bw {
                    buffer.resize(rows * bw, 0);
                }
            }
            let (row_sums, row_squares) = (&mut scratch.sums, &mut scratch.squares);
            for r in 0..rows {
                let line = row_of(r);
                let (mut sum, mut square) = line_sums(&line[..w]);
                row_sums[r * bw] = sum;
                row_squares[r * bw] = square;
                for i in 1..bw {
                    let (old, new) = (u32::from(line[i - 1]), u32::from(line[i - 1 + w]));
                    sum = sum + new - old;
                    square = square + new * new - old * old;
                    row_sums[r * bw + i] = sum;
                    row_squares[r * bw + i] = square;
                }
            }
            for i in 0..bw {
                let column = |values: &[u32], from: usize| -> u64 {
                    (from..from + h)
                        .map(|v| u64::from(values[v * bw + i]))
                        .sum()
                };
                let (mut sum, mut square) = (column(row_sums, 0), column(row_squares, 0));
                out[i].0 = sum;
                out[i].1 = square;
                for j in 1..bh {
                    let (leaving, entering) = ((j - 1) * bw + i, (j - 1 + h) * bw + i);
                    sum = sum + u64::from(row_sums[entering]) - u64::from(row_sums[leaving]);
                    square =
                        square + u64::from(row_squares[entering]) - u64::from(row_squares[leaving]);
                    out[j * bw + i].0 = sum;
                    out[j * bw + i].1 = square;
                }
            }

            // Products, a strip of 8 position columns at a time. Every pair
            // of the strip's rows is written before it is read, so the
            // buffer is only grown, never cleared.
            let cols = 8 + 2 * half;
            let chunk = (super::MAX_AREA_32 / w).max(1);
            if scratch.pairs.len() < rows * cols {
                scratch.pairs.resize(rows * cols, 0);
            }
            let block = &mut scratch.pairs[..rows * cols];
            let passes: [ProductsRows; 8] = [
                products_rows::<1>,
                products_rows::<2>,
                products_rows::<3>,
                products_rows::<4>,
                products_rows::<5>,
                products_rows::<6>,
                products_rows::<7>,
                products_rows::<8>,
            ];
            for strip in (0..bw).step_by(8) {
                let lanes = (bw - strip).min(8);
                for r in 0..rows {
                    // The whole plane row (less of the last if the slice
                    // ends early): pairs reaching past it read zeros, which
                    // meet only zero weights or positions outside the block.
                    let start = (y0 + r) * stride;
                    let line = &pixels[start..(start + stride).min(pixels.len())];
                    pair_up(line, x0 + strip, &mut block[r * cols..(r + 1) * cols]);
                }
                for v0 in (0..h).step_by(chunk) {
                    let v1 = (v0 + chunk).min(h);
                    let mut j0 = 0;
                    while j0 < bh {
                        let n = (bh - j0).min(8);
                        let mut lanes_of = [[0u32; 8]; 8];
                        // SAFETY: AVX2; rows `j0 + j + v < rows`, and every
                        // load ends within a row of `block`
                        // (`2 (half - 1) + 8 <= cols`).
                        unsafe {
                            passes[n - 1](block, cols, j0, pairs, half, v0..v1, &mut lanes_of)
                        };
                        for (j, lane) in lanes_of.iter().enumerate().take(n) {
                            let at = (j0 + j) * bw + strip;
                            for (o, &p) in out[at..at + lanes].iter_mut().zip(lane) {
                                o.2 += u64::from(p);
                            }
                        }
                        j0 += n;
                    }
                }
            }
        });
    }

    /// The sum of `line`'s pixels and of their squares: 32 a step, a line
    /// that is not a whole number of steps ending with one more over its
    /// last 32 pixels, the ones already counted masked out.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn line_sums(line: &[u8]) -> (u32, u32) {
        let n = line.len();
        if n < 32 {
            return line.iter().fold((0, 0), |(s, q), &p| {
                (s + u32::from(p), q + u32::from(p) * u32::from(p))
            });
        }
        let zero = _mm256_setzero_si256();
        let (mut sums, mut squares) = (zero, zero);
        let mut add = |p: __m256i| {
            sums = _mm256_add_epi64(sums, _mm256_sad_epu8(p, zero));
            let lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(p));
            let hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(p));
            squares = _mm256_add_epi32(squares, _mm256_madd_epi16(lo, lo));
            squares = _mm256_add_epi32(squares, _mm256_madd_epi16(hi, hi));
        };
        for i in 0..n / 32 {
            // SAFETY: `i * 32 + 32 <= n`.
            add(unsafe { _mm256_loadu_si256(line.as_ptr().add(i * 32).cast()) });
        }
        let (tail, mask) = tail_mask_32(n);
        if tail > 0 {
            // SAFETY: `n >= 32`.
            let p = unsafe { _mm256_loadu_si256(line.as_ptr().add(n - 32).cast()) };
            add(_mm256_and_si256(p, mask));
        }
        (sum_u64x4(sums) as u32, sum_i32x8(squares) as u32)
    }

    /// `out[c] = line[x + c] | line[x + c + 1] << 16`, zero past the line:
    /// each pixel beside its right-hand neighbour, as 16-bit halves.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn pair_up(line: &[u8], x: usize, out: &mut [u32]) {
        let mut c = 0;
        // 16 pairs a step, from two loads of 16 pixels a pixel apart.
        while c + 16 <= out.len() && x + c + 17 <= line.len() {
            // SAFETY: `x + c + 17 <= line.len()` and `c + 16 <= out.len()`.
            unsafe {
                let at = line.as_ptr().add(x + c);
                let here = _mm256_cvtepu8_epi16(_mm_loadu_si128(at.cast()));
                let next = _mm256_cvtepu8_epi16(_mm_loadu_si128(at.add(1).cast()));
                let low = _mm256_unpacklo_epi16(here, next);
                let high = _mm256_unpackhi_epi16(here, next);
                let target = out.as_mut_ptr().add(c);
                _mm256_storeu_si256(target.cast(), _mm256_permute2x128_si256::<0x20>(low, high));
                _mm256_storeu_si256(
                    target.add(8).cast(),
                    _mm256_permute2x128_si256::<0x31>(low, high),
                );
            }
            c += 16;
        }
        for (k, pair) in out.iter_mut().enumerate().skip(c) {
            let a = line.get(x + k).map_or(0, |&p| u32::from(p));
            let b = line.get(x + k + 1).map_or(0, |&p| u32::from(p));
            *pair = a | (b << 16);
        }
    }

    /// The products of `ROWS` position rows from `j0` with template rows
    /// `rows`, into the first `ROWS` of `out`: for each pair of template
    /// pixels, one broadcast, and for each position row a load and a
    /// multiply-add, the accumulators staying in registers throughout.
    ///
    /// # Safety
    ///
    /// AVX2, and `block` holds rows `j0 + ROWS - 1 + rows.end - 1` of
    /// `cols >= 2 half + 6` pairs.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn products_rows<const ROWS: usize>(
        block: &[u32],
        cols: usize,
        j0: usize,
        pairs: &[u32],
        half: usize,
        rows: std::ops::Range<usize>,
        out: &mut [[u32; 8]; 8],
    ) {
        debug_assert!(block.len() >= (j0 + ROWS - 1 + rows.end) * cols);
        let mut acc = [_mm256_setzero_si256(); ROWS];
        let base = block.as_ptr();
        for v in rows {
            for (k, &pair) in pairs[v * half..(v + 1) * half].iter().enumerate() {
                let weight = _mm256_set1_epi32(pair as i32);
                for (j, sum) in acc.iter_mut().enumerate() {
                    // SAFETY: by the caller's guarantee.
                    let p =
                        unsafe { _mm256_loadu_si256(base.add((j0 + j + v) * cols + 2 * k).cast()) };
                    *sum = _mm256_add_epi32(*sum, _mm256_madd_epi16(p, weight));
                }
            }
        }
        for (lane, sum) in out.iter_mut().zip(acc) {
            // SAFETY: a lane array is 32 bytes, and the store unaligned.
            unsafe { _mm256_storeu_si256(lane.as_mut_ptr().cast(), sum) };
        }
    }

    /// 8 pixels a step: the alpha byte's top bit, spread over its lane by
    /// an arithmetic shift, masks the transparent pixels out, and each
    /// channel is added in 32-bit lanes (moved to 64 bits every row).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn opaque_sums_avx2(
        raw: &[u8],
        stride: usize,
        (x, y): (usize, usize),
        (w, h): (usize, usize),
    ) -> ([u64; 3], u64) {
        assert!(raw.len() >= (y + h - 1) * stride + (x + w) * 4);
        let byte = _mm256_set1_epi32(0xff);
        let (mut sums, mut count) = ([0u64; 3], 0u64);
        let steps = w / 8;
        for row in y..y + h {
            let start = row * stride + x * 4;
            let line = &raw[start..start + w * 4];
            let (mut r, mut g, mut b, mut n) = (
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
                _mm256_setzero_si256(),
            );
            for i in 0..steps {
                // SAFETY: `i * 32 + 32 <= w * 4`, the line's length.
                let px = unsafe { _mm256_loadu_si256(line.as_ptr().add(i * 32).cast()) };
                let opaque = _mm256_srai_epi32::<31>(px);
                let kept = _mm256_and_si256(px, opaque);
                r = _mm256_add_epi32(r, _mm256_and_si256(kept, byte));
                g = _mm256_add_epi32(g, _mm256_and_si256(_mm256_srli_epi32::<8>(kept), byte));
                b = _mm256_add_epi32(b, _mm256_and_si256(_mm256_srli_epi32::<16>(kept), byte));
                n = _mm256_sub_epi32(n, opaque);
            }
            sums[0] += sum_i32x8(r);
            sums[1] += sum_i32x8(g);
            sums[2] += sum_i32x8(b);
            count += sum_i32x8(n);
            for p in line[steps * 32..].as_chunks::<4>().0 {
                if p[3] >= 128 {
                    for (sum, &v) in sums.iter_mut().zip(&p[..3]) {
                        *sum += u64::from(v);
                    }
                    count += 1;
                }
            }
        }
        (sums, count)
    }

    /// 16 outputs a step: the row loaded three times, a pixel pair apart,
    /// and each load split into its even and odd pixels as 16-bit lanes
    /// (a mask and a shift), so the four taps of 16 outputs are four whole
    /// vectors. The first output, and the last few, which reach past the
    /// row's ends, go one at a time.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn halve_across_avx2(row: &[u8], out: &mut [u16]) {
        let width = out.len();
        assert_eq!(width, row.len() / 2);
        if width == 0 {
            return;
        }
        let low = _mm256_set1_epi16(0x00ff);
        let mut i = 1;
        // A step at `i` reads `row[2i - 2 .. 2i + 34]` and writes
        // `out[i .. i + 16]`.
        while i + 16 <= width && 2 * i + 34 <= row.len() {
            // SAFETY: within both slices, by the loop's bounds.
            unsafe {
                let at = row.as_ptr().add(2 * i - 2);
                let before = _mm256_loadu_si256(at.cast());
                let here = _mm256_loadu_si256(at.add(2).cast());
                let after = _mm256_loadu_si256(at.add(4).cast());
                // Little-endian lanes: the low byte is the even pixel.
                let left = _mm256_srli_epi16::<8>(before);
                let middle =
                    _mm256_add_epi16(_mm256_and_si256(here, low), _mm256_srli_epi16::<8>(here));
                let right = _mm256_and_si256(after, low);
                let sum = _mm256_add_epi16(
                    _mm256_add_epi16(left, right),
                    _mm256_add_epi16(middle, _mm256_add_epi16(middle, middle)),
                );
                _mm256_storeu_si256(out.as_mut_ptr().add(i).cast(), sum);
            }
            i += 16;
        }
        super::halve_across_scalar(row, out, 0, 1);
        super::halve_across_scalar(row, out, i.max(1), width);
    }

    /// 32 outputs a step: two vectors of 16-bit totals, shifted down and
    /// packed to bytes (the pack interleaves the halves of the two vectors;
    /// a permute puts them back in order).
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn combine_down_avx2(
        a: &[u16],
        b: &[u16],
        c: &[u16],
        d: &[u16],
        out: &mut Vec<u8>,
    ) {
        let n = a.len();
        assert!(b.len() == n && c.len() == n && d.len() == n);
        out.reserve(n);
        let start = out.len();
        let round = _mm256_set1_epi16(32);
        // SAFETY: each slice holds `k + 16` values when called below.
        let down = |k: usize| unsafe {
            let load = |s: &[u16]| _mm256_loadu_si256(s.as_ptr().add(k).cast());
            let middle = _mm256_add_epi16(load(b), load(c));
            let total = _mm256_add_epi16(
                _mm256_add_epi16(load(a), load(d)),
                _mm256_add_epi16(
                    _mm256_add_epi16(middle, middle),
                    _mm256_add_epi16(middle, round),
                ),
            );
            _mm256_srli_epi16::<6>(total)
        };
        // SAFETY: `reserve` made room for `n` more bytes past `start`.
        let target = unsafe { out.as_mut_ptr().add(start) };
        let mut i = 0;
        while i + 32 <= n {
            let packed = _mm256_packus_epi16(down(i), down(i + 16));
            let ordered = _mm256_permute4x64_epi64::<0b11_01_10_00>(packed);
            // SAFETY: `i + 32 <= n`, within the room reserved.
            unsafe { _mm256_storeu_si256(target.add(i).cast(), ordered) };
            i += 32;
        }
        for k in i..n {
            let value = ((a[k] + 3 * b[k] + 3 * c[k] + d[k] + 32) >> 6) as u8;
            // SAFETY: `k < n`, within the room reserved.
            unsafe { target.add(k).write(value) };
        }
        // SAFETY: all `n` bytes past `start` were written above.
        unsafe { out.set_len(start + n) };
    }

    /// `scores_everywhere` compiled for AVX2: the column totals 8 at a time,
    /// the running totals by a scan within vectors, and each row scored by
    /// `scores_row_avx2`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn scores_everywhere_avx2(
        plane: &[u8],
        width: usize,
        height: usize,
        (w, h): (usize, usize),
        products: &[u32],
        (t_sum, t_spread): (i64, i64),
        out: &mut Vec<f32>,
    ) {
        let n = (w * h) as i64;
        let wide = super::zmm_scores();
        super::scores_everywhere_body(
            &plane[..width * height],
            width,
            (w, h),
            products,
            (t_sum, t_spread),
            out,
            // SAFETY: AVX2, here, and AVX-512 when `zmm_scores` says so;
            // the rows are as long as each other.
            |row, sums, squares, out| unsafe {
                if wide {
                    scores_row_zmm(row, sums, squares, (n, t_sum, t_spread), out)
                } else {
                    scores_row_avx2(row, sums, squares, (n, t_sum, t_spread), out)
                }
            },
            |values, totals| running_totals(values, totals),
        );
    }

    /// `totals[i + 1] = totals[i] + values[i]` (wrapping) from
    /// `totals[0] = 0`: 8 at a time, each vector's prefix taken within it in
    /// three shifted additions (the halves joined by a fourth) and the total
    /// so far added to all of it.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) fn running_totals(values: &[u32], totals: &mut [u32]) {
        assert!(totals.len() > values.len());
        totals[0] = 0;
        let mut carry = _mm256_setzero_si256();
        let last = _mm256_set1_epi32(7);
        let mut i = 0;
        while i + 8 <= values.len() {
            // SAFETY: 8 values from `i`; 8 totals from `i + 1`.
            unsafe {
                let mut x = _mm256_loadu_si256(values.as_ptr().add(i).cast());
                x = _mm256_add_epi32(x, _mm256_slli_si256::<4>(x));
                x = _mm256_add_epi32(x, _mm256_slli_si256::<8>(x));
                // The low half's total into every lane of the high half.
                let low_total = _mm256_shuffle_epi32::<0xff>(x);
                x = _mm256_add_epi32(x, _mm256_permute2x128_si256::<0x08>(low_total, low_total));
                x = _mm256_add_epi32(x, carry);
                _mm256_storeu_si256(totals.as_mut_ptr().add(i + 1).cast(), x);
                carry = _mm256_permutevar8x32_epi32(x, last);
            }
            i += 8;
        }
        let mut total = totals[i];
        for (t, &v) in totals[i + 1..].iter_mut().zip(&values[i..]) {
            total = total.wrapping_add(v);
            *t = total;
        }
    }

    /// `scores_row_avx2` on 512-bit vectors, 8 positions at a time: the
    /// integer terms in 64-bit lanes, converted straight to single
    /// precision (rounded once, as `as f32` rounds), then the same
    /// single-precision product, square root, division and clamp.
    #[target_feature(enable = "avx512f,avx512dq,avx512vl")]
    pub(super) unsafe fn scores_row_zmm(
        products: &[u32],
        sums: &[u32],
        squares: &[u32],
        (n, t_sum, t_spread): (i64, i64, i64),
        out: &mut Vec<f32>,
    ) {
        let len = products.len();
        assert!(sums.len() >= len && squares.len() >= len);
        assert!(n > 0 && n <= super::MAX_AREA_32 as i64 && (0..1 << 32).contains(&t_sum));
        out.reserve(len);
        let start = out.len();
        let n_v = _mm512_set1_epi64(n);
        let nn = _mm512_set1_epi64(n * n);
        let t_sum_v = _mm512_set1_epi64(t_sum);
        let t_spread_v = _mm256_set1_ps(t_spread as f32);
        let (low, high) = (_mm256_set1_ps(-1.0), _mm256_set1_ps(1.0));
        // SAFETY: `reserve` made room for `len` more scores past `start`.
        let target = unsafe { out.as_mut_ptr().add(start) };
        let mut i = 0;
        while i + 8 <= len {
            // SAFETY: eight values from `i` in each slice, eight into
            // `target`.
            unsafe {
                let load =
                    |v: &[u32]| _mm512_cvtepu32_epi64(_mm256_loadu_si256(v.as_ptr().add(i).cast()));
                let (p, s, q) = (load(products), load(sums), load(squares));
                let spread = _mm512_sub_epi64(_mm512_mul_epu32(n_v, q), _mm512_mul_epu32(s, s));
                let covariance =
                    _mm512_sub_epi64(_mm512_mul_epu32(n_v, p), _mm512_mul_epu32(s, t_sum_v));
                // `4 spread > n²`: not flat.
                let varied = _mm512_cmpgt_epi64_mask(_mm512_slli_epi64::<2>(spread), nn);
                let denominator =
                    _mm256_sqrt_ps(_mm256_mul_ps(_mm512_cvtepi64_ps(spread), t_spread_v));
                let score = _mm256_min_ps(
                    _mm256_max_ps(
                        _mm256_div_ps(_mm512_cvtepi64_ps(covariance), denominator),
                        low,
                    ),
                    high,
                );
                // A flat window scores 0 (its quotient, whatever it was, dropped).
                _mm256_storeu_ps(target.add(i), _mm256_maskz_mov_ps(varied, score));
            }
            i += 8;
        }
        for k in i..len {
            let score = super::ncc(
                n,
                t_sum,
                t_spread,
                products[k].into(),
                sums[k].into(),
                squares[k].into(),
            );
            // SAFETY: `k < len`, within the room reserved.
            unsafe { target.add(k).write(score) };
        }
        // SAFETY: all `len` scores past `start` were written above.
        unsafe { out.set_len(start + len) };
    }

    /// `ncc` for a row of positions, 4 at a time: the integer terms in
    /// 64-bit lanes (32-bit operands multiplied whole), converted to float
    /// exactly through doubles (an integer below 2^51 added into the
    /// mantissa of 2^52 + 2^51 and that taken off again) and rounded once to
    /// single precision, as `as f32` rounds; then the same single-precision
    /// product, square root, division and clamp, which are correctly
    /// rounded in every lane as they are one at a time.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn scores_row_avx2(
        products: &[u32],
        sums: &[u32],
        squares: &[u32],
        (n, t_sum, t_spread): (i64, i64, i64),
        out: &mut Vec<f32>,
    ) {
        let len = products.len();
        assert!(sums.len() >= len && squares.len() >= len);
        // Within the exact range of the conversion: every term is below
        // 2^51 for templates of at most 2^32 / 255² pixels.
        assert!(n > 0 && n <= super::MAX_AREA_32 as i64 && (0..1 << 32).contains(&t_sum));
        out.reserve(len);
        let start = out.len();
        let n_v = _mm256_set1_epi64x(n);
        let nn = _mm256_set1_epi64x(n * n);
        let t_sum_v = _mm256_set1_epi64x(t_sum);
        let t_spread_v = _mm_set1_ps(t_spread as f32);
        let magic_i = _mm256_set1_epi64x(0x4338_0000_0000_0000);
        let magic_f = _mm256_set1_pd(6_755_399_441_055_744.0);
        let (low, high) = (_mm_set1_ps(-1.0), _mm_set1_ps(1.0));
        // The low 32 bits of each 64-bit lane, as four 32-bit lanes.
        let evens = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
        let to_f32 = |x: __m256i| {
            let exact = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(x, magic_i)), magic_f);
            _mm256_cvtpd_ps(exact)
        };
        let four = |p: __m128i, s: __m128i, q: __m128i| -> __m128 {
            let (p, s, q) = (
                _mm256_cvtepu32_epi64(p),
                _mm256_cvtepu32_epi64(s),
                _mm256_cvtepu32_epi64(q),
            );
            let spread = _mm256_sub_epi64(_mm256_mul_epu32(n_v, q), _mm256_mul_epu32(s, s));
            let covariance =
                _mm256_sub_epi64(_mm256_mul_epu32(n_v, p), _mm256_mul_epu32(s, t_sum_v));
            // `4 spread > n²`: not flat.
            let varied = _mm256_cmpgt_epi64(_mm256_slli_epi64::<2>(spread), nn);
            let varied = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(varied, evens));
            let denominator = _mm_sqrt_ps(_mm_mul_ps(to_f32(spread), t_spread_v));
            let score = _mm_min_ps(
                _mm_max_ps(_mm_div_ps(to_f32(covariance), denominator), low),
                high,
            );
            // A flat window scores 0 (and its quotient, whatever it was, is
            // dropped).
            _mm_and_ps(score, _mm_castsi128_ps(varied))
        };
        // SAFETY: `reserve` made room for `len` more scores past `start`.
        let target = unsafe { out.as_mut_ptr().add(start) };
        let mut i = 0;
        while i + 4 <= len {
            // SAFETY: four values from `i` in each slice, four into `target`.
            unsafe {
                let load = |v: &[u32]| _mm_loadu_si128(v.as_ptr().add(i).cast());
                _mm_storeu_ps(
                    target.add(i),
                    four(load(products), load(sums), load(squares)),
                );
            }
            i += 4;
        }
        for k in i..len {
            let score = super::ncc(
                n,
                t_sum,
                t_spread,
                products[k].into(),
                sums[k].into(),
                squares[k].into(),
            );
            // SAFETY: `k < len`, within the room reserved.
            unsafe { target.add(k).write(score) };
        }
        // SAFETY: all `len` scores past `start` were written above.
        unsafe { out.set_len(start + len) };
    }

    /// 8 pixels a step: the channels out of each 32-bit pixel by shifts and
    /// masks, then `HsvParts` in vectors — each quotient, maximum and
    /// difference the one the scalar code computes — and the hue's three
    /// cases all computed and the right one kept. (The scalar code's
    /// `% 6.0` leaves its operand, which lies in [-1, 1], unchanged.) Sets
    /// the bits of the whole steps in `out` (zeroed, a word per 64 pixels)
    /// and returns how many pixels it did.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn color_mask_avx2(
        pixels: &[u8],
        (hue_min, hue_max): (f32, f32),
        min_saturation: f32,
        min_value: f32,
        out: &mut [u64],
    ) -> usize {
        let count = pixels.len() / 4;
        assert!(out.len() >= count.div_ceil(64));
        let byte = _mm256_set1_epi32(0xff);
        let scale = _mm256_set1_ps(255.0);
        let zero = _mm256_setzero_ps();
        let (sixty, two, four, full) = (
            _mm256_set1_ps(60.0),
            _mm256_set1_ps(2.0),
            _mm256_set1_ps(4.0),
            _mm256_set1_ps(360.0),
        );
        let (value_v, saturation_v) = (_mm256_set1_ps(min_value), _mm256_set1_ps(min_saturation));
        let (lo, hi) = (_mm256_set1_ps(hue_min), _mm256_set1_ps(hue_max));
        let half_alpha = _mm256_set1_epi32(127);
        let mut i = 0;
        while i + 8 <= count {
            // SAFETY: 8 pixels from `i` within `pixels`.
            let px = unsafe { _mm256_loadu_si256(pixels.as_ptr().add(4 * i).cast()) };
            let channel = |shift: i32| -> __m256 {
                let v = match shift {
                    0 => px,
                    8 => _mm256_srli_epi32::<8>(px),
                    _ => _mm256_srli_epi32::<16>(px),
                };
                _mm256_div_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, byte)), scale)
            };
            let (rf, gf, bf) = (channel(0), channel(8), channel(16));
            // Alpha at least 128 is `alpha / 255 >= 0.5`.
            let opaque =
                _mm256_castsi256_ps(_mm256_cmpgt_epi32(_mm256_srli_epi32::<24>(px), half_alpha));
            let max = _mm256_max_ps(_mm256_max_ps(rf, gf), bf);
            let min = _mm256_min_ps(_mm256_min_ps(rf, gf), bf);
            let delta = _mm256_sub_ps(max, min);
            let bright = _mm256_cmp_ps::<_CMP_GE_OQ>(max, value_v);
            let black = _mm256_cmp_ps::<_CMP_EQ_OQ>(max, zero);
            let saturation = _mm256_blendv_ps(_mm256_div_ps(delta, max), zero, black);
            let saturated = _mm256_cmp_ps::<_CMP_GE_OQ>(saturation, saturation_v);
            let red = _mm256_mul_ps(sixty, _mm256_div_ps(_mm256_sub_ps(gf, bf), delta));
            let green = _mm256_mul_ps(
                sixty,
                _mm256_add_ps(_mm256_div_ps(_mm256_sub_ps(bf, rf), delta), two),
            );
            let blue = _mm256_mul_ps(
                sixty,
                _mm256_add_ps(_mm256_div_ps(_mm256_sub_ps(rf, gf), delta), four),
            );
            let mut hue = _mm256_blendv_ps(blue, green, _mm256_cmp_ps::<_CMP_EQ_OQ>(max, gf));
            hue = _mm256_blendv_ps(hue, red, _mm256_cmp_ps::<_CMP_EQ_OQ>(max, rf));
            hue = _mm256_blendv_ps(hue, zero, _mm256_cmp_ps::<_CMP_EQ_OQ>(delta, zero));
            hue = _mm256_blendv_ps(
                hue,
                _mm256_add_ps(hue, full),
                _mm256_cmp_ps::<_CMP_LT_OQ>(hue, zero),
            );
            let above = _mm256_cmp_ps::<_CMP_GE_OQ>(hue, lo);
            let below = _mm256_cmp_ps::<_CMP_LE_OQ>(hue, hi);
            let in_range = if hue_min <= hue_max {
                _mm256_and_ps(above, below)
            } else {
                _mm256_or_ps(above, below)
            };
            let matched = _mm256_and_ps(
                _mm256_and_ps(opaque, bright),
                _mm256_and_ps(saturated, in_range),
            );
            let bits = _mm256_movemask_ps(matched) as u64;
            out[i / 64] |= bits << (i % 64);
            i += 8;
        }
        i
    }

    /// 8 values a step: a comparison, its signs as bits, and the set bits'
    /// positions. Returns how many values it did.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn at_least_avx2(values: &[f32], floor: f32, out: &mut Vec<usize>) -> usize {
        let floor = _mm256_set1_ps(floor);
        let mut i = 0;
        while i + 8 <= values.len() {
            // SAFETY: 8 values from `i`.
            let v = unsafe { _mm256_loadu_ps(values.as_ptr().add(i)) };
            let mut bits = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_GE_OQ>(v, floor)) as u32;
            while bits != 0 {
                out.push(i + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
            i += 8;
        }
        i
    }

    thread_local! {
        static QUADS_SCRATCH: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Positions a block of the VNNI correlation covers: 4 vectors.
    const VNNI_BLOCK: usize = 64;

    /// The exhaustive correlation on dot-product instructions. Each pixel,
    /// less 128 so that it fits a signed byte, is stored with the next
    /// three as one 32-bit lane; one instruction then multiplies four
    /// template pixels (unsigned bytes) into it and adds the four products
    /// to the lane: `Σ t (p - 128)`, to which `128 Σ t` is added back. The
    /// lanes wrap, and the true sums fit 32 bits, so the totals are exact.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn correlate_everywhere_vnni(
        vnni: super::Vnni,
        out: &mut [u32],
        plane: &[u8],
        width: usize,
        height: usize,
        weights: &[u8],
        w: usize,
        h: usize,
    ) {
        let (columns, rows) = (width - w + 1, height - h + 1);
        assert!(out.len() >= columns * rows);
        assert!(plane.len() >= width * height && weights.len() >= w * h);
        let (tq, bias) = super::weight_quads(weights, w, h);
        let per_row = w.div_ceil(4);
        // Every load of a block stays within its row of quads: the last
        // block starts before `columns` and reads `VNNI_BLOCK` lanes from
        // each of the template's quads.
        let stride = columns.next_multiple_of(VNNI_BLOCK) + 4 * per_row;
        QUADS_SCRATCH.with_borrow_mut(|quads| {
            if quads.len() < height * stride {
                quads.resize(height * stride, 0);
            }
            for (r, line) in plane[..width * height].chunks_exact(width).enumerate() {
                quad_up(line, &mut quads[r * stride..(r + 1) * stride]);
            }
            let quads = &quads[..height * stride];
            // SAFETY: as checked above; `vnni` was detected on this CPU.
            unsafe {
                match vnni {
                    super::Vnni::Zmm => correlate_quads_zmm(
                        out, quads, stride, columns, rows, &tq, per_row, h, bias,
                    ),
                    super::Vnni::Ymm => correlate_quads_ymm(
                        out, quads, stride, columns, rows, &tq, per_row, h, bias,
                    ),
                }
            }
        });
    }

    /// `out[c]`: the bytes `line[c .. c + 4]`, each less 128 (flipping its
    /// top bit makes it the signed byte `p - 128`), zero past the line.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn quad_up(line: &[u8], out: &mut [u32]) {
        // 8 quads a step from one 16-byte load in both halves of a vector:
        // the low half takes bytes 0..7, the high half bytes 4..11.
        let order = _mm256_setr_epi8(
            0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, //
            4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10,
        );
        let flip = _mm256_set1_epi8(-128);
        let mut c = 0;
        while c + 16 <= line.len() && c + 8 <= out.len() {
            // SAFETY: 16 bytes from `c` within the line; 8 quads within `out`.
            unsafe {
                let bytes =
                    _mm256_broadcastsi128_si256(_mm_loadu_si128(line.as_ptr().add(c).cast()));
                let quads = _mm256_xor_si256(_mm256_shuffle_epi8(bytes, order), flip);
                _mm256_storeu_si256(out.as_mut_ptr().add(c).cast(), quads);
            }
            c += 8;
        }
        for (k, quad) in out.iter_mut().enumerate().skip(c) {
            let mut value = 0u32;
            for b in 0..4 {
                let p = line.get(k + b).map_or(0, |&p| u32::from(p ^ 0x80));
                value |= p << (8 * b);
            }
            *quad = value;
        }
    }

    /// Where each of a template's quads is read from, relative to a
    /// block's first position: `v · stride + 4 k` for quad `k` of row `v`,
    /// in the order the quads are listed.
    fn quad_offsets(stride: usize, per_row: usize, h: usize) -> Vec<usize> {
        (0..h * per_row)
            .map(|j| (j / per_row) * stride + 4 * (j % per_row))
            .collect()
    }

    /// 64 positions a block, 16 a vector, the template's quads broadcast
    /// once for the four vectors. A dot-product instruction adds into its
    /// accumulator, so each accumulator is a chain of them, one after the
    /// other: the quads go round four sets of accumulators in turn, four
    /// chains a vector instead of one, and the sets are added at the end.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn correlate_quads_zmm(
        out: &mut [u32],
        quads: &[u32],
        stride: usize,
        columns: usize,
        rows: usize,
        tq: &[u32],
        per_row: usize,
        h: usize,
        bias: u32,
    ) {
        assert!(quads.len() >= (rows - 1 + h) * stride && tq.len() >= per_row * h);
        assert!(stride >= columns.next_multiple_of(VNNI_BLOCK) + 4 * per_row);
        let offsets = quad_offsets(stride, per_row, h);
        let n = offsets.len();
        let bias = _mm512_set1_epi32(bias as i32);
        let base = quads.as_ptr();
        for (y, out_row) in out[..columns * rows].chunks_exact_mut(columns).enumerate() {
            let mut x = 0;
            while x < columns {
                // SAFETY: every load below is at `y · stride + x` plus an
                // offset into row `y + v` of the quads, and ends by
                // `x + 64 + 4 per_row <= stride` within it.
                let start = unsafe { base.add(y * stride + x) };
                let mut acc = [[_mm512_setzero_si512(); 4]; 4];
                let step = |acc: &mut [__m512i; 4], j: usize| {
                    let t = _mm512_set1_epi32(tq[j] as i32);
                    // SAFETY: as above.
                    let at = unsafe { start.add(offsets[j]) };
                    for (lane, sum) in acc.iter_mut().enumerate() {
                        // SAFETY: as above.
                        let p = unsafe { _mm512_loadu_si512(at.add(16 * lane).cast()) };
                        *sum = _mm512_dpbusd_epi32(*sum, t, p);
                    }
                };
                let mut j = 0;
                while j + 4 <= n {
                    for (part, set) in acc.iter_mut().enumerate() {
                        step(set, j + part);
                    }
                    j += 4;
                }
                for (part, set) in acc.iter_mut().enumerate().take(n - j) {
                    step(set, j + part);
                }
                #[allow(clippy::needless_range_loop)] // `lane` picks one of each set
                for lane in 0..4 {
                    let at = x + 16 * lane;
                    if at >= columns {
                        break;
                    }
                    let total = _mm512_add_epi32(
                        _mm512_add_epi32(acc[0][lane], acc[1][lane]),
                        _mm512_add_epi32(_mm512_add_epi32(acc[2][lane], acc[3][lane]), bias),
                    );
                    let n = (columns - at).min(16);
                    let mask: __mmask16 = if n == 16 { u16::MAX } else { (1 << n) - 1 };
                    // SAFETY: the mask keeps the store within the row.
                    unsafe {
                        _mm512_mask_storeu_epi32(out_row.as_mut_ptr().add(at).cast(), mask, total)
                    };
                }
                x += VNNI_BLOCK;
            }
        }
    }

    /// The same on 256-bit vectors: 8 positions a vector, 4 vectors a
    /// block of 32, two sets of accumulators taking turns (16 registers
    /// hold no more).
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn correlate_quads_ymm(
        out: &mut [u32],
        quads: &[u32],
        stride: usize,
        columns: usize,
        rows: usize,
        tq: &[u32],
        per_row: usize,
        h: usize,
        bias: u32,
    ) {
        assert!(quads.len() >= (rows - 1 + h) * stride && tq.len() >= per_row * h);
        assert!(stride >= columns.next_multiple_of(VNNI_BLOCK) + 4 * per_row);
        let offsets = quad_offsets(stride, per_row, h);
        let n = offsets.len();
        let bias = _mm256_set1_epi32(bias as i32);
        let base = quads.as_ptr();
        for (y, out_row) in out[..columns * rows].chunks_exact_mut(columns).enumerate() {
            let mut x = 0;
            while x < columns {
                // SAFETY: as in `correlate_quads_zmm`, with blocks of 32.
                let start = unsafe { base.add(y * stride + x) };
                let mut acc = [[_mm256_setzero_si256(); 4]; 2];
                let step = |acc: &mut [__m256i; 4], j: usize| {
                    let t = _mm256_set1_epi32(tq[j] as i32);
                    // SAFETY: as above.
                    let at = unsafe { start.add(offsets[j]) };
                    for (lane, sum) in acc.iter_mut().enumerate() {
                        // SAFETY: as above.
                        let p = unsafe { _mm256_loadu_si256(at.add(8 * lane).cast()) };
                        *sum = _mm256_dpbusd_avx_epi32(*sum, t, p);
                    }
                };
                let mut j = 0;
                while j + 2 <= n {
                    let [first, second] = &mut acc;
                    step(first, j);
                    step(second, j + 1);
                    j += 2;
                }
                if j < n {
                    step(&mut acc[0], j);
                }
                let mut lanes = [0u32; 32];
                #[allow(clippy::needless_range_loop)] // `lane` picks one of each set
                for lane in 0..4 {
                    let total =
                        _mm256_add_epi32(_mm256_add_epi32(acc[0][lane], acc[1][lane]), bias);
                    // SAFETY: 8 lanes into an array of 32.
                    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().add(8 * lane).cast(), total) };
                }
                let n = (columns - x).min(32);
                out_row[x..x + n].copy_from_slice(&lanes[..n]);
                x += 32;
            }
        }
    }

    /// The row correlation compiled for AVX2: the compiler vectorises the
    /// inner loop itself, 32 pixels a step.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn correlate_row_avx2(totals: &mut [u32], source: &[u8], weights: &[u8]) {
        super::correlate_row_scalar(totals, source, weights);
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    fn sum_u64x2(v: __m128i) -> u64 {
        let mut out = [0u64; 2];
        // SAFETY: `out` is 16 bytes, and the store is unaligned.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), v) };
        out[0] + out[1]
    }

    #[inline]
    #[target_feature(enable = "sse2")]
    fn sum_i32x4(v: __m128i) -> u64 {
        let mut out = [0u32; 4];
        // SAFETY: `out` is 16 bytes, and the store is unaligned.
        unsafe { _mm_storeu_si128(out.as_mut_ptr().cast(), v) };
        out.iter().map(|&x| u64::from(x)).sum()
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn sum_u64x4(v: __m256i) -> u64 {
        let mut out = [0u64; 4];
        // SAFETY: `out` is 32 bytes, and the store is unaligned.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), v) };
        out.iter().sum()
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn sum_i32x8(v: __m256i) -> u64 {
        let mut out = [0u32; 8];
        // SAFETY: `out` is 32 bytes, and the store is unaligned.
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), v) };
        out.iter().map(|&x| u64::from(x)).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic plane of "random" bytes.
    fn noise(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (state >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn every_path_sums_the_same_window() {
        let (stride, rows) = (200usize, 90usize);
        let plane = noise(stride * rows, 7);
        for &(w, h) in &[
            (1usize, 1usize),
            (15, 3),
            (16, 16),
            (56, 56),
            (57, 41),
            (100, 70),
        ] {
            let weights = noise(w * h, w as u32 * 31 + h as u32);
            let (x, y) = (stride - w - 3, rows - h - 5);
            let expected = window_stats_scalar(&plane, stride, x, y, &weights, w, h);
            let got = window_stats(&plane, stride, x, y, &weights, w, h);
            assert_eq!(got, expected, "{w}x{h} on {:?}", backend());
            #[cfg(target_arch = "x86_64")]
            {
                // SAFETY: SSE2 is part of x86-64.
                let sse2 = unsafe { x86::window_stats_sse2(&plane, stride, x, y, &weights, w, h) };
                assert_eq!(sse2, expected, "{w}x{h} on sse2");
                if is_x86_feature_detected!("avx2") {
                    // SAFETY: the CPU reported AVX2.
                    let avx2 =
                        unsafe { x86::window_stats_avx2(&plane, stride, x, y, &weights, w, h) };
                    assert_eq!(avx2, expected, "{w}x{h} on avx2");
                }
            }
        }
    }

    #[test]
    fn correlated_rows_add_up_the_same() {
        let source = noise(1000, 3);
        let weights = noise(56, 9);
        for columns in [1usize, 7, 42, 331, 945] {
            let mut expected = vec![7u32; columns];
            let mut got = expected.clone();
            correlate_row_scalar(&mut expected, &source[..columns + 55], &weights);
            correlate_row(&mut got, &source[..columns + 55], &weights);
            assert_eq!(got, expected, "{columns} columns");
        }
    }

    #[test]
    fn correlations_everywhere_add_up_the_same() {
        let (width, height) = (173usize, 61usize);
        let plane = noise(width * height, 11);
        for &(w, h) in &[
            (1usize, 1usize),
            (2, 3),
            (7, 5),
            (16, 16),
            (17, 9),
            (32, 32),
            (48, 33),
            (64, 60),
            (173, 1),
            (140, 61),
        ] {
            let weights = noise(w * h, (w * 131 + h) as u32);
            let mut expected = vec![0u32; (width - w + 1) * (height - h + 1)];
            correlate_everywhere_rows(&mut expected, &plane, width, &weights, w, h);
            let got = correlate_everywhere(&plane, width, height, &weights, w, h);
            assert_eq!(got, expected, "{w}x{h} on {:?}", backend());
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") {
                let mut avx2 = vec![0u32; expected.len()];
                // SAFETY: the CPU reported AVX2.
                unsafe {
                    x86::correlate_everywhere_avx2(&mut avx2, &plane, width, height, &weights, w, h)
                };
                assert_eq!(avx2, expected, "{w}x{h} on avx2");
            }
            #[cfg(target_arch = "x86_64")]
            for vnni in available_vnni() {
                let mut got = vec![0u32; expected.len()];
                // SAFETY: the CPU reported the instructions.
                unsafe {
                    x86::correlate_everywhere_vnni(
                        vnni, &mut got, &plane, width, height, &weights, w, h,
                    )
                };
                assert_eq!(got, expected, "{w}x{h} on {vnni:?}");
            }
        }
        // Planes narrower than a block, and every width around one.
        for width in [13usize, 64, 65, 77, 129] {
            let plane = noise(width * 23, width as u32);
            let weights = noise(13 * 20, 5);
            let mut expected = vec![0u32; (width - 12) * 4];
            correlate_everywhere_rows(&mut expected, &plane, width, &weights, 13, 20);
            assert_eq!(
                correlate_everywhere(&plane, width, 23, &weights, 13, 20),
                expected,
                "{width} wide"
            );
            #[cfg(target_arch = "x86_64")]
            for vnni in available_vnni() {
                let mut got = vec![0u32; expected.len()];
                // SAFETY: the CPU reported the instructions.
                unsafe {
                    x86::correlate_everywhere_vnni(
                        vnni, &mut got, &plane, width, 23, &weights, 13, 20,
                    )
                };
                assert_eq!(got, expected, "{width} wide on {vnni:?}");
            }
        }
        // All 255s: the largest sums, and lanes past 2^31.
        let (width, height) = (300usize, 260usize);
        let white = vec![255u8; width * height];
        let weights = vec![255u8; 256 * 256];
        let got = correlate_everywhere(&white, width, height, &weights, 256, 256);
        assert!(got.iter().all(|&t| t == 256 * 256 * 255 * 255));
        #[cfg(target_arch = "x86_64")]
        for vnni in available_vnni() {
            let mut got = vec![0u32; 45 * 5];
            // SAFETY: the CPU reported the instructions.
            unsafe {
                x86::correlate_everywhere_vnni(
                    vnni, &mut got, &white, width, height, &weights, 256, 256,
                )
            };
            assert!(got.iter().all(|&t| t == 256 * 256 * 255 * 255), "{vnni:?}");
            // And all zeros: every pixel at the far end of the signed range.
            let black = vec![0u8; width * height];
            unsafe {
                x86::correlate_everywhere_vnni(
                    vnni, &mut got, &black, width, height, &weights, 256, 256,
                )
            };
            assert!(got.iter().all(|&t| t == 0), "{vnni:?} on black");
        }
    }

    /// The dot-product instructions this CPU has, each to test.
    #[cfg(target_arch = "x86_64")]
    fn available_vnni() -> Vec<Vnni> {
        let (zmm, ymm) = vnni_support();
        let mut out = Vec::new();
        if zmm {
            out.push(Vnni::Zmm);
        }
        if ymm {
            out.push(Vnni::Ymm);
        }
        out
    }

    #[test]
    fn widened_weights_sum_the_same_window() {
        let (stride, rows) = (400usize, 300usize);
        let plane = noise(stride * rows, 17);
        let white = vec![255u8; stride * rows];
        for &(w, h) in &[
            (1usize, 1usize),
            (15, 3),
            (31, 2),
            (32, 1),
            (33, 5),
            (56, 56),
            (63, 7),
            (64, 64),
            (77, 77),
            (158, 158),
            (390, 290),
        ] {
            for (name, pixels) in [("noise", &plane), ("white", &white)] {
                let weights = if name == "white" {
                    vec![255u8; w * h]
                } else {
                    noise(w * h, (w * 3 + h) as u32)
                };
                let wide: Vec<i16> = weights.iter().map(|&v| i16::from(v)).collect();
                for (x, y) in [
                    (0, 0),
                    (stride - w, rows - h),
                    ((stride - w) / 3, (rows - h) / 2),
                ] {
                    let expected = window_stats_scalar(pixels, stride, x, y, &weights, w, h);
                    let got = window_stats_wide(pixels, stride, x, y, &weights, &wide, w, h);
                    assert_eq!(
                        got,
                        expected,
                        "{name} {w}x{h} at {x},{y} on {:?}",
                        backend()
                    );
                    #[cfg(target_arch = "x86_64")]
                    if is_x86_feature_detected!("avx2") && w >= 32 {
                        // SAFETY: the CPU reported AVX2; the window fits.
                        let avx2 = unsafe {
                            x86::window_stats_wide_avx2(pixels, stride, x, y, &wide, w, h)
                        };
                        assert_eq!(avx2, expected, "{name} {w}x{h} at {x},{y} on avx2");
                    }
                }
            }
        }
    }

    #[test]
    fn a_block_sums_what_its_windows_sum() {
        let (stride, rows) = (331usize, 300usize);
        let plane = noise(stride * rows, 21);
        let white = vec![255u8; stride * rows];
        for &(w, h, bw, bh) in &[
            (1usize, 1usize, 1usize, 1usize),
            (2, 3, 5, 5),
            (7, 5, 7, 7),
            (16, 16, 8, 8),
            (17, 9, 9, 11),
            (37, 37, 5, 5),
            (56, 56, 7, 7),
            (76, 75, 17, 3),
            (157, 157, 7, 7),
            // Wider than a 32-bit sum of products allows in one pass: the
            // template's rows go in chunks.
            (300, 240, 3, 2),
        ] {
            let weights = noise(w * h, (w * 7 + h) as u32);
            let pairs = weight_pairs(&weights, w, h);
            let (x0, y0) = (stride - w - bw + 1, rows - h - bh + 1);
            for (name, pixels) in [("noise", &plane), ("white", &white)] {
                let weights = if name == "white" {
                    vec![255u8; w * h]
                } else {
                    weights.clone()
                };
                let pairs = if name == "white" {
                    weight_pairs(&weights, w, h)
                } else {
                    pairs.clone()
                };
                let mut expected = Vec::new();
                for y in y0..y0 + bh {
                    for x in x0..x0 + bw {
                        expected.push(window_stats_scalar(pixels, stride, x, y, &weights, w, h));
                    }
                }
                let mut got = Vec::new();
                block_stats(
                    pixels,
                    stride,
                    (x0, y0),
                    (bw, bh),
                    &weights,
                    &pairs,
                    (w, h),
                    &mut got,
                );
                assert_eq!(
                    got,
                    expected,
                    "{name} {w}x{h}, block {bw}x{bh} on {:?}",
                    backend()
                );
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    let mut avx2 = Vec::new();
                    // SAFETY: the CPU reported AVX2; the block fits.
                    unsafe {
                        x86::block_stats_avx2(
                            pixels,
                            stride,
                            (x0, y0),
                            (bw, bh),
                            &pairs,
                            (w, h),
                            &mut avx2,
                        )
                    };
                    assert_eq!(avx2, expected, "{name} {w}x{h}, block {bw}x{bh} on avx2");
                }
                // And from the top left, where the pairs start at column 0.
                let mut expected = Vec::new();
                for y in 0..bh {
                    for x in 0..bw {
                        expected.push(window_stats_scalar(pixels, stride, x, y, &weights, w, h));
                    }
                }
                block_stats(
                    pixels,
                    stride,
                    (0, 0),
                    (bw, bh),
                    &weights,
                    &pairs,
                    (w, h),
                    &mut got,
                );
                assert_eq!(got, expected, "{name} {w}x{h} at the corner");
            }
        }
    }

    #[test]
    fn opaque_sums_skip_the_transparent_and_count_the_rest() {
        let (width, height) = (53usize, 9usize);
        let raw = noise(width * height * 4, 33);
        for &(x, y, w, h) in &[
            (0usize, 0usize, 53usize, 9usize),
            (3, 2, 17, 5),
            (40, 8, 13, 1),
            (5, 5, 0, 3),
        ] {
            let mut expected = ([0u64; 3], 0u64);
            for row in y..y + h {
                for col in x..x + w {
                    let p = &raw[(row * width + col) * 4..][..4];
                    if p[3] >= 128 {
                        for (sum, &v) in expected.0.iter_mut().zip(p) {
                            *sum += u64::from(v);
                        }
                        expected.1 += 1;
                    }
                }
            }
            assert_eq!(
                opaque_sums(&raw, width * 4, (x, y), (w, h)),
                expected,
                "{x},{y} {w}x{h}"
            );
            assert_eq!(
                opaque_sums_scalar(&raw, width * 4, (x, y), (w, h)),
                expected
            );
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && w > 0 {
                // SAFETY: the CPU reported AVX2; the region fits.
                let avx2 = unsafe { x86::opaque_sums_avx2(&raw, width * 4, (x, y), (w, h)) };
                assert_eq!(avx2, expected, "{x},{y} {w}x{h} on avx2");
            }
        }
    }

    #[test]
    fn scores_in_vectors_are_the_scores_one_at_a_time() {
        // Windows of every kind: noise, flat (one value, or nearly), the
        // extremes, and products that make the correlation exactly ±1.
        let cases: [(usize, i64, i64); 4] = [
            (260, 260 * 77, 4_000_000),
            (1, 9, 0),
            (66_051, 66_051 * 255, 1),
            (7056, 7056 * 128, 70_560_000_000),
        ];
        for (n, t_sum, t_spread) in cases {
            let mut state = n as u32;
            let mut next = |limit: u64| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                (u64::from(state) * limit) >> 32
            };
            let mut products = Vec::new();
            let mut sums = Vec::new();
            let mut squares = Vec::new();
            for i in 0..203u64 {
                let n64 = n as u64;
                let sum = match i % 5 {
                    0 => 0,
                    1 => 255 * n64,
                    _ => next(255 * n64 + 1),
                };
                let mean = sum / n64;
                // Squares between flat (n mean²) and the most spread out.
                let square = match i % 7 {
                    0 => sum * sum / n64,
                    1 => (sum * sum).div_ceil(n64),
                    _ => (sum * sum / n64 + next(sum * 255 - sum * sum / n64 + 1)).min(sum * 255),
                };
                let product = match i % 3 {
                    0 => 0,
                    _ => next((255 * sum).min(u64::from(u32::MAX)) + 1),
                };
                let _ = mean;
                products.push(product as u32);
                sums.push(sum as u32);
                squares.push(square as u32);
            }
            let expected: Vec<f32> = (0..products.len())
                .map(|i| {
                    ncc(
                        n as i64,
                        t_sum,
                        t_spread,
                        products[i].into(),
                        sums[i].into(),
                        squares[i].into(),
                    )
                })
                .collect();
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512dq")
                && is_x86_feature_detected!("avx512vl")
            {
                let mut got = Vec::new();
                // SAFETY: the CPU reported AVX-512 F, DQ and VL.
                unsafe {
                    x86::scores_row_zmm(
                        &products,
                        &sums,
                        &squares,
                        (n as i64, t_sum, t_spread),
                        &mut got,
                    )
                };
                for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                    assert_eq!(g.to_bits(), e.to_bits(), "zmm, n {n} at {i}: {g} vs {e}");
                }
            }
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") {
                let mut got = vec![7.0f32];
                // SAFETY: the CPU reported AVX2.
                unsafe {
                    x86::scores_row_avx2(
                        &products,
                        &sums,
                        &squares,
                        (n as i64, t_sum, t_spread),
                        &mut got,
                    )
                };
                assert_eq!(got[0], 7.0, "appends");
                for (i, (g, e)) in got[1..].iter().zip(&expected).enumerate() {
                    assert_eq!(g.to_bits(), e.to_bits(), "n {n} at {i}: {g} vs {e}");
                }
            }
        }
    }

    #[test]
    fn running_totals_in_vectors_wrap_like_one_at_a_time() {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            for len in [0usize, 1, 7, 8, 9, 15, 16, 17, 100, 3840] {
                let values: Vec<u32> = noise(4 * len, len as u32)
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                    .collect();
                let mut expected = vec![0u32; len + 1];
                for i in 0..len {
                    expected[i + 1] = expected[i].wrapping_add(values[i]);
                }
                let mut got = vec![9u32; len + 1];
                // SAFETY: the CPU reported AVX2.
                unsafe { x86::running_totals(&values, &mut got) };
                assert_eq!(got, expected, "{len}");
            }
        }
    }

    /// The colour test on vectors against the one at a time: a grid through
    /// every colour, every alpha around the cut, and the parameters the HUD
    /// is found with (including a hue range that wraps past 360).
    #[test]
    fn the_colour_test_in_vectors_answers_as_one_at_a_time() {
        colour_masks_agree(23);
    }

    /// Every colour, not a grid (slow without optimisation).
    #[test]
    #[ignore]
    fn the_colour_test_in_vectors_answers_as_one_at_a_time_everywhere() {
        colour_masks_agree(1);
    }

    fn colour_masks_agree(step: usize) {
        let mut pixels = Vec::new();
        for r in (0..256).step_by(step).chain([254, 255]) {
            for g in (0..256).step_by(step).chain([1, 255]) {
                for b in (0..256).step_by(step).chain([0, 128, 255]) {
                    pixels.extend_from_slice(&[r as u8, g as u8, b as u8, 255]);
                }
            }
        }
        for a in [0u8, 127, 128, 129, 254] {
            pixels.extend_from_slice(&[200, 30, 30, a, 30, 30, 200, a]);
        }
        // An odd count, so the last few go one at a time.
        pixels.extend_from_slice(&[255, 0, 0, 255, 0, 255, 0, 255, 9, 9, 9, 255]);
        for (hue, saturation, value) in [
            ((340.0, 30.0), 0.35, 0.30),
            ((190.0, 250.0), 0.30, 0.30),
            ((40.0, 80.0), 0.25, 0.25),
            ((320.0, 350.0), 0.25, 0.35),
            ((0.0, 360.0), 0.0, 0.0),
            ((0.0, 0.0), 0.0, 0.0),
        ] {
            let expected: Vec<bool> = pixels
                .as_chunks::<4>()
                .0
                .iter()
                .map(|p| crate::color::is_color_pixel(&image::Rgba(*p), hue, saturation, value))
                .collect();
            let bits = |words: &[u64]| -> Vec<bool> {
                (0..expected.len())
                    .map(|i| words[i / 64] & (1 << (i % 64)) != 0)
                    .collect()
            };
            let mut got = vec![u64::MAX; 3];
            color_mask(&pixels, hue, saturation, value, &mut got);
            assert_eq!(got.len(), expected.len().div_ceil(64));
            assert_eq!(
                bits(&got),
                expected,
                "{hue:?} {saturation} {value} on {:?}",
                backend()
            );
            // No bit set past the last pixel.
            let tail = expected.len() % 64;
            assert!(
                tail == 0 || got[got.len() - 1] >> tail == 0,
                "bits past the end"
            );
        }
    }

    /// Halving across, the reference: each tap read with the row's ends
    /// repeated.
    fn halve_across_reference(row: &[u8]) -> Vec<u16> {
        let at = |k: isize| u16::from(row[k.clamp(0, row.len() as isize - 1) as usize]);
        (0..row.len() as isize / 2)
            .map(|i| at(2 * i - 1) + 3 * at(2 * i) + 3 * at(2 * i + 1) + at(2 * i + 2))
            .collect()
    }

    #[test]
    fn every_path_halves_a_row_the_same() {
        let source = noise(2000, 5);
        let white = vec![255u8; 300];
        for len in [
            0usize, 1, 2, 3, 4, 5, 33, 34, 35, 36, 37, 64, 65, 66, 67, 99, 1366, 1999,
        ] {
            for row in [&source[..len], &white[..len.min(300)]] {
                let expected = halve_across_reference(row);
                let mut got = vec![0u16; row.len() / 2];
                halve_across(row, &mut got);
                assert_eq!(got, expected, "{} pixels on {:?}", row.len(), backend());
                let mut scalar = vec![0u16; row.len() / 2];
                halve_across_scalar(row, &mut scalar, 0, row.len() / 2);
                assert_eq!(scalar, expected, "{} pixels, scalar", row.len());
                #[cfg(target_arch = "x86_64")]
                if is_x86_feature_detected!("avx2") {
                    let mut avx2 = vec![0u16; row.len() / 2];
                    // SAFETY: the CPU reported AVX2.
                    unsafe { x86::halve_across_avx2(row, &mut avx2) };
                    assert_eq!(avx2, expected, "{} pixels on avx2", row.len());
                }
            }
        }
    }

    #[test]
    fn every_path_combines_rows_down_the_same() {
        let wide = |seed: u32, n: usize| -> Vec<u16> {
            noise(2 * n, seed)
                .chunks_exact(2)
                .map(|p| u16::from_le_bytes([p[0], p[1]]) % 2041)
                .collect()
        };
        for n in [0usize, 1, 15, 16, 31, 32, 33, 64, 100, 683] {
            let (a, b, c, d) = (wide(1, n), wide(2, n), wide(3, n), wide(4, n));
            let expected: Vec<u8> = (0..n)
                .map(|i| {
                    ((u32::from(a[i])
                        + 3 * u32::from(b[i])
                        + 3 * u32::from(c[i])
                        + u32::from(d[i])
                        + 32)
                        / 64) as u8
                })
                .collect();
            let mut got = vec![9u8];
            combine_down(&a, &b, &c, &d, &mut got);
            assert_eq!(got[0], 9, "appends");
            assert_eq!(&got[1..], &expected[..], "{n} on {:?}", backend());
            let mut scalar = Vec::new();
            combine_down_scalar(&a, &b, &c, &d, &mut scalar);
            assert_eq!(scalar, expected, "{n}, scalar");
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") {
                let mut avx2 = Vec::new();
                // SAFETY: the CPU reported AVX2.
                unsafe { x86::combine_down_avx2(&a, &b, &c, &d, &mut avx2) };
                assert_eq!(avx2, expected, "{n} on avx2");
            }
        }
        // The largest totals still round down to 255.
        let full = vec![2040u16; 40];
        let mut got = Vec::new();
        combine_down(&full, &full, &full, &full, &mut got);
        assert!(got.iter().all(|&v| v == 255));
    }

    #[test]
    fn the_backend_is_named() {
        assert!(!backend().name().is_empty());
    }
}
