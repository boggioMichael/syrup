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

/// Whether a refinement's products run on AVX-VNNI's 256-bit dot products
/// (8 positions a vector, as the AVX2 path has them): when dot products are
/// in use at all and the CPU has that form of them. Decided once.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn block_vnni() -> bool {
    static CHOSEN: OnceLock<bool> = OnceLock::new();
    *CHOSEN.get_or_init(|| vnni().is_some() && vnni_support().1)
}

/// Whether the scores are computed on 512-bit vectors too: with AVX-512
/// in use for the dot products, and its double-word and 64-bit integer
/// conversions (DQ) and 256-bit forms (VL) there as well. Decided once.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
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
///
/// With AVX-512 dot products, a window whose sums fit 32 bits is taken row
/// by row in 64-byte vectors from the template's signed rows (`rows64`,
/// [`weight_rows64`]), however narrow: the same three sums, in a fraction
/// of the instructions of either 256-bit path (a 13×20 window: about 160
/// against 1000).
#[cfg_attr(not(target_arch = "x86_64"), allow(unused_variables))]
pub(crate) fn window_stats_wide(
    pixels: &[u8],
    stride: usize,
    (x, y): (usize, usize),
    (weights, wide, rows64): (&[u8], &[i16], &[u8]),
    (w, h): (usize, usize),
) -> (u64, u64, u64) {
    #[cfg(target_arch = "x86_64")]
    if w * h <= MAX_AREA_32 && vnni() == Some(Vnni::Zmm) {
        // SAFETY: AVX-512 F, BW and VNNI were detected (`vnni()`); the
        // kernel checks the sizes.
        return unsafe { x86::window_stats_zmm(pixels, stride, (x, y), rows64, (w, h)) };
    }
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

/// A template as [`block_stats`] takes it: its pixels (`w`×`h`, row by
/// row), and the same in the forms its vector paths multiply.
pub(crate) struct BlockWeights<'a> {
    pub(crate) pixels: &'a [u8],
    /// In neighbouring pairs ([`weight_pairs`]).
    pub(crate) pairs: &'a [u32],
    /// In fours, and `128 Σt` ([`weight_quads`]).
    pub(crate) quads: (&'a [u32], u32),
    /// Each row less 128 as signed bytes, in whole 64-byte vectors
    /// ([`weight_rows64`]).
    pub(crate) rows64: &'a [u8],
}

/// Each row of a `w`×`h` template less 128, as signed bytes (the top bit
/// flipped), padded with zeros to `w.next_multiple_of(64)` bytes: the
/// template as the wide dot-product path of [`block_stats`] takes it.
pub(crate) fn weight_rows64(weights: &[u8], w: usize, h: usize) -> Vec<u8> {
    let stride = w.next_multiple_of(64);
    let mut out = vec![0u8; stride * h];
    for (row, line) in out
        .chunks_exact_mut(stride)
        .zip(weights[..w * h].chunks_exact(w.max(1)))
    {
        for (o, &t) in row.iter_mut().zip(line) {
            *o = t ^ 0x80;
        }
    }
    out
}

/// Template rows at least this wide take the wide dot-product path of
/// [`block_stats`] (where 512-bit dot products are there): the block's
/// pixels are used as they lie, where the narrower path rearranges them
/// first, which costs more than the part of each 64-byte vector a row
/// leaves unused. (On Zen 5, an 8×5 block of a 28×105 template: 1.6 µs
/// against 2.5; of a 13×20 one, 0.58 against 0.42.)
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const ROWS64_MIN_WIDTH: usize = 20;

/// [`window_stats`] at every position of a block: for the `bw`×`bh`
/// positions from `(x0, y0)`, row by row, `out[j·bw + i]` is the window
/// whose top left is `(x0 + i, y0 + j)`. The same three sums, computed
/// together: on AVX2 the products of 8 positions a vector, each row of
/// pixels loaded once for every position row it reaches, or on AVX-512
/// each position's window row by row, 64 pixels a dot product; and the
/// pixel sums slid along columns, where a window at a time sums every
/// pixel once per position.
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
    template: &BlockWeights<'_>,
    (w, h): (usize, usize),
    out: &mut Vec<(u64, u64, u64)>,
) {
    out.clear();
    if bw == 0 || bh == 0 {
        return;
    }
    let BlockWeights {
        pixels: weights,
        pairs,
        quads,
        rows64,
    } = *template;
    assert!(w > 0 && h > 0 && x0 + bw - 1 + w <= stride);
    assert!(pixels.len() >= (y0 + bh - 1 + h - 1) * stride + x0 + bw - 1 + w);
    assert!(weights.len() >= w * h && pairs.len() >= w.div_ceil(2) * h);
    assert!(quads.0.len() >= w.div_ceil(4) * h);
    assert!(rows64.len() >= w.next_multiple_of(64) * h);
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `backend()` reported AVX2 on this CPU (and `vnni()`
        // AVX-512 VNNI, or `block_vnni()` AVX-VNNI, when that is used);
        // the sizes are checked above.
        Backend::Avx2 if w <= MAX_AREA_32 => unsafe {
            let fits = w * h <= MAX_AREA_32;
            let products = if fits && w >= ROWS64_MIN_WIDTH && vnni() == Some(Vnni::Zmm) {
                x86::BlockProducts::Rows64(rows64)
            } else if fits && block_vnni() {
                x86::BlockProducts::Quads(quads)
            } else {
                x86::BlockProducts::Pairs
            };
            x86::block_stats_avx2(
                pixels,
                stride,
                (x0, y0),
                (bw, bh),
                pairs,
                products,
                (w, h),
                out,
            )
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
    let mut out = Vec::new();
    correlate_everywhere_into(plane, width, height, weights, w, h, &mut out);
    out
}

/// [`correlate_everywhere`] into `out` (replacing what it held, and
/// reusing its room).
pub(crate) fn correlate_everywhere_into(
    plane: &[u8],
    width: usize,
    height: usize,
    weights: &[u8],
    w: usize,
    h: usize,
    out: &mut Vec<u32>,
) {
    assert!(w > 0 && h > 0 && w <= width && h <= height);
    assert!(plane.len() >= width * height && weights.len() >= w * h);
    let (columns, rows) = (width - w + 1, height - h + 1);
    out.resize(columns * rows, 0);
    #[cfg(target_arch = "x86_64")]
    if let Some(vnni) = vnni() {
        // SAFETY: `vnni()` reported the instructions on this CPU; the sizes
        // are checked above. The kernel writes every position.
        unsafe { x86::correlate_everywhere_vnni(vnni, out, plane, width, height, weights, w, h) };
        return;
    }
    // The paths below add into the totals.
    out.fill(0);
    match backend() {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `backend()` reported AVX2 on this CPU; the sizes are
        // checked above.
        Backend::Avx2 => unsafe {
            x86::correlate_everywhere_avx2(out, plane, width, height, weights, w, h)
        },
        _ => correlate_everywhere_rows(out, plane, width, weights, w, h),
    }
}

/// [`correlate_everywhere_into`] for several templates of one size (a
/// picture and its mirror image), `outs[i]` the totals of `templates[i]`:
/// on the dot-product path the plane is rearranged for them once.
pub(crate) fn correlate_alike_into(
    plane: &[u8],
    width: usize,
    height: usize,
    templates: &[&[u8]],
    (w, h): (usize, usize),
    outs: &mut [Vec<u32>],
) {
    assert_eq!(templates.len(), outs.len());
    assert!(w > 0 && h > 0 && w <= width && h <= height);
    assert!(plane.len() >= width * height && templates.iter().all(|t| t.len() >= w * h));
    #[cfg(target_arch = "x86_64")]
    if let Some(vnni) = vnni() {
        let (columns, rows) = (width - w + 1, height - h + 1);
        let mut slices: Vec<&mut [u32]> = outs
            .iter_mut()
            .map(|out| {
                out.resize(columns * rows, 0);
                out.as_mut_slice()
            })
            .collect();
        // SAFETY: `vnni()` reported the instructions on this CPU; the sizes
        // are checked above. The kernels write every position.
        unsafe {
            x86::correlate_alike_vnni(vnni, &mut slices, plane, width, height, templates, w, h)
        };
        return;
    }
    for (out, weights) in outs.iter_mut().zip(templates) {
        correlate_everywhere_into(plane, width, height, weights, w, h, out);
    }
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
    running: impl FnMut(&[u32], &mut [u32]),
) {
    let columns = width - w + 1;
    window_rows(plane, width, (w, h), running, |y, sums, squares| {
        score_row(
            &products[y * columns..(y + 1) * columns],
            sums,
            squares,
            out,
        )
    });
}

/// A position whose score reached the floor of a search: its column, its
/// row and the score.
pub(crate) type Hit = (u32, u32, f32);

/// How far below `floor²` a position's squared correlation may seem, in
/// double precision, and still be scored: far more than the single
/// precision of the score itself can move it (a few parts in 10⁷), so the
/// test drops no position that scores at least the floor.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
const HIT_MARGIN: f64 = 1e-5;

/// The squared covariance a position needs, as a multiple of its spread,
/// to be scored against `floor` (positive) with a template of spread
/// `t_spread`: `floor² (1 - HIT_MARGIN) t_spread`.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
fn hit_threshold(floor: f32, t_spread: i64) -> f64 {
    f64::from(floor) * f64::from(floor) * (1.0 - HIT_MARGIN) * t_spread as f64
}

/// The positions of a `width`×`height` plane where each of `templates`
/// scores at least `floor`, with their scores, in scan order, into the
/// matching list of `hits` (replacing what it held). Each template is
/// `w`×`h` and given by its products at every position (as
/// [`correlate_everywhere`] gives them), its pixel sum and its spread; the
/// windows' sums are taken once for them all.
///
/// Exactly the positions and scores of [`scores_everywhere`] at or above
/// the floor, which must be positive: a position's covariance must then be
/// positive and its square at least `floor²` times the product of the
/// spreads, and in vectors that test is made on the exact integer terms
/// (in double precision, with [`HIT_MARGIN`] to spare) and only the
/// positions passing it — few, on most searches — are scored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn hits_everywhere(
    plane: &[u8],
    width: usize,
    height: usize,
    (w, h): (usize, usize),
    templates: &[(&[u32], i64, i64)],
    floor: f32,
    hits: &mut [Vec<Hit>],
) {
    let n = w * h;
    assert!(w > 0 && h > 0 && w <= width && h <= height && n <= MAX_AREA_32);
    assert!(plane.len() >= width * height);
    assert!(floor > 0.0 && hits.len() == templates.len());
    let (columns, rows) = (width - w + 1, height - h + 1);
    assert!(templates.iter().all(|t| t.0.len() >= columns * rows));
    assert!(rows <= u32::MAX as usize && columns <= u32::MAX as usize);
    hits.iter_mut().for_each(Vec::clear);
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx2 {
        // SAFETY: `backend()` reported AVX2 on this CPU; sizes checked.
        unsafe { x86::hits_everywhere_avx2(plane, width, height, (w, h), templates, floor, hits) };
        return;
    }
    window_rows(
        &plane[..width * height],
        width,
        (w, h),
        |values, totals| {
            let mut total = 0u32;
            for (t, &v) in totals[1..].iter_mut().zip(values) {
                total = total.wrapping_add(v);
                *t = total;
            }
        },
        |y, sums, squares| {
            for (&(products, t_sum, t_spread), list) in templates.iter().zip(hits.iter_mut()) {
                let row = &products[y * columns..(y + 1) * columns];
                for (x, ((&p, &s), &q)) in row.iter().zip(sums).zip(squares).enumerate() {
                    let score = ncc(n as i64, t_sum, t_spread, p.into(), s.into(), q.into());
                    if score >= floor {
                        list.push((x as u32, y as u32, score));
                    }
                }
            }
        },
    );
}

/// Calls `row(y, sums, squares)` for each row `y` of positions of a
/// `w`×`h` window over a `width`-wide plane, in order, with the pixel sum
/// and the sum of squares of the window at each position along the row.
/// The sums come from column totals over the window's height, slid down a
/// row at a time, and running totals of those along the row (`running`:
/// `totals[i + 1] = totals[i] + values[i]`, wrapping, `totals[0]` left at
/// 0; a window's total, the difference of two, fits 32 bits).
#[inline(always)]
fn window_rows(
    plane: &[u8],
    width: usize,
    (w, h): (usize, usize),
    mut running: impl FnMut(&[u32], &mut [u32]),
    mut row: impl FnMut(usize, &[u32], &[u32]),
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
        row(y, &window_sums, &window_squares);
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
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    let mut from = 0;
    #[cfg(target_arch = "x86_64")]
    if zmm_scores() {
        // SAFETY: `zmm_scores()` reported AVX-512 F, BW, DQ and VL.
        from = unsafe { x86::color_mask_zmm(pixels, hue_range, min_saturation, min_value, out) };
    } else if backend() == Backend::Avx2 {
        // SAFETY: `backend()` reported AVX2 on this CPU.
        from = unsafe { x86::color_mask_avx2(pixels, hue_range, min_saturation, min_value, out) };
    }
    for (i, p) in pixels.as_chunks::<4>().0.iter().enumerate().skip(from) {
        if crate::color::is_color_pixel(&image::Rgba(*p), hue_range, min_saturation, min_value) {
            out[i / 64] |= 1 << (i % 64);
        }
    }
}

/// How a colour test takes a pixel's hue: within a range (`min > max`
/// wraps past 360, as [`crate::color::is_color_pixel`] has it), or at most
/// a distance round the circle from a hue ([`crate::bars::hue_distance`]).
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
#[derive(Clone, Copy, Debug)]
pub(crate) enum HueTest {
    Within(f32, f32),
    Near(f32, f32),
}

/// A bar's fill test ([`crate::bars::BarModel`]'s) for each pixel of
/// `pixels` (RGBA, 4 bytes a pixel), as bits as [`color_mask`] sets them:
/// saturation at least `min_saturation`, value at least `min_value` and
/// hue within `tolerance` of `centre`, round the circle. On AVX-512 16 at
/// a time (for a centre in `[0, 360)`), with the very arithmetic of the
/// pixel-at-a-time test.
pub(crate) fn fill_mask(
    pixels: &[u8],
    (centre, tolerance): (f32, f32),
    (min_saturation, min_value): (f32, f32),
    out: &mut Vec<u64>,
) {
    let count = pixels.len() / 4;
    out.clear();
    out.resize(count.div_ceil(64), 0);
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    let mut from = 0;
    #[cfg(target_arch = "x86_64")]
    if zmm_scores() && (0.0..360.0).contains(&centre) {
        // SAFETY: `zmm_scores()` reported AVX-512 F, BW, DQ and VL.
        from = unsafe {
            x86::colour_test_zmm(
                pixels,
                HueTest::Near(centre, tolerance),
                false,
                (min_saturation, min_value),
                out,
            )
        };
    }
    for (i, p) in pixels.as_chunks::<4>().0.iter().enumerate().skip(from) {
        if crate::bars::is_fill_pixel(p, (centre, tolerance), (min_saturation, min_value)) {
            out[i / 64] |= 1 << (i % 64);
        }
    }
}

/// The positions of `values` that are at least `floor`, in order, into
/// `out` (replacing what it held): 8 compared at once on AVX2.
pub(crate) fn at_least(values: &[f32], floor: f32, out: &mut Vec<usize>) {
    out.clear();
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
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
pub(crate) fn weight_quads(weights: &[u8], w: usize, h: usize) -> (Vec<u32>, u32) {
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

/// A template's columns in fours down the rows, as the vertical form of the
/// dot-product correlation takes them: `out[j·w + c]` holds the pixels of
/// column `c` in rows `4j .. 4j + 4` (`t(c, 4j) | t(c, 4j + 1) << 8 | …`,
/// zero past the last row), `h.div_ceil(4)` rows of `w`; and `128 Σt`.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn weight_vquads(weights: &[u8], w: usize, h: usize) -> (Vec<u32>, u32) {
    let rows = h.div_ceil(4);
    let mut out = vec![0u32; rows * w];
    let mut sum = 0u32;
    for (v, line) in weights[..w * h].chunks_exact(w).enumerate() {
        for (c, &t) in line.iter().enumerate() {
            out[(v / 4) * w + c] |= u32::from(t) << (8 * (v % 4));
            sum = sum.wrapping_add(u32::from(t));
        }
    }
    (out, sum.wrapping_mul(128))
}

/// Whether the vertical form of the dot-product correlation takes fewer
/// dot products than the horizontal one for a `w`×`h` template: fours of
/// a column (`h.div_ceil(4) · w` a position) against fours of a row
/// (`h · w.div_ceil(4)`); a 13×20 template, 65 against 80.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) fn vertical_quads_pay(w: usize, h: usize) -> bool {
    h.div_ceil(4) * w < h * w.div_ceil(4)
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

/// BGRA pixels (as Windows hands them over) into RGBA ones, opaque: blue
/// and red exchanged and alpha 255, `dst` as long as `src`, four bytes a
/// pixel. A byte shuffle on 512- or 256-bit vectors where the CPU has one;
/// the same bytes as [`bgra_to_rgba_word`] a pixel at a time.
// The capture of Windows frames is what takes them.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn bgra_to_rgba(src: &[u8], dst: &mut [u8]) {
    assert_eq!(src.len(), dst.len());
    assert!(src.len().is_multiple_of(4));
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx2 {
        // SAFETY: AVX-512 BW where `vnni()` reported 512-bit vectors, else
        // AVX2 (`backend()`); both slices hold `src.len()` bytes.
        done = unsafe {
            if vnni() == Some(Vnni::Zmm) {
                x86::bgra_to_rgba_zmm(src.as_ptr(), dst.as_mut_ptr(), src.len())
            } else {
                x86::bgra_to_rgba_avx2(src.as_ptr(), dst.as_mut_ptr(), src.len())
            }
        };
    }
    for (s, d) in src[done..]
        .as_chunks::<4>()
        .0
        .iter()
        .zip(dst[done..].as_chunks_mut::<4>().0)
    {
        *d = bgra_to_rgba_word(*s);
    }
}

/// [`bgra_to_rgba`] in place.
// The capture of Windows frames is what takes them.
#[cfg_attr(not(windows), allow(dead_code))]
pub(crate) fn bgra_to_rgba_in_place(pixels: &mut [u8]) {
    assert!(pixels.len().is_multiple_of(4));
    #[cfg_attr(not(target_arch = "x86_64"), allow(unused_mut))]
    let mut done = 0;
    #[cfg(target_arch = "x86_64")]
    if backend() == Backend::Avx2 {
        let at = pixels.as_mut_ptr();
        // SAFETY: as in `bgra_to_rgba`; each vector is read before the same
        // bytes are written.
        done = unsafe {
            if vnni() == Some(Vnni::Zmm) {
                x86::bgra_to_rgba_zmm(at, at, pixels.len())
            } else {
                x86::bgra_to_rgba_avx2(at, at, pixels.len())
            }
        };
    }
    for pixel in pixels[done..].as_chunks_mut::<4>().0 {
        *pixel = bgra_to_rgba_word(*pixel);
    }
}

/// One BGRA pixel as RGBA, opaque: shifts and masks on a 32-bit word.
// The capture of Windows frames is what takes them.
#[cfg_attr(not(windows), allow(dead_code))]
#[inline(always)]
pub(crate) fn bgra_to_rgba_word(bgra: [u8; 4]) -> [u8; 4] {
    let x = u32::from_le_bytes(bgra);
    (((x & 0x0000_00ff) << 16) | (x & 0x0000_ff00) | ((x >> 16) & 0x0000_00ff) | 0xff00_0000)
        .to_le_bytes()
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// `bgra_to_rgba` 32 bytes a step: a byte shuffle putting each pixel's
    /// red, green and blue in order with a zero for alpha, then alpha set.
    /// Returns how many bytes it did (a whole number of steps); `dst` may
    /// be `src`.
    ///
    /// # Safety
    ///
    /// AVX2; `src` and `dst` hold `len` bytes.
    #[cfg_attr(not(windows), allow(dead_code))]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn bgra_to_rgba_avx2(src: *const u8, dst: *mut u8, len: usize) -> usize {
        let order = _mm256_setr_epi8(
            2, 1, 0, -128, 6, 5, 4, -128, 10, 9, 8, -128, 14, 13, 12, -128, //
            2, 1, 0, -128, 6, 5, 4, -128, 10, 9, 8, -128, 14, 13, 12, -128,
        );
        let alpha = _mm256_set1_epi32(0xff00_0000_u32 as i32);
        let mut i = 0;
        while i + 32 <= len {
            // SAFETY: 32 bytes from `i` within both.
            unsafe {
                let v = _mm256_loadu_si256(src.add(i).cast());
                let v = _mm256_or_si256(_mm256_shuffle_epi8(v, order), alpha);
                _mm256_storeu_si256(dst.add(i).cast(), v);
            }
            i += 32;
        }
        i
    }

    /// `bgra_to_rgba_avx2` 64 bytes a step.
    ///
    /// # Safety
    ///
    /// AVX-512 F and BW; `src` and `dst` hold `len` bytes.
    #[cfg_attr(not(windows), allow(dead_code))]
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn bgra_to_rgba_zmm(src: *const u8, dst: *mut u8, len: usize) -> usize {
        let lane = [
            2i8, 1, 0, -128, 6, 5, 4, -128, 10, 9, 8, -128, 14, 13, 12, -128,
        ];
        let mut bytes = [0i8; 64];
        for (k, b) in bytes.iter_mut().enumerate() {
            *b = lane[k % 16];
        }
        // SAFETY: `bytes` is 64 bytes.
        let order = unsafe { _mm512_loadu_si512(bytes.as_ptr().cast()) };
        let alpha = _mm512_set1_epi32(0xff00_0000_u32 as i32);
        let mut i = 0;
        while i + 64 <= len {
            // SAFETY: 64 bytes from `i` within both.
            unsafe {
                let v = _mm512_loadu_si512(src.add(i).cast());
                let v = _mm512_or_si512(_mm512_shuffle_epi8(v, order), alpha);
                _mm512_storeu_si512(dst.add(i).cast(), v);
            }
            i += 64;
        }
        i
    }

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
        if w < 16 {
            // Rows narrower than half a vector: two at a time, a row in
            // each half, its first `w` bytes kept (the rows read whole
            // where 16 bytes lie within the slices, else copied out).
            let keep = {
                let mut bytes = [0u8; 16];
                bytes[..w].fill(0xFF);
                // SAFETY: a 16-byte array.
                unsafe { _mm_loadu_si128(bytes.as_ptr().cast()) }
            };
            let row16 = |slice: &[u8], at: usize| -> __m128i {
                let bytes = if at + 16 <= slice.len() {
                    // SAFETY: 16 bytes from `at` lie within the slice.
                    unsafe { _mm_loadu_si128(slice.as_ptr().add(at).cast()) }
                } else {
                    let mut copy = [0u8; 16];
                    copy[..w].copy_from_slice(&slice[at..at + w]);
                    // SAFETY: a 16-byte array.
                    unsafe { _mm_loadu_si128(copy.as_ptr().cast()) }
                };
                _mm_and_si128(bytes, keep)
            };
            let mut v = 0;
            while v < h {
                let pair = |slice: &[u8], first: usize, second: Option<usize>| {
                    let low = row16(slice, first);
                    let high = second.map_or(_mm_setzero_si128(), |at| row16(slice, at));
                    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(low), high)
                };
                let next = (v + 1 < h).then_some(v + 1);
                let p = pair(
                    pixels,
                    (y + v) * stride + x,
                    next.map(|n| (y + n) * stride + x),
                );
                let t = pair(weights, v * w, next.map(|n| n * w));
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
                v += 2;
            }
            return (sum_u64x4(sums), sum_i32x8(squares), sum_i32x8(products));
        }
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

    /// One window's three sums on AVX-512 dot products, row by row, 64
    /// pixels a step straight from the plane (the last step of a row
    /// loaded under a mask, so nothing past the window is read): the
    /// products `Σ p (t - 128)` with the template's signed rows (`rows64`,
    /// `w.next_multiple_of(64)` bytes a row), the squares `Σ p (p - 128)`,
    /// and the pixel sums of eight bytes at a time; `128 Σ p` added back to
    /// the first two. The 32-bit lanes wrap, and the true sums fit 32 bits.
    ///
    /// # Safety
    ///
    /// AVX-512 F, BW and VNNI.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    pub(super) unsafe fn window_stats_zmm(
        pixels: &[u8],
        stride: usize,
        (x, y): (usize, usize),
        rows64: &[u8],
        (w, h): (usize, usize),
    ) -> (u64, u64, u64) {
        let row = w.next_multiple_of(64);
        assert!(w > 0 && h > 0 && w * h <= super::MAX_AREA_32);
        assert!(rows64.len() >= row * h);
        assert!(pixels.len() >= (y + h - 1) * stride + x + w);
        let chunks = row / 64;
        let last: u64 = if w.is_multiple_of(64) {
            u64::MAX
        } else {
            (1 << (w % 64)) - 1
        };
        let flip = _mm512_set1_epi8(-128);
        let zero = _mm512_setzero_si512();
        let (mut sums, mut squares, mut products) = (zero, zero, zero);
        let (base, weights) = (pixels.as_ptr(), rows64.as_ptr());
        for v in 0..h {
            // SAFETY: row `v` of the window and of the weights, by the
            // checks above; the mask keeps the last load within the row.
            unsafe {
                let line = base.add((y + v) * stride + x);
                let t = weights.add(v * row);
                for k in 0..chunks {
                    let mask = if k + 1 == chunks { last } else { u64::MAX };
                    let p = _mm512_maskz_loadu_epi8(mask, line.add(64 * k).cast());
                    products =
                        _mm512_dpbusd_epi32(products, p, _mm512_loadu_si512(t.add(64 * k).cast()));
                    squares = _mm512_dpbusd_epi32(squares, p, _mm512_xor_si512(p, flip));
                    sums = _mm512_add_epi64(sums, _mm512_sad_epu8(p, zero));
                }
            }
        }
        let sum = _mm512_reduce_add_epi64(sums) as u64;
        let bias = (sum as u32).wrapping_mul(128);
        let squares = (_mm512_reduce_add_epi32(squares) as u32).wrapping_add(bias);
        let products = (_mm512_reduce_add_epi32(products) as u32).wrapping_add(bias);
        (sum, u64::from(squares), u64::from(products))
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
        vectors: Vec<__m512i>,
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
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn block_stats_avx2(
        pixels: &[u8],
        stride: usize,
        (x0, y0): (usize, usize),
        (bw, bh): (usize, usize),
        pairs: &[u32],
        products: BlockProducts<'_>,
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
            if w * h <= super::MAX_AREA_32 && matches!(products, BlockProducts::Rows64(_)) {
                // Every window's sums fit 32 bits, and 512-bit dot products
                // are there: 8 positions at a time, a row's pixels 64 at a
                // time.
                // SAFETY: AVX-512 VNNI, as the products' path says; the
                // block fits, as checked above.
                unsafe {
                    block_sums_zmm(
                        pixels,
                        stride,
                        (x0, y0),
                        (bw, bh),
                        (w, h),
                        &mut scratch.vectors,
                        out,
                    )
                };
            } else if w * h <= super::MAX_AREA_32 {
                // Every window's sums fit 32 bits: 8 positions at a time.
                // SAFETY: AVX2; the block fits, as checked above.
                unsafe {
                    block_sums_avx2(
                        pixels,
                        stride,
                        (x0, y0),
                        (bw, bh),
                        (w, h),
                        &mut scratch.sums,
                        out,
                    )
                };
            } else {
                block_sums_scalar(&row_of, (bw, bh), (w, h), rows, scratch, out);
            }
            block_products(
                pixels,
                stride,
                (x0, y0),
                (bw, bh),
                pairs,
                products,
                (w, h),
                rows,
                scratch,
                out,
            );
        });
    }

    /// The sums and squares of a block's windows one position at a time:
    /// each block row's at every position along it (the first window's 32
    /// pixels a step, then slid a pixel at a time), and those summed down
    /// each column of positions over the template's height, then slid
    /// down a row at a time, in 64 bits. (Every value is written before it
    /// is read: the buffers are only grown, never cleared.)
    #[target_feature(enable = "avx2")]
    fn block_sums_scalar<'a>(
        row_of: &impl Fn(usize) -> &'a [u8],
        (bw, bh): (usize, usize),
        (w, h): (usize, usize),
        rows: usize,
        scratch: &mut BlockScratch,
        out: &mut [(u64, u64, u64)],
    ) {
        {
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
        }
    }

    /// The sums and squares of a block's windows, 8 positions along a row
    /// at a time, in 32 bits (every window's fit): for each plane row of
    /// the block, the first window's (`line_sums`) and each next one's as
    /// that plus the pixels entering less those leaving, as running totals
    /// within a vector; then those summed down over the template's height,
    /// and slid down a row at a time. `scratch` holds each plane row's 8
    /// sums and 8 squares.
    ///
    /// # Safety
    ///
    /// AVX2; every window lies within `pixels`; `w · h <= MAX_AREA_32`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    unsafe fn block_sums_avx2(
        pixels: &[u8],
        stride: usize,
        (x0, y0): (usize, usize),
        (bw, bh): (usize, usize),
        (w, h): (usize, usize),
        scratch: &mut Vec<u32>,
        out: &mut [(u64, u64, u64)],
    ) {
        let rows = bh - 1 + h;
        assert!(w * h <= super::MAX_AREA_32 && out.len() >= bw * bh);
        assert!(pixels.len() >= (y0 + rows - 1) * stride + x0 + bw - 1 + w);
        if scratch.len() < 16 * rows {
            scratch.resize(16 * rows, 0);
        }
        // 8 pixels from `at`, as 32-bit lanes; zero past the slice.
        let eight = |at: usize| -> __m256i {
            let bytes = if at + 8 <= pixels.len() {
                // SAFETY: 8 bytes from `at` lie within the slice.
                unsafe { _mm_loadl_epi64(pixels.as_ptr().add(at).cast()) }
            } else {
                let mut copy = [0u8; 8];
                let have = pixels.len().saturating_sub(at).min(8);
                copy[..have].copy_from_slice(&pixels[at..at + have]);
                // SAFETY: an 8-byte array.
                unsafe { _mm_loadl_epi64(copy.as_ptr().cast()) }
            };
            _mm256_cvtepu8_epi32(bytes)
        };
        // Each lane the total of the lanes before it.
        let before = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
        let exclusive = |d: __m256i| {
            let mut x = _mm256_blend_epi32::<1>(
                _mm256_permutevar8x32_epi32(d, before),
                _mm256_setzero_si256(),
            );
            x = _mm256_add_epi32(x, _mm256_slli_si256::<4>(x));
            x = _mm256_add_epi32(x, _mm256_slli_si256::<8>(x));
            let low_total = _mm256_shuffle_epi32::<0xff>(x);
            _mm256_add_epi32(x, _mm256_permute2x128_si256::<0x08>(low_total, low_total))
        };
        for strip in (0..bw).step_by(8) {
            let lanes = (bw - strip).min(8);
            let xs = x0 + strip;
            for r in 0..rows {
                let start = (y0 + r) * stride + xs;
                let (s0, q0) = line_sums(&pixels[start..start + w]);
                let (leaving, entering) = (eight(start), eight(start + w));
                let d = _mm256_sub_epi32(entering, leaving);
                let d2 = _mm256_sub_epi32(
                    _mm256_mullo_epi32(entering, entering),
                    _mm256_mullo_epi32(leaving, leaving),
                );
                let sums = _mm256_add_epi32(_mm256_set1_epi32(s0 as i32), exclusive(d));
                let squares = _mm256_add_epi32(_mm256_set1_epi32(q0 as i32), exclusive(d2));
                // SAFETY: 16 values from `16 r` lie within `scratch`.
                unsafe {
                    _mm256_storeu_si256(scratch.as_mut_ptr().add(16 * r).cast(), sums);
                    _mm256_storeu_si256(scratch.as_mut_ptr().add(16 * r + 8).cast(), squares);
                }
            }
            // SAFETY: rows `r < rows` of `scratch`.
            let load = |r: usize, half: usize| unsafe {
                _mm256_loadu_si256(scratch.as_ptr().add(16 * r + 8 * half).cast())
            };
            let (mut sums, mut squares) = (_mm256_setzero_si256(), _mm256_setzero_si256());
            for r in 0..h {
                sums = _mm256_add_epi32(sums, load(r, 0));
                squares = _mm256_add_epi32(squares, load(r, 1));
            }
            for j in 0..bh {
                if j > 0 {
                    sums = _mm256_sub_epi32(
                        _mm256_add_epi32(sums, load(j - 1 + h, 0)),
                        load(j - 1, 0),
                    );
                    squares = _mm256_sub_epi32(
                        _mm256_add_epi32(squares, load(j - 1 + h, 1)),
                        load(j - 1, 1),
                    );
                }
                let (mut s, mut q) = ([0u32; 8], [0u32; 8]);
                // SAFETY: two 8-lane arrays.
                unsafe {
                    _mm256_storeu_si256(s.as_mut_ptr().cast(), sums);
                    _mm256_storeu_si256(q.as_mut_ptr().cast(), squares);
                }
                for (i, o) in out[j * bw + strip..][..lanes].iter_mut().enumerate() {
                    *o = (u64::from(s[i]), u64::from(q[i]), 0);
                }
            }
        }
    }

    /// `block_sums_avx2` on 512-bit vectors, without a sum across a vector
    /// for every plane row. Each plane row of the block gives, for the
    /// first of 8 positions along it: its pixels' sum in a vector of
    /// partial sums (byte sums of eights), and their squares' less `128 Σ
    /// p` as another (dot products `Σ p (p - 128)`, the row 64 pixels at
    /// a time); and for all 8, the pixels entering less those leaving
    /// (`d[i]`: column `i + w` less column `i`), and the same of their
    /// squares. Summed down over the template's height and slid down a row
    /// at a time, those give each position row's first window's sums
    /// (taken across the vectors once a position row), and each next
    /// window's as that plus the differences before it.
    ///
    /// # Safety
    ///
    /// AVX-512 F, BW, VL and VNNI; every window lies within `pixels`; `w ·
    /// h <= MAX_AREA_32`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
    unsafe fn block_sums_zmm(
        pixels: &[u8],
        stride: usize,
        (x0, y0): (usize, usize),
        (bw, bh): (usize, usize),
        (w, h): (usize, usize),
        scratch: &mut Vec<__m512i>,
        out: &mut [(u64, u64, u64)],
    ) {
        let rows = bh - 1 + h;
        assert!(w * h <= super::MAX_AREA_32 && out.len() >= bw * bh);
        assert!(pixels.len() >= (y0 + rows - 1) * stride + x0 + bw - 1 + w);
        if scratch.len() < 3 * rows {
            scratch.resize(3 * rows, _mm512_setzero_si512());
        }
        let chunks = w.div_ceil(64);
        let last: u64 = if w.is_multiple_of(64) {
            u64::MAX
        } else {
            (1 << (w % 64)) - 1
        };
        let flip = _mm512_set1_epi8(-128);
        let zero = _mm512_setzero_si512();
        // 8 pixels from `at`, as 32-bit lanes; zero past the slice.
        let eight = |at: usize| -> __m256i {
            let have = pixels.len().saturating_sub(at).min(8);
            let mask: __mmask16 = (1 << have) - 1;
            // SAFETY: the mask keeps the load within the slice.
            _mm256_cvtepu8_epi32(unsafe {
                _mm_maskz_loadu_epi8(mask, pixels.as_ptr().add(at.min(pixels.len())).cast())
            })
        };
        // Each lane the total of the lanes before it.
        let before = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
        let exclusive = |d: __m256i| {
            let mut x = _mm256_blend_epi32::<1>(
                _mm256_permutevar8x32_epi32(d, before),
                _mm256_setzero_si256(),
            );
            x = _mm256_add_epi32(x, _mm256_slli_si256::<4>(x));
            x = _mm256_add_epi32(x, _mm256_slli_si256::<8>(x));
            let low_total = _mm256_shuffle_epi32::<0xff>(x);
            _mm256_add_epi32(x, _mm256_permute2x128_si256::<0x08>(low_total, low_total))
        };
        for strip in (0..bw).step_by(8) {
            let lanes = (bw - strip).min(8);
            let xs = x0 + strip;
            for r in 0..rows {
                let start = (y0 + r) * stride + xs;
                let (mut sums, mut squares) = (zero, zero);
                for k in 0..chunks {
                    let mask = if k + 1 == chunks { last } else { u64::MAX };
                    // SAFETY: the mask keeps the load within the window's
                    // row, which lies within `pixels`.
                    let p = unsafe {
                        _mm512_maskz_loadu_epi8(mask, pixels.as_ptr().add(start + 64 * k).cast())
                    };
                    sums = _mm512_add_epi64(sums, _mm512_sad_epu8(p, zero));
                    // Masked-out pixels are 0, and 0 (0 - 128) adds nothing.
                    squares = _mm512_dpbusd_epi32(squares, p, _mm512_xor_si512(p, flip));
                }
                let (leaving, entering) = (eight(start), eight(start + w));
                let d = _mm256_sub_epi32(entering, leaving);
                let d2 = _mm256_sub_epi32(
                    _mm256_mullo_epi32(entering, entering),
                    _mm256_mullo_epi32(leaving, leaving),
                );
                scratch[3 * r] = sums;
                scratch[3 * r + 1] = squares;
                scratch[3 * r + 2] = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(d), d2);
            }
            let (mut sums, mut squares, mut deltas) = (zero, zero, zero);
            for r in 0..h {
                sums = _mm512_add_epi64(sums, scratch[3 * r]);
                squares = _mm512_add_epi32(squares, scratch[3 * r + 1]);
                deltas = _mm512_add_epi32(deltas, scratch[3 * r + 2]);
            }
            for j in 0..bh {
                if j > 0 {
                    let (old, new) = (3 * (j - 1), 3 * (j - 1 + h));
                    sums = _mm512_sub_epi64(_mm512_add_epi64(sums, scratch[new]), scratch[old]);
                    squares = _mm512_sub_epi32(
                        _mm512_add_epi32(squares, scratch[new + 1]),
                        scratch[old + 1],
                    );
                    deltas = _mm512_sub_epi32(
                        _mm512_add_epi32(deltas, scratch[new + 2]),
                        scratch[old + 2],
                    );
                }
                // The first window's sums (wrapping: the true ones fit).
                let sum = _mm512_reduce_add_epi64(sums) as u32;
                let square =
                    (_mm512_reduce_add_epi32(squares) as u32).wrapping_add(sum.wrapping_mul(128));
                let s = _mm256_add_epi32(
                    _mm256_set1_epi32(sum as i32),
                    exclusive(_mm512_castsi512_si256(deltas)),
                );
                let q = _mm256_add_epi32(
                    _mm256_set1_epi32(square as i32),
                    exclusive(_mm512_extracti64x4_epi64::<1>(deltas)),
                );
                let (mut s_lanes, mut q_lanes) = ([0u32; 8], [0u32; 8]);
                // SAFETY: two 8-lane arrays.
                unsafe {
                    _mm256_storeu_si256(s_lanes.as_mut_ptr().cast(), s);
                    _mm256_storeu_si256(q_lanes.as_mut_ptr().cast(), q);
                }
                for (i, o) in out[j * bw + strip..][..lanes].iter_mut().enumerate() {
                    *o = (u64::from(s_lanes[i]), u64::from(q_lanes[i]), 0);
                }
            }
        }
    }

    /// The products of a block's windows into `out`'s third sums, by the
    /// path `products` names (`rows`: the block's plane rows).
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    fn block_products(
        pixels: &[u8],
        stride: usize,
        (x0, y0): (usize, usize),
        (bw, bh): (usize, usize),
        pairs: &[u32],
        products: BlockProducts<'_>,
        (w, h): (usize, usize),
        rows: usize,
        scratch: &mut BlockScratch,
        out: &mut [(u64, u64, u64)],
    ) {
        let half = w.div_ceil(2);
        {
            match products {
                BlockProducts::Rows64(rows64) => {
                    // SAFETY: AVX-512 VNNI, as `block_stats` checked; the
                    // block fits, as checked above.
                    unsafe {
                        block_products_rows64(
                            pixels,
                            stride,
                            (x0, y0),
                            (bw, bh),
                            rows64,
                            (w, h),
                            out,
                        )
                    };
                    return;
                }
                BlockProducts::Quads((tq, bias)) => {
                    // SAFETY: AVX-VNNI, as `block_stats` checked; the block
                    // fits, as checked above.
                    unsafe {
                        block_products_vnni(
                            pixels,
                            stride,
                            (x0, y0),
                            (bw, bh),
                            tq,
                            bias,
                            (w, h),
                            &mut scratch.pairs,
                            out,
                        )
                    };
                    return;
                }
                BlockProducts::Pairs => {}
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
        }
    }

    /// How `block_stats_avx2` takes a block's products: from the template
    /// in neighbouring pairs (16-bit multiply-adds), in fours (AVX-VNNI),
    /// or row by row in 64-byte vectors (AVX-512 VNNI).
    #[derive(Clone, Copy)]
    pub(super) enum BlockProducts<'a> {
        Pairs,
        Quads((&'a [u32], u32)),
        Rows64(&'a [u8]),
    }

    /// Columns of positions `rows64_columns` takes at once: each of the
    /// template's vectors is loaded once for all of them.
    const ROWS64_COLUMNS: usize = 4;

    /// The products of a block on AVX-512 VNNI, into `out`'s third sums
    /// (its first, the windows' pixel sums, already there): for each
    /// position, each row of its window multiplied with the template's,
    /// 64 pixels a dot product straight from the plane (the last vector of
    /// a row loaded under a mask, so nothing past the window is read):
    /// `Σ p (t - 128)`, the template's rows as signed bytes (`rows64`, a
    /// whole number of vectors a row), to which `128 Σ p` is added back.
    /// The lanes wrap, and the true sums fit 32 bits.
    ///
    /// Positions go in groups of `ROWS64_COLUMNS` columns by up to 5 rows:
    /// a vector of a plane row is loaded once for every position row of
    /// the group it reaches (each with its own row of the template), and a
    /// vector of the template once for every column.
    ///
    /// # Safety
    ///
    /// AVX-512 F, BW and VNNI; every window lies within `pixels`;
    /// `rows64` holds `h` rows of `w.next_multiple_of(64)` bytes; `out`
    /// holds the block's `bw · bh` positions with their sums.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn block_products_rows64(
        pixels: &[u8],
        stride: usize,
        (x0, y0): (usize, usize),
        (bw, bh): (usize, usize),
        rows64: &[u8],
        (w, h): (usize, usize),
        out: &mut [(u64, u64, u64)],
    ) {
        let row = w.next_multiple_of(64);
        assert!(rows64.len() >= row * h && out.len() >= bw * bh);
        assert!(pixels.len() >= (y0 + bh - 1 + h - 1) * stride + x0 + bw - 1 + w);
        let mut x = 0;
        while x < bw {
            let columns = (bw - x).min(ROWS64_COLUMNS);
            let mut j = 0;
            while j < bh {
                let n = (bh - j).min(5);
                let group = Group {
                    pixels,
                    stride,
                    at: (x0 + x, y0 + j),
                    columns,
                    rows64,
                    row,
                    w,
                    h,
                };
                // SAFETY: by this function's guarantees, for the group's
                // positions (`x .. x + columns`, `j .. j + n`).
                let dots: [[u32; ROWS64_COLUMNS]; 5] = unsafe {
                    match n {
                        1 => widen(rows64_columns::<1>(&group)),
                        2 => widen(rows64_columns::<2>(&group)),
                        3 => widen(rows64_columns::<3>(&group)),
                        4 => widen(rows64_columns::<4>(&group)),
                        _ => rows64_columns::<5>(&group),
                    }
                };
                for (r, dots) in dots.iter().enumerate().take(n) {
                    for (c, &dot) in dots.iter().enumerate().take(columns) {
                        let o = &mut out[(j + r) * bw + x + c];
                        o.2 = u64::from(dot.wrapping_add((o.0 as u32).wrapping_mul(128)));
                    }
                }
                j += n;
            }
            x += columns;
        }
    }

    /// `ROWS` rows of dot products, padded to 5.
    fn widen<const ROWS: usize>(dots: [[u32; ROWS64_COLUMNS]; ROWS]) -> [[u32; ROWS64_COLUMNS]; 5] {
        let mut out = [[0; ROWS64_COLUMNS]; 5];
        out[..ROWS].copy_from_slice(&dots);
        out
    }

    /// A group of positions for `rows64_columns`: the plane, the top left
    /// position, how many of the `ROWS64_COLUMNS` columns are real (the
    /// rest repeat the last, and what they compute is dropped), and the
    /// template's signed rows (`row` bytes each) and size.
    struct Group<'a> {
        pixels: &'a [u8],
        stride: usize,
        at: (usize, usize),
        columns: usize,
        rows64: &'a [u8],
        row: usize,
        w: usize,
        h: usize,
    }

    /// The dot products `Σ p (t - 128)` of a group's `ROWS64_COLUMNS` ×
    /// `ROWS` positions, wrapping, as `block_products_rows64` takes them.
    ///
    /// # Safety
    ///
    /// AVX-512 F, BW and VNNI; the group's windows lie within `pixels`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn rows64_columns<const ROWS: usize>(g: &Group<'_>) -> [[u32; ROWS64_COLUMNS]; ROWS] {
        let chunks = g.row / 64;
        let last: u64 = if g.w.is_multiple_of(64) {
            u64::MAX
        } else {
            (1 << (g.w % 64)) - 1
        };
        let mut acc = [[_mm512_setzero_si512(); ROWS64_COLUMNS]; ROWS];
        let (x, y) = g.at;
        // Plane rows `y + r`: position row `i` takes it with template row
        // `r - i`, when that is one.
        for r in 0..g.h + ROWS - 1 {
            let line = (y + r) * g.stride + x;
            for k in 0..chunks {
                let mask = if k + 1 == chunks { last } else { u64::MAX };
                let mut p = [_mm512_setzero_si512(); ROWS64_COLUMNS];
                for (c, v) in p.iter_mut().enumerate() {
                    let c = c.min(g.columns - 1);
                    // SAFETY: the mask keeps the load within this window
                    // row, which lies within `pixels`.
                    *v = unsafe {
                        _mm512_maskz_loadu_epi8(
                            mask,
                            g.pixels.as_ptr().add(line + c + 64 * k).cast(),
                        )
                    };
                }
                for (i, sums) in acc.iter_mut().enumerate() {
                    let Some(v) = r.checked_sub(i).filter(|&v| v < g.h) else {
                        continue;
                    };
                    // SAFETY: row `v < h` of `rows64` holds `chunks`
                    // vectors.
                    let t = unsafe {
                        _mm512_loadu_si512(g.rows64.as_ptr().add(v * g.row + 64 * k).cast())
                    };
                    for (sum, &p) in sums.iter_mut().zip(&p) {
                        *sum = _mm512_dpbusd_epi32(*sum, p, t);
                    }
                }
            }
        }
        acc.map(|sums| sums.map(|sum| _mm512_reduce_add_epi32(sum) as u32))
    }

    /// The products of a block on AVX-VNNI, into `out`'s third sums: the
    /// block's pixels, less 128, in fours (`quad_up`), 8 positions a
    /// vector, and for each of the template's quads one broadcast and, for
    /// each position row, a load and a dot product of four pixels at once
    /// (`Σ t (p - 128)`, to which `128 Σ t` is added back; the lanes wrap,
    /// and the true sums fit 32 bits). `scratch` holds the block's quads.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn block_products_vnni(
        pixels: &[u8],
        stride: usize,
        (x0, y0): (usize, usize),
        (bw, bh): (usize, usize),
        tq: &[u32],
        bias: u32,
        (w, h): (usize, usize),
        scratch: &mut Vec<u32>,
        out: &mut [(u64, u64, u64)],
    ) {
        let rows = bh - 1 + h;
        let per_row = w.div_ceil(4);
        assert!(tq.len() >= per_row * h && out.len() >= bw * bh && w * h <= super::MAX_AREA_32);
        let cols = 8 + 4 * per_row;
        if scratch.len() < rows * cols {
            scratch.resize(rows * cols, 0);
        }
        let block = &mut scratch[..rows * cols];
        for strip in (0..bw).step_by(8) {
            let lanes = (bw - strip).min(8);
            for r in 0..rows {
                // The whole plane row (less of the last if the slice ends
                // early): quads reaching past it hold zeros, which meet only
                // zero weights or positions outside the block.
                let start = (y0 + r) * stride;
                let line = &pixels[start..(start + stride).min(pixels.len())];
                quad_up(
                    &line[(x0 + strip).min(line.len())..],
                    &mut block[r * cols..(r + 1) * cols],
                );
            }
            let mut j0 = 0;
            while j0 < bh {
                let n = (bh - j0).min(8);
                let mut lanes_of = [[0u32; 8]; 8];
                // SAFETY: AVX-VNNI; rows `j0 + j + v < rows`, and every
                // load ends within a row of `block` (`4 (per_row - 1) + 8 <=
                // cols`).
                unsafe {
                    match n {
                        1 => quad_rows::<1>(block, cols, j0, tq, per_row, h, bias, &mut lanes_of),
                        2 => quad_rows::<2>(block, cols, j0, tq, per_row, h, bias, &mut lanes_of),
                        3 => quad_rows::<3>(block, cols, j0, tq, per_row, h, bias, &mut lanes_of),
                        4 => quad_rows::<4>(block, cols, j0, tq, per_row, h, bias, &mut lanes_of),
                        5 => quad_rows::<5>(block, cols, j0, tq, per_row, h, bias, &mut lanes_of),
                        6 => quad_rows::<6>(block, cols, j0, tq, per_row, h, bias, &mut lanes_of),
                        7 => quad_rows::<7>(block, cols, j0, tq, per_row, h, bias, &mut lanes_of),
                        _ => quad_rows::<8>(block, cols, j0, tq, per_row, h, bias, &mut lanes_of),
                    }
                }
                for (j, lane) in lanes_of.iter().enumerate().take(n) {
                    let at = (j0 + j) * bw + strip;
                    for (o, &p) in out[at..at + lanes].iter_mut().zip(lane) {
                        o.2 = u64::from(p);
                    }
                }
                j0 += n;
            }
        }
    }

    /// `ROWS` position rows of a block's products from its quads, every
    /// template row and quad, the accumulators in registers throughout.
    ///
    /// # Safety
    ///
    /// AVX-VNNI, and `block` holds rows `j0 + ROWS - 1 + h - 1` of `cols >=
    /// 4 (per_row - 1) + 8` quads.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn quad_rows<const ROWS: usize>(
        block: &[u32],
        cols: usize,
        j0: usize,
        tq: &[u32],
        per_row: usize,
        h: usize,
        bias: u32,
        out: &mut [[u32; 8]; 8],
    ) {
        debug_assert!(block.len() >= (j0 + ROWS - 1 + h) * cols);
        let mut acc = [_mm256_setzero_si256(); ROWS];
        let base = block.as_ptr();
        for v in 0..h {
            for (k, &quad) in tq[v * per_row..(v + 1) * per_row].iter().enumerate() {
                let t = _mm256_set1_epi32(quad as i32);
                for (j, sum) in acc.iter_mut().enumerate() {
                    // SAFETY: by the caller's guarantee.
                    let p =
                        unsafe { _mm256_loadu_si256(base.add((j0 + j + v) * cols + 4 * k).cast()) };
                    *sum = _mm256_dpbusd_avx_epi32(*sum, t, p);
                }
            }
        }
        let bias = _mm256_set1_epi32(bias as i32);
        for (lane, sum) in out.iter_mut().zip(acc) {
            // SAFETY: a lane array is 32 bytes, and the store unaligned.
            unsafe { _mm256_storeu_si256(lane.as_mut_ptr().cast(), _mm256_add_epi32(sum, bias)) };
        }
    }

    /// The sum of `line`'s pixels and of their squares: 32 a step, a line
    /// that is not a whole number of steps ending with one more over its
    /// last 32 pixels, the ones already counted masked out.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn line_sums(line: &[u8]) -> (u32, u32) {
        let n = line.len();
        let zero = _mm256_setzero_si256();
        let (mut sums, mut squares) = (zero, zero);
        let mut add = |p: __m256i| {
            sums = _mm256_add_epi64(sums, _mm256_sad_epu8(p, zero));
            let lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(p));
            let hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(p));
            squares = _mm256_add_epi32(squares, _mm256_madd_epi16(lo, lo));
            squares = _mm256_add_epi32(squares, _mm256_madd_epi16(hi, hi));
        };
        if n < 32 {
            // Under a step: the line copied into a zeroed one.
            let mut copy = [0u8; 32];
            copy[..n].copy_from_slice(line);
            // SAFETY: a 32-byte array.
            add(unsafe { _mm256_loadu_si256(copy.as_ptr().cast()) });
            return (sum_u64x4(sums) as u32, sum_i32x8(squares) as u32);
        }
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
                _mm256_storeu_si256(
                    totals.as_mut_ptr().add(i + 1).cast(),
                    _mm256_add_epi32(x, carry),
                );
                // The vector's own total joins the carry: one addition from
                // step to step.
                carry = _mm256_add_epi32(carry, _mm256_permutevar8x32_epi32(x, last));
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

    /// `hits_everywhere` compiled for AVX2: the windows' sums as
    /// `scores_everywhere_avx2` takes them, and each row of each template
    /// tested by `hits_row_zmm` or `hits_row_avx2`.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn hits_everywhere_avx2(
        plane: &[u8],
        width: usize,
        height: usize,
        (w, h): (usize, usize),
        templates: &[(&[u32], i64, i64)],
        floor: f32,
        hits: &mut [Vec<super::Hit>],
    ) {
        let n = (w * h) as i64;
        let columns = width - w + 1;
        let wide = super::zmm_scores();
        if wide {
            // SAFETY: `zmm_scores` reported AVX-512 F, BW, DQ and VL.
            unsafe { hits_everywhere_zmm(plane, width, height, (w, h), templates, floor, hits) };
            return;
        }
        super::window_rows(
            &plane[..width * height],
            width,
            (w, h),
            |values, totals| running_totals(values, totals),
            |y, sums, squares| {
                for (&(products, t_sum, t_spread), list) in templates.iter().zip(hits.iter_mut()) {
                    let row = &products[y * columns..(y + 1) * columns];
                    let template = (n, t_sum, t_spread);
                    // SAFETY: AVX2, here, and AVX-512 when `zmm_scores`
                    // says so; the rows are as long as each other.
                    unsafe {
                        if wide {
                            hits_row_zmm(row, sums, squares, template, floor, y as u32, list)
                        } else {
                            hits_row_avx2(row, sums, squares, template, floor, y as u32, list)
                        }
                    }
                }
            },
        );
    }

    /// `hits_everywhere` on 512-bit vectors, a row of positions in two
    /// passes, taken in turn over stretches of `HIT_STRETCH` columns so
    /// that what one writes the other reads from the first-level cache.
    /// The first slides the column totals down a row and takes their
    /// running totals along it, 16 columns a step: each vector's own prefix
    /// in four shifted additions, and the total so far carried as a vector
    /// of its own, so that the only chain from step to step is one
    /// addition. The second, `w` columns behind, takes each window's sums
    /// as the difference of two running totals, 16 positions a step, and
    /// tests and scores the positions of every template as `hits_row_zmm`
    /// does, the windows' terms shared.
    ///
    /// # Safety
    ///
    /// AVX-512 F, BW, DQ and VL; sizes as `hits_everywhere` checks them.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
    unsafe fn hits_everywhere_zmm(
        plane: &[u8],
        width: usize,
        height: usize,
        (w, h): (usize, usize),
        templates: &[(&[u32], i64, i64)],
        floor: f32,
        hits: &mut [Vec<super::Hit>],
    ) {
        const HIT_STRETCH: usize = 256;
        let n = (w * h) as i64;
        let (columns, rows) = (width - w + 1, height - h + 1);
        assert!(plane.len() >= width * height && n <= super::MAX_AREA_32 as i64);
        assert!(templates.iter().all(|t| t.0.len() >= columns * rows));
        assert!(templates.iter().all(|t| (0..1 << 32).contains(&t.1)));
        // Column totals over the window's height, a vector per 16 columns
        // (aligned, as the vectors are); and their running totals, `run[16
        // + x]` the total of columns `0 ..= x` and the 16 before it zero,
        // padded so a vector read from `x + w + 15` stays within them for
        // every `x < columns`.
        let blocks = width.div_ceil(16);
        let mut column_sums = vec![_mm512_setzero_si512(); blocks];
        let mut column_squares = vec![_mm512_setzero_si512(); blocks];
        let mut run_sums = vec![_mm512_setzero_si512(); blocks + 3];
        let mut run_squares = vec![_mm512_setzero_si512(); blocks + 3];
        {
            // SAFETY: a vector is 16 lanes of 32 bits, any of them valid.
            let (sums, squares) = unsafe {
                (
                    std::slice::from_raw_parts_mut(
                        column_sums.as_mut_ptr().cast::<u32>(),
                        16 * blocks,
                    ),
                    std::slice::from_raw_parts_mut(
                        column_squares.as_mut_ptr().cast::<u32>(),
                        16 * blocks,
                    ),
                )
            };
            for line in plane[..width * h].chunks_exact(width) {
                for ((s, q), &p) in sums.iter_mut().zip(squares.iter_mut()).zip(line) {
                    *s += u32::from(p);
                    *q += u32::from(p) * u32::from(p);
                }
            }
        }
        let last = _mm512_set1_epi32(15);
        let zero = _mm512_setzero_si512();
        // A vector's own running totals: each lane plus every lane before it.
        let prefix = |mut v: __m512i| {
            v = _mm512_add_epi32(v, _mm512_alignr_epi32::<15>(v, zero));
            v = _mm512_add_epi32(v, _mm512_alignr_epi32::<14>(v, zero));
            v = _mm512_add_epi32(v, _mm512_alignr_epi32::<12>(v, zero));
            _mm512_add_epi32(v, _mm512_alignr_epi32::<8>(v, zero))
        };
        let run_s = run_sums.as_ptr().cast::<u32>();
        let run_q = run_squares.as_ptr().cast::<u32>();
        for y in 0..rows {
            let slide = (y > 0).then(|| {
                (
                    &plane[(y - 1) * width..y * width],
                    &plane[(y - 1 + h) * width..(y + h) * width],
                )
            });
            let mut carry_s = zero;
            let mut carry_q = zero;
            // Positions scored so far along the row.
            let mut scored = 0;
            let mut block = 0;
            while block < blocks {
                // Pass 1 over a stretch of columns.
                let end = (block + HIT_STRETCH / 16).min(blocks);
                for b in block..end {
                    let x = 16 * b;
                    let k = (width - x).min(16);
                    let mask: __mmask16 = if k == 16 { u16::MAX } else { (1 << k) - 1 };
                    let mut s = column_sums[b];
                    let mut q = column_squares[b];
                    if let Some((leaving, entering)) = slide {
                        // SAFETY: the masks keep the loads within the rows.
                        let (old, new) = unsafe {
                            (
                                _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(
                                    mask,
                                    leaving.as_ptr().add(x).cast(),
                                )),
                                _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(
                                    mask,
                                    entering.as_ptr().add(x).cast(),
                                )),
                            )
                        };
                        s = _mm512_sub_epi32(_mm512_add_epi32(s, new), old);
                        // new² - old² = (new - old)(new + old).
                        q = _mm512_add_epi32(
                            q,
                            _mm512_mullo_epi32(
                                _mm512_sub_epi32(new, old),
                                _mm512_add_epi32(new, old),
                            ),
                        );
                        column_sums[b] = s;
                        column_squares[b] = q;
                    }
                    let (ps, pq) = (prefix(s), prefix(q));
                    run_sums[b + 1] = _mm512_add_epi32(ps, carry_s);
                    run_squares[b + 1] = _mm512_add_epi32(pq, carry_q);
                    carry_s = _mm512_add_epi32(carry_s, _mm512_permutexvar_epi32(last, ps));
                    carry_q = _mm512_add_epi32(carry_q, _mm512_permutexvar_epi32(last, pq));
                }
                block = end;
                // Pass 2 over the positions whose windows end in columns
                // done — those before `16 block - w + 1` — a whole vector
                // at a time; or all the rest at the end.
                let ready = if block == blocks {
                    columns
                } else {
                    (16 * block + 1).saturating_sub(w).min(columns) / 16 * 16
                };
                if scored < ready {
                    let row = Row {
                        y,
                        columns,
                        w,
                        n,
                        floor,
                        run: (run_s, run_q),
                    };
                    // SAFETY: as `hits_stretch` asks: the running totals
                    // of row `y` are done up to `16 block`, which positions
                    // up to `ready` reach.
                    unsafe {
                        match (templates, &mut *hits) {
                            ([a], [list]) => hits_stretch([*a], [list], &row, scored..ready),
                            ([a, b], [list_a, list_b]) => {
                                hits_stretch([*a, *b], [list_a, list_b], &row, scored..ready)
                            }
                            _ => {
                                for (&t, list) in templates.iter().zip(hits.iter_mut()) {
                                    hits_stretch([t], [list], &row, scored..ready);
                                }
                            }
                        }
                    }
                    scored = ready.next_multiple_of(16);
                }
            }
        }
    }

    /// A row of positions for `hits_stretch`: its index, how many
    /// positions it has, the window's width and area, the floor, and the
    /// running totals of the column sums and squares along it (`run[16 +
    /// x]` the total of columns `0 ..= x`, the 16 before it zero).
    struct Row {
        y: usize,
        columns: usize,
        w: usize,
        n: i64,
        floor: f32,
        run: (*const u32, *const u32),
    }

    /// The second pass of `hits_everywhere_zmm` over positions `range` of
    /// a row, for `T` templates at once (each its products, pixel sum and
    /// spread, and the list of hits it adds to): 16 positions a step, the
    /// windows' sums and spreads taken once for all of them.
    ///
    /// # Safety
    ///
    /// AVX-512 F, BW, DQ and VL; `range` starts at a multiple of 16 and
    /// ends by `row.columns`; the running totals hold at least `w + 31`
    /// values past `range.end`, final for every column a position of
    /// `range` reaches (`x + w - 1`, rounded up to a whole vector of
    /// positions, unless the range ends the row); each template's products
    /// hold the row.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
    unsafe fn hits_stretch<const T: usize>(
        templates: [(&[u32], i64, i64); T],
        hits: [&mut Vec<super::Hit>; T],
        row: &Row,
        range: std::ops::Range<usize>,
    ) {
        let Row {
            y,
            columns,
            w,
            n,
            floor,
            run: (run_s, run_q),
        } = *row;
        let n_v = _mm512_set1_pd(n as f64);
        let zero_pd = _mm512_setzero_pd();
        let t_sums = templates.map(|t| _mm512_set1_pd(t.1 as f64));
        let thresholds = templates.map(|t| _mm512_set1_pd(super::hit_threshold(floor, t.2)));
        let rows = templates.map(|t| t.0[y * columns..(y + 1) * columns].as_ptr());
        let halves = |v: __m512i| {
            (
                _mm512_cvtepu32_pd(_mm512_castsi512_si256(v)),
                _mm512_cvtepu32_pd(_mm512_extracti64x4_epi64::<1>(v)),
            )
        };
        let mut x = range.start;
        while x < range.end {
            let k = (columns - x).min(16);
            let mask: __mmask16 = if k == 16 { u16::MAX } else { (1 << k) - 1 };
            // SAFETY: by the caller's guarantees; the masks keep the
            // products' loads within the row.
            unsafe {
                let window = |run: *const u32| {
                    _mm512_sub_epi32(
                        _mm512_loadu_si512(run.add(x + w + 15).cast()),
                        _mm512_loadu_si512(run.add(x + 15).cast()),
                    )
                };
                let (s_lo, s_hi) = halves(window(run_s));
                let (q_lo, q_hi) = halves(window(run_q));
                let spread_lo = _mm512_sub_pd(_mm512_mul_pd(n_v, q_lo), _mm512_mul_pd(s_lo, s_lo));
                let spread_hi = _mm512_sub_pd(_mm512_mul_pd(n_v, q_hi), _mm512_mul_pd(s_hi, s_hi));
                let mut passed = [0u16; T];
                for t in 0..T {
                    let p = _mm512_maskz_loadu_epi32(mask, rows[t].add(x).cast());
                    let (p_lo, p_hi) = halves(p);
                    let test = |p: __m512d, s: __m512d, spread: __m512d| {
                        let covariance =
                            _mm512_sub_pd(_mm512_mul_pd(n_v, p), _mm512_mul_pd(s, t_sums[t]));
                        let positive = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(covariance, zero_pd);
                        _mm512_mask_cmp_pd_mask::<_CMP_GE_OQ>(
                            positive,
                            _mm512_mul_pd(covariance, covariance),
                            _mm512_mul_pd(thresholds[t], spread),
                        )
                    };
                    passed[t] = (u16::from(test(p_lo, s_lo, spread_lo))
                        | u16::from(test(p_hi, s_hi, spread_hi)) << 8)
                        & mask;
                }
                if passed.iter().any(|&p| p != 0) {
                    let (sums, squares) = (window(run_s), window(run_q));
                    for t in 0..T {
                        if passed[t] == 0 {
                            continue;
                        }
                        // The 16 positions' scores as `ncc` gives them.
                        let (_, t_sum, t_spread) = templates[t];
                        let p = _mm512_maskz_loadu_epi32(mask, rows[t].add(x).cast());
                        let mut scores = [0f32; 16];
                        _mm512_storeu_ps(
                            scores.as_mut_ptr(),
                            scores16(p, sums, squares, (n, t_sum, t_spread)),
                        );
                        let mut bits = passed[t];
                        while bits != 0 {
                            let lane = bits.trailing_zeros() as usize;
                            if scores[lane] >= floor {
                                hits[t].push(((x + lane) as u32, y as u32, scores[lane]));
                            }
                            bits &= bits - 1;
                        }
                    }
                }
            }
            x += 16;
        }
    }

    /// `ncc` of 16 positions from their products, pixel sums and sums of
    /// squares (32-bit lanes): `scores_row_zmm`'s arithmetic, 8 lanes a
    /// half, so the same scores.
    #[inline]
    #[target_feature(enable = "avx512f,avx512dq,avx512vl")]
    pub(super) fn scores16(
        products: __m512i,
        sums: __m512i,
        squares: __m512i,
        (n, t_sum, t_spread): (i64, i64, i64),
    ) -> __m512 {
        let n_v = _mm512_set1_epi64(n);
        let nn = _mm512_set1_epi64(n * n);
        let t_sum_v = _mm512_set1_epi64(t_sum);
        let t_spread_v = _mm256_set1_ps(t_spread as f32);
        let (low, high) = (_mm256_set1_ps(-1.0), _mm256_set1_ps(1.0));
        let half = |v: __m512i, upper: bool| {
            _mm512_cvtepu32_epi64(if upper {
                _mm512_extracti64x4_epi64::<1>(v)
            } else {
                _mm512_castsi512_si256(v)
            })
        };
        let eight = |upper: bool| {
            let (p, s, q) = (
                half(products, upper),
                half(sums, upper),
                half(squares, upper),
            );
            let spread = _mm512_sub_epi64(_mm512_mul_epu32(n_v, q), _mm512_mul_epu32(s, s));
            let covariance =
                _mm512_sub_epi64(_mm512_mul_epu32(n_v, p), _mm512_mul_epu32(s, t_sum_v));
            // `4 spread > n²`: not flat.
            let varied = _mm512_cmpgt_epi64_mask(_mm512_slli_epi64::<2>(spread), nn);
            let denominator = _mm256_sqrt_ps(_mm256_mul_ps(_mm512_cvtepi64_ps(spread), t_spread_v));
            let score = _mm256_min_ps(
                _mm256_max_ps(
                    _mm256_div_ps(_mm512_cvtepi64_ps(covariance), denominator),
                    low,
                ),
                high,
            );
            // A flat window scores 0.
            _mm256_maskz_mov_ps(varied, score)
        };
        _mm512_insertf32x8::<1>(_mm512_castps256_ps512(eight(false)), eight(true))
    }

    /// The positions of a row (`y`) that score at least `floor`, pushed to
    /// `out` as `(column, y, score)`: 8 at a time, each tested on its
    /// integer terms in double precision — all exact: the operands below
    /// 2^32 and every product and difference below 2^51 — and only those
    /// passing scored, by `ncc` itself.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f,avx512dq,avx512vl")]
    pub(super) unsafe fn hits_row_zmm(
        products: &[u32],
        sums: &[u32],
        squares: &[u32],
        (n, t_sum, t_spread): (i64, i64, i64),
        floor: f32,
        y: u32,
        out: &mut Vec<super::Hit>,
    ) {
        let len = products.len();
        assert!(sums.len() >= len && squares.len() >= len);
        assert!(n > 0 && n <= super::MAX_AREA_32 as i64 && (0..1 << 32).contains(&t_sum));
        let n_v = _mm512_set1_pd(n as f64);
        let t_sum_v = _mm512_set1_pd(t_sum as f64);
        let threshold = _mm512_set1_pd(super::hit_threshold(floor, t_spread));
        let zero = _mm512_setzero_pd();
        let score = |k: usize, out: &mut Vec<super::Hit>| {
            let score = super::ncc(
                n,
                t_sum,
                t_spread,
                products[k].into(),
                sums[k].into(),
                squares[k].into(),
            );
            if score >= floor {
                out.push((k as u32, y, score));
            }
        };
        let mut i = 0;
        while i + 8 <= len {
            // SAFETY: eight values from `i` in each slice.
            let passed = unsafe {
                let load =
                    |v: &[u32]| _mm512_cvtepu32_pd(_mm256_loadu_si256(v.as_ptr().add(i).cast()));
                let (p, s, q) = (load(products), load(sums), load(squares));
                let spread = _mm512_sub_pd(_mm512_mul_pd(n_v, q), _mm512_mul_pd(s, s));
                let covariance = _mm512_sub_pd(_mm512_mul_pd(n_v, p), _mm512_mul_pd(s, t_sum_v));
                let positive = _mm512_cmp_pd_mask::<_CMP_GT_OQ>(covariance, zero);
                _mm512_mask_cmp_pd_mask::<_CMP_GE_OQ>(
                    positive,
                    _mm512_mul_pd(covariance, covariance),
                    _mm512_mul_pd(threshold, spread),
                )
            };
            let mut passed = passed;
            while passed != 0 {
                score(i + passed.trailing_zeros() as usize, out);
                passed &= passed - 1;
            }
            i += 8;
        }
        for k in i..len {
            score(k, out);
        }
    }

    /// `hits_row_zmm` on 256-bit vectors, 4 positions at a time; each
    /// 32-bit operand made a double exactly by placing it in the mantissa
    /// of 2^52 and taking 2^52 off.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn hits_row_avx2(
        products: &[u32],
        sums: &[u32],
        squares: &[u32],
        (n, t_sum, t_spread): (i64, i64, i64),
        floor: f32,
        y: u32,
        out: &mut Vec<super::Hit>,
    ) {
        let len = products.len();
        assert!(sums.len() >= len && squares.len() >= len);
        assert!(n > 0 && n <= super::MAX_AREA_32 as i64 && (0..1 << 32).contains(&t_sum));
        let n_v = _mm256_set1_pd(n as f64);
        let t_sum_v = _mm256_set1_pd(t_sum as f64);
        let threshold = _mm256_set1_pd(super::hit_threshold(floor, t_spread));
        let zero = _mm256_setzero_pd();
        let magic_i = _mm256_set1_epi64x(0x4330_0000_0000_0000);
        let magic_f = _mm256_set1_pd(4_503_599_627_370_496.0);
        let score = |k: usize, out: &mut Vec<super::Hit>| {
            let score = super::ncc(
                n,
                t_sum,
                t_spread,
                products[k].into(),
                sums[k].into(),
                squares[k].into(),
            );
            if score >= floor {
                out.push((k as u32, y, score));
            }
        };
        let mut i = 0;
        while i + 4 <= len {
            // SAFETY: four values from `i` in each slice.
            let passed = unsafe {
                let load = |v: &[u32]| {
                    let wide = _mm256_cvtepu32_epi64(_mm_loadu_si128(v.as_ptr().add(i).cast()));
                    _mm256_sub_pd(_mm256_castsi256_pd(_mm256_or_si256(wide, magic_i)), magic_f)
                };
                let (p, s, q) = (load(products), load(sums), load(squares));
                let spread = _mm256_sub_pd(_mm256_mul_pd(n_v, q), _mm256_mul_pd(s, s));
                let covariance = _mm256_sub_pd(_mm256_mul_pd(n_v, p), _mm256_mul_pd(s, t_sum_v));
                let positive = _mm256_cmp_pd::<_CMP_GT_OQ>(covariance, zero);
                let strong = _mm256_cmp_pd::<_CMP_GE_OQ>(
                    _mm256_mul_pd(covariance, covariance),
                    _mm256_mul_pd(threshold, spread),
                );
                _mm256_movemask_pd(_mm256_and_pd(positive, strong)) as u32
            };
            let mut passed = passed;
            while passed != 0 {
                score(i + passed.trailing_zeros() as usize, out);
                passed &= passed - 1;
            }
            i += 4;
        }
        for k in i..len {
            score(k, out);
        }
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

    /// `color_mask_avx2` on 512-bit vectors, 16 pixels a step, and the
    /// work in the order `is_color_pixel` does it, a step dropping out as
    /// soon as none of its pixels is left: the alpha and the value tested
    /// on the channels' integers (`c / 255` only grows with `c`, so the
    /// value is at least `min_value` exactly when the largest channel is
    /// at least the least integer whose quotient is), then the saturation
    /// with one division, then the hue with one more (the numerator of
    /// the pixel's case chosen first). Each `c / 255` is the double
    /// product `c · (1 / 255)` rounded once to single precision, the very
    /// quotient a division gives for every `c` (checked for all 256 in the
    /// tests). Sets the bits of the whole steps in `out` (zeroed, a word
    /// per 64 pixels) and returns how many pixels it did.
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
    pub(super) unsafe fn color_mask_zmm(
        pixels: &[u8],
        hue_range: (f32, f32),
        min_saturation: f32,
        min_value: f32,
        out: &mut [u64],
    ) -> usize {
        // SAFETY: as this function's.
        unsafe {
            colour_test_zmm(
                pixels,
                super::HueTest::Within(hue_range.0, hue_range.1),
                true,
                (min_saturation, min_value),
                out,
            )
        }
    }

    /// `color_mask_zmm` with the hue tested as `hue_test` says, and the
    /// alpha only when `opaque` (at least 128, as `is_color_pixel` asks; a
    /// bar's fill test asks nothing of it).
    #[target_feature(enable = "avx512f,avx512bw,avx512dq,avx512vl")]
    pub(super) unsafe fn colour_test_zmm(
        pixels: &[u8],
        hue_test: super::HueTest,
        opaque: bool,
        (min_saturation, min_value): (f32, f32),
        out: &mut [u64],
    ) -> usize {
        let count = pixels.len() / 4;
        assert!(out.len() >= count.div_ceil(64));
        // The least channel value bright enough (256: none is): from a
        // guess a step below, up (the quotients only grow).
        let bright = |c: u32| (c as f32 / 255.0).clamp(0.0, 1.0) >= min_value;
        let guess = if min_value.is_nan() {
            256
        } else {
            ((min_value.clamp(0.0, 1.0) * 255.0) as u32).saturating_sub(1)
        };
        let least = (guess..=255).find(|&c| bright(c)).unwrap_or(256);
        let least = _mm512_set1_epi32(least as i32);
        let byte = _mm512_set1_epi32(0xff);
        let half_alpha = _mm512_set1_epi32(128);
        let inverse = _mm512_set1_pd(1.0 / 255.0);
        let quotient = |c: __m512i| -> __m512 {
            let low = _mm512_cvtpd_ps(_mm512_mul_pd(
                _mm512_cvtepu32_pd(_mm512_castsi512_si256(c)),
                inverse,
            ));
            let high = _mm512_cvtpd_ps(_mm512_mul_pd(
                _mm512_cvtepu32_pd(_mm512_extracti64x4_epi64::<1>(c)),
                inverse,
            ));
            _mm512_insertf32x8::<1>(_mm512_castps256_ps512(low), high)
        };
        let zero = _mm512_setzero_ps();
        let saturation_v = _mm512_set1_ps(min_saturation);
        let (lo, hi) = match hue_test {
            super::HueTest::Within(min, max) => (_mm512_set1_ps(min), _mm512_set1_ps(max)),
            super::HueTest::Near(centre, tolerance) => {
                (_mm512_set1_ps(centre), _mm512_set1_ps(tolerance))
            }
        };
        let (sixty, two, four, full) = (
            _mm512_set1_ps(60.0),
            _mm512_set1_ps(2.0),
            _mm512_set1_ps(4.0),
            _mm512_set1_ps(360.0),
        );
        let mut i = 0;
        while i + 16 <= count {
            // SAFETY: 16 pixels from `i` within `pixels`.
            let px = unsafe { _mm512_loadu_si512(pixels.as_ptr().add(4 * i).cast()) };
            let r = _mm512_and_si512(px, byte);
            let g = _mm512_and_si512(_mm512_srli_epi32::<8>(px), byte);
            let b = _mm512_and_si512(_mm512_srli_epi32::<16>(px), byte);
            let high = _mm512_max_epu32(_mm512_max_epu32(r, g), b);
            let mut left = _mm512_cmpge_epu32_mask(high, least);
            if opaque {
                // Alpha at least 128 is `alpha / 255 >= 0.5`.
                left &= _mm512_cmpge_epu32_mask(_mm512_srli_epi32::<24>(px), half_alpha);
            }
            if left != 0 {
                let low = _mm512_min_epu32(_mm512_min_epu32(r, g), b);
                let (max, min) = (quotient(high), quotient(low));
                let delta = _mm512_sub_ps(max, min);
                // A black pixel's saturation is 0.
                let saturation = _mm512_maskz_div_ps(_mm512_cmpneq_ps_mask(max, zero), delta, max);
                left &= _mm512_cmp_ps_mask::<_CMP_GE_OQ>(saturation, saturation_v);
            }
            if left != 0 {
                let (rf, gf, bf) = (quotient(r), quotient(g), quotient(b));
                let (red, green) = (
                    _mm512_cmpeq_epi32_mask(high, r),
                    _mm512_cmpeq_epi32_mask(high, g),
                );
                let delta = _mm512_sub_ps(
                    quotient(high),
                    quotient(_mm512_min_epu32(_mm512_min_epu32(r, g), b)),
                );
                // The numerator of the pixel's case, red over green over blue.
                let mut top = _mm512_sub_ps(rf, gf);
                top = _mm512_mask_mov_ps(top, green, _mm512_sub_ps(bf, rf));
                top = _mm512_mask_mov_ps(top, red, _mm512_sub_ps(gf, bf));
                let q = _mm512_div_ps(top, delta);
                let mut turn = _mm512_add_ps(q, four);
                turn = _mm512_mask_mov_ps(turn, green, _mm512_add_ps(q, two));
                turn = _mm512_mask_mov_ps(turn, red, q);
                let mut hue = _mm512_mul_ps(sixty, turn);
                hue = _mm512_maskz_mov_ps(_mm512_cmpneq_ps_mask(delta, zero), hue);
                hue =
                    _mm512_mask_add_ps(hue, _mm512_cmp_ps_mask::<_CMP_LT_OQ>(hue, zero), hue, full);
                left &= match hue_test {
                    super::HueTest::Within(min, max) => {
                        let above = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(hue, lo);
                        let below = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(hue, hi);
                        if min <= max {
                            above & below
                        } else {
                            above | below
                        }
                    }
                    super::HueTest::Near(..) => {
                        // `bars::hue_distance`: the difference taken round
                        // the circle (`rem_euclid` of a difference within
                        // ±360: itself, 0 at ±360, or 360 more when
                        // negative), then the shorter way.
                        let (centre, tolerance) = (lo, hi);
                        let x = _mm512_sub_ps(hue, centre);
                        let whole = _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(_mm512_abs_ps(x), full);
                        let r = _mm512_mask_mov_ps(x, whole, zero);
                        let d = _mm512_mask_add_ps(
                            r,
                            _mm512_cmp_ps_mask::<_CMP_LT_OQ>(r, zero),
                            r,
                            full,
                        );
                        let distance = _mm512_min_ps(d, _mm512_sub_ps(full, d));
                        _mm512_cmp_ps_mask::<_CMP_LE_OQ>(distance, tolerance)
                    }
                };
            }
            out[i / 64] |= u64::from(left) << (i % 64);
            i += 16;
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
        // SAFETY: as this function's.
        unsafe { correlate_alike_vnni(vnni, &mut [out], plane, width, height, &[weights], w, h) }
    }

    /// [`correlate_everywhere_vnni`] for several templates of one size (a
    /// picture and its mirror image): the plane's quads made once for all
    /// of them, `outs[i]` the totals of `templates[i]`.
    ///
    /// # Safety
    ///
    /// `vnni` was detected on this CPU.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn correlate_alike_vnni(
        vnni: super::Vnni,
        outs: &mut [&mut [u32]],
        plane: &[u8],
        width: usize,
        height: usize,
        templates: &[&[u8]],
        w: usize,
        h: usize,
    ) {
        let (columns, rows) = (width - w + 1, height - h + 1);
        assert_eq!(outs.len(), templates.len());
        assert!(outs.iter().all(|out| out.len() >= columns * rows));
        assert!(plane.len() >= width * height && templates.iter().all(|t| t.len() >= w * h));
        if vnni == super::Vnni::Zmm && super::vertical_quads_pay(w, h) {
            // SAFETY: `Zmm` is reported only when the CPU has AVX-512 F, BW
            // and VNNI; the sizes are checked above.
            unsafe { correlate_alike_vquads(outs, plane, width, height, templates, w, h) };
            return;
        }
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
            for (out, weights) in outs.iter_mut().zip(templates) {
                let (tq, bias) = super::weight_quads(weights, w, h);
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
            }
        });
    }

    /// The exhaustive correlation on 512-bit dot products with the quads
    /// taken down the columns: each pixel, less 128, stored with the three
    /// below it as one 32-bit lane (`vquad_rows`), and the template's
    /// columns in fours the same way (`weight_vquads`). A template whose
    /// width is not a multiple of four leaves fewer lanes empty so (13×20:
    /// 65 dot products a position, against 80). The same exact totals:
    /// `Σ t (p - 128)`, wrapping, to which `128 Σ t` is added back.
    ///
    /// # Safety
    ///
    /// AVX-512 F, BW and VNNI; each of `outs` holds `columns · rows`
    /// positions.
    #[allow(clippy::too_many_arguments)]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn correlate_alike_vquads(
        outs: &mut [&mut [u32]],
        plane: &[u8],
        width: usize,
        height: usize,
        templates: &[&[u8]],
        w: usize,
        h: usize,
    ) {
        let (columns, rows) = (width - w + 1, height - h + 1);
        assert!(outs.iter().all(|out| out.len() >= columns * rows));
        assert!(plane.len() >= width * height && templates.iter().all(|t| t.len() >= w * h));
        // A block's loads reach `VNNI_BLOCK` lanes past its first column
        // and `w` columns along; the rows are written whole, in 64-lane
        // steps.
        let stride = (columns.next_multiple_of(VNNI_BLOCK) + w).max(width.next_multiple_of(64));
        QUADS_SCRATCH.with_borrow_mut(|vq| {
            if vq.len() < height * stride {
                vq.resize(height * stride, 0);
            }
            // SAFETY: AVX-512 F and BW, by this function's guarantee; `vq`
            // holds `height` rows of `stride >= width.next_multiple_of(64)`.
            unsafe { vquad_rows(plane, width, height, &mut vq[..height * stride], stride) };
            let vq = &vq[..height * stride];
            for (out, weights) in outs.iter_mut().zip(templates) {
                // SAFETY: as this function's; `out` holds every position.
                unsafe { vquads_into(out, vq, stride, (columns, rows), weights, (w, h)) };
            }
        });
    }

    /// The vertical correlation of one template, from the plane's quads
    /// (`vquad_rows`), into `out`.
    ///
    /// # Safety
    ///
    /// AVX-512 F, BW and VNNI; `vq` holds the quad rows of a plane
    /// `columns + w - 1` wide and `rows + h - 1` high, `stride >=
    /// columns.next_multiple_of(64) + w` lanes apart; `out` holds `columns
    /// · rows` positions.
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn vquads_into(
        out: &mut [u32],
        vq: &[u32],
        stride: usize,
        (columns, rows): (usize, usize),
        weights: &[u8],
        (w, h): (usize, usize),
    ) {
        assert!(out.len() >= columns * rows && weights.len() >= w * h);
        assert!(stride >= columns.next_multiple_of(VNNI_BLOCK) + w);
        assert!(vq.len() >= (rows + h - 1) * stride);
        let (tv, bias) = super::weight_vquads(weights, w, h);
        let fours = h.div_ceil(4);
        {
            let bias = _mm512_set1_epi32(bias as i32);
            let mut store = |y: usize, x: usize, sums: &[__m512i; 4]| {
                for (lane, &sum) in sums.iter().enumerate() {
                    let at = x + 16 * lane;
                    if at >= columns {
                        break;
                    }
                    let n = (columns - at).min(16);
                    let mask: __mmask16 = if n == 16 { u16::MAX } else { (1 << n) - 1 };
                    // SAFETY: the mask keeps the store within row `y`.
                    unsafe {
                        _mm512_mask_storeu_epi32(
                            out.as_mut_ptr().add(y * columns + at).cast(),
                            mask,
                            _mm512_add_epi32(sum, bias),
                        )
                    };
                }
            };
            // Position rows four apart go together, so that each row of
            // quads loaded serves up to four of them (each with its own
            // fours of the template's rows); sixteen rows are four such sets.
            let mut y0 = 0;
            while y0 < rows {
                for phase in 0..4 {
                    let first = y0 + phase;
                    if first >= rows {
                        break;
                    }
                    let r = (rows - first).div_ceil(4).min(4);
                    let mut x = 0;
                    while x < columns {
                        // SAFETY: position rows `first + 4a < rows` take
                        // quad rows up to `first + 4(a + fours - 1) <=
                        // height - 1`; the loads end by `x + 64 + w <=
                        // stride`; `tv` holds `fours · w`.
                        unsafe {
                            match r {
                                4 => {
                                    let sums =
                                        vquad_block::<4>(vq, stride, (x, first), &tv, (w, fours));
                                    for (a, sums) in sums.iter().enumerate() {
                                        store(first + 4 * a, x, sums);
                                    }
                                }
                                3 => {
                                    let sums =
                                        vquad_block::<3>(vq, stride, (x, first), &tv, (w, fours));
                                    for (a, sums) in sums.iter().enumerate() {
                                        store(first + 4 * a, x, sums);
                                    }
                                }
                                2 => {
                                    let sums =
                                        vquad_block::<2>(vq, stride, (x, first), &tv, (w, fours));
                                    for (a, sums) in sums.iter().enumerate() {
                                        store(first + 4 * a, x, sums);
                                    }
                                }
                                _ => {
                                    let sums =
                                        vquad_block::<1>(vq, stride, (x, first), &tv, (w, fours));
                                    store(first, x, &sums[0]);
                                }
                            }
                        }
                        x += VNNI_BLOCK;
                    }
                }
                y0 += 16;
            }
        }
    }

    /// `vq[y·stride + x]`: the pixels at column `x` of rows `y .. y + 4`,
    /// each less 128 (its top bit flipped), as the bytes of one lane;
    /// rows past the plane's last count as zero (they meet only zero
    /// weights), and so do columns past its width, up to the next 64.
    ///
    /// # Safety
    ///
    /// AVX-512 F and BW; `vq` holds `height` rows of `stride >=
    /// width.next_multiple_of(64)` lanes.
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn vquad_rows(plane: &[u8], width: usize, height: usize, vq: &mut [u32], stride: usize) {
        assert!(plane.len() >= width * height);
        assert!(stride >= width.next_multiple_of(64) && vq.len() >= height * stride);
        let flip = _mm512_set1_epi8(-128);
        for y in 0..height {
            let mut x = 0;
            while x < width {
                let n = (width - x).min(64);
                let mask: u64 = if n == 64 { u64::MAX } else { (1 << n) - 1 };
                let row = |r: usize| {
                    if y + r < height {
                        // SAFETY: the mask keeps the load within row `y + r`.
                        let bytes = unsafe {
                            _mm512_maskz_loadu_epi8(
                                mask,
                                plane.as_ptr().add((y + r) * width + x).cast(),
                            )
                        };
                        _mm512_xor_si512(bytes, flip)
                    } else {
                        _mm512_setzero_si512()
                    }
                };
                let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
                // Within each 128-bit lane: bytes of rows 0 and 1, then 2
                // and 3, side by side, then those pairs side by side — the
                // lane's 16 columns as four vectors' quarters, put in order
                // by exchanging quarters.
                let (a, b) = (_mm512_unpacklo_epi8(r0, r1), _mm512_unpackhi_epi8(r0, r1));
                let (c, d) = (_mm512_unpacklo_epi8(r2, r3), _mm512_unpackhi_epi8(r2, r3));
                let q0 = _mm512_unpacklo_epi16(a, c);
                let q1 = _mm512_unpackhi_epi16(a, c);
                let q2 = _mm512_unpacklo_epi16(b, d);
                let q3 = _mm512_unpackhi_epi16(b, d);
                let t0 = _mm512_shuffle_i64x2::<0b01_00_01_00>(q0, q1);
                let t1 = _mm512_shuffle_i64x2::<0b01_00_01_00>(q2, q3);
                let t2 = _mm512_shuffle_i64x2::<0b11_10_11_10>(q0, q1);
                let t3 = _mm512_shuffle_i64x2::<0b11_10_11_10>(q2, q3);
                let lanes = [
                    _mm512_shuffle_i64x2::<0b10_00_10_00>(t0, t1),
                    _mm512_shuffle_i64x2::<0b11_01_11_01>(t0, t1),
                    _mm512_shuffle_i64x2::<0b10_00_10_00>(t2, t3),
                    _mm512_shuffle_i64x2::<0b11_01_11_01>(t2, t3),
                ];
                for (k, &v) in lanes.iter().enumerate() {
                    // SAFETY: 16 lanes from `x + 16k < x + 64 <= stride`
                    // within row `y`.
                    unsafe {
                        _mm512_storeu_si512(vq.as_mut_ptr().add(y * stride + x + 16 * k).cast(), v)
                    };
                }
                x += 64;
            }
        }
    }

    /// The vertical correlation's inner loop: 64 positions from `x` in each
    /// of `R` position rows four apart from `y`. Quad row `y + 4m` serves
    /// position row `a` with the template's fours `m - a`, so each load is
    /// used by up to `R` dot products.
    ///
    /// # Safety
    ///
    /// AVX-512 F, BW and VNNI; `vq` holds quad rows `y .. y + 4(R - 1 +
    /// fours - 1)` of `stride >= x + 64 + w` lanes; `tv` holds `fours · w`.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn vquad_block<const R: usize>(
        vq: &[u32],
        stride: usize,
        (x, y): (usize, usize),
        tv: &[u32],
        (w, fours): (usize, usize),
    ) -> [[__m512i; 4]; R] {
        debug_assert!(vq.len() >= (y + 4 * (R + fours - 2)) * stride + x + 64 + w);
        debug_assert!(tv.len() >= fours * w);
        let mut acc = [[_mm512_setzero_si512(); 4]; R];
        let base = vq.as_ptr();
        for m in 0..fours + R - 1 {
            // SAFETY: quad row `y + 4m`, by the caller's guarantee.
            let row = unsafe { base.add((y + 4 * m) * stride + x) };
            for c in 0..w {
                let mut p = [_mm512_setzero_si512(); 4];
                for (lane, v) in p.iter_mut().enumerate() {
                    // SAFETY: within the row, by the caller's guarantee.
                    *v = unsafe { _mm512_loadu_si512(row.add(c + 16 * lane).cast()) };
                }
                for (a, sums) in acc.iter_mut().enumerate() {
                    let Some(j) = m.checked_sub(a).filter(|&j| j < fours) else {
                        continue;
                    };
                    // SAFETY: `j < fours` and `c < w`, so within `tv`.
                    let t = _mm512_set1_epi32(unsafe { *tv.get_unchecked(j * w + c) } as i32);
                    for (sum, &p) in sums.iter_mut().zip(&p) {
                        *sum = _mm512_dpbusd_epi32(*sum, t, p);
                    }
                }
            }
        }
        acc
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

    /// The exhaustive correlation's inner loop on 512-bit vectors: `R`
    /// position rows from `y` (and 64 positions from `x` along them) at
    /// once. Each of the plane's quads is loaded once, for every position
    /// row it reaches, so a load serves up to `R` dot products; and the
    /// `4 R` accumulators are as many independent chains of them.
    ///
    /// # Safety
    ///
    /// AVX-512 F, BW and VNNI; `quads` holds rows `y .. y + R - 1 + h` of
    /// `stride >= x + 64 + 4 per_row` quads; `tq` holds `h · per_row`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw,avx512vnni")]
    unsafe fn quad_block_zmm<const R: usize>(
        quads: &[u32],
        stride: usize,
        (x, y): (usize, usize),
        tq: &[u32],
        per_row: usize,
        h: usize,
    ) -> [[__m512i; 4]; R] {
        debug_assert!(quads.len() >= (y + R - 1 + h) * stride && tq.len() >= per_row * h);
        let mut acc = [[_mm512_setzero_si512(); 4]; R];
        let base = quads.as_ptr();
        // Plane rows `y + r` for `r` in `0 .. h + R - 1`: position row `i`
        // takes it with template row `r - i`, when that is one.
        for r in 0..h + R - 1 {
            // SAFETY: row `y + r` of the quads, by the caller's guarantee.
            let row = unsafe { base.add((y + r) * stride + x) };
            for k in 0..per_row {
                let mut p = [_mm512_setzero_si512(); 4];
                for (lane, v) in p.iter_mut().enumerate() {
                    // SAFETY: within the row, by the caller's guarantee.
                    *v = unsafe { _mm512_loadu_si512(row.add(4 * k + 16 * lane).cast()) };
                }
                for (i, sums) in acc.iter_mut().enumerate() {
                    let Some(v) = r.checked_sub(i).filter(|&v| v < h) else {
                        continue;
                    };
                    // SAFETY: `v < h`, so within `tq`.
                    let t = _mm512_set1_epi32(unsafe { *tq.get_unchecked(v * per_row + k) } as i32);
                    for (sum, &p) in sums.iter_mut().zip(&p) {
                        *sum = _mm512_dpbusd_epi32(*sum, t, p);
                    }
                }
            }
        }
        acc
    }

    /// 64 positions a block along the row, 16 a vector, and four position
    /// rows at a time (`quad_block_zmm`).
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
        assert!(out.len() >= columns * rows);
        let bias = _mm512_set1_epi32(bias as i32);
        let mut store = |y: usize, x: usize, sums: &[__m512i; 4]| {
            for (lane, &sum) in sums.iter().enumerate() {
                let at = x + 16 * lane;
                if at >= columns {
                    break;
                }
                let n = (columns - at).min(16);
                let mask: __mmask16 = if n == 16 { u16::MAX } else { (1 << n) - 1 };
                // SAFETY: the mask keeps the store within row `y`.
                unsafe {
                    _mm512_mask_storeu_epi32(
                        out.as_mut_ptr().add(y * columns + at).cast(),
                        mask,
                        _mm512_add_epi32(sum, bias),
                    )
                };
            }
        };
        let mut y = 0;
        while y < rows {
            let r = (rows - y).min(4);
            let mut x = 0;
            while x < columns {
                // SAFETY: rows `y .. y + r - 1 + h` exist, as `y + r <=
                // rows`; every block's loads end by `x + 64 + 4 per_row <=
                // stride`.
                unsafe {
                    match r {
                        4 => {
                            for (i, sums) in
                                quad_block_zmm::<4>(quads, stride, (x, y), tq, per_row, h)
                                    .iter()
                                    .enumerate()
                            {
                                store(y + i, x, sums);
                            }
                        }
                        3 => {
                            for (i, sums) in
                                quad_block_zmm::<3>(quads, stride, (x, y), tq, per_row, h)
                                    .iter()
                                    .enumerate()
                            {
                                store(y + i, x, sums);
                            }
                        }
                        2 => {
                            for (i, sums) in
                                quad_block_zmm::<2>(quads, stride, (x, y), tq, per_row, h)
                                    .iter()
                                    .enumerate()
                            {
                                store(y + i, x, sums);
                            }
                        }
                        _ => {
                            for (i, sums) in
                                quad_block_zmm::<1>(quads, stride, (x, y), tq, per_row, h)
                                    .iter()
                                    .enumerate()
                            {
                                store(y + i, x, sums);
                            }
                        }
                    }
                }
                x += VNNI_BLOCK;
            }
            y += r;
        }
    }

    /// `quad_block_zmm` on 256-bit vectors (16 registers hold no more than
    /// two position rows of 32 positions).
    ///
    /// # Safety
    ///
    /// AVX-VNNI; as `quad_block_zmm`, with 32 positions.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn quad_block_ymm<const R: usize>(
        quads: &[u32],
        stride: usize,
        (x, y): (usize, usize),
        tq: &[u32],
        per_row: usize,
        h: usize,
    ) -> [[__m256i; 4]; R] {
        debug_assert!(quads.len() >= (y + R - 1 + h) * stride && tq.len() >= per_row * h);
        let mut acc = [[_mm256_setzero_si256(); 4]; R];
        let base = quads.as_ptr();
        for r in 0..h + R - 1 {
            // SAFETY: row `y + r` of the quads, by the caller's guarantee.
            let row = unsafe { base.add((y + r) * stride + x) };
            for k in 0..per_row {
                let mut p = [_mm256_setzero_si256(); 4];
                for (lane, v) in p.iter_mut().enumerate() {
                    // SAFETY: within the row, by the caller's guarantee.
                    *v = unsafe { _mm256_loadu_si256(row.add(4 * k + 8 * lane).cast()) };
                }
                for (i, sums) in acc.iter_mut().enumerate() {
                    let Some(v) = r.checked_sub(i).filter(|&v| v < h) else {
                        continue;
                    };
                    // SAFETY: `v < h`, so within `tq`.
                    let t = _mm256_set1_epi32(unsafe { *tq.get_unchecked(v * per_row + k) } as i32);
                    for (sum, &p) in sums.iter_mut().zip(&p) {
                        *sum = _mm256_dpbusd_avx_epi32(*sum, t, p);
                    }
                }
            }
        }
        acc
    }

    /// The same on 256-bit vectors: 32 positions a block, 8 a vector, two
    /// position rows at a time.
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
        assert!(out.len() >= columns * rows);
        let bias = _mm256_set1_epi32(bias as i32);
        let mut store = |y: usize, x: usize, sums: &[__m256i; 4]| {
            let mut lanes = [0u32; 32];
            for (lane, &sum) in sums.iter().enumerate() {
                // SAFETY: 8 lanes into an array of 32.
                unsafe {
                    _mm256_storeu_si256(
                        lanes.as_mut_ptr().add(8 * lane).cast(),
                        _mm256_add_epi32(sum, bias),
                    )
                };
            }
            let n = (columns - x).min(32);
            out[y * columns + x..y * columns + x + n].copy_from_slice(&lanes[..n]);
        };
        let mut y = 0;
        while y < rows {
            let r = (rows - y).min(2);
            let mut x = 0;
            while x < columns {
                // SAFETY: as in `correlate_quads_zmm`, with blocks of 32.
                unsafe {
                    if r == 2 {
                        for (i, sums) in quad_block_ymm::<2>(quads, stride, (x, y), tq, per_row, h)
                            .iter()
                            .enumerate()
                        {
                            store(y + i, x, sums);
                        }
                    } else {
                        for (i, sums) in quad_block_ymm::<1>(quads, stride, (x, y), tq, per_row, h)
                            .iter()
                            .enumerate()
                        {
                            store(y + i, x, sums);
                        }
                    }
                }
                x += 32;
            }
            y += r;
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

    #[test]
    fn bgra_becomes_opaque_rgba_on_every_path() {
        for len in [
            0usize, 1, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 1000, 3840,
        ] {
            let src = noise(4 * len, len as u32 + 1);
            let expected: Vec<u8> = src
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|&p| [p[2], p[1], p[0], 255])
                .collect();
            let mut out = vec![7u8; src.len()];
            bgra_to_rgba(&src, &mut out);
            assert_eq!(out, expected, "{len} pixels");
            let mut in_place = src.clone();
            bgra_to_rgba_in_place(&mut in_place);
            assert_eq!(in_place, expected, "{len} pixels in place");
            #[cfg(target_arch = "x86_64")]
            {
                if is_x86_feature_detected!("avx2") {
                    let mut out = expected.clone();
                    // SAFETY: the CPU reported AVX2; both hold `src.len()`.
                    let done = unsafe {
                        x86::bgra_to_rgba_avx2(src.as_ptr(), out.as_mut_ptr(), src.len())
                    };
                    assert_eq!(done, src.len() - src.len() % 32);
                    assert_eq!(out, expected, "{len} pixels on avx2");
                }
                if is_x86_feature_detected!("avx512bw") {
                    let mut out = expected.clone();
                    // SAFETY: the CPU reported AVX-512 BW; both hold `src.len()`.
                    let done =
                        unsafe { x86::bgra_to_rgba_zmm(src.as_ptr(), out.as_mut_ptr(), src.len()) };
                    assert_eq!(done, src.len() - src.len() % 64);
                    assert_eq!(out, expected, "{len} pixels on avx-512");
                }
            }
        }
    }

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
            (13, 20),
            (13, 1),
            (2, 7),
            (8, 8),
            (16, 16),
            (56, 56),
            (57, 41),
            (100, 70),
        ] {
            let weights = noise(w * h, w as u32 * 31 + h as u32);
            // Inside the plane, and in its last corner, where a row's
            // vector would reach past the end.
            for (x, y) in [(stride - w - 3, rows - h - 5), (stride - w, rows - h)] {
                let expected = window_stats_scalar(&plane, stride, x, y, &weights, w, h);
                let got = window_stats(&plane, stride, x, y, &weights, w, h);
                assert_eq!(got, expected, "{w}x{h} on {:?}", backend());
                #[cfg(target_arch = "x86_64")]
                {
                    // SAFETY: SSE2 is part of x86-64.
                    let sse2 =
                        unsafe { x86::window_stats_sse2(&plane, stride, x, y, &weights, w, h) };
                    assert_eq!(sse2, expected, "{w}x{h} on sse2");
                    if is_x86_feature_detected!("avx2") {
                        // SAFETY: the CPU reported AVX2.
                        let avx2 =
                            unsafe { x86::window_stats_avx2(&plane, stride, x, y, &weights, w, h) };
                        assert_eq!(avx2, expected, "{w}x{h} at {x},{y} on avx2");
                    }
                }
            }
        }
    }

    #[test]
    fn alike_templates_correlate_as_each_alone() {
        // A picture and its mirror, and a third of the same size: the
        // plane's quads made once give each the totals it gets alone, on
        // the column-wise fours (13×20) and the row-wise ones (16×16).
        let (width, height) = (331usize, 47usize);
        let plane = noise(width * height, 29);
        for &(w, h) in &[(13usize, 20usize), (16, 16), (7, 9)] {
            let picture = noise(w * h, (w * 7 + h) as u32);
            let mirror: Vec<u8> = picture
                .chunks_exact(w)
                .flat_map(|row| row.iter().rev().copied().collect::<Vec<_>>())
                .collect();
            let third = noise(w * h, 3);
            let templates = [picture.as_slice(), mirror.as_slice(), third.as_slice()];
            let mut outs = vec![Vec::new(); 3];
            correlate_alike_into(&plane, width, height, &templates, (w, h), &mut outs);
            for (t, out) in templates.iter().zip(&outs) {
                let mut expected = vec![0u32; (width - w + 1) * (height - h + 1)];
                correlate_everywhere_rows(&mut expected, &plane, width, t, w, h);
                assert_eq!(out, &expected, "{w}x{h}");
            }
            #[cfg(target_arch = "x86_64")]
            for vnni in available_vnni() {
                let mut got = vec![vec![0u32; outs[0].len()]; 3];
                let mut slices: Vec<&mut [u32]> =
                    got.iter_mut().map(|g| g.as_mut_slice()).collect();
                // SAFETY: the CPU reported the instructions.
                unsafe {
                    x86::correlate_alike_vnni(
                        vnni,
                        &mut slices,
                        &plane,
                        width,
                        height,
                        &templates,
                        w,
                        h,
                    )
                };
                assert_eq!(got, outs, "{w}x{h} on {vnni:?}");
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
            // Shapes the column-wise fours take (`vertical_quads_pay`):
            // every height around a multiple of four, and position rows
            // that do not fill the last set of sixteen.
            (1, 4),
            (3, 4),
            (5, 8),
            (9, 12),
            (13, 20),
            (13, 21),
            (13, 22),
            (13, 23),
            (21, 37),
            (101, 58),
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
        // The same on the column-wise fours: a 255×256 template, all 255s.
        let weights = vec![255u8; 255 * 256];
        assert!(vertical_quads_pay(255, 256));
        let got = correlate_everywhere(&white, width, height, &weights, 255, 256);
        assert!(got.iter().all(|&t| t == 255 * 256 * 255 * 255));
        let black = vec![0u8; width * height];
        let got = correlate_everywhere(&black, width, height, &weights, 255, 256);
        assert!(got.iter().all(|&t| t == 0));
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
                let rows64 = weight_rows64(&weights, w, h);
                for (x, y) in [
                    (0, 0),
                    (stride - w, rows - h),
                    ((stride - w) / 3, (rows - h) / 2),
                ] {
                    let expected = window_stats_scalar(pixels, stride, x, y, &weights, w, h);
                    let got = window_stats_wide(
                        pixels,
                        stride,
                        (x, y),
                        (&weights, &wide, &rows64),
                        (w, h),
                    );
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
                    #[cfg(target_arch = "x86_64")]
                    if vnni() == Some(Vnni::Zmm) && w * h <= MAX_AREA_32 {
                        // SAFETY: the CPU reported AVX-512 VNNI; the window
                        // fits.
                        let zmm = unsafe {
                            x86::window_stats_zmm(pixels, stride, (x, y), &rows64, (w, h))
                        };
                        assert_eq!(zmm, expected, "{name} {w}x{h} at {x},{y} on avx-512");
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
                let quads = weight_quads(&weights, w, h);
                let rows64 = weight_rows64(&weights, w, h);
                let template = BlockWeights {
                    pixels: &weights,
                    pairs: &pairs,
                    quads: (&quads.0, quads.1),
                    rows64: &rows64,
                };
                block_stats(
                    pixels,
                    stride,
                    (x0, y0),
                    (bw, bh),
                    &template,
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
                            x86::BlockProducts::Pairs,
                            (w, h),
                            &mut avx2,
                        )
                    };
                    assert_eq!(avx2, expected, "{name} {w}x{h}, block {bw}x{bh} on avx2");
                    if is_x86_feature_detected!("avxvnni") && w * h <= MAX_AREA_32 {
                        let mut vnni = Vec::new();
                        // SAFETY: the CPU reported AVX-VNNI; the block fits.
                        unsafe {
                            x86::block_stats_avx2(
                                pixels,
                                stride,
                                (x0, y0),
                                (bw, bh),
                                &pairs,
                                x86::BlockProducts::Quads((&quads.0, quads.1)),
                                (w, h),
                                &mut vnni,
                            )
                        };
                        assert_eq!(
                            vnni, expected,
                            "{name} {w}x{h}, block {bw}x{bh} on avx-vnni"
                        );
                    }
                    if available_vnni().contains(&Vnni::Zmm) && w * h <= MAX_AREA_32 {
                        let mut zmm = Vec::new();
                        // SAFETY: the CPU reported AVX-512 VNNI; the block
                        // fits.
                        unsafe {
                            x86::block_stats_avx2(
                                pixels,
                                stride,
                                (x0, y0),
                                (bw, bh),
                                &pairs,
                                x86::BlockProducts::Rows64(&rows64),
                                (w, h),
                                &mut zmm,
                            )
                        };
                        assert_eq!(
                            zmm, expected,
                            "{name} {w}x{h}, block {bw}x{bh} on avx-512 rows"
                        );
                    }
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
                    &template,
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
            // (Compared with on x86-64 only.)
            let _ = &expected;
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
            if is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512dq")
                && is_x86_feature_detected!("avx512vl")
            {
                for start in (0..products.len() - 16).step_by(16) {
                    let load = |v: &[u32]| {
                        // SAFETY: 16 values from `start`; AVX-512 F reported.
                        unsafe { std::arch::x86_64::_mm512_loadu_si512(v[start..].as_ptr().cast()) }
                    };
                    let mut got = [0f32; 16];
                    // SAFETY: the CPU reported AVX-512 F, DQ and VL.
                    unsafe {
                        std::arch::x86_64::_mm512_storeu_ps(
                            got.as_mut_ptr(),
                            x86::scores16(
                                load(&products),
                                load(&sums),
                                load(&squares),
                                (n as i64, t_sum, t_spread),
                            ),
                        )
                    };
                    for (k, g) in got.iter().enumerate() {
                        let e = expected[start + k];
                        assert_eq!(
                            g.to_bits(),
                            e.to_bits(),
                            "16 at once, n {n} at {}: {g} vs {e}",
                            start + k
                        );
                    }
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

    /// Noise blurred over 3×3: structure at several scales, so a window cut
    /// from it correlates strongly near where it was cut and across the
    /// whole range elsewhere.
    fn textured(width: usize, height: usize, seed: u32) -> Vec<u8> {
        let raw = noise(width * height, seed);
        let mut out = vec![0u8; width * height];
        for y in 0..height {
            for x in 0..width {
                let mut sum = 0u32;
                for dy in 0..3 {
                    for dx in 0..3 {
                        let (yy, xx) = ((y + dy).min(height - 1), (x + dx).min(width - 1));
                        sum += u32::from(raw[yy * width + xx]);
                    }
                }
                out[y * width + x] = (sum / 9) as u8;
            }
        }
        out
    }

    #[test]
    fn hits_are_the_scores_at_or_above_the_floor() {
        // Wide enough for the vectors to take the rows in several
        // stretches, and narrow.
        for (width, height, sizes) in [
            (157, 61, &[(13, 20), (8, 8), (31, 9), (1, 2)][..]),
            (1201, 27, &[(13, 20), (40, 3), (300, 5), (1201, 2)][..]),
        ] {
            hits_agree(width, height, sizes);
        }
    }

    fn hits_agree(width: usize, height: usize, sizes: &[(usize, usize)]) {
        let plane = textured(width, height, 5);
        for &(w, h) in sizes {
            let columns = width - w + 1;
            // A window of the plane and its mirror image.
            let (cx, cy) = ((width - w).min(40), (height - h).min(17));
            let cut: Vec<u8> = (0..h)
                .flat_map(|y| plane[(y + cy) * width + cx..][..w].to_vec())
                .collect();
            let mirror: Vec<u8> = cut
                .chunks_exact(w)
                .flat_map(|row| row.iter().rev().copied().collect::<Vec<_>>())
                .collect();
            let templates: Vec<(Vec<u32>, i64, i64)> = [&cut, &mirror]
                .iter()
                .map(|t| {
                    let n = t.len() as i64;
                    let sum: i64 = t.iter().map(|&v| i64::from(v)).sum();
                    let squares: i64 = t.iter().map(|&v| i64::from(v).pow(2)).sum();
                    let products = correlate_everywhere(&plane, width, height, t, w, h);
                    (products, sum, n * squares - sum * sum)
                })
                .collect();
            let refs: Vec<(&[u32], i64, i64)> = templates
                .iter()
                .map(|(p, sum, spread)| (p.as_slice(), *sum, *spread))
                .collect();
            let scores: Vec<Vec<f32>> = refs
                .iter()
                .map(|&(products, sum, spread)| {
                    let mut out = Vec::new();
                    scores_everywhere(
                        &plane,
                        width,
                        height,
                        (w, h),
                        products,
                        (sum, spread),
                        &mut out,
                    );
                    out
                })
                .collect();
            // Floors across the range, and at, just above and just below
            // scores that occur.
            let mut floors = vec![0.05f32, 0.3, 0.58, 0.9, 1.0];
            let mut by_score: Vec<f32> = scores[0].iter().copied().filter(|&s| s > 0.0).collect();
            by_score.sort_by(f32::total_cmp);
            for k in [
                0,
                by_score.len() / 3,
                by_score.len() / 2,
                by_score.len() - 2,
                by_score.len() - 1,
            ] {
                let s = by_score[k];
                floors.extend([s, s.next_up(), s.next_down()]);
            }
            let wanted = |t: usize, floor: f32| -> Vec<(u32, u32, u32)> {
                scores[t]
                    .iter()
                    .enumerate()
                    .filter(|&(_, &s)| s >= floor)
                    .map(|(i, &s)| ((i % columns) as u32, (i / columns) as u32, s.to_bits()))
                    .collect()
            };
            let bits = |hits: &[Hit]| -> Vec<(u32, u32, u32)> {
                hits.iter().map(|&(x, y, s)| (x, y, s.to_bits())).collect()
            };
            for &floor in &floors {
                let mut hits = vec![vec![(9, 9, 9.0)], Vec::new()];
                hits_everywhere(&plane, width, height, (w, h), &refs, floor, &mut hits);
                for (t, got) in hits.iter().enumerate() {
                    assert_eq!(
                        bits(got),
                        wanted(t, floor),
                        "{w}x{h}, template {t}, floor {floor}"
                    );
                }
                // One template alone finds what it finds beside another.
                let mut alone = vec![Vec::new()];
                hits_everywhere(&plane, width, height, (w, h), &refs[1..], floor, &mut alone);
                assert_eq!(
                    bits(&alone[0]),
                    wanted(1, floor),
                    "{w}x{h} alone, floor {floor}"
                );
                // Each kernel, row by row, on the windows' sums.
                #[cfg(target_arch = "x86_64")]
                for (t, &(products, sum, spread)) in refs.iter().enumerate() {
                    let n = (w * h) as i64;
                    let mut ymm = Vec::new();
                    let mut zmm = Vec::new();
                    let zmm_ok = is_x86_feature_detected!("avx512f")
                        && is_x86_feature_detected!("avx512dq")
                        && is_x86_feature_detected!("avx512vl");
                    window_rows(
                        &plane,
                        width,
                        (w, h),
                        |values, totals| {
                            let mut total = 0u32;
                            for (t, &v) in totals[1..].iter_mut().zip(values) {
                                total = total.wrapping_add(v);
                                *t = total;
                            }
                        },
                        |y, sums, squares| {
                            let row = &products[y * columns..(y + 1) * columns];
                            // SAFETY: each only when the CPU reported it.
                            unsafe {
                                if is_x86_feature_detected!("avx2") {
                                    x86::hits_row_avx2(
                                        row,
                                        sums,
                                        squares,
                                        (n, sum, spread),
                                        floor,
                                        y as u32,
                                        &mut ymm,
                                    );
                                }
                                if zmm_ok {
                                    x86::hits_row_zmm(
                                        row,
                                        sums,
                                        squares,
                                        (n, sum, spread),
                                        floor,
                                        y as u32,
                                        &mut zmm,
                                    );
                                }
                            }
                        },
                    );
                    if is_x86_feature_detected!("avx2") {
                        assert_eq!(bits(&ymm), wanted(t, floor), "avx2 {w}x{h} {t} {floor}");
                    }
                    if zmm_ok {
                        assert_eq!(bits(&zmm), wanted(t, floor), "zmm {w}x{h} {t} {floor}");
                    }
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
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| u32::from_le_bytes(b))
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

    /// The vector colour test takes each channel's `c / 255` as the double
    /// product `c · (1 / 255)` rounded once: the same as the division for
    /// every channel value.
    #[test]
    fn a_channel_over_255_by_a_double_product_is_the_quotient() {
        for c in 0..=255u32 {
            assert_eq!(((c as f64) * (1.0 / 255.0)) as f32, c as f32 / 255.0, "{c}");
        }
    }

    /// The bar fill test on vectors against the one at a time, over a grid
    /// through every colour (every colour in the ignored test), for hues
    /// at and around the wrap, tolerances, and least saturations and values.
    #[test]
    fn the_fill_test_in_vectors_answers_as_one_at_a_time() {
        fill_masks_agree(17);
    }

    #[test]
    #[ignore = "every colour: a few seconds"]
    fn the_fill_test_in_vectors_answers_as_one_at_a_time_everywhere() {
        fill_masks_agree(1);
    }

    fn fill_masks_agree(step: usize) {
        let mut pixels = Vec::new();
        for r in (0..256).step_by(step).chain([254, 255]) {
            for g in (0..256).step_by(step).chain([1, 255]) {
                for b in (0..256).step_by(step).chain([0, 128, 255]) {
                    // Alpha plays no part in this test: vary it anyway.
                    pixels.extend_from_slice(&[r as u8, g as u8, b as u8, (r ^ b) as u8]);
                }
            }
        }
        pixels.extend_from_slice(&[255, 0, 0, 255, 0, 255, 0, 255, 9, 9, 9, 255]);
        for (centre, tolerance) in [
            (0.0, 16.0),
            (359.9, 16.0),
            (180.0, 16.0),
            (48.0, 0.0),
            (5.0, 400.0),
            (300.5, 12.5),
        ] {
            for least in [(0.35, 0.28), (0.0, 0.0), (0.21, 0.6)] {
                let expected: Vec<bool> = pixels
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| crate::bars::is_fill_pixel(p, (centre, tolerance), least))
                    .collect();
                let mut got = vec![u64::MAX; 2];
                fill_mask(&pixels, (centre, tolerance), least, &mut got);
                assert_eq!(got.len(), expected.len().div_ceil(64));
                for (i, &e) in expected.iter().enumerate() {
                    assert_eq!(
                        got[i / 64] & (1 << (i % 64)) != 0,
                        e,
                        "{centre} {tolerance} {least:?} at {i}: {:?}",
                        &pixels[4 * i..4 * i + 4]
                    );
                }
                let tail = expected.len() % 64;
                assert!(
                    tail == 0 || got[got.len() - 1] >> tail == 0,
                    "bits past the end"
                );
            }
        }
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
            // Each vector path on its own, over the whole steps it takes.
            #[cfg(target_arch = "x86_64")]
            {
                let words = expected.len().div_ceil(64);
                let check = |name: &str, did: usize, words_got: &[u64]| {
                    for (i, &e) in expected.iter().enumerate().take(did) {
                        assert_eq!(
                            words_got[i / 64] & (1 << (i % 64)) != 0,
                            e,
                            "{name} {hue:?} {saturation} {value} at {i}"
                        );
                    }
                };
                if is_x86_feature_detected!("avx2") {
                    let mut words_got = vec![0u64; words];
                    // SAFETY: the CPU reported AVX2.
                    let did = unsafe {
                        x86::color_mask_avx2(&pixels, hue, saturation, value, &mut words_got)
                    };
                    check("avx2", did, &words_got);
                }
                if is_x86_feature_detected!("avx512f")
                    && is_x86_feature_detected!("avx512bw")
                    && is_x86_feature_detected!("avx512dq")
                    && is_x86_feature_detected!("avx512vl")
                {
                    let mut words_got = vec![0u64; words];
                    // SAFETY: the CPU reported AVX-512 F, BW, DQ and VL.
                    let did = unsafe {
                        x86::color_mask_zmm(&pixels, hue, saturation, value, &mut words_got)
                    };
                    check("zmm", did, &words_got);
                }
            }
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
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&p| u16::from_le_bytes(p) % 2041)
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

/// The kernels behind template matching, for timing them on their own
/// (feature `bench-internals`). Not a stable API.
#[cfg(feature = "bench-internals")]
#[doc(hidden)]
pub mod bench {
    /// [`super::correlate_everywhere`].
    pub fn correlate_everywhere(
        plane: &[u8],
        width: usize,
        height: usize,
        weights: &[u8],
        w: usize,
        h: usize,
    ) -> Vec<u32> {
        super::correlate_everywhere(plane, width, height, weights, w, h)
    }

    /// [`super::scores_everywhere`], for a template whose pixels sum to
    /// `t_sum` with spread `t_spread`.
    pub fn scores_everywhere(
        plane: &[u8],
        width: usize,
        height: usize,
        (w, h): (usize, usize),
        products: &[u32],
        (t_sum, t_spread): (i64, i64),
    ) -> Vec<f32> {
        let mut out = Vec::new();
        super::scores_everywhere(
            plane,
            width,
            height,
            (w, h),
            products,
            (t_sum, t_spread),
            &mut out,
        );
        out
    }

    /// [`super::block_stats`] of a template given by its pixels.
    pub fn block_stats(
        pixels: &[u8],
        stride: usize,
        at: (usize, usize),
        block: (usize, usize),
        weights: &[u8],
        (w, h): (usize, usize),
    ) -> Vec<(u64, u64, u64)> {
        let pairs = super::weight_pairs(weights, w, h);
        let quads = super::weight_quads(weights, w, h);
        let rows64 = super::weight_rows64(weights, w, h);
        let template = super::BlockWeights {
            pixels: weights,
            pairs: &pairs,
            quads: (&quads.0, quads.1),
            rows64: &rows64,
        };
        let mut out = Vec::new();
        super::block_stats(pixels, stride, at, block, &template, (w, h), &mut out);
        out
    }

    /// A block's sums from its plane, stride, top left and size, into the
    /// given list.
    pub type BlockStats<'a> =
        Box<dyn Fn(&[u8], usize, (usize, usize), (usize, usize), &mut Vec<(u64, u64, u64)>) + 'a>;

    /// The same with the template's forms made once: call the returned
    /// closure to take the block's sums.
    pub fn block_stats_prepared<'a>(weights: &'a [u8], (w, h): (usize, usize)) -> BlockStats<'a> {
        let pairs = super::weight_pairs(weights, w, h);
        let quads = super::weight_quads(weights, w, h);
        let rows64 = super::weight_rows64(weights, w, h);
        Box::new(move |pixels, stride, at, block, out| {
            let template = super::BlockWeights {
                pixels: weights,
                pairs: &pairs,
                quads: (&quads.0, quads.1),
                rows64: &rows64,
            };
            super::block_stats(pixels, stride, at, block, &template, (w, h), out);
        })
    }

    /// [`super::hits_everywhere`]: for each template (its products, pixel
    /// sum and spread), the positions scoring at least `floor`.
    pub fn hits_everywhere(
        plane: &[u8],
        width: usize,
        height: usize,
        (w, h): (usize, usize),
        templates: &[(&[u32], i64, i64)],
        floor: f32,
    ) -> Vec<Vec<(u32, u32, f32)>> {
        let mut hits = vec![Vec::new(); templates.len()];
        super::hits_everywhere(plane, width, height, (w, h), templates, floor, &mut hits);
        hits
    }
}
