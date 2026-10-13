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
        let mut pairs = vec![0u32; width * height];
        for (paired, row) in pairs.chunks_exact_mut(width).zip(plane.chunks_exact(width)) {
            for ((pair, &here), &next) in paired.iter_mut().zip(row).zip(&row[1..]) {
                *pair = u32::from(here) | (u32::from(next) << 16);
            }
            paired[width - 1] = u32::from(row[width - 1]);
        }
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

        // Sums and squares.
        let (mut sums, mut squares) = (vec![0u32; span], vec![0u32; span]);
        for v in 0..h {
            for ((s, q), &p) in sums.iter_mut().zip(squares.iter_mut()).zip(row_of(v)) {
                *s += u32::from(p);
                *q += u32::from(p) * u32::from(p);
            }
        }
        for j in 0..bh {
            if j > 0 {
                let (leaving, entering) = (row_of(j - 1), row_of(j - 1 + h));
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
            let mut sum: u64 = sums[..w].iter().map(|&s| u64::from(s)).sum();
            let mut square: u64 = squares[..w].iter().map(|&q| u64::from(q)).sum();
            for i in 0..bw {
                if i > 0 {
                    sum = sum + u64::from(sums[i - 1 + w]) - u64::from(sums[i - 1]);
                    square = square + u64::from(squares[i - 1 + w]) - u64::from(squares[i - 1]);
                }
                let o = &mut out[j * bw + i];
                o.0 = sum;
                o.1 = square;
            }
        }

        // Products, a strip of 8 position columns at a time.
        let cols = 8 + 2 * half;
        let chunk = (super::MAX_AREA_32 / w).max(1);
        let mut block = vec![0u32; rows * cols];
        for strip in (0..bw).step_by(8) {
            let lanes = (bw - strip).min(8);
            for r in 0..rows {
                // The whole plane row (less of the last if the slice ends
                // early): pairs reaching past it read zeros, which meet
                // only zero weights or positions outside the block.
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
                    // SAFETY: rows `j0 + j + v < rows` and every load ends
                    // within a row of `block` (`2 (half - 1) + 8 <= cols`).
                    unsafe {
                        match n {
                            1 => products_rows::<1>(
                                &block,
                                cols,
                                j0,
                                pairs,
                                half,
                                v0..v1,
                                &mut lanes_of,
                            ),
                            2 => products_rows::<2>(
                                &block,
                                cols,
                                j0,
                                pairs,
                                half,
                                v0..v1,
                                &mut lanes_of,
                            ),
                            3 => products_rows::<3>(
                                &block,
                                cols,
                                j0,
                                pairs,
                                half,
                                v0..v1,
                                &mut lanes_of,
                            ),
                            4 => products_rows::<4>(
                                &block,
                                cols,
                                j0,
                                pairs,
                                half,
                                v0..v1,
                                &mut lanes_of,
                            ),
                            5 => products_rows::<5>(
                                &block,
                                cols,
                                j0,
                                pairs,
                                half,
                                v0..v1,
                                &mut lanes_of,
                            ),
                            6 => products_rows::<6>(
                                &block,
                                cols,
                                j0,
                                pairs,
                                half,
                                v0..v1,
                                &mut lanes_of,
                            ),
                            7 => products_rows::<7>(
                                &block,
                                cols,
                                j0,
                                pairs,
                                half,
                                v0..v1,
                                &mut lanes_of,
                            ),
                            _ => products_rows::<8>(
                                &block,
                                cols,
                                j0,
                                pairs,
                                half,
                                v0..v1,
                                &mut lanes_of,
                            ),
                        }
                    }
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
        }
        // All 255s: the largest sums, and lanes past 2^31.
        let (width, height) = (300usize, 260usize);
        let white = vec![255u8; width * height];
        let weights = vec![255u8; 256 * 256];
        let got = correlate_everywhere(&white, width, height, &weights, 256, 256);
        assert!(got.iter().all(|&t| t == 256 * 256 * 255 * 255));
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
