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
    fn the_backend_is_named() {
        assert!(!backend().name().is_empty());
    }
}
