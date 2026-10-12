//! Resizing a grey image with the triangle (bilinear) filter, byte for byte
//! as `image::imageops::resize(.., FilterType::Triangle)` does it.
//!
//! The cascade detector shrinks the frame once per scale, a few dozen times
//! a frame, and the `image` crate's resize was a third of its work: it
//! filters four channels of 32-bit floats whatever the pixel type, and its
//! loops walk the image column by column. This is the same computation —
//! the same weights, computed the same way, the same float operations in
//! the same order for every pixel, and the same rounding — on one channel,
//! row by row, so that every result is identical and the detector's
//! output does not move. The tests hold it to the `image` crate's bytes.

use image::GrayImage;

/// The taps of one output row or column: the first input index and the
/// normalised weights from there on (the zero-weight taps at either end
/// left out: adding a zero product changes no sum here, all being
/// non-negative).
struct Taps {
    first: usize,
    weights: Vec<f32>,
}

/// The `image` crate's triangle kernel.
fn triangle(x: f32) -> f32 {
    if x.abs() < 1.0 { 1.0 - x.abs() } else { 0.0 }
}

/// The taps for resampling `from` input positions to `to` outputs, computed
/// exactly as `image::imageops::sample` computes them (same expressions,
/// same `f32` operations, same order).
fn taps(from: u32, to: u32) -> Vec<Taps> {
    let ratio = from as f32 / to as f32;
    let sratio = if ratio < 1.0 { 1.0 } else { ratio };
    let support = 1.0 * sratio;
    (0..to)
        .map(|out| {
            let input = (out as f32 + 0.5) * ratio;
            let left = ((input - support).floor() as i64).clamp(0, i64::from(from) - 1);
            let right = ((input + support).ceil() as i64).clamp(left + 1, i64::from(from));
            let input = input - 0.5;
            let mut weights: Vec<f32> = (left..right)
                .map(|i| triangle((i as f32 - input) / sratio))
                .collect();
            let mut sum = 0.0;
            for &w in &weights {
                sum += w;
            }
            for w in &mut weights {
                *w /= sum;
            }
            let lead = weights.iter().take_while(|&&w| w == 0.0).count();
            let trail = weights[lead..]
                .iter()
                .rev()
                .take_while(|&&w| w == 0.0)
                .count();
            weights.truncate(weights.len() - trail);
            weights.drain(..lead);
            Taps {
                first: left as usize + lead,
                weights,
            }
        })
        .collect()
}

/// `image` rounds a filtered value by clamping it to 0..=255 and then
/// rounding half away from zero (`f32::round`). For a value in that range
/// this is the integer part plus one when the fraction is at least a half;
/// the fraction is exact (`t - trunc(t)` loses nothing), and this avoids
/// the call `f32::round` compiles to.
#[inline]
fn to_byte(t: f32) -> u8 {
    let t = t.clamp(0.0, 255.0);
    let whole = t as u32;
    (whole + u32::from(t - whole as f32 >= 0.5)) as u8
}

/// `gray` resized to `width` x `height` with the triangle filter: the same
/// bytes as `image::imageops::resize(gray, width, height, Triangle)`.
pub(crate) fn resize_triangle(gray: &GrayImage, width: u32, height: u32) -> GrayImage {
    let (w, h) = gray.dimensions();
    if (width, height) == (w, h) {
        return gray.clone();
    }
    if width == 0 || height == 0 || w == 0 || h == 0 {
        return GrayImage::new(width, height);
    }
    let (wu, nw) = (w as usize, width as usize);
    let src = gray.as_raw();
    // Rows first, into floats, as the `image` crate does.
    let rows = taps(h, height);
    let mut between = vec![0.0f32; height as usize * wu];
    for (out, taps) in between.chunks_exact_mut(wu).zip(&rows) {
        accumulate_rows(out, src, wu, taps);
    }
    // Then columns, rounding to bytes.
    let columns = taps(w, width);
    let mut out = vec![0u8; height as usize * nw];
    for (line, row) in out.chunks_exact_mut(nw).zip(between.chunks_exact(wu)) {
        for (pixel, taps) in line.iter_mut().zip(&columns) {
            let inputs = &row[taps.first..taps.first + taps.weights.len()];
            let mut t = 0.0f32;
            for (&v, &weight) in inputs.iter().zip(&taps.weights) {
                t += v * weight;
            }
            *pixel = to_byte(t);
        }
    }
    GrayImage::from_raw(width, height, out).expect("buffer of the right size")
}

/// One output row of the vertical pass: `out[x] = Σ src[first + i][x] ·
/// weights[i]`, each pixel's terms added in tap order from zero. Each
/// pixel's sum is its own sequence of operations whatever the vector
/// width, so the wide build gives the same floats as the narrow one.
fn accumulate_rows(out: &mut [f32], src: &[u8], width: usize, taps: &Taps) {
    #[cfg(target_arch = "x86_64")]
    if crate::kernels::backend() == crate::kernels::Backend::Avx2 {
        // SAFETY: `backend()` reported AVX2 on this CPU.
        unsafe { accumulate_rows_avx2(out, src, width, taps) };
        return;
    }
    accumulate_rows_plain(out, src, width, taps);
}

#[inline(always)]
fn accumulate_rows_plain(out: &mut [f32], src: &[u8], width: usize, taps: &Taps) {
    for (i, &weight) in taps.weights.iter().enumerate() {
        let row = &src[(taps.first + i) * width..(taps.first + i + 1) * width];
        for (o, &p) in out.iter_mut().zip(row) {
            *o += f32::from(p) * weight;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accumulate_rows_avx2(out: &mut [f32], src: &[u8], width: usize, taps: &Taps) {
    // Multiplies and adds stay separate (Rust never fuses them), so this is
    // the plain loop at 8 floats a step.
    accumulate_rows_plain(out, src, width, taps);
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::imageops::{self, FilterType};

    fn noise(width: u32, height: u32, seed: u32) -> GrayImage {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        GrayImage::from_fn(width, height, |x, y| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Smooth gradients, hard edges and noise together.
            let v = match (state >> 29) % 3 {
                0 => (x * 7 + y * 3) % 256,
                1 => {
                    if (x / 5 + y / 3) % 2 == 0 {
                        0
                    } else {
                        255
                    }
                }
                _ => state >> 24,
            };
            image::Luma([v as u8])
        })
    }

    #[test]
    fn same_bytes_as_the_image_crate() {
        let mut state = 12_345u32;
        let mut next = move |n: u32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state % n
        };
        for trial in 0..400 {
            let (w, h) = (1 + next(90), 1 + next(70));
            let (nw, nh) = (1 + next(120), 1 + next(100));
            let image = noise(w, h, trial);
            let want = imageops::resize(&image, nw, nh, FilterType::Triangle);
            let got = resize_triangle(&image, nw, nh);
            assert_eq!(got.dimensions(), want.dimensions());
            assert!(
                got.as_raw() == want.as_raw(),
                "trial {trial}: {w}x{h} -> {nw}x{nh}"
            );
        }
    }

    /// The sizes the cascade asks for: a frame shrunk by 1.1, 1.21, ...
    #[test]
    fn same_bytes_down_a_cascade_pyramid() {
        let image = noise(360, 288, 99);
        let mut scale = 1.0f32;
        while scale < 14.0 {
            let (nw, nh) = (
                (360.0 / scale).round() as u32,
                (288.0 / scale).round() as u32,
            );
            let want = imageops::resize(&image, nw, nh, FilterType::Triangle);
            assert!(
                resize_triangle(&image, nw, nh).as_raw() == want.as_raw(),
                "scale {scale}"
            );
            scale *= 1.1;
        }
    }

    #[test]
    fn the_wide_and_plain_paths_agree() {
        let image = noise(97, 61, 5);
        for (nw, nh) in [(40, 30), (97, 20), (13, 61), (150, 90)] {
            for taps in taps(61, nh) {
                let mut plain = vec![0.0f32; 97];
                let mut wide = vec![0.0f32; 97];
                accumulate_rows_plain(&mut plain, image.as_raw(), 97, &taps);
                accumulate_rows(&mut wide, image.as_raw(), 97, &taps);
                assert_eq!(
                    plain.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    wide.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{nw}x{nh}"
                );
            }
        }
    }

    #[test]
    fn rounding_matches_f32_round() {
        for i in 0..=(256 * 64) {
            let t = i as f32 / 64.0 - 1.0;
            let want = t.clamp(0.0, 255.0).round() as u8;
            assert_eq!(to_byte(t), want, "{t}");
        }
        for t in [0.49999997f32, 0.5, 254.5, 254.49998, 255.00002, -0.0, 1e-30] {
            assert_eq!(to_byte(t), t.clamp(0.0, 255.0).round() as u8, "{t}");
        }
    }
}
