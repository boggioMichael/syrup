//! Reading a face once it is found: where the eyes and mouth are, whether
//! the eyes are open, how open the mouth is — the small measurements that
//! blink counting, speaking detection and lip-shape matching are built on.
//!
//! The detectors in [`crate::cascade`] give a face box and eye boxes.
//! Everything else here is geometry and intensity: a mouth sits about one
//! inter-ocular distance below the eye line, an open eye shows a dark
//! iris band several rows tall while a closed one shows a thin dark line,
//! an open mouth shows a dark cavity. These are honest, cheap proxies —
//! they say *what the pixels show*, not what the person feels or means,
//! and they are sensitive to lighting, glasses and pose. A learned
//! landmark model behind the same [`FaceFrame`] is the natural upgrade.

use image::{GrayImage, RgbaImage};

use crate::cascade::{Cascade, CascadeOptions};
use crate::geometry::Rect;
use crate::threshold::{Channel, channel_image};

/// Where a face and its parts are in a frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceFrame {
    pub face: Rect,
    /// Eye boxes, left to right in the image; zero, one or two.
    pub eyes: Vec<Rect>,
    /// Where the mouth is expected: from the eyes when both were found,
    /// from face proportions otherwise.
    pub mouth: Rect,
    /// Whether `mouth` came from the eyes (true) or from the face box.
    pub mouth_from_eyes: bool,
}

impl FaceFrame {
    /// Midpoint between the eyes, or the face's upper-third centre.
    pub fn eye_line(&self) -> (f32, f32) {
        match self.eyes.as_slice() {
            [a, b] => {
                let (ax, ay) = a.center();
                let (bx, by) = b.center();
                ((ax + bx) / 2.0, (ay + by) / 2.0)
            }
            _ => (
                self.face.x as f32 + self.face.w as f32 / 2.0,
                self.face.y as f32 + self.face.h as f32 * 0.38,
            ),
        }
    }
}

/// Where to look for the mouth relative to the eyes: its centre is this
/// many inter-ocular distances below the eye line, and it is this wide
/// and tall in the same unit.
const MOUTH_BELOW_EYES: f32 = 1.05;
const MOUTH_WIDTH: f32 = 1.0;
const MOUTH_HEIGHT: f32 = 0.5;

/// Find the largest frontal face in `region` and lay out its parts.
pub fn locate(image: &RgbaImage, region: Rect, options: &CascadeOptions) -> Option<FaceFrame> {
    let faces = Cascade::frontal_face().detect_in(image, region, options);
    let face = faces.iter().max_by_key(|m| m.bounds.area())?.bounds;
    Some(lay_out(image, face, options))
}

/// Lay out the parts of a face already found.
pub fn lay_out(image: &RgbaImage, face: Rect, options: &CascadeOptions) -> FaceFrame {
    let eyes = find_eyes(image, face, options);
    let (mouth, mouth_from_eyes) = mouth_box(face, &eyes);
    FaceFrame {
        face,
        eyes,
        mouth: clip(image, mouth),
        mouth_from_eyes,
    }
}

/// The eye boxes inside `face`, left to right: at most two, and never two
/// on top of each other. An eye is between a tenth and half of the face's
/// width, whatever size limits `options` carries for faces.
pub fn find_eyes(image: &RgbaImage, face: Rect, options: &CascadeOptions) -> Vec<Rect> {
    // Eyes live in the upper half; the cascade sees them there far more
    // reliably than over the whole face.
    let upper = Rect {
        h: face.h / 2 + face.h / 8,
        ..face
    };
    let sized = CascadeOptions {
        min_size: face.w / 10,
        max_size: Some(face.w / 2),
        ..*options
    };
    let mut eyes: Vec<Rect> = Cascade::eye()
        .detect_in(image, upper, &sized)
        .into_iter()
        .map(|m| m.bounds)
        .collect();
    eyes.sort_by_key(|e| std::cmp::Reverse(e.area()));
    eyes.truncate(2);
    eyes.sort_by_key(|e| e.x);
    // Two boxes on top of each other are one eye seen twice.
    if let [a, b] = eyes.as_slice()
        && (a.center().0 - b.center().0).abs() < a.w.min(b.w) as f32 * 0.5
    {
        eyes.truncate(1);
    }
    eyes
}

/// Where the mouth should be: from two eyes when there are two (`true`),
/// from the face's proportions otherwise (`false`).
pub fn mouth_box(face: Rect, eyes: &[Rect]) -> (Rect, bool) {
    match eyes {
        [a, b] => {
            let (ax, ay) = a.center();
            let (bx, by) = b.center();
            let distance = ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt().max(4.0);
            let (cx, cy) = (
                (ax + bx) / 2.0,
                (ay + by) / 2.0 + distance * MOUTH_BELOW_EYES,
            );
            let (w, h) = (distance * MOUTH_WIDTH, distance * MOUTH_HEIGHT);
            (
                Rect {
                    x: (cx - w / 2.0).max(0.0).round() as u32,
                    y: (cy - h / 2.0).max(0.0).round() as u32,
                    w: w.round().max(2.0) as u32,
                    h: h.round().max(2.0) as u32,
                },
                true,
            )
        }
        _ => (
            Rect {
                x: face.x + face.w / 4,
                y: face.y + face.h * 13 / 20,
                w: face.w / 2,
                h: face.h / 4,
            },
            false,
        ),
    }
}

fn clip(image: &RgbaImage, r: Rect) -> Rect {
    let x = r.x.min(image.width().saturating_sub(1));
    let y = r.y.min(image.height().saturating_sub(1));
    Rect {
        x,
        y,
        w: r.w.min(image.width() - x).max(1),
        h: r.h.min(image.height() - y).max(1),
    }
}

/// The darkest tenth of a row, on average: where an iris, a lash line or a
/// mouth cavity is, this drops well below the skin around it.
fn row_floor(gray: &GrayImage, y: u32) -> f32 {
    let width = gray.width() as usize;
    let mut row: Vec<u8> = gray.as_raw()[y as usize * width..(y as usize + 1) * width].to_vec();
    let keep = (row.len() / 10).max(1);
    row.select_nth_unstable(keep - 1);
    row[..keep].iter().map(|&v| v as f32).sum::<f32>() / keep as f32
}

/// How open an eye is, in `[0, 1]`: the share of the box's rows that carry
/// a dark band (iris and lashes) rather than lid or skin. An open eye in
/// a well-placed box scores around 0.4–0.7; a closed one, where the dark
/// band collapses to a line, around 0.1–0.2. Compare against the eye's own
/// recent values rather than a fixed number: a blink is a brief drop to
/// well under half of the running level.
pub fn eye_openness(image: &RgbaImage, eye: Rect) -> f32 {
    let gray = channel_image(image, eye, Channel::Luma);
    let (width, height) = gray.dimensions();
    if width < 3 || height < 3 {
        return 0.0;
    }
    let floors: Vec<f32> = (0..height).map(|y| row_floor(&gray, y)).collect();
    let brightest = floors.iter().cloned().fold(0.0f32, f32::max);
    let darkest = floors.iter().cloned().fold(255.0f32, f32::min);
    if brightest - darkest < 12.0 {
        // No contrast at all: nothing eye-like in the box.
        return 0.0;
    }
    // Rows whose floor is closer to the darkest row than to the brightest.
    let threshold = darkest + (brightest - darkest) * 0.45;
    let dark_rows = floors.iter().filter(|&&f| f <= threshold).count();
    dark_rows as f32 / height as f32
}

/// The shape of a mouth, as three numbers a matcher can compare.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MouthShape {
    /// Share of the box's rows crossed by the dark cavity band, `[0, 1]`.
    pub openness: f32,
    /// Share of the box's columns the dark band spans, `[0, 1]`.
    pub width: f32,
    /// How dark the cavity is against the lips, `[0, 1]`.
    pub darkness: f32,
}

impl MouthShape {
    pub const CLOSED: MouthShape = MouthShape {
        openness: 0.0,
        width: 0.0,
        darkness: 0.0,
    };

    /// The shape as a feature vector, for [`crate::sequence`].
    pub fn features(&self) -> [f32; 3] {
        [self.openness, self.width, self.darkness]
    }
}

/// Measure the mouth in `mouth`: an open mouth shows a dark cavity (and
/// often bright teeth) between the lips; closed lips show neither.
pub fn mouth_shape(image: &RgbaImage, mouth: Rect) -> MouthShape {
    let gray = channel_image(image, mouth, Channel::Luma);
    let (width, height) = gray.dimensions();
    if width < 4 || height < 3 {
        return MouthShape::CLOSED;
    }
    let floors: Vec<f32> = (0..height).map(|y| row_floor(&gray, y)).collect();
    let brightest = floors.iter().cloned().fold(0.0f32, f32::max);
    let darkest = floors.iter().cloned().fold(255.0f32, f32::min);
    let contrast = brightest - darkest;
    if contrast < 20.0 {
        return MouthShape::CLOSED;
    }
    let threshold = darkest + contrast * 0.4;
    let dark_rows: Vec<u32> = (0..height)
        .filter(|&y| floors[y as usize] <= threshold)
        .collect();
    let openness = dark_rows.len() as f32 / height as f32;
    // Width: in the darkest row, how many columns are within the cavity's
    // darkness.
    let widest = dark_rows
        .iter()
        .map(|&y| {
            let row =
                &gray.as_raw()[y as usize * width as usize..(y as usize + 1) * width as usize];
            let dark = row
                .iter()
                .filter(|&&v| (v as f32) <= threshold + 10.0)
                .count();
            dark as f32 / width as f32
        })
        .fold(0.0f32, f32::max);
    MouthShape {
        openness,
        width: widest,
        darkness: (contrast / 255.0).min(1.0),
    }
}

/// The mouth's appearance as a small standardised patch: the region in
/// grey, resampled to `width` x `height`, then shifted to zero mean and
/// scaled to unit variance so lighting and skin tone drop out and only the
/// shape of the lips, teeth and cavity remain. Two patches compare by
/// Euclidean distance; scaled by `1 / sqrt(width * height)` so the
/// distance between unrelated mouths is about 1.4 whatever the size.
///
/// This is what a matcher should compare across frames rather than the
/// three numbers of [`mouth_shape`] alone: the numbers say how open, the
/// patch says how — rounded, spread, teeth showing.
pub fn mouth_patch(image: &RgbaImage, mouth: Rect, width: u32, height: u32) -> Vec<f32> {
    let gray = channel_image(image, mouth, Channel::Luma);
    let n = (width * height) as usize;
    if gray.width() == 0 || gray.height() == 0 || n == 0 {
        return vec![0.0; n];
    }
    let small =
        image::imageops::resize(&gray, width, height, image::imageops::FilterType::Triangle);
    let values: Vec<f32> = small.as_raw().iter().map(|&v| f32::from(v)).collect();
    let mean = values.iter().sum::<f32>() / n as f32;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n as f32;
    if variance < 1.0 {
        return vec![0.0; n];
    }
    let scale = 1.0 / (variance.sqrt() * (n as f32).sqrt());
    values.iter().map(|v| (v - mean) * scale).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn astronaut() -> RgbaImage {
        image::load_from_memory(include_bytes!("../tests/fixtures/astronaut_320.jpg"))
            .expect("fixture decodes")
            .to_rgba8()
    }

    #[test]
    fn the_astronauts_face_is_laid_out_from_her_eyes() {
        let image = astronaut();
        let whole = Rect {
            x: 0,
            y: 0,
            w: 320,
            h: 320,
        };
        let frame = locate(&image, whole, &CascadeOptions::default()).expect("a face");
        assert_eq!(frame.eyes.len(), 2, "{frame:?}");
        assert!(frame.mouth_from_eyes);
        // The mouth box sits below the eyes, inside the face, roughly centred.
        let (ex, ey) = frame.eye_line();
        let (mx, my) = frame.mouth.center();
        assert!(
            my > ey + 15.0 && my < frame.face.y as f32 + frame.face.h as f32,
            "{frame:?}"
        );
        assert!((mx - ex).abs() < 6.0, "{frame:?}");
        assert!(frame.mouth.w >= 20 && frame.mouth.h >= 8, "{frame:?}");
    }

    #[test]
    fn open_eyes_score_higher_than_painted_over_ones() {
        let image = astronaut();
        let whole = Rect {
            x: 0,
            y: 0,
            w: 320,
            h: 320,
        };
        let frame = locate(&image, whole, &CascadeOptions::default()).unwrap();
        let open: Vec<f32> = frame
            .eyes
            .iter()
            .map(|&e| eye_openness(&image, e))
            .collect();
        assert!(open.iter().all(|&o| o > 0.25), "open eyes: {open:?}");

        // Paint the eyes over with the skin tone beside them: closed lids.
        let mut closed = image.clone();
        for eye in &frame.eyes {
            let skin = *image.get_pixel(eye.x + eye.w / 2, eye.y + eye.h + 6);
            for y in eye.y + eye.h / 4..eye.y + eye.h * 3 / 4 {
                for x in eye.x..eye.x + eye.w {
                    closed.put_pixel(x, y, skin);
                }
            }
            // A thin lash line.
            for x in eye.x + 3..eye.x + eye.w - 3 {
                closed.put_pixel(x, eye.y + eye.h / 2, image::Rgba([60, 40, 40, 255]));
            }
        }
        let shut: Vec<f32> = frame
            .eyes
            .iter()
            .map(|&e| eye_openness(&closed, e))
            .collect();
        for (o, s) in open.iter().zip(&shut) {
            assert!(*s < *o * 0.6, "open {o} vs painted shut {s}");
        }
    }

    #[test]
    fn a_dark_cavity_reads_as_an_open_mouth() {
        let image = astronaut();
        let whole = Rect {
            x: 0,
            y: 0,
            w: 320,
            h: 320,
        };
        let frame = locate(&image, whole, &CascadeOptions::default()).unwrap();
        let smiling = mouth_shape(&image, frame.mouth);

        // Paint a dark open-mouth ellipse in the mouth box.
        let mut opened = image.clone();
        let m = frame.mouth;
        let (cx, cy) = m.center();
        for y in m.y..m.y + m.h {
            for x in m.x..m.x + m.w {
                let dx = (x as f32 - cx) / (m.w as f32 / 2.0);
                let dy = (y as f32 - cy) / (m.h as f32 / 2.0);
                if dx * dx + dy * dy < 0.8 {
                    opened.put_pixel(x, y, image::Rgba([30, 15, 15, 255]));
                }
            }
        }
        let open = mouth_shape(&opened, frame.mouth);
        assert!(
            open.openness > smiling.openness + 0.3,
            "open {open:?} vs smiling {smiling:?}"
        );
        assert!(open.width > 0.5, "{open:?}");

        // Paint the mouth over with skin: closed.
        let mut shut = image.clone();
        let skin = *image.get_pixel(m.x + m.w / 2, m.y.saturating_sub(4));
        for y in m.y..m.y + m.h {
            for x in m.x..m.x + m.w {
                shut.put_pixel(x, y, skin);
            }
        }
        assert_eq!(mouth_shape(&shut, frame.mouth), MouthShape::CLOSED);
    }

    #[test]
    fn mouth_patches_ignore_brightness_and_tell_shapes_apart() {
        let image = astronaut();
        let whole = Rect {
            x: 0,
            y: 0,
            w: 320,
            h: 320,
        };
        let frame = locate(&image, whole, &CascadeOptions::default()).unwrap();
        let patch = mouth_patch(&image, frame.mouth, 16, 8);
        assert_eq!(patch.len(), 128);
        let norm: f32 = patch.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "unit length, got {norm}");

        // The same mouth, darker: the same patch.
        let darker = image::imageops::brighten(&image, -60);
        let dark_patch = mouth_patch(&darker, frame.mouth, 16, 8);
        let distance = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt()
        };
        assert!(
            distance(&patch, &dark_patch) < 0.2,
            "{}",
            distance(&patch, &dark_patch)
        );

        // A painted-open mouth: a different patch.
        let mut opened = image.clone();
        let m = frame.mouth;
        let (cx, cy) = m.center();
        for y in m.y..m.y + m.h {
            for x in m.x..m.x + m.w {
                let dx = (x as f32 - cx) / (m.w as f32 / 2.0);
                let dy = (y as f32 - cy) / (m.h as f32 / 2.0);
                if dx * dx + dy * dy < 0.8 {
                    opened.put_pixel(x, y, image::Rgba([30, 15, 15, 255]));
                }
            }
        }
        let open_patch = mouth_patch(&opened, frame.mouth, 16, 8);
        assert!(
            distance(&patch, &open_patch) > 0.8,
            "{}",
            distance(&patch, &open_patch)
        );

        // A flat region has no shape: all zeros rather than noise.
        let flat = RgbaImage::from_pixel(40, 20, image::Rgba([120, 100, 90, 255]));
        assert!(
            mouth_patch(
                &flat,
                Rect {
                    x: 0,
                    y: 0,
                    w: 40,
                    h: 20
                },
                16,
                8
            )
            .iter()
            .all(|&v| v == 0.0)
        );
    }

    #[test]
    fn a_face_without_eyes_still_gets_a_mouth_box() {
        let image = astronaut();
        // Pretend the detector gave a face box but the eyes are missing:
        // lay_out over a box where the eye cascade finds nothing.
        let box_without_eyes = Rect {
            x: 0,
            y: 200,
            w: 60,
            h: 60,
        };
        let frame = lay_out(&image, box_without_eyes, &CascadeOptions::default());
        assert!(frame.eyes.is_empty());
        assert!(!frame.mouth_from_eyes);
        assert_eq!(
            frame.mouth,
            Rect {
                x: 15,
                y: 239,
                w: 30,
                h: 15
            }
        );
    }
}
