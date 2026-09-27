//! The implementations behind each [`Plan`](super::Plan): plain functions
//! over the primitives, called both by the in-process executor and by the
//! source that [`codegen`](super::codegen) writes, so the two can never
//! disagree.

use image::{Rgba, RgbaImage};

use crate::cascade::{Cascade, CascadeOptions};
use crate::color::is_color_pixel;
use crate::detection::{Confidence, Detection, Reliability};
use crate::geometry::{Rect, find_color_regions, find_text_block, measure_bar_fill};
use crate::intent::{Match, Outcome};
use crate::ocr::{OcrConfig, is_ocr_available, ocr_region_with};

/// Saturation and value below which a pixel is not "coloured" for the
/// colour plans: greys, shadows and near-black are never a red bar.
const COLOR_SATURATION: f32 = 0.35;
const COLOR_VALUE: f32 = 0.35;

/// A bar is at least this many times wider than tall.
const BAR_ASPECT: u32 = 3;

fn clip(image: &RgbaImage, region: Rect) -> Option<Rect> {
    let x_end = region.x.saturating_add(region.w).min(image.width());
    let y_end = region.y.saturating_add(region.h).min(image.height());
    (region.x < x_end && region.y < y_end).then(|| Rect {
        x: region.x,
        y: region.y,
        w: x_end - region.x,
        h: y_end - region.y,
    })
}

fn centre(r: Rect) -> (f32, f32) {
    r.center()
}

/// Faces in `region`, largest support first.
pub fn find_faces(image: &RgbaImage, region: Rect, options: &CascadeOptions) -> Detection<Outcome> {
    let Some(region) = clip(image, region) else {
        return Detection::missing("find_face", "the region lies outside the image");
    };
    let found = Cascade::frontal_face().detect_in(image, region, options);
    let matches: Vec<Match> = found
        .iter()
        .map(|m| Match {
            bounds: m.bounds,
            // Support saturates: 3 agreeing windows is weak evidence, 30 is
            // as sure as this detector gets.
            score: 1.0 - (-(m.neighbors as f32) / 10.0).exp(),
            centre: centre(m.bounds),
        })
        .collect();
    let best = matches.first().map(|m| m.score).unwrap_or(0.0);
    let reliability = if found.iter().any(|m| m.neighbors >= 10) {
        Reliability::Corroborated
    } else {
        Reliability::Heuristic
    };
    let mut detection = Detection::found(
        Outcome::Matches(matches),
        Confidence::new(if found.is_empty() { 0.6 } else { best }),
        "find_face",
        reliability,
    );
    if found.is_empty() {
        detection.failure_reason = Some("no frontal face; the cascade sees roughly upright, roughly front-on faces of 20px or more".into());
    }
    detection
}

/// Regions of one colour, largest first.
pub fn find_blobs(image: &RgbaImage, region: Rect, hue: (f32, f32)) -> Detection<Outcome> {
    let Some(region) = clip(image, region) else {
        return Detection::missing("find_blob", "the region lies outside the image");
    };
    let mut rects = find_color_regions(image, region, hue, COLOR_SATURATION, COLOR_VALUE);
    rects.sort_by_key(|r| std::cmp::Reverse(r.area()));
    let largest = rects.first().map(|r| r.area()).unwrap_or(1).max(1) as f32;
    let matches: Vec<Match> = rects
        .iter()
        .map(|&r| Match {
            bounds: r,
            score: r.area() as f32 / largest,
            centre: centre(r),
        })
        .collect();
    let mut detection = Detection::found(
        Outcome::Matches(matches),
        Confidence::new(if rects.is_empty() { 0.6 } else { 0.8 }),
        "find_blob",
        Reliability::Heuristic,
    );
    if rects.is_empty() {
        detection.failure_reason = Some("no region of that colour".into());
    }
    detection
}

/// Horizontal bars of one colour: coloured regions much wider than tall.
pub fn find_bars(image: &RgbaImage, region: Rect, hue: (f32, f32)) -> Detection<Outcome> {
    let mut detection = find_blobs(image, region, hue);
    if let Some(Outcome::Matches(matches)) = &mut detection.value {
        matches.retain(|m| m.bounds.w >= m.bounds.h * BAR_ASPECT);
        if matches.is_empty() {
            detection.failure_reason = Some("no bar-shaped region of that colour".into());
        }
    }
    detection.source = "find_bar";
    detection
}

/// The block of text-like pixels in `region`, as one match.
pub fn find_text(image: &RgbaImage, region: Rect) -> Detection<Outcome> {
    let Some(region) = clip(image, region) else {
        return Detection::missing("find_text", "the region lies outside the image");
    };
    match find_text_block(image, region) {
        Some(block) => Detection::found(
            Outcome::Matches(vec![Match {
                bounds: block,
                score: 0.7,
                centre: centre(block),
            }]),
            Confidence::new(0.7),
            "find_text",
            Reliability::Heuristic,
        ),
        None => {
            let mut detection = Detection::found(
                Outcome::Matches(Vec::new()),
                Confidence::new(0.6),
                "find_text",
                Reliability::Heuristic,
            );
            detection.failure_reason = Some("too few text-like pixels".into());
            detection
        }
    }
}

/// How full the largest bar of one colour is, as a percentage of its
/// track.
pub fn measure_bar(image: &RgbaImage, region: Rect, hue: (f32, f32)) -> Detection<Outcome> {
    let Some(region) = clip(image, region) else {
        return Detection::missing("measure_bar", "the region lies outside the image");
    };
    let bars = find_bars(image, region, hue);
    let Some(Outcome::Matches(bars)) = bars.value else {
        return Detection::missing("measure_bar", bars.failure_reason.unwrap_or_default());
    };
    let Some(bar) = bars.first() else {
        return Detection::missing("measure_bar", "no bar of that colour to measure");
    };
    let is_fill = |p: &Rgba<u8>| is_color_pixel(p, hue, COLOR_SATURATION, COLOR_VALUE);
    match measure_bar_fill(image, bar.bounds, region, is_fill) {
        Some(percent) => Detection::found(
            Outcome::Scalar(percent),
            Confidence::new(0.8),
            "measure_bar",
            Reliability::Heuristic,
        ),
        None => Detection::missing(
            "measure_bar",
            "found the fill but not the track it sits in, so the percentage would be a guess",
        ),
    }
}

/// Text in `region`, through the OCR engine.
pub fn read_text(image: &RgbaImage, region: Rect) -> Detection<Outcome> {
    let Some(region) = clip(image, region) else {
        return Detection::missing("read_text", "the region lies outside the image");
    };
    if !is_ocr_available() {
        return Detection::missing(
            "read_text",
            "no OCR engine: install Tesseract or set TESSERACT_BIN",
        );
    }
    match ocr_region_with(
        image,
        region.x,
        region.y,
        region.w,
        region.h,
        &OcrConfig::default(),
    ) {
        Some(result) if !result.text.trim().is_empty() => {
            let confidence = result.confidence();
            Detection::found(
                Outcome::Text(result.text.trim().to_string()),
                confidence,
                "read_text",
                Reliability::Heuristic,
            )
        }
        _ => Detection::missing("read_text", "the engine recognised no text"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn astronaut() -> RgbaImage {
        image::load_from_memory(include_bytes!("../../tests/fixtures/astronaut_320.jpg"))
            .expect("fixture decodes")
            .to_rgba8()
    }

    #[test]
    fn faces_are_found_with_saturating_scores() {
        let image = astronaut();
        let whole = Rect {
            x: 0,
            y: 0,
            w: 320,
            h: 320,
        };
        let detection = find_faces(&image, whole, &CascadeOptions::default());
        let Some(Outcome::Matches(faces)) = detection.value else {
            panic!("expected matches, got {detection:?}");
        };
        assert_eq!(faces.len(), 1);
        assert!(faces[0].score > 0.9, "score {}", faces[0].score);
        assert_eq!(detection.reliability, Reliability::Corroborated);
        assert!(faces[0].bounds.x > 100 && faces[0].bounds.x < 120);
    }

    #[test]
    fn a_region_outside_the_image_cannot_be_searched() {
        let image = astronaut();
        let outside = Rect {
            x: 400,
            y: 400,
            w: 10,
            h: 10,
        };
        for detection in [
            find_faces(&image, outside, &CascadeOptions::default()),
            find_blobs(&image, outside, (0.0, 30.0)),
            measure_bar(&image, outside, (0.0, 30.0)),
            find_text(&image, outside),
        ] {
            assert!(detection.value.is_none());
            assert!(detection.failure_reason.unwrap().contains("outside"));
        }
    }

    #[test]
    fn bars_are_the_wide_blobs() {
        let mut image = RgbaImage::from_pixel(100, 100, Rgba([20, 20, 20, 255]));
        for y in 10..18 {
            for x in 10..90 {
                image.put_pixel(x, y, Rgba([30, 200, 40, 255]));
            }
        }
        for y in 40..70 {
            for x in 40..70 {
                image.put_pixel(x, y, Rgba([30, 200, 40, 255]));
            }
        }
        let whole = Rect {
            x: 0,
            y: 0,
            w: 100,
            h: 100,
        };
        let blobs = find_blobs(&image, whole, (70.0, 170.0));
        assert_eq!(
            blobs.value.as_ref().map(|o| match o {
                Outcome::Matches(m) => m.len(),
                _ => 0,
            }),
            Some(2)
        );
        let bars = find_bars(&image, whole, (70.0, 170.0));
        let Some(Outcome::Matches(bars)) = bars.value else {
            panic!()
        };
        assert_eq!(bars.len(), 1);
        assert_eq!(
            bars[0].bounds,
            Rect {
                x: 10,
                y: 10,
                w: 80,
                h: 8
            }
        );
    }

    #[test]
    fn an_absent_bar_is_missing_not_zero() {
        let image = RgbaImage::from_pixel(50, 50, Rgba([20, 20, 20, 255]));
        let detection = measure_bar(
            &image,
            Rect {
                x: 0,
                y: 0,
                w: 50,
                h: 50,
            },
            (340.0, 20.0),
        );
        assert!(detection.value.is_none());
        assert!(detection.failure_reason.unwrap().contains("no bar"));
    }
}
