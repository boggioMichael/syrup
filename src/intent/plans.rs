//! The implementations behind each [`Plan`](super::Plan): plain functions
//! over the primitives, called both by the in-process executor and by the
//! source that [`codegen`](super::codegen) writes, so the two can never
//! disagree.

use image::{Rgba, RgbaImage};

use crate::cascade::{Cascade, CascadeMatch, CascadeOptions};
use crate::color::is_color_pixel;
use crate::components::{Connectivity, label_pixels};
use crate::detection::{Confidence, Detection, Reliability};
use crate::geometry::{Rect, find_text_block, measure_bar_fill};
use crate::intent::{Match, Outcome, Pick, State};
use crate::motion::{MotionConfig, MotionDetector};
use crate::ocr::{OcrConfig, is_ocr_available, ocr_region_with};
use crate::template::{self, Template};
use crate::threshold::Channel;
use crate::tracking::ObjectTracker;

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

/// A found-nothing result: the search ran, the list is empty, and the
/// reason says what would have counted.
fn nothing(source: &'static str, reason: &str) -> Detection<Outcome> {
    let mut detection = Detection::found(
        Outcome::Matches(Vec::new()),
        Confidence::new(0.6),
        source,
        Reliability::Heuristic,
    );
    detection.failure_reason = Some(reason.to_string());
    detection
}

/// Cascade matches as intent matches. Support saturates: 3 agreeing
/// windows is weak evidence, 30 is as sure as a cascade gets.
fn cascade_matches(found: &[CascadeMatch]) -> Vec<Match> {
    found
        .iter()
        .map(|m| Match::new(m.bounds, 1.0 - (-(m.neighbors as f32) / 10.0).exp()))
        .collect()
}

fn cascade_detection(
    source: &'static str,
    found: &[CascadeMatch],
    none: &str,
) -> Detection<Outcome> {
    if found.is_empty() {
        return nothing(source, none);
    }
    let matches = cascade_matches(found);
    let best = matches[0].score;
    let reliability = if found.iter().any(|m| m.neighbors >= 10) {
        Reliability::Corroborated
    } else {
        Reliability::Heuristic
    };
    Detection::found(
        Outcome::Matches(matches),
        Confidence::new(best),
        source,
        reliability,
    )
}

/// Faces in `region`, best-supported first: frontal, or turned to the side
/// with `profile`.
pub fn find_faces(
    image: &RgbaImage,
    region: Rect,
    profile: bool,
    options: &CascadeOptions,
) -> Detection<Outcome> {
    let Some(region) = clip(image, region) else {
        return Detection::missing("find_face", "the region lies outside the image");
    };
    let cascade = if profile {
        Cascade::profile_face()
    } else {
        Cascade::frontal_face()
    };
    let found = cascade.detect_in(image, region, options);
    cascade_detection(
        "find_face",
        &found,
        if profile {
            "no profile face; the cascade sees faces turned to one side, 20px or more"
        } else {
            "no frontal face; the cascade sees roughly upright, roughly front-on faces of 20px or more"
        },
    )
}

/// Eyes in `region`: the eye cascade run inside each frontal face, so
/// eye-shaped texture elsewhere does not count. Best-supported first.
pub fn find_eyes(image: &RgbaImage, region: Rect, options: &CascadeOptions) -> Detection<Outcome> {
    let Some(region) = clip(image, region) else {
        return Detection::missing("find_eye", "the region lies outside the image");
    };
    let faces = Cascade::frontal_face().detect_in(image, region, options);
    if faces.is_empty() {
        return nothing("find_eye", "no frontal face to look for eyes in");
    }
    let mut found: Vec<CascadeMatch> = Vec::new();
    for face in &faces {
        // Eyes sit in the upper half of a face; searching only there halves
        // the work and the false positives.
        let upper = Rect {
            h: face.bounds.h / 2 + face.bounds.h / 8,
            ..face.bounds
        };
        found.extend(Cascade::eye().detect_in(image, upper, options));
    }
    found.sort_by_key(|m| std::cmp::Reverse(m.neighbors));
    cascade_detection(
        "find_eye",
        &found,
        "faces were found but no eyes inside them",
    )
}

/// Every place `template` appears in `region`, strongest first.
pub fn find_icon(image: &RgbaImage, region: Rect, template: &RgbaImage) -> Detection<Outcome> {
    let Some(region) = clip(image, region) else {
        return Detection::missing("find_icon", "the region lies outside the image");
    };
    let Some(template) = Template::from_image(template, Channel::Luma) else {
        return Detection::missing(
            "find_icon",
            "the picture has no structure to match (uniform or empty)",
        );
    };
    let found = template::find_all(image, region, &template, ICON_MIN_SCORE);
    if found.is_empty() {
        return nothing(
            "find_icon",
            "nothing correlated with the picture strongly enough",
        );
    }
    let matches: Vec<Match> = found
        .iter()
        .map(|m| Match {
            bounds: m.bounds,
            score: m.score,
            centre: m.centre,
            id: None,
        })
        .collect();
    let best = matches[0].score;
    Detection::found(
        Outcome::Matches(matches),
        Confidence::new(best),
        "find_icon",
        if best > 0.9 {
            Reliability::Corroborated
        } else {
            Reliability::Heuristic
        },
    )
}

/// Normalised correlation below which a template match is noise.
const ICON_MIN_SCORE: f32 = 0.7;

/// Regions that changed since the previous call, with the motion
/// detector's stable ids. The first call only records the frame.
pub fn find_motion(state: &mut State, image: &RgbaImage, region: Rect) -> Detection<Outcome> {
    let Some(region) = clip(image, region) else {
        return Detection::missing("find_motion", "the region lies outside the image");
    };
    let detector = state
        .motion
        .get_or_insert_with(|| MotionDetector::new(MotionConfig::default()));
    let view = image::imageops::crop_imm(image, region.x, region.y, region.w, region.h).to_image();
    let detection = detector.detect(&view);
    let Some(blobs) = detection.value else {
        return Detection::missing("find_motion", detection.failure_reason.unwrap_or_default());
    };
    let matches: Vec<Match> = blobs
        .iter()
        .map(|b| {
            let bounds = Rect {
                x: b.bounds.x + region.x,
                y: b.bounds.y + region.y,
                ..b.bounds
            };
            Match {
                id: Some(b.id),
                ..Match::new(bounds, if b.is_predicted { 0.4 } else { 0.8 })
            }
        })
        .collect();
    let mut result = Detection::found(
        Outcome::Matches(matches),
        detection.confidence,
        "find_motion",
        detection.reliability,
    );
    result.failure_reason = detection.failure_reason;
    result
}

/// The matches of `found` with a stable id per object across calls.
pub fn track(state: &mut State, found: Detection<Outcome>) -> Detection<Outcome> {
    let Some(Outcome::Matches(matches)) = &found.value else {
        return found;
    };
    let tracker = state
        .tracker
        .get_or_insert_with(|| ObjectTracker::new(TRACK_REACH, TRACK_GRACE));
    let detections: Vec<(f32, f32, f32, f32)> = matches
        .iter()
        .map(|m| (m.centre.0, m.centre.1, m.bounds.w as f32, m.bounds.h as f32))
        .collect();
    let tracks = tracker.update(&detections);
    let tracked: Vec<Match> = tracks
        .iter()
        .map(|t| {
            let bounds = Rect {
                x: (t.position.x - t.width / 2.0).max(0.0).round() as u32,
                y: (t.position.y - t.height / 2.0).max(0.0).round() as u32,
                w: t.width.max(1.0).round() as u32,
                h: t.height.max(1.0).round() as u32,
            };
            Match {
                bounds,
                score: t.confidence.value(),
                centre: (t.position.x, t.position.y),
                id: Some(t.id),
            }
        })
        .collect();
    let mut result = Detection::found(
        Outcome::Matches(tracked),
        found.confidence,
        found.source,
        found.reliability,
    );
    result.failure_reason = found.failure_reason;
    result
}

/// Pixels a tracked object may move between calls and still be itself,
/// and calls it may go unseen before its id is retired.
const TRACK_REACH: f32 = 64.0;
const TRACK_GRACE: u32 = 5;

/// Keep only the largest or smallest match.
pub fn keep(pick: Pick, mut found: Detection<Outcome>) -> Detection<Outcome> {
    if let Some(Outcome::Matches(matches)) = &mut found.value {
        let chosen = match pick {
            Pick::Largest => matches.iter().max_by_key(|m| m.bounds.area()),
            Pick::Smallest => matches.iter().min_by_key(|m| m.bounds.area()),
        }
        .cloned();
        *matches = chosen.into_iter().collect();
    }
    found
}

/// How many matches `found` holds, as a number.
pub fn count(found: Detection<Outcome>) -> Detection<Outcome> {
    let Detection {
        value,
        confidence,
        timestamp,
        source,
        reliability,
        failure_reason,
    } = found;
    Detection {
        value: value.map(|o| match o {
            Outcome::Matches(m) => Outcome::Scalar(m.len() as f32),
            other => other,
        }),
        confidence,
        timestamp,
        source,
        reliability,
        failure_reason,
    }
}

/// Connected regions of one colour, largest first. Each match's score is
/// its area relative to the largest, and its centre is the region's
/// centroid rather than the middle of its box.
pub fn find_blobs(image: &RgbaImage, region: Rect, hue: (f32, f32)) -> Detection<Outcome> {
    let Some(region) = clip(image, region) else {
        return Detection::missing("find_blob", "the region lies outside the image");
    };
    let (_, components) = label_pixels(image, region, Connectivity::Eight, |p| {
        is_color_pixel(p, hue, COLOR_SATURATION, COLOR_VALUE)
    });
    let mut components: Vec<_> = components
        .into_iter()
        .filter(|c| c.area >= MIN_BLOB_AREA)
        .collect();
    components.sort_by_key(|c| std::cmp::Reverse(c.area));
    let largest = components.first().map(|c| c.area).unwrap_or(1).max(1) as f32;
    let matches: Vec<Match> = components
        .iter()
        .map(|c| Match {
            centre: c.centroid,
            ..Match::new(c.bounds, c.area as f32 / largest)
        })
        .collect();
    if matches.is_empty() {
        return nothing("find_blob", "no region of that colour");
    }
    Detection::found(
        Outcome::Matches(matches),
        Confidence::new(0.8),
        "find_blob",
        Reliability::Heuristic,
    )
}

/// Fewer pixels than this is speckle, not a region.
const MIN_BLOB_AREA: u32 = 16;

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
            Outcome::Matches(vec![Match::new(block, 0.7)]),
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
        let detection = find_faces(&image, whole, false, &CascadeOptions::default());
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
            find_faces(&image, outside, false, &CascadeOptions::default()),
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
