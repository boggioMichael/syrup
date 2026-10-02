//! Reference semantics for plans. Validation runs every compiled module
//! against this interpreter; it never serves real calls. It is written
//! independently of the code generator, and finds colours with the core's
//! own primitives, so a slip in either shows up as a mismatch.

use std::cmp::Ordering;

use image::{Rgba, RgbaImage};
use syrup::color::is_color_pixel;
use syrup::geometry::segment_row;

use crate::abi::{SYRUP_MAX_KEYPOINTS, SyrupDetection, SyrupParams, SyrupRect, SyrupRun};
use crate::catalog::Capability;
use crate::contract::ImageInput;
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::intent::{OrderKey, PixelRect, RegionSpec};
use crate::plan::{Measurement, Plan, Step};

/// What the host gives a plan: detectors, the core's region grouping, and
/// the core's measurements.
pub trait Host {
    /// `view` is in input pixels and never empty.
    fn detect(&mut self, capability: Capability, view: PixelRect) -> Result<Vec<SyrupDetection>>;
    fn group(&mut self, runs: &[SyrupRun], min_height: u32, max_gap: u32) -> Vec<SyrupRect>;
    /// One value per box, NaN where it cannot be measured; `boxes` is never
    /// empty and is in `view`'s pixels.
    fn measure(
        &mut self,
        what: Measurement,
        view: PixelRect,
        boxes: &[SyrupDetection],
    ) -> Result<Vec<f32>>;
}

/// The core's grouping, as the real host and the validator's mock use it.
pub fn group_runs(runs: &[SyrupRun], min_height: u32, max_gap: u32) -> Vec<SyrupRect> {
    let rows = runs.iter().map(|r| (r.y, r.x0, r.x1)).collect();
    syrup::geometry::group_segments(rows, min_height, max_gap)
        .into_iter()
        .map(|r| SyrupRect {
            x: r.x,
            y: r.y,
            w: r.w,
            h: r.h,
        })
        .collect()
}

/// The pixels of `view`, as RGBA.
fn window(image: &ImageInput<'_>, view: PixelRect) -> RgbaImage {
    let (data, stride, c) = (image.data(), image.stride(), image.channels() as usize);
    RgbaImage::from_fn(view.w, view.h, |x, y| {
        let at = (view.y + y) as usize * stride + (view.x + x) as usize * c;
        let p = &data[at..at + c];
        Rgba(match c {
            1 => [p[0], p[0], p[0], 255],
            3 => [p[0], p[1], p[2], 255],
            _ => [p[0], p[1], p[2], p[3]],
        })
    })
}

enum Value {
    View(PixelRect),
    Boxes(Vec<SyrupDetection>),
    Done,
}

fn invalid(reason: impl Into<String>) -> SyrupError {
    SyrupError::new(Stage::Plan, ErrorKind::InvalidPlan, reason)
}

pub fn run(
    plan: &Plan,
    image: &ImageInput<'_>,
    params: &SyrupParams,
    host: &mut dyn Host,
) -> Result<Vec<SyrupDetection>> {
    let (width, height) = (image.width(), image.height());
    let mut values: Vec<Value> = Vec::with_capacity(plan.steps.len());
    let view = |values: &[Value], at: usize| -> Result<PixelRect> {
        match values.get(at) {
            Some(Value::View(rect)) => Ok(*rect),
            _ => Err(invalid(format!("value {at} is not a view"))),
        }
    };
    let take = |values: &mut [Value], at: usize| -> Result<Vec<SyrupDetection>> {
        match values
            .get_mut(at)
            .map(|v| std::mem::replace(v, Value::Done))
        {
            Some(Value::Boxes(boxes)) => Ok(boxes),
            _ => Err(invalid(format!(
                "value {at} is not boxes, or was already consumed"
            ))),
        }
    };

    for step in &plan.steps {
        let value = match *step {
            Step::Input => Value::View(PixelRect {
                x: 0,
                y: 0,
                w: width,
                h: height,
            }),
            Step::SelectRegion { view: at, region } => {
                let parent = view(&values, at)?;
                let rect = match region {
                    RegionSpec::Fixed { rect } => {
                        let local = rect.pixels(parent.w, parent.h);
                        PixelRect {
                            x: parent.x + local.x,
                            y: parent.y + local.y,
                            w: local.w,
                            h: local.h,
                        }
                    }
                    RegionSpec::Caller => {
                        if params.has_region != 1 {
                            return Err(invalid("a caller region is required"));
                        }
                        let x0 = params.region_x.min(parent.w);
                        let y0 = params.region_y.min(parent.h);
                        let x1 = (params.region_x as u64 + params.region_w as u64)
                            .min(parent.w as u64) as u32;
                        let y1 = (params.region_y as u64 + params.region_h as u64)
                            .min(parent.h as u64) as u32;
                        PixelRect {
                            x: parent.x + x0,
                            y: parent.y + y0,
                            w: x1.saturating_sub(x0),
                            h: y1.saturating_sub(y0),
                        }
                    }
                };
                Value::View(rect)
            }
            Step::Detect {
                view: at,
                capability,
            } => {
                let rect = view(&values, at)?;
                if rect.w == 0 || rect.h == 0 {
                    Value::Boxes(Vec::new())
                } else {
                    Value::Boxes(host.detect(capability, rect)?)
                }
            }
            Step::FindColor {
                view: at,
                hue,
                min_saturation_pct,
                min_value_pct,
                min_run,
                min_height,
                max_gap,
            } => {
                let rect = view(&values, at)?;
                if rect.w == 0 || rect.h == 0 {
                    Value::Boxes(Vec::new())
                } else {
                    let pixels = window(image, rect);
                    let hue = (hue.0 as f32, hue.1 as f32);
                    let (sat, val) = (
                        min_saturation_pct as f32 / 100.0,
                        min_value_pct as f32 / 100.0,
                    );
                    let matches = |p: &Rgba<u8>| is_color_pixel(p, hue, sat, val);
                    let mut runs = vec![];
                    for y in 0..rect.h {
                        for (x0, x1) in segment_row(&pixels, y, 0, rect.w - 1, min_run, matches) {
                            runs.push(SyrupRun { y, x0, x1 });
                        }
                    }
                    let boxes = host
                        .group(&runs, min_height, max_gap)
                        .into_iter()
                        .map(|r| {
                            let mut hits = 0u32;
                            for y in r.y..r.y + r.h {
                                for x in r.x..r.x + r.w {
                                    hits += matches(pixels.get_pixel(x, y)) as u32;
                                }
                            }
                            SyrupDetection {
                                x: r.x as f32,
                                y: r.y as f32,
                                w: r.w as f32,
                                h: r.h as f32,
                                score: hits as f32 / (r.w * r.h) as f32,
                                value: 0.0,
                                n_keypoints: 0,
                                keypoints: [0.0; 2 * SYRUP_MAX_KEYPOINTS],
                                payload: 0,
                            }
                        })
                        .collect();
                    Value::Boxes(boxes)
                }
            }
            Step::Whole { view: at } => {
                let rect = view(&values, at)?;
                let mut boxes = vec![];
                if rect.w > 0 && rect.h > 0 {
                    boxes.push(SyrupDetection {
                        x: 0.0,
                        y: 0.0,
                        w: rect.w as f32,
                        h: rect.h as f32,
                        score: 1.0,
                        value: 0.0,
                        n_keypoints: 0,
                        keypoints: [0.0; 2 * SYRUP_MAX_KEYPOINTS],
                        payload: 0,
                    });
                }
                Value::Boxes(boxes)
            }
            Step::Measure {
                boxes,
                view: at,
                what,
            } => {
                let rect = view(&values, at)?;
                let mut kept = take(&mut values, boxes)?;
                if !kept.is_empty() {
                    let local: Vec<SyrupDetection> = kept
                        .iter()
                        .map(|d| SyrupDetection {
                            x: d.x - rect.x as f32,
                            y: d.y - rect.y as f32,
                            ..*d
                        })
                        .collect();
                    let measured = host.measure(what, rect, &local)?;
                    if measured.len() != kept.len() {
                        return Err(invalid("the host measured a different number of boxes"));
                    }
                    for (d, value) in kept.iter_mut().zip(measured) {
                        d.value = value;
                    }
                    kept.retain(|d| d.value.is_finite());
                }
                Value::Boxes(kept)
            }
            Step::Restore { boxes, view: at } => {
                let rect = view(&values, at)?;
                Value::Boxes(restore(take(&mut values, boxes)?, rect))
            }
            Step::FilterConfidence { boxes } => {
                let mut kept = take(&mut values, boxes)?;
                kept.retain(|d| d.score >= params.min_confidence);
                Value::Boxes(kept)
            }
            Step::FilterAspect { boxes, min } => {
                let mut kept = take(&mut values, boxes)?;
                kept.retain(|d| d.w >= min as f32 * d.h);
                Value::Boxes(kept)
            }
            Step::FilterArea {
                boxes,
                min_pct,
                max_pct,
            } => {
                let total = width as f64 * height as f64;
                let mut kept = take(&mut values, boxes)?;
                kept.retain(|d| {
                    let area = d.w as f64 * d.h as f64 * 100.0;
                    min_pct.is_none_or(|p| area >= p as f64 * total)
                        && max_pct.is_none_or(|p| area < p as f64 * total)
                });
                Value::Boxes(kept)
            }
            Step::Order { boxes, key } => {
                let mut sorted = take(&mut values, boxes)?;
                sorted.sort_by(|a, b| compare(key, a, b));
                Value::Boxes(sorted)
            }
            Step::Limit { boxes, n } => {
                let mut kept = take(&mut values, boxes)?;
                kept.truncate(n as usize);
                Value::Boxes(kept)
            }
            Step::LimitParam { boxes } => {
                let mut kept = take(&mut values, boxes)?;
                if params.max_results > 0 {
                    kept.truncate(params.max_results as usize);
                }
                Value::Boxes(kept)
            }
            Step::Emit { boxes } => return take(&mut values, boxes),
        };
        values.push(value);
    }
    Err(invalid("the plan never emits"))
}

pub fn restore(boxes: Vec<SyrupDetection>, view: PixelRect) -> Vec<SyrupDetection> {
    let (left, top) = (view.x as f32, view.y as f32);
    let (right, bottom) = ((view.x + view.w) as f32, (view.y + view.h) as f32);
    let clamp = |v: f32, lo: f32, hi: f32| v.max(lo).min(hi);
    boxes
        .into_iter()
        .filter_map(|d| {
            let (ax, ay) = (d.x + left, d.y + top);
            let (bx, by) = (ax + d.w, ay + d.h);
            let x0 = clamp(ax, left, right);
            let y0 = clamp(ay, top, bottom);
            let w = clamp(bx, left, right) - x0;
            let h = clamp(by, top, bottom) - y0;
            if !(w > 0.0 && h > 0.0) {
                return None;
            }
            let n = (d.n_keypoints as usize).min(SYRUP_MAX_KEYPOINTS);
            let mut keypoints = [0.0f32; 2 * SYRUP_MAX_KEYPOINTS];
            for i in 0..n {
                keypoints[2 * i] = clamp(d.keypoints[2 * i] + left, left, right);
                keypoints[2 * i + 1] = clamp(d.keypoints[2 * i + 1] + top, top, bottom);
            }
            Some(SyrupDetection {
                x: x0,
                y: y0,
                w,
                h,
                score: d.score,
                value: d.value,
                n_keypoints: n as u32,
                keypoints,
                payload: d.payload,
            })
        })
        .collect()
}

/// The key, then confidence ↓, y ↑, x ↑, h ↑, w ↑, so the order is total.
pub fn compare(key: OrderKey, a: &SyrupDetection, b: &SyrupDetection) -> Ordering {
    let primary = match key {
        OrderKey::ConfidenceDesc => Ordering::Equal,
        OrderKey::AreaDesc => (b.w * b.h).total_cmp(&(a.w * a.h)),
        OrderKey::AreaAsc => (a.w * a.h).total_cmp(&(b.w * b.h)),
        OrderKey::LeftToRight => a.x.total_cmp(&b.x),
        OrderKey::RightToLeft => (b.x + b.w).total_cmp(&(a.x + a.w)),
        OrderKey::TopToBottom => a.y.total_cmp(&b.y),
        OrderKey::BottomToTop => (b.y + b.h).total_cmp(&(a.y + a.h)),
    };
    primary.then_with(|| {
        b.score
            .total_cmp(&a.score)
            .then(a.y.total_cmp(&b.y))
            .then(a.x.total_cmp(&b.x))
            .then(a.h.total_cmp(&b.h))
            .then(a.w.total_cmp(&b.w))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::parse;

    fn det(x: f32, y: f32, w: f32, h: f32, score: f32) -> SyrupDetection {
        SyrupDetection {
            x,
            y,
            w,
            h,
            score,
            value: 0.0,
            n_keypoints: 0,
            keypoints: [0.0; 10],
            payload: 0,
        }
    }

    fn params() -> SyrupParams {
        SyrupParams {
            min_confidence: 0.5,
            max_results: 0,
            has_region: 0,
            region_x: 0,
            region_y: 0,
            region_w: 0,
            region_h: 0,
        }
    }

    struct Fixed {
        boxes: Vec<SyrupDetection>,
        seen: Vec<PixelRect>,
    }

    impl Host for Fixed {
        fn detect(&mut self, _: Capability, view: PixelRect) -> Result<Vec<SyrupDetection>> {
            self.seen.push(view);
            Ok(self.boxes.clone())
        }

        fn group(&mut self, runs: &[SyrupRun], min_height: u32, max_gap: u32) -> Vec<SyrupRect> {
            group_runs(runs, min_height, max_gap)
        }

        /// The box's width as a fraction of 100, or NaN for boxes wider than 50.
        fn measure(
            &mut self,
            _: Measurement,
            _: PixelRect,
            boxes: &[SyrupDetection],
        ) -> Result<Vec<f32>> {
            Ok(boxes
                .iter()
                .map(|d| if d.w > 50.0 { f32::NAN } else { d.w / 100.0 })
                .collect())
        }
    }

    fn run_named(
        name: &str,
        w: u32,
        h: u32,
        boxes: Vec<SyrupDetection>,
        p: SyrupParams,
    ) -> (Vec<SyrupDetection>, Vec<PixelRect>) {
        let plan = Plan::build(&parse(name).unwrap()).unwrap();
        let mut detector = Fixed {
            boxes,
            seen: vec![],
        };
        let pixels = vec![0u8; (w * h * 3) as usize];
        let image = ImageInput::new(&pixels, w, h, 3).unwrap();
        let out = run(&plan, &image, &p, &mut detector).unwrap();
        (out, detector.seen)
    }

    #[test]
    fn colour_regions_are_found_scored_and_restored() {
        // Grey 60x40 with a solid red 20x6 bar at (10, 30) and a 10x10 red ring.
        let mut pixels = vec![120u8; 60 * 40 * 3];
        let mut paint =
            |x: usize, y: usize| pixels[(y * 60 + x) * 3..][..3].copy_from_slice(&[220, 30, 30]);
        for y in 30..36 {
            for x in 10..30 {
                paint(x, y);
            }
        }
        for y in 2..12 {
            for x in 40..50 {
                if !(6..8).contains(&y) || !(44..46).contains(&x) {
                    paint(x, y);
                }
            }
        }
        let image = ImageInput::new(&pixels, 60, 40, 3).unwrap();
        let run_plan = |name: &str| {
            let plan = Plan::build(&parse(name).unwrap()).unwrap();
            let mut host = Fixed {
                boxes: vec![],
                seen: vec![],
            };
            let mut p = params();
            p.min_confidence = 0.0;
            run(&plan, &image, &p, &mut host).unwrap()
        };

        let regions = run_plan("find_red_regions_by_size");
        assert_eq!(regions.len(), 2);
        assert_eq!(
            (
                regions[0].x,
                regions[0].y,
                regions[0].w,
                regions[0].h,
                regions[0].score
            ),
            (10.0, 30.0, 20.0, 6.0, 1.0)
        );
        assert_eq!(
            (regions[1].x, regions[1].y, regions[1].w, regions[1].h),
            (40.0, 2.0, 10.0, 10.0)
        );
        assert!(
            (regions[1].score - 0.96).abs() < 1e-6,
            "the ring fills 96 of 100 pixels"
        );

        let bars = run_plan("find_red_bars");
        assert_eq!(bars.len(), 1, "only the bar is 3x wider than tall");
        let lower = run_plan("find_red_regions_in_bottom_half");
        assert_eq!((lower.len(), lower[0].y), (1, 30.0));
        assert!(run_plan("find_blue_regions").is_empty());
    }

    #[test]
    fn region_results_come_back_in_input_coordinates() {
        // A box at (10, 10) inside the bottom half of a 100x100 image is at
        // (10, 60) in the image.
        let (out, seen) = run_named(
            "find_faces_in_bottom_half",
            100,
            100,
            vec![det(10.0, 10.0, 20.0, 20.0, 0.9)],
            params(),
        );
        assert_eq!(
            seen,
            vec![PixelRect {
                x: 0,
                y: 50,
                w: 100,
                h: 50
            }]
        );
        assert_eq!(
            (out[0].x, out[0].y, out[0].w, out[0].h),
            (10.0, 60.0, 20.0, 20.0)
        );
    }

    #[test]
    fn boxes_are_clipped_to_the_searched_region_and_empty_ones_dropped() {
        let (out, _) = run_named(
            "find_faces_in_top_half",
            100,
            100,
            vec![
                det(-5.0, 40.0, 20.0, 20.0, 0.9),
                det(200.0, 0.0, 5.0, 5.0, 0.9),
            ],
            params(),
        );
        assert_eq!(out.len(), 1, "the box entirely outside is dropped");
        assert_eq!(
            (out[0].x, out[0].y, out[0].w, out[0].h),
            (0.0, 40.0, 15.0, 10.0)
        );
    }

    #[test]
    fn ordering_is_total_and_selectors_limit() {
        let boxes = vec![
            det(50.0, 0.0, 10.0, 10.0, 0.7),
            det(0.0, 0.0, 30.0, 30.0, 0.8),
            det(20.0, 0.0, 20.0, 20.0, 0.9),
            det(20.0, 0.0, 20.0, 20.0, 0.4), // below min_confidence
        ];
        let (out, _) = run_named("find_faces", 100, 100, boxes.clone(), params());
        assert_eq!(
            out.iter().map(|d| d.score).collect::<Vec<_>>(),
            vec![0.9, 0.8, 0.7]
        );
        let (out, _) = run_named("find_faces_by_size", 100, 100, boxes.clone(), params());
        assert_eq!(
            out.iter().map(|d| d.w).collect::<Vec<_>>(),
            vec![30.0, 20.0, 10.0]
        );
        let (out, _) = run_named("find_rightmost_face", 100, 100, boxes.clone(), params());
        assert_eq!((out.len(), out[0].x), (1, 50.0));
        let (out, _) = run_named("find_2_smallest_faces", 100, 100, boxes, params());
        assert_eq!(
            out.iter().map(|d| d.w).collect::<Vec<_>>(),
            vec![10.0, 20.0]
        );
    }

    #[test]
    fn area_filters_use_percent_of_the_whole_image() {
        // 100x100 image: 10x10 is exactly 1%.
        let boxes = vec![
            det(0.0, 0.0, 10.0, 10.0, 0.9),
            det(0.0, 0.0, 9.0, 10.0, 0.9),
        ];
        let (out, _) = run_named(
            "find_faces_larger_than_1pct",
            100,
            100,
            boxes.clone(),
            params(),
        );
        assert_eq!(out.len(), 1);
        let (out, _) = run_named("find_faces_smaller_than_1pct", 100, 100, boxes, params());
        assert_eq!((out.len(), out[0].w), (1, 9.0));
    }

    #[test]
    fn caller_regions_and_max_results() {
        let mut p = params();
        p.has_region = 1;
        (p.region_x, p.region_y, p.region_w, p.region_h) = (30, 40, 500, 10);
        p.max_results = 1;
        let (out, seen) = run_named(
            "find_faces_in_region",
            100,
            100,
            vec![det(1.0, 1.0, 2.0, 2.0, 0.6), det(1.0, 1.0, 2.0, 2.0, 0.9)],
            p,
        );
        assert_eq!(
            seen,
            vec![PixelRect {
                x: 30,
                y: 40,
                w: 70,
                h: 10
            }],
            "region clipped to the image"
        );
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].x, out[0].score), (31.0, 0.9));
    }

    #[test]
    fn an_empty_region_searches_nothing() {
        let (out, seen) = run_named(
            "find_faces_in_left_third",
            1,
            10,
            vec![det(0.0, 0.0, 1.0, 1.0, 0.9)],
            params(),
        );
        assert!(
            seen.is_empty(),
            "the detector is not called on an empty view"
        );
        assert!(out.is_empty());
    }
}
