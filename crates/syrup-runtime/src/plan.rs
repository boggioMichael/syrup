//! The typed plan an intent compiles to. Step `i` defines value `i`; values
//! carry their coordinate space, so the checker can prove results are in the
//! caller's image coordinates before any code is generated.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::abi::{SYRUP_MEASURE_FILL, SYRUP_MEASURE_SHARPNESS, SyrupMeasure};
use crate::catalog::{self, Capability, Finder, Quantity};
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::intent::{Intent, NormRect, OrderKey, RegionSpec};

pub const IR_VERSION: u32 = 1;

/// What a `Measure` step asks the host for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "quantity", rename_all = "snake_case")]
pub enum Measurement {
    Sharpness,
    /// Filled pixels are those of the bar's colour, as `FindColor` matches them.
    Fill {
        hue: (u32, u32),
        min_saturation_pct: u32,
        min_value_pct: u32,
    },
}

impl Measurement {
    pub fn abi(self) -> SyrupMeasure {
        match self {
            Measurement::Sharpness => SyrupMeasure {
                kind: SYRUP_MEASURE_SHARPNESS,
                hue_lo: 0,
                hue_hi: 0,
                min_saturation_pct: 0,
                min_value_pct: 0,
            },
            Measurement::Fill {
                hue,
                min_saturation_pct,
                min_value_pct,
            } => SyrupMeasure {
                kind: SYRUP_MEASURE_FILL,
                hue_lo: hue.0,
                hue_hi: hue.1,
                min_saturation_pct,
                min_value_pct,
            },
        }
    }

    pub fn quantity(self) -> Quantity {
        match self {
            Measurement::Sharpness => Quantity::Sharpness,
            Measurement::Fill { .. } => Quantity::Fill,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Step {
    Input,
    SelectRegion {
        view: usize,
        region: RegionSpec,
    },
    Detect {
        view: usize,
        capability: Capability,
    },
    /// Runs of pixels whose hue lies in `hue` (degrees, wrapping when the
    /// start is larger), grouped into regions by the core's grouping. A
    /// region's score is the share of its pixels that match.
    FindColor {
        view: usize,
        hue: (u32, u32),
        min_saturation_pct: u32,
        min_value_pct: u32,
        min_run: u32,
        min_height: u32,
        max_gap: u32,
    },
    /// The whole of `view` as one box with score 1, or none if it is empty.
    Whole {
        view: usize,
    },
    /// Moves boxes from `view`'s space to input space and clips them to the
    /// view; boxes left without area are dropped.
    Restore {
        boxes: usize,
        view: usize,
    },
    FilterConfidence {
        boxes: usize,
    },
    /// Keeps boxes at least `min` times wider than tall.
    FilterAspect {
        boxes: usize,
        min: u32,
    },
    FilterArea {
        boxes: usize,
        min_pct: Option<u32>,
        max_pct: Option<u32>,
    },
    Order {
        boxes: usize,
        key: OrderKey,
    },
    Limit {
        boxes: usize,
        n: u32,
    },
    /// Truncates to the run's `max_results`.
    LimitParam {
        boxes: usize,
    },
    /// Sets each box's value; boxes the host cannot measure are dropped.
    /// The boxes are in input space; the host measures them within `view`,
    /// the searched region, so a fill is never read from outside it.
    Measure {
        boxes: usize,
        view: usize,
        what: Measurement,
    },
    Emit {
        boxes: usize,
    },
}

impl Step {
    pub fn inputs(&self) -> Vec<usize> {
        match *self {
            Step::Input => vec![],
            Step::SelectRegion { view, .. }
            | Step::Detect { view, .. }
            | Step::FindColor { view, .. }
            | Step::Whole { view } => vec![view],
            Step::Restore { boxes, view } | Step::Measure { boxes, view, .. } => vec![boxes, view],
            Step::FilterConfidence { boxes }
            | Step::FilterAspect { boxes, .. }
            | Step::FilterArea { boxes, .. }
            | Step::Order { boxes, .. }
            | Step::Limit { boxes, .. }
            | Step::LimitParam { boxes }
            | Step::Emit { boxes } => vec![boxes],
        }
    }
}

/// `Region(i)` is the space of the region selected by step `i`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Space {
    Input,
    Region(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    View(Space),
    Boxes { space: Space, clipped: bool },
    Unit,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
pub struct Plan {
    pub ir_version: u32,
    pub steps: Vec<Step>,
}

fn invalid(reason: impl Into<String>) -> SyrupError {
    SyrupError::new(Stage::Plan, ErrorKind::InvalidPlan, reason)
}

impl Plan {
    pub fn build(intent: &Intent) -> Result<Plan> {
        intent.check()?;
        let mut steps = vec![Step::Input];
        let mut push = |step: Step| {
            steps.push(step);
            steps.len() - 1
        };
        let view = match intent.region {
            Some(region) => push(Step::SelectRegion { view: 0, region }),
            None => 0,
        };
        let finder = catalog::entry(intent.target).finder;
        let mut boxes = match (finder, intent.color) {
            (Finder::Detect(capability), _) => push(Step::Detect { view, capability }),
            (
                Finder::Color {
                    min_run,
                    min_height,
                    max_gap,
                    ..
                },
                Some(color),
            ) => push(Step::FindColor {
                view,
                hue: catalog::color(color).hue,
                min_saturation_pct: catalog::MIN_SATURATION_PCT,
                min_value_pct: catalog::MIN_VALUE_PCT,
                min_run,
                min_height,
                max_gap,
            }),
            (Finder::Color { .. }, None) => {
                return Err(invalid("a colour target without a colour"));
            }
            (Finder::Whole, _) => push(Step::Whole { view }),
        };
        boxes = push(Step::Restore { boxes, view });
        boxes = push(Step::FilterConfidence { boxes });
        if let Finder::Color {
            min_aspect: Some(min),
            ..
        } = finder
        {
            boxes = push(Step::FilterAspect { boxes, min });
        }
        if intent.min_area_pct.is_some() || intent.max_area_pct.is_some() {
            boxes = push(Step::FilterArea {
                boxes,
                min_pct: intent.min_area_pct,
                max_pct: intent.max_area_pct,
            });
        }
        boxes = push(Step::Order {
            boxes,
            key: intent.order,
        });
        if let Some(n) = intent.limit {
            boxes = push(Step::Limit { boxes, n });
        }
        boxes = push(Step::LimitParam { boxes });
        if let Some(quantity) = intent.measure {
            let what = match (quantity, intent.color) {
                (Quantity::Sharpness, _) => Measurement::Sharpness,
                (Quantity::Fill, Some(color)) => Measurement::Fill {
                    hue: catalog::color(color).hue,
                    min_saturation_pct: catalog::MIN_SATURATION_PCT,
                    min_value_pct: catalog::MIN_VALUE_PCT,
                },
                (Quantity::Fill, None) => return Err(invalid("fill needs the bar's colour")),
            };
            boxes = push(Step::Measure { boxes, view, what });
        }
        push(Step::Emit { boxes });

        let plan = Plan {
            ir_version: IR_VERSION,
            steps,
        };
        plan.check()?;
        Ok(plan)
    }

    pub fn check(&self) -> Result<Vec<ValueType>> {
        if self.ir_version != IR_VERSION {
            return Err(invalid(format!(
                "plan version {} is not {IR_VERSION}",
                self.ir_version
            )));
        }
        if self.steps.first() != Some(&Step::Input)
            || !matches!(self.steps.last(), Some(Step::Emit { .. }))
        {
            return Err(invalid("a plan starts with its input and ends by emitting"));
        }
        let mut types: Vec<ValueType> = Vec::with_capacity(self.steps.len());
        let mut uses = vec![0usize; self.steps.len()];
        for (i, step) in self.steps.iter().enumerate() {
            for input in step.inputs() {
                if input >= i {
                    return Err(invalid(format!(
                        "step {i} uses value {input}, which is not defined yet"
                    )));
                }
                uses[input] += 1;
            }
            // Boxes that geometry depends on must already be in input space.
            let restored = |at: usize| match types[at] {
                ValueType::Boxes {
                    space: Space::Input,
                    clipped: true,
                } => Ok(types[at]),
                other => Err(invalid(format!(
                    "step {i} needs boxes restored to input space, got {other:?}"
                ))),
            };
            let ty = match *step {
                Step::Input if i == 0 => ValueType::View(Space::Input),
                Step::Input => return Err(invalid("a plan has one input")),
                Step::SelectRegion { view, region } => {
                    if types[view] != ValueType::View(Space::Input) {
                        return Err(invalid(format!(
                            "step {i} selects a region of something other than the input"
                        )));
                    }
                    if let RegionSpec::Fixed { rect } = region
                        && NormRect::new(rect.x, rect.y, rect.w, rect.h).is_none()
                    {
                        return Err(invalid(format!(
                            "step {i}: region {rect} is empty or leaves the image"
                        )));
                    }
                    ValueType::View(Space::Region(i))
                }
                Step::FindColor {
                    hue,
                    min_saturation_pct,
                    min_value_pct,
                    min_run,
                    min_height,
                    ..
                } if hue.0 >= 360
                    || hue.1 >= 360
                    || min_saturation_pct > 100
                    || min_value_pct > 100
                    || min_run == 0
                    || min_height == 0 =>
                {
                    return Err(invalid(format!("step {i}: colour thresholds out of range")));
                }
                Step::Measure {
                    what:
                        Measurement::Fill {
                            hue,
                            min_saturation_pct,
                            min_value_pct,
                        },
                    ..
                } if hue.0 >= 360
                    || hue.1 >= 360
                    || min_saturation_pct > 100
                    || min_value_pct > 100 =>
                {
                    return Err(invalid(format!("step {i}: fill thresholds out of range")));
                }
                Step::Measure { boxes, view, .. } => {
                    if !matches!(types[view], ValueType::View(_)) {
                        return Err(invalid(format!(
                            "step {i} measures within {:?}",
                            types[view]
                        )));
                    }
                    restored(boxes)?
                }
                Step::Detect { view, .. } | Step::FindColor { view, .. } | Step::Whole { view } => {
                    match types[view] {
                        ValueType::View(space) => ValueType::Boxes {
                            space,
                            clipped: false,
                        },
                        other => return Err(invalid(format!("step {i} detects in {other:?}"))),
                    }
                }
                Step::Restore { boxes, view } => match (types[boxes], types[view]) {
                    (ValueType::Boxes { space, .. }, ValueType::View(view_space))
                        if space == view_space =>
                    {
                        ValueType::Boxes {
                            space: Space::Input,
                            clipped: true,
                        }
                    }
                    (b, v) => {
                        return Err(invalid(format!("step {i} restores {b:?} against {v:?}")));
                    }
                },
                Step::FilterConfidence { boxes } => match types[boxes] {
                    ty @ ValueType::Boxes { .. } => ty,
                    other => return Err(invalid(format!("step {i} filters {other:?}"))),
                },
                Step::FilterArea {
                    boxes,
                    min_pct,
                    max_pct,
                } => {
                    let ok = |p: Option<u32>| p.is_none_or(|p| (1..=100).contains(&p));
                    let empty = matches!((min_pct, max_pct), (Some(min), Some(max)) if min >= max);
                    if (min_pct, max_pct) == (None, None) || !ok(min_pct) || !ok(max_pct) || empty {
                        return Err(invalid(format!("step {i}: bad area bounds")));
                    }
                    restored(boxes)?
                }
                Step::Limit { n: 0, .. } | Step::FilterAspect { min: 0, .. } => {
                    return Err(invalid(format!("step {i}: a bound of 0")));
                }
                Step::FilterAspect { boxes, .. } => restored(boxes)?,
                Step::Order { boxes, .. }
                | Step::Limit { boxes, .. }
                | Step::LimitParam { boxes } => restored(boxes)?,
                Step::Emit { boxes } => {
                    restored(boxes)?;
                    ValueType::Unit
                }
            };
            types.push(ty);
        }
        // Boxes are consumed exactly once (generated code moves them); views
        // may be read several times; nothing is left unused.
        for (i, step) in self.steps.iter().enumerate() {
            let fine = match step {
                Step::Emit { .. } => uses[i] == 0 && i == self.steps.len() - 1,
                Step::Input | Step::SelectRegion { .. } => uses[i] >= 1,
                _ => uses[i] == 1,
            };
            if !fine {
                return Err(invalid(format!("value {i} is used {} times", uses[i])));
            }
        }
        Ok(types)
    }

    /// SHA-256 of the canonical JSON, in hex.
    pub fn hash(&self) -> String {
        hex(&Sha256::digest(
            serde_json::to_vec(self).expect("plans serialize"),
        ))
    }

    pub fn capabilities(&self) -> BTreeSet<Capability> {
        self.steps
            .iter()
            .filter_map(|step| match step {
                Step::Detect { capability, .. } => Some(*capability),
                _ => None,
            })
            .collect()
    }

    pub fn needs_caller_region(&self) -> bool {
        self.steps.iter().any(|step| {
            matches!(
                step,
                Step::SelectRegion {
                    region: RegionSpec::Caller,
                    ..
                }
            )
        })
    }

    pub fn describe(&self) -> String {
        let mut out = String::new();
        for (i, step) in self.steps.iter().enumerate() {
            let line = match step {
                Step::Input => "input image".to_string(),
                Step::SelectRegion { view, region } => format!("select {region} of v{view}"),
                Step::Detect { view, capability } => {
                    format!("detect {} in v{view}", capability.as_str())
                }
                Step::Whole { view } => format!("all of v{view}, as one item"),
                Step::Measure { boxes, view, what } => format!(
                    "measure the {} of each of v{boxes} in v{view}; drop those that cannot be measured",
                    what.quantity().name()
                ),
                Step::Restore { boxes, view } => {
                    format!("restore v{boxes} from v{view} to input coordinates, clipped")
                }
                Step::FindColor {
                    view,
                    hue,
                    min_saturation_pct,
                    min_value_pct,
                    min_run,
                    min_height,
                    max_gap,
                } => format!(
                    "find hue {}-{} deg (saturation >= {min_saturation_pct}%, value >= \
                     {min_value_pct}%) in v{view}: runs >= {min_run} px, regions >= \
                     {min_height} rows, gaps <= {max_gap}",
                    hue.0, hue.1
                ),
                Step::FilterConfidence { boxes } => {
                    format!("keep v{boxes} with confidence >= min_confidence")
                }
                Step::FilterAspect { boxes, min } => {
                    format!("keep v{boxes} at least {min}x wider than tall")
                }
                Step::FilterArea {
                    boxes,
                    min_pct,
                    max_pct,
                } => {
                    let mut bounds = vec![];
                    if let Some(p) = min_pct {
                        bounds.push(format!("area >= {p}%"));
                    }
                    if let Some(p) = max_pct {
                        bounds.push(format!("area < {p}%"));
                    }
                    format!("keep v{boxes} with {}", bounds.join(" and "))
                }
                Step::Order { boxes, key } => format!("order v{boxes} by {}", key.describe()),
                Step::Limit { boxes, n } => format!("first {n} of v{boxes}"),
                Step::LimitParam { boxes } => format!("first max_results of v{boxes}"),
                Step::Emit { boxes } => format!("emit v{boxes}"),
            };
            let _ = writeln!(out, "v{i} = {line}");
        }
        out
    }
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, b| {
            let _ = write!(out, "{b:02x}");
            out
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::parse;

    fn plan(name: &str) -> Plan {
        Plan::build(&parse(name).unwrap()).unwrap()
    }

    fn bad(steps: Vec<Step>) -> SyrupError {
        let e = Plan {
            ir_version: IR_VERSION,
            steps,
        }
        .check()
        .unwrap_err();
        assert_eq!((e.stage, e.kind), (Stage::Plan, ErrorKind::InvalidPlan));
        e
    }

    #[test]
    fn equivalent_names_share_a_hash() {
        assert_eq!(plan("find_face").hash(), plan("detect_faces").hash());
        assert_ne!(
            plan("find_face").hash(),
            plan("find_faces_in_top_half").hash()
        );
        assert_ne!(
            plan("find_faces_in_top_half").hash(),
            plan("find_faces_in_bottom_half").hash()
        );
    }

    #[test]
    fn unrestored_boxes_are_rejected() {
        let e = bad(vec![
            Step::Input,
            Step::SelectRegion {
                view: 0,
                region: RegionSpec::Caller,
            },
            Step::Detect {
                view: 1,
                capability: Capability::FaceDetection,
            },
            Step::Order {
                boxes: 2,
                key: OrderKey::AreaDesc,
            },
            Step::Emit { boxes: 3 },
        ]);
        assert!(e.reason.contains("restored"), "{e}");
    }

    #[test]
    fn restoring_against_the_wrong_view_is_rejected() {
        bad(vec![
            Step::Input,
            Step::SelectRegion {
                view: 0,
                region: RegionSpec::Caller,
            },
            Step::Detect {
                view: 1,
                capability: Capability::FaceDetection,
            },
            Step::Restore { boxes: 2, view: 0 },
            Step::Emit { boxes: 3 },
        ]);
    }

    #[test]
    fn dangling_and_forward_values_are_rejected() {
        let mut steps = plan("find_face").steps;
        steps.insert(
            1,
            Step::Detect {
                view: 0,
                capability: Capability::FaceDetection,
            },
        );
        bad(steps);
        bad(vec![Step::Input, Step::Emit { boxes: 2 }]);
    }
}
