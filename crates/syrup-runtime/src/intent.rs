//! Operation names are parsed against a small grammar (docs/contract.md),
//! not looked up, so clauses compose into operations nobody wrote. Every word
//! has to be understood: ambiguous or unknown words fail resolution instead of
//! being dropped.

use std::fmt;

use serde::Serialize;

use crate::catalog::{self, Color, Finder, Quantity, Target};
use crate::error::{ErrorKind, Result, Stage, SyrupError};

/// An exact fraction. Regions use these so the interpreter and generated
/// code compute identical pixel bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct Ratio {
    pub num: u32,
    pub den: u32,
}

impl Ratio {
    pub fn new(num: u32, den: u32) -> Ratio {
        assert!(den != 0);
        let g = gcd(num as u64, den as u64) as u32;
        Ratio {
            num: num / g,
            den: den / g,
        }
    }

    /// Rounded to the nearest 1/10000.
    pub fn from_f64(value: f64) -> Option<Ratio> {
        (0.0..=1.0)
            .contains(&value)
            .then(|| Ratio::new((value * 10_000.0).round() as u32, 10_000))
    }

    pub fn checked_add(self, other: Ratio) -> Option<Ratio> {
        let num = self.num as u64 * other.den as u64 + other.num as u64 * self.den as u64;
        let den = self.den as u64 * other.den as u64;
        let g = gcd(num, den);
        Some(Ratio {
            num: u32::try_from(num / g).ok()?,
            den: u32::try_from(den / g).ok()?,
        })
    }

    /// `round(self * extent)` with halves rounded up.
    pub fn of(self, extent: u32) -> u64 {
        (2 * self.num as u64 * extent as u64 + self.den as u64) / (2 * self.den as u64)
    }
}

impl fmt::Display for Ratio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.den {
            1 => write!(f, "{}", self.num),
            den => write!(f, "{}/{den}", self.num),
        }
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a.max(1) } else { gcd(b, a % b) }
}

/// Half-open pixel rectangle `[x, x+w) × [y, y+h)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct PixelRect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

/// A rectangle as fractions of the image size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
pub struct NormRect {
    pub x: Ratio,
    pub y: Ratio,
    pub w: Ratio,
    pub h: Ratio,
}

impl NormRect {
    pub fn new(x: Ratio, y: Ratio, w: Ratio, h: Ratio) -> Option<NormRect> {
        let fits = |r: Option<Ratio>| r.is_some_and(|r| r.num <= r.den);
        (w.num > 0 && h.num > 0 && fits(x.checked_add(w)) && fits(y.checked_add(h)))
            .then_some(NormRect { x, y, w, h })
    }

    pub fn right(&self) -> Ratio {
        self.x
            .checked_add(self.w)
            .expect("checked in NormRect::new")
    }

    pub fn bottom(&self) -> Ratio {
        self.y
            .checked_add(self.h)
            .expect("checked in NormRect::new")
    }

    pub fn pixels(&self, width: u32, height: u32) -> PixelRect {
        let (x0, y0) = (self.x.of(width) as u32, self.y.of(height) as u32);
        let (x1, y1) = (
            self.right().of(width) as u32,
            self.bottom().of(height) as u32,
        );
        PixelRect {
            x: x0,
            y: y0,
            w: x1 - x0,
            h: y1 - y0,
        }
    }

    pub fn name(&self) -> Option<&'static str> {
        NAMED_REGIONS
            .iter()
            .find(|(_, rect)| rect() == *self)
            .map(|(names, _)| names[0])
    }
}

impl fmt::Display for NormRect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = self.name() {
            write!(f, "{name} ")?;
        }
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.w, self.h)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum RegionSpec {
    Fixed {
        rect: NormRect,
    },
    /// Passed by the caller on every run.
    Caller,
}

impl fmt::Display for RegionSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegionSpec::Fixed { rect } => write!(f, "{rect}"),
            RegionSpec::Caller => f.write_str("the caller's region"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OrderKey {
    ConfidenceDesc,
    AreaDesc,
    AreaAsc,
    LeftToRight,
    RightToLeft,
    TopToBottom,
    BottomToTop,
}

impl OrderKey {
    /// Accepts the serialized names and the grammar's `by_` keys.
    pub fn parse(word: &str) -> Option<OrderKey> {
        Some(match word {
            "confidence_desc" | "confidence" | "score" => OrderKey::ConfidenceDesc,
            "area_desc" | "area" | "size" => OrderKey::AreaDesc,
            "area_asc" => OrderKey::AreaAsc,
            "left_to_right" => OrderKey::LeftToRight,
            "right_to_left" => OrderKey::RightToLeft,
            "top_to_bottom" => OrderKey::TopToBottom,
            "bottom_to_top" => OrderKey::BottomToTop,
            _ => return None,
        })
    }

    pub fn describe(self) -> &'static str {
        match self {
            OrderKey::ConfidenceDesc => "confidence, highest first",
            OrderKey::AreaDesc => "area, largest first",
            OrderKey::AreaAsc => "area, smallest first",
            OrderKey::LeftToRight => "left to right",
            OrderKey::RightToLeft => "right to left",
            OrderKey::TopToBottom => "top to bottom",
            OrderKey::BottomToTop => "bottom to top",
        }
    }
}

/// What an operation means. Names that resolve to the same intent are the
/// same operation and share one compiled artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct Intent {
    pub target: Target,
    /// Required by targets found by colour, refused by the others.
    pub color: Option<Color>,
    pub region: Option<RegionSpec>,
    pub order: OrderKey,
    pub limit: Option<u32>,
    /// Percent of the image area.
    pub min_area_pct: Option<u32>,
    pub max_area_pct: Option<u32>,
    /// Set for `measure_*` operations: each item carries this quantity.
    pub measure: Option<Quantity>,
    /// Set for `track_*` operations, which run in sessions over frames and
    /// give each item an identity that lasts across them.
    pub track: bool,
}

impl Intent {
    pub fn find(target: Target) -> Intent {
        Intent {
            target,
            color: None,
            region: None,
            order: OrderKey::ConfidenceDesc,
            limit: None,
            min_area_pct: None,
            max_area_pct: None,
            measure: None,
            track: false,
        }
    }

    pub fn check(&self) -> Result<()> {
        let entry = catalog::entry(self.target);
        let error = |kind, reason: String| Err(SyrupError::new(Stage::Resolve, kind, reason));
        if self.target == Target::MovingRegion && !self.track {
            return Err(SyrupError::new(
                Stage::Resolve,
                ErrorKind::Unsupported,
                "moving regions only exist between frames",
            )
            .with_hint("track_moving_regions, in a session"));
        }
        if self.track && self.measure.is_some() {
            return error(
                ErrorKind::Conflicting,
                "an operation either tracks or measures".into(),
            );
        }
        match (self.measure, self.target) {
            (None, Target::Image) => {
                return error(
                    ErrorKind::Malformed,
                    "the whole image is only an item to measure".into(),
                );
            }
            (Some(Quantity::Fill), Target::Image) => {
                return error(
                    ErrorKind::Ambiguous,
                    "fill of what? e.g. measure_fill_of_red_bars".into(),
                );
            }
            (Some(Quantity::Fill), target) if target != Target::Bar => {
                return error(
                    ErrorKind::Unsupported,
                    format!("fill is measured on bars, not {}", entry.plural[0]),
                );
            }
            (Some(q), Target::Image)
                if self.order != OrderKey::ConfidenceDesc
                    || self.limit.is_some()
                    || self.min_area_pct.is_some()
                    || self.max_area_pct.is_some() =>
            {
                return error(
                    ErrorKind::Conflicting,
                    format!(
                        "{} of the image or a region is one value, with nothing to order or filter",
                        q.name()
                    ),
                );
            }
            _ => {}
        }
        match (entry.finder, self.color) {
            (Finder::Color { .. }, None) => {
                return Err(SyrupError::new(
                    Stage::Resolve,
                    ErrorKind::Ambiguous,
                    format!("{} of which colour?", entry.plural[0]),
                )
                .with_hint(format!(
                    "e.g. red_{}; colours: {}",
                    entry.plural[0],
                    color_names()
                )));
            }
            (Finder::Detect(_), Some(color)) => {
                return Err(SyrupError::new(
                    Stage::Resolve,
                    ErrorKind::Unsupported,
                    format!(
                        "{} {} cannot be told apart by colour",
                        color.name(),
                        entry.plural[0]
                    ),
                ));
            }
            _ => {}
        }
        let conflict = |reason: String| {
            Err(SyrupError::new(
                Stage::Resolve,
                ErrorKind::Conflicting,
                reason,
            ))
        };
        if self.limit == Some(0) {
            return conflict("a limit of 0 can never return anything".into());
        }
        if let Some(pct) = [self.min_area_pct, self.max_area_pct]
            .into_iter()
            .flatten()
            .find(|pct| !(1..=100).contains(pct))
        {
            return conflict(format!("area percentages are 1 to 100, got {pct}"));
        }
        if let (Some(min), Some(max)) = (self.min_area_pct, self.max_area_pct)
            && min >= max
        {
            return conflict(format!(
                "larger_than_{min}pct and smaller_than_{max}pct never both hold"
            ));
        }
        Ok(())
    }
}

impl fmt::Display for Intent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.measure {
            Some(q) => write!(f, "measure {} of ", q.name())?,
            None if self.track => f.write_str("track ")?,
            None => f.write_str("find ")?,
        }
        if let Some(color) = self.color {
            write!(f, "{} ", color.name())?;
        }
        f.write_str(catalog::entry(self.target).label)?;
        if let Some(region) = &self.region {
            write!(f, " in {region}")?;
        }
        if let Some(pct) = self.min_area_pct {
            write!(f, ", area >= {pct}%")?;
        }
        if let Some(pct) = self.max_area_pct {
            write!(f, ", area < {pct}%")?;
        }
        write!(f, ", by {}", self.order.describe())?;
        if let Some(limit) = self.limit {
            write!(f, ", at most {limit}")?;
        }
        Ok(())
    }
}

const FIND_VERBS: &[&str] = &["find", "detect", "locate"];

const OTHER_VERBS: &[(&str, &str)] = &[
    ("count", "use len(find_...) for now"),
    ("read", "reading text is not in the catalog yet"),
    (
        "recognize",
        "Syrup locates things, it does not recognise identities",
    ),
    (
        "identify",
        "Syrup locates things, it does not identify people",
    ),
    ("classify", "classification is not in the catalog yet"),
    ("segment", "segmentation is not in the catalog yet"),
];

const SELECTORS: &[(&[&str], OrderKey)] = &[
    (&["most", "confident"], OrderKey::ConfidenceDesc),
    (&["largest"], OrderKey::AreaDesc),
    (&["biggest"], OrderKey::AreaDesc),
    (&["smallest"], OrderKey::AreaAsc),
    (&["leftmost"], OrderKey::LeftToRight),
    (&["rightmost"], OrderKey::RightToLeft),
    (&["topmost"], OrderKey::TopToBottom),
    (&["bottommost"], OrderKey::BottomToTop),
];

const DIRECTIONS: &[(&[&str], OrderKey)] = &[
    (&["left", "to", "right"], OrderKey::LeftToRight),
    (&["right", "to", "left"], OrderKey::RightToLeft),
    (&["top", "to", "bottom"], OrderKey::TopToBottom),
    (&["bottom", "to", "top"], OrderKey::BottomToTop),
];

const ORDER_KEYS: &[(&str, OrderKey)] = &[
    ("size", OrderKey::AreaDesc),
    ("area", OrderKey::AreaDesc),
    ("confidence", OrderKey::ConfidenceDesc),
    ("score", OrderKey::ConfidenceDesc),
];

fn rect(x: (u32, u32), y: (u32, u32), w: (u32, u32), h: (u32, u32)) -> NormRect {
    let r = |(n, d)| Ratio::new(n, d);
    NormRect {
        x: r(x),
        y: r(y),
        w: r(w),
        h: r(h),
    }
}

type NamedRegion = (&'static [&'static str], fn() -> NormRect);

// First spelling is canonical; longer phrases come first.
const NAMED_REGIONS: &[NamedRegion] = &[
    (&["top_left_quadrant", "top_left"], || {
        rect((0, 1), (0, 1), (1, 2), (1, 2))
    }),
    (&["top_right_quadrant", "top_right"], || {
        rect((1, 2), (0, 1), (1, 2), (1, 2))
    }),
    (&["bottom_left_quadrant", "bottom_left"], || {
        rect((0, 1), (1, 2), (1, 2), (1, 2))
    }),
    (&["bottom_right_quadrant", "bottom_right"], || {
        rect((1, 2), (1, 2), (1, 2), (1, 2))
    }),
    (&["top_half", "upper_half"], || {
        rect((0, 1), (0, 1), (1, 1), (1, 2))
    }),
    (&["bottom_half", "lower_half"], || {
        rect((0, 1), (1, 2), (1, 1), (1, 2))
    }),
    (&["left_half"], || rect((0, 1), (0, 1), (1, 2), (1, 1))),
    (&["right_half"], || rect((1, 2), (0, 1), (1, 2), (1, 1))),
    (&["top_third", "upper_third"], || {
        rect((0, 1), (0, 1), (1, 1), (1, 3))
    }),
    (&["bottom_third", "lower_third"], || {
        rect((0, 1), (2, 3), (1, 1), (1, 3))
    }),
    (&["left_third"], || rect((0, 1), (0, 1), (1, 3), (1, 1))),
    (&["right_third"], || rect((2, 3), (0, 1), (1, 3), (1, 1))),
    (&["center", "centre"], || {
        rect((1, 4), (1, 4), (1, 2), (1, 2))
    }),
];

const AMBIGUOUS_WORDS: &[(&str, &str)] = &[
    (
        "best",
        "best by what? Say largest, most_confident, leftmost, ...",
    ),
    ("main", "main by what? Say largest, most_confident, ..."),
    (
        "primary",
        "primary by what? Say largest, most_confident, ...",
    ),
    (
        "important",
        "importance is not measurable; say largest, most_confident, ...",
    ),
    (
        "relevant",
        "relevance is not measurable; say largest, most_confident, ...",
    ),
    (
        "good",
        "good by what? Use min_confidence or larger_than_<n>pct",
    ),
    (
        "first",
        "first in which order? Say leftmost, topmost, ... or add left_to_right",
    ),
    (
        "last",
        "last in which order? Say rightmost, bottommost, ...",
    ),
    (
        "highest",
        "by position or by confidence? Say topmost or most_confident",
    ),
    ("lowest", "by position or by confidence? Say bottommost"),
    (
        "top",
        "top by what? Say topmost or most_confident, or use in_top_half",
    ),
    ("large", "large compared to what? Say larger_than_<n>pct"),
    ("big", "big compared to what? Say larger_than_<n>pct"),
    ("huge", "huge compared to what? Say larger_than_<n>pct"),
    ("small", "small compared to what? Say smaller_than_<n>pct"),
    ("tiny", "tiny compared to what? Say smaller_than_<n>pct"),
    ("nearest", "Syrup has no depth or reference point"),
    ("closest", "Syrup has no depth or reference point"),
    ("some", "how many? Say e.g. find_3_largest_faces"),
    ("several", "how many? Say e.g. find_3_largest_faces"),
    ("few", "how many? Say e.g. find_3_largest_faces"),
    (
        "middle",
        "horizontally or vertically? Say center for the middle of both",
    ),
];

// Properties no capability in the catalog can judge.
const UNSUPPORTED_QUALIFIERS: &[&str] = &[
    "smiling",
    "frowning",
    "happy",
    "sad",
    "angry",
    "surprised",
    "male",
    "female",
    "man",
    "woman",
    "men",
    "women",
    "boy",
    "girl",
    "young",
    "old",
    "child",
    "children",
    "kid",
    "adult",
    "baby",
    "masked",
    "bearded",
    "blurry",
    "sharp",
    "profile",
    "frontal",
    "known",
    "unknown",
    "familiar",
    "real",
    "fake",
    "cat",
    "dog",
    "animal",
    "cartoon",
];

// Colour words without a hue range.
const NON_HUES: &[&str] = &[
    "white", "black", "grey", "gray", "brown", "pink", "gold", "silver", "beige", "dark", "light",
];

fn color_names() -> String {
    catalog::COLORS
        .iter()
        .map(|c| c.names[0])
        .collect::<Vec<_>>()
        .join(", ")
}

const NUMBER_WORDS: &[&str] = &[
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
];

/// Canonical region names, `region` (the caller's) last.
pub fn region_names() -> Vec<&'static str> {
    NAMED_REGIONS
        .iter()
        .map(|(names, _)| names[0])
        .chain(["region"])
        .collect()
}

/// A region by its grammar name, e.g. `top_half` or `region`.
pub fn region_named(name: &str) -> Option<RegionSpec> {
    if name == "region" {
        return Some(RegionSpec::Caller);
    }
    NAMED_REGIONS
        .iter()
        .find(|(names, _)| names.contains(&name))
        .map(|(_, rect)| RegionSpec::Fixed { rect: rect() })
}

/// Whether the grammar already gives `word` a meaning, so a target added at
/// run time may not use it in its noun.
pub fn is_reserved(word: &str) -> bool {
    const KEYWORDS: &[&str] = &[
        "all", "in", "by", "than", "to", "pct", "percent", "larger", "bigger", "smaller",
    ];
    FIND_VERBS.contains(&word)
        || ["measure", "track", "of", "moving"].contains(&word)
        || Quantity::named(word).is_some()
        || OTHER_VERBS.iter().any(|(verb, _)| *verb == word)
        || SELECTORS.iter().any(|(words, _)| words.contains(&word))
        || KEYWORDS.contains(&word)
        || NUMBER_WORDS.contains(&word)
        || catalog::color_named(word).is_some()
        || word.bytes().all(|b| b.is_ascii_digit())
}

fn quantity_names() -> String {
    Quantity::ALL.map(Quantity::name).join("|")
}

pub fn grammar_summary() -> String {
    let regions: Vec<&str> = NAMED_REGIONS.iter().map(|(names, _)| names[0]).collect();
    format!(
        "<find|detect|locate>_[<count>_<selector>_|<selector>_]<target>[_<clause>...]; \
         targets: {}; selectors: largest, smallest, leftmost, rightmost, topmost, bottommost, \
         most_confident; clauses: in_<region>, by_size, by_confidence, left_to_right, \
         right_to_left, top_to_bottom, bottom_to_top, larger_than_<n>pct, smaller_than_<n>pct; \
         regions: {}, region; measure_<{}>[_of_<what find takes>|_in_<region>]",
        catalog::known_targets(),
        regions.join(", "),
        quantity_names(),
    )
}

struct Parser<'a> {
    name: &'a str,
    tokens: Vec<&'a str>,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn error(&self, kind: ErrorKind, reason: impl Into<String>) -> SyrupError {
        SyrupError::new(Stage::Resolve, kind, reason).for_operation(self.name)
    }

    fn peek(&self) -> &'a str {
        self.tokens.get(self.pos).copied().unwrap_or("")
    }

    fn at(&self, words: &[&str]) -> bool {
        self.tokens.get(self.pos..self.pos + words.len()) == Some(words)
    }

    fn eat(&mut self, words: &[&str]) -> bool {
        let found = self.at(words);
        if found {
            self.pos += words.len();
        }
        found
    }

    fn eat_phrase(&mut self, phrase: &str) -> bool {
        self.eat(&phrase.split('_').collect::<Vec<_>>())
    }

    /// Longest target phrase starting at `pos`, with whether it was plural.
    fn target_at(&self, pos: usize) -> Option<(usize, Target, bool)> {
        let mut best: Option<(usize, Target, bool)> = None;
        for entry in catalog::all() {
            for (phrases, plural) in [(entry.singular, false), (entry.plural, true)] {
                for phrase in phrases {
                    let words: Vec<&str> = phrase.split('_').collect();
                    if self.tokens.get(pos..pos + words.len()) == Some(&words[..])
                        && best.is_none_or(|(len, ..)| words.len() > len)
                    {
                        best = Some((words.len(), entry.target, plural));
                    }
                }
            }
        }
        best
    }

    fn count(&mut self) -> Result<Option<u32>> {
        let token = self.peek();
        let n = match NUMBER_WORDS.iter().position(|w| *w == token) {
            Some(i) => i as u32 + 1,
            None if !token.is_empty() && token.bytes().all(|b| b.is_ascii_digit()) => {
                match token.parse() {
                    Ok(n @ 1..=100) => n,
                    _ => {
                        return Err(self.error(
                            ErrorKind::Malformed,
                            format!("count {token} is not 1 to 100"),
                        ));
                    }
                }
            }
            None => return Ok(None),
        };
        self.pos += 1;
        Ok(Some(n))
    }

    fn percent(&mut self, clause: &str) -> Result<u32> {
        let token = self.peek();
        let digits = if let Some(digits) = token.strip_suffix("pct") {
            self.pos += 1;
            digits
        } else if matches!(self.tokens.get(self.pos + 1), Some(&"pct" | &"percent")) {
            self.pos += 2;
            token
        } else {
            return Err(self.error(
                ErrorKind::Malformed,
                format!("{clause} needs a percentage, e.g. {clause}_5pct"),
            ));
        };
        match digits.parse() {
            Ok(pct @ 1..=100) => Ok(pct),
            _ => Err(self.error(
                ErrorKind::Malformed,
                format!("{clause} percentage {digits} is not 1 to 100"),
            )),
        }
    }

    fn region(&mut self) -> Result<RegionSpec> {
        if self.eat_phrase("region") {
            return Ok(RegionSpec::Caller);
        }
        for (names, rect) in NAMED_REGIONS {
            if names.iter().any(|name| self.eat_phrase(name)) {
                return Ok(RegionSpec::Fixed { rect: rect() });
            }
        }
        let word = self.peek();
        let error = if [
            "top", "bottom", "left", "right", "upper", "lower", "corner", "side",
        ]
        .contains(&word)
        {
            self.error(
                ErrorKind::Ambiguous,
                format!("in_{word} does not say how much of the image"),
            )
        } else if let Some((_, why)) = AMBIGUOUS_WORDS.iter().find(|(w, _)| *w == word) {
            self.error(
                ErrorKind::Ambiguous,
                format!("in_{word} is ambiguous: {why}"),
            )
        } else {
            self.error(
                ErrorKind::Unsupported,
                format!("in_{word} is not a region Syrup knows"),
            )
        };
        Err(error.with_hint(grammar_summary()))
    }

    fn ambiguous(&self, word: &str) -> Option<SyrupError> {
        AMBIGUOUS_WORDS
            .iter()
            .find(|(w, _)| *w == word)
            .map(|(_, why)| self.error(ErrorKind::Ambiguous, format!("{word} is ambiguous: {why}")))
    }
}

pub fn parse(name: &str) -> Result<Intent> {
    let well_formed = !name.is_empty()
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
        && name.split('_').all(|word| !word.is_empty())
        && !name.as_bytes()[0].is_ascii_digit();
    let mut p = Parser {
        name,
        tokens: name.split('_').collect(),
        pos: 0,
    };
    if !well_formed {
        return Err(malformed(
            &p,
            "names are lowercase words joined by single underscores".into(),
        ));
    }

    let verb = p.peek();
    p.pos += 1;
    let intent = if verb == "measure" {
        measure(&mut p)?
    } else if verb == "track" {
        Intent {
            track: true,
            ..find_body(&mut p)?
        }
    } else if let Some((_, why)) = OTHER_VERBS.iter().find(|(v, _)| *v == verb) {
        return Err(p.error(
            ErrorKind::Unsupported,
            format!("{verb} is not supported: {why}"),
        ));
    } else if FIND_VERBS.contains(&verb) {
        find_body(&mut p)?
    } else {
        return Err(malformed(&p, format!("{verb} is not a verb Syrup knows")));
    };
    intent.check().map_err(|e| e.for_operation(name))?;
    Ok(intent)
}

fn malformed(p: &Parser, reason: String) -> SyrupError {
    p.error(ErrorKind::Malformed, reason)
        .with_hint(grammar_summary())
}

/// `<quantity>[_of_<find body>|_in_<region>]`
fn measure(p: &mut Parser) -> Result<Intent> {
    let word = p.peek();
    let Some(quantity) = Quantity::named(word) else {
        let (kind, reason) = match word {
            "" => (ErrorKind::Malformed, "measure what?".to_string()),
            _ => (
                ErrorKind::Unsupported,
                format!("Syrup cannot measure {word}"),
            ),
        };
        return Err(p
            .error(kind, reason)
            .with_hint(format!("quantities: {}", quantity_names())));
    };
    p.pos += 1;
    let mut intent = if p.eat(&["of"]) {
        find_body(p)?
    } else {
        let mut intent = Intent::find(Target::Image);
        clauses(p, &mut intent)?;
        intent
    };
    intent.measure = Some(quantity);
    Ok(intent)
}

/// Everything after a find verb: the target with its count, selector and
/// colour, then clauses.
fn find_body(p: &mut Parser) -> Result<Intent> {
    let mut count = None;
    let mut selector: Option<(OrderKey, &str)> = None;
    let mut all = false;
    let mut color = None;
    let (target, plural) = loop {
        if let Some((len, target, plural)) = p.target_at(p.pos) {
            p.pos += len;
            break (target, plural);
        }
        if p.pos == p.tokens.len() {
            return Err(malformed(p, "the name has no target".into()));
        }
        if let Some(n) = p.count()? {
            if count.replace(n).is_some() {
                return Err(p.error(ErrorKind::Conflicting, "more than one count"));
            }
            continue;
        }
        if let Some((words, key)) = SELECTORS.iter().find(|(words, _)| p.at(words)) {
            p.pos += words.len();
            if selector.replace((*key, words[0])).is_some() {
                return Err(p.error(ErrorKind::Conflicting, "more than one selector"));
            }
            continue;
        }
        if p.eat(&["all"]) {
            all = true;
            continue;
        }
        let word = p.peek();
        if let Some(c) = catalog::color_named(word) {
            p.pos += 1;
            if color.replace(c).is_some() {
                return Err(p.error(ErrorKind::Conflicting, "more than one colour"));
            }
            continue;
        }
        if NON_HUES.contains(&word) {
            return Err(p
                .error(
                    ErrorKind::Unsupported,
                    format!("{word} has no hue Syrup can match"),
                )
                .with_hint(format!("colours: {}", color_names())));
        }
        if let Some(e) = p.ambiguous(word) {
            return Err(e);
        }
        if UNSUPPORTED_QUALIFIERS.contains(&word) {
            return Err(p.error(
                ErrorKind::Unsupported,
                format!(
                    "{word} needs a capability Syrup does not have, and it will not be ignored"
                ),
            ));
        }
        let target_later = (p.pos..p.tokens.len()).any(|at| p.target_at(at).is_some());
        let reason = if target_later {
            format!("{word} is not a word Syrup understands, and unknown words are not dropped")
        } else {
            format!(
                "nothing in the catalog finds {}",
                p.tokens[p.pos..].join("_")
            )
        };
        return Err(p
            .error(ErrorKind::Unsupported, reason)
            .with_hint(format!("known targets: {}", catalog::known_targets())));
    };

    let mut intent = Intent::find(target);
    intent.color = color;
    let order_clause = clauses(p, &mut intent)?;

    match (selector, count) {
        (Some((key, word)), count) => {
            if all {
                return Err(p.error(ErrorKind::Conflicting, format!("all contradicts {word}")));
            }
            if let Some(clause) = order_clause {
                return Err(p.error(
                    ErrorKind::Conflicting,
                    format!("{word} already orders the results; {clause} contradicts it"),
                ));
            }
            if plural && count.is_none() {
                let entry = catalog::entry(target);
                return Err(p
                    .error(ErrorKind::Ambiguous, format!("how many {word} results?"))
                    .with_hint(format!(
                        "find_{word}_{} for one, find_3_{word}_{} for three",
                        entry.singular[0], entry.plural[0]
                    )));
            }
            intent.order = key;
            intent.limit = Some(count.unwrap_or(1));
        }
        (None, Some(n)) => {
            let plural = catalog::entry(target).plural[0];
            return Err(p
                .error(
                    ErrorKind::Ambiguous,
                    format!("{n} of which? A count needs an ordering"),
                )
                .with_hint(format!(
                    "find_{n}_largest_{plural} or find_{n}_most_confident_{plural}"
                )));
        }
        (None, None) => {}
    }

    Ok(intent)
}

/// Clauses after the target; returns the ordering clause, if any.
fn clauses<'a>(p: &mut Parser<'a>, intent: &mut Intent) -> Result<Option<&'a str>> {
    let mut order_clause: Option<&'a str> = None;
    while p.pos < p.tokens.len() {
        if p.eat(&["in"]) {
            if intent.region.is_some() {
                return Err(p.error(ErrorKind::Conflicting, "more than one region"));
            }
            intent.region = Some(p.region()?);
        } else if p.eat(&["by"]) {
            let word = p.peek();
            let Some((_, key)) = ORDER_KEYS.iter().find(|(w, _)| *w == word) else {
                let kind = if ["position", "location"].contains(&word) {
                    ErrorKind::Ambiguous
                } else {
                    ErrorKind::Unsupported
                };
                return Err(p
                    .error(kind, format!("cannot order by {word}"))
                    .with_hint("by_size, by_confidence, left_to_right, top_to_bottom, ..."));
            };
            p.pos += 1;
            if order_clause.replace(word).is_some() {
                return Err(p.error(ErrorKind::Conflicting, "more than one ordering"));
            }
            intent.order = *key;
        } else if let Some((words, key)) = DIRECTIONS.iter().find(|(words, _)| p.at(words)) {
            p.pos += words.len();
            if order_clause.replace(words[0]).is_some() {
                return Err(p.error(ErrorKind::Conflicting, "more than one ordering"));
            }
            intent.order = *key;
        } else if p.eat(&["larger", "than"]) || p.eat(&["bigger", "than"]) {
            if intent
                .min_area_pct
                .replace(p.percent("larger_than")?)
                .is_some()
            {
                return Err(p.error(ErrorKind::Conflicting, "more than one larger_than"));
            }
        } else if p.eat(&["smaller", "than"]) {
            if intent
                .max_area_pct
                .replace(p.percent("smaller_than")?)
                .is_some()
            {
                return Err(p.error(ErrorKind::Conflicting, "more than one smaller_than"));
            }
        } else if let Some(e) = p.ambiguous(p.peek()) {
            return Err(e);
        } else {
            let rest = p.tokens[p.pos..].join("_");
            return Err(malformed(
                p,
                format!("{rest} is not a clause Syrup understands"),
            ));
        }
    }

    Ok(order_clause)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(name: &str) -> Intent {
        parse(name).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    fn kind(name: &str) -> ErrorKind {
        let e = parse(name).expect_err(name);
        assert_eq!(
            (e.stage, e.operation.as_deref()),
            (Stage::Resolve, Some(name))
        );
        e.kind
    }

    #[test]
    fn synonyms_and_plurals_share_one_intent() {
        for name in [
            "find_faces",
            "detect_faces",
            "locate_face",
            "find_human_faces",
            "find_all_faces",
        ] {
            assert_eq!(ok(name), ok("find_face"), "{name}");
        }
        assert_eq!(ok("find_face"), Intent::find(Target::Face));
        assert_eq!(
            ok("find_faces_in_center_by_size"),
            ok("find_faces_by_size_in_centre")
        );
    }

    #[test]
    fn clauses_compose() {
        let intent = ok("find_2_largest_faces_in_top_half_larger_than_2pct");
        assert_eq!(
            (intent.order, intent.limit, intent.min_area_pct),
            (OrderKey::AreaDesc, Some(2), Some(2))
        );
        assert!(
            matches!(intent.region, Some(RegionSpec::Fixed { rect }) if rect.name() == Some("top_half"))
        );

        assert_eq!(ok("find_faces_left_to_right").order, OrderKey::LeftToRight);
        assert_eq!(ok("find_most_confident_face").limit, Some(1));
        assert_eq!(ok("find_three_leftmost_faces").limit, Some(3));
        assert_eq!(ok("find_faces_in_region").region, Some(RegionSpec::Caller));
        let bars = ok("find_2_largest_red_bars_in_bottom_third");
        assert_eq!(
            (bars.target, bars.color, bars.limit),
            (Target::Bar, Some(Color::Red), Some(2))
        );
        assert_eq!(ok("find_violet_blobs"), ok("find_purple_regions"));
        assert_eq!(
            ok("find_faces_smaller_than_10_percent").max_area_pct,
            Some(10)
        );
    }

    #[test]
    fn refuses_to_guess() {
        let cases = [
            ("find_best_face", ErrorKind::Ambiguous),
            ("find_main_face", ErrorKind::Ambiguous),
            ("find_first_face", ErrorKind::Ambiguous),
            ("find_highest_face", ErrorKind::Ambiguous),
            ("find_large_faces", ErrorKind::Ambiguous),
            ("find_two_faces", ErrorKind::Ambiguous),
            ("find_largest_faces", ErrorKind::Ambiguous),
            ("find_faces_in_middle", ErrorKind::Ambiguous),
            ("find_faces_in_top", ErrorKind::Ambiguous),
            ("find_faces_by_position", ErrorKind::Ambiguous),
            ("find_smiling_faces", ErrorKind::Unsupported),
            ("find_cat_faces", ErrorKind::Unsupported),
            ("find_wobbly_faces", ErrorKind::Unsupported),
            ("find_cars", ErrorKind::Unsupported),
            ("find_people", ErrorKind::Unsupported),
            ("count_faces", ErrorKind::Unsupported),
            ("find_regions", ErrorKind::Ambiguous),
            ("find_white_regions", ErrorKind::Unsupported),
            ("find_red_faces", ErrorKind::Unsupported),
            ("find_red_blue_regions", ErrorKind::Conflicting),
            (
                "find_faces_in_top_half_in_left_half",
                ErrorKind::Conflicting,
            ),
            ("find_largest_face_by_confidence", ErrorKind::Conflicting),
            ("find_all_largest_face", ErrorKind::Conflicting),
            ("find_faces_by_size_left_to_right", ErrorKind::Conflicting),
            (
                "find_faces_larger_than_10pct_smaller_than_5pct",
                ErrorKind::Conflicting,
            ),
            ("find_0_largest_faces", ErrorKind::Malformed),
            ("find_101_largest_faces", ErrorKind::Malformed),
            ("find_faces_larger_than", ErrorKind::Malformed),
            ("find_faces_larger_than_0pct", ErrorKind::Malformed),
            ("find_faces_please", ErrorKind::Malformed),
            ("frobnicate_face", ErrorKind::Malformed),
            ("find", ErrorKind::Malformed),
        ];
        for (name, expected) in cases {
            assert_eq!(kind(name), expected, "{name}");
        }
        for name in [
            "",
            "Find_face",
            "find__face",
            "_find_face",
            "find_face_",
            "2find_face",
        ] {
            assert_eq!(
                parse(name).expect_err(name).kind,
                ErrorKind::Malformed,
                "{name:?}"
            );
        }
    }

    #[test]
    fn measurements_compose_with_what_find_takes() {
        let sharpness = ok("measure_sharpness");
        assert_eq!(
            (sharpness.target, sharpness.measure),
            (Target::Image, Some(Quantity::Sharpness))
        );
        assert!(ok("measure_sharpness_in_center").region.is_some());
        let words = ok("measure_sharpness_of_words_left_to_right");
        assert_eq!(
            (words.target, words.order),
            (Target::Word, OrderKey::LeftToRight)
        );
        let fill = ok("measure_fill_of_largest_red_bar_in_bottom_third");
        assert_eq!(
            (fill.target, fill.color, fill.limit, fill.measure),
            (Target::Bar, Some(Color::Red), Some(1), Some(Quantity::Fill))
        );
        assert_ne!(ok("measure_sharpness_of_faces"), ok("find_faces"));
        assert!(ok("track_faces").track && !ok("find_faces").track);
        assert_eq!(ok("track_moving_blobs").target, Target::MovingRegion);

        for (name, expected) in [
            ("measure", ErrorKind::Malformed),
            ("measure_weight", ErrorKind::Unsupported),
            ("measure_fill", ErrorKind::Ambiguous),
            ("measure_fill_of_faces", ErrorKind::Unsupported),
            ("measure_fill_of_bars", ErrorKind::Ambiguous),
            ("measure_sharpness_by_size", ErrorKind::Conflicting),
            ("measure_sharpness_larger_than_5pct", ErrorKind::Conflicting),
            ("measure_sharpness_of", ErrorKind::Malformed),
            ("find_sharpness", ErrorKind::Unsupported),
            ("find_moving_regions", ErrorKind::Unsupported),
        ] {
            assert_eq!(kind(name), expected, "{name}");
        }
    }

    #[test]
    fn adjacent_regions_partition_the_image() {
        let pixels = |name: &str, h: u32| match ok(name).region {
            Some(RegionSpec::Fixed { rect }) => rect.pixels(10, h),
            other => panic!("{other:?}"),
        };
        for h in [1, 2, 5, 7, 480, 481] {
            let (top, bottom) = (
                pixels("find_faces_in_top_half", h),
                pixels("find_faces_in_bottom_half", h),
            );
            assert_eq!(
                (top.y, top.y + top.h, bottom.y + bottom.h),
                (0, bottom.y, h),
                "height {h}"
            );
        }
        assert_eq!(pixels("find_faces_in_top_half", 5).h, 3, "2.5 rounds up");
    }

    #[test]
    fn ratios_are_exact() {
        assert_eq!(Ratio::new(2, 4), Ratio::new(1, 2));
        assert_eq!(Ratio::new(2, 3).of(10), 7);
        assert_eq!(Ratio::new(1, 2).of(5), 3);
        assert_eq!(Ratio::from_f64(0.5), Some(Ratio::new(1, 2)));
        assert_eq!(Ratio::from_f64(1.5), None);
    }
}
