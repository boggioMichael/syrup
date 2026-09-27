//! Functions you name instead of write.
//!
//! Declare the function you wish existed, and the library works out what
//! it means from its name, builds it out of the primitives, and runs it:
//!
//! ```
//! use syrup::Detection;
//! use syrup::image::RgbaImage;
//! use syrup::intent::Match;
//!
//! syrup::intent!(fn find_face(image: &RgbaImage) -> Detection<Vec<Match>>);
//!
//! let frame = RgbaImage::new(64, 64);
//! let faces = find_face(&frame);
//! assert_eq!(faces.value.map(|f| f.len()), Some(0)); // searched, nothing there
//! ```
//!
//! The name is parsed against a small closed vocabulary — a verb, optional
//! qualifiers, a noun — and turned into a [`Plan`]: a composition of the
//! library's primitives with concrete parameters. A plan can run in-process,
//! or be written out as Rust source, compiled to a shared library and
//! loaded back, which is how an invented function becomes a `.so`/`.dll`
//! any process can use ([`Resolved::compile`], [`native`]).
//!
//! ```text
//!   "find_red_bar"
//!      │ parse      Intent { verb: Find, qualifiers: [Red], noun: Bar }
//!      │ plan       Plan::ColorBars { hue: (340, 20), … }
//!      ├ execute    in-process: plans::find_bars(image, region, …)
//!      └ compile    src → cargo build (cdylib) → libloading → same call, native
//! ```
//!
//! # What a call returns
//!
//! | verb        | outcome                    |
//! |-------------|----------------------------|
//! | `find_*`    | `Vec<Match>` — 0 or more places, strongest first |
//! | `track_*`   | `Vec<Match>` — the same, with a stable `id` per object across calls |
//! | `count_*`   | `f32` — how many `find_*` would return |
//! | `measure_*` | `f32` — a percentage or distance |
//! | `read_*`    | `String` — text |
//!
//! Qualifiers go between the verb and the noun: a colour (`red`, `green`,
//! …), a position (`top`, `bottom`, `left`, `right`: that half of the
//! region), a pick (`largest`, `smallest`: just that one), `profile` for
//! faces turned to the side, and for icons the picture's name
//! (`find_boss_icon` uses the template registered as `boss`; see
//! [`register_template`]).
//!
//! An empty `find_*` result with `value: Some(vec![])` means the search ran
//! and found nothing; `value: None` with a `failure_reason` means it could
//! not run — the two are never conflated.
//!
//! # When it refuses
//!
//! Resolution fails, with an [`IntentError`] that says why, when the name
//! is outside the vocabulary ([`IntentError::Unparsed`]), when the library
//! knows what is meant but cannot do it yet ([`IntentError::Unsupported`]),
//! or when a native build or load fails. It never guesses at a meaning.
//! At run time, an implementation reports `Detection::missing` with a
//! reason rather than a low-confidence answer dressed up as a result.

pub mod codegen;
pub mod native;
pub mod plans;
mod templates;

use std::fmt;
use std::sync::{Arc, Mutex, RwLock};

use image::RgbaImage;

use crate::cascade::CascadeOptions;
use crate::detection::Detection;
use crate::geometry::Rect;
pub use templates::{register_template, template_named};

/// Something found by a `find_*` or `track_*` intent.
#[derive(Debug, Clone, PartialEq)]
pub struct Match {
    pub bounds: Rect,
    /// How strong the evidence is, in `[0, 1]`; meaning depends on the plan
    /// (correlation, cluster support, area).
    pub score: f32,
    pub centre: (f32, f32),
    /// A stable identity across calls, from `track_*` intents only.
    pub id: Option<u64>,
}

impl Match {
    pub fn new(bounds: Rect, score: f32) -> Self {
        Self {
            bounds,
            score,
            centre: bounds.center(),
            id: None,
        }
    }
}

/// What an intent produces; which variant depends on its verb.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Matches(Vec<Match>),
    Scalar(f32),
    Text(String),
}

/// Which of [`Outcome`]'s shapes a verb yields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutcomeKind {
    Matches,
    Scalar,
    Text,
}

impl fmt::Display for OutcomeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            OutcomeKind::Matches => "a list of matches",
            OutcomeKind::Scalar => "a number",
            OutcomeKind::Text => "text",
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verb {
    Find,
    Track,
    Count,
    Measure,
    Read,
}

impl Verb {
    pub fn outcome(self) -> OutcomeKind {
        match self {
            Verb::Find | Verb::Track => OutcomeKind::Matches,
            Verb::Count | Verb::Measure => OutcomeKind::Scalar,
            Verb::Read => OutcomeKind::Text,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Noun {
    Face,
    Eye,
    Bar,
    Blob,
    Text,
    Icon,
    Motion,
}

/// Which half of the region a position qualifier selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Half {
    Top,
    Bottom,
    Left,
    Right,
}

impl Half {
    /// That half of `region`.
    pub fn of(self, region: Rect) -> Rect {
        let (w2, h2) = (region.w / 2, region.h / 2);
        match self {
            Half::Top => Rect { h: h2, ..region },
            Half::Bottom => Rect {
                y: region.y + h2,
                h: region.h - h2,
                ..region
            },
            Half::Left => Rect { w: w2, ..region },
            Half::Right => Rect {
                x: region.x + w2,
                w: region.w - w2,
                ..region
            },
        }
    }
}

/// Which single match a pick qualifier keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pick {
    Largest,
    Smallest,
}

/// A named colour, as a hue window in degrees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Color {
    Red,
    Orange,
    Yellow,
    Green,
    Cyan,
    Blue,
    Purple,
    Magenta,
    Pink,
}

impl Color {
    /// The hue window `(from, to)` in degrees, wrapping through 360.
    pub fn hue(self) -> (f32, f32) {
        match self {
            Color::Red => (340.0, 20.0),
            Color::Orange => (20.0, 45.0),
            Color::Yellow => (45.0, 70.0),
            Color::Green => (70.0, 170.0),
            Color::Cyan => (170.0, 200.0),
            Color::Blue => (200.0, 260.0),
            Color::Purple => (260.0, 300.0),
            Color::Magenta => (300.0, 330.0),
            Color::Pink => (320.0, 350.0),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Color::Red => "red",
            Color::Orange => "orange",
            Color::Yellow => "yellow",
            Color::Green => "green",
            Color::Cyan => "cyan",
            Color::Blue => "blue",
            Color::Purple => "purple",
            Color::Magenta => "magenta",
            Color::Pink => "pink",
        }
    }
}

const VERBS: &[(&str, Verb)] = &[
    ("find", Verb::Find),
    ("track", Verb::Track),
    ("count", Verb::Count),
    ("measure", Verb::Measure),
    ("read", Verb::Read),
];

const NOUNS: &[(&str, Noun)] = &[
    ("face", Noun::Face),
    ("eye", Noun::Eye),
    ("bar", Noun::Bar),
    ("blob", Noun::Blob),
    ("region", Noun::Blob),
    ("text", Noun::Text),
    ("icon", Noun::Icon),
    ("motion", Noun::Motion),
];

const HALVES: &[(&str, Half)] = &[
    ("top", Half::Top),
    ("bottom", Half::Bottom),
    ("left", Half::Left),
    ("right", Half::Right),
];

const PICKS: &[(&str, Pick)] = &[
    ("largest", Pick::Largest),
    ("biggest", Pick::Largest),
    ("smallest", Pick::Smallest),
];

const COLORS: &[(&str, Color)] = &[
    ("red", Color::Red),
    ("orange", Color::Orange),
    ("yellow", Color::Yellow),
    ("green", Color::Green),
    ("cyan", Color::Cyan),
    ("blue", Color::Blue),
    ("purple", Color::Purple),
    ("violet", Color::Purple),
    ("magenta", Color::Magenta),
    ("pink", Color::Pink),
];

/// The meaning of a function name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Intent {
    pub name: String,
    pub verb: Verb,
    pub noun: Noun,
    pub colors: Vec<Color>,
    pub half: Option<Half>,
    pub pick: Option<Pick>,
    /// Qualifier words that are none of the above, kept for plans that can
    /// use them (`find_boss_icon` keeps `boss`; `find_profile_face` keeps
    /// `profile`).
    pub labels: Vec<String>,
}

/// Why an intent could not be turned into a running function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntentError {
    /// The name does not parse against the vocabulary.
    Unparsed { name: String, hint: String },
    /// The meaning is clear but nothing here can carry it out.
    Unsupported { name: String, missing: String },
    /// The declared return type does not match what the verb yields.
    WrongShape {
        name: String,
        yields: OutcomeKind,
        declared: OutcomeKind,
    },
    /// Compiling the native implementation failed; `log` is the compiler's.
    BuildFailed { name: String, log: String },
    /// The compiled library could not be loaded or does not export the ABI.
    LoadFailed { name: String, reason: String },
}

impl fmt::Display for IntentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntentError::Unparsed { name, hint } => {
                write!(f, "`{name}` is not an intent I understand: {hint}")
            }
            IntentError::Unsupported { name, missing } => {
                write!(f, "`{name}` is understood but cannot be built: {missing}")
            }
            IntentError::WrongShape {
                name,
                yields,
                declared,
            } => {
                write!(
                    f,
                    "`{name}` yields {yields}, but it was declared to return {declared}"
                )
            }
            IntentError::BuildFailed { name, log } => {
                write!(f, "building `{name}` natively failed:\n{log}")
            }
            IntentError::LoadFailed { name, reason } => {
                write!(f, "loading `{name}` failed: {reason}")
            }
        }
    }
}

impl std::error::Error for IntentError {}

/// The vocabulary, for error hints and documentation.
pub fn vocabulary() -> String {
    let list = |items: &[&str]| items.join(", ");
    format!(
        "names are <verb>_[qualifiers_]<noun>; verbs: {}; nouns: {}; qualifiers: colours ({}), positions ({}), picks ({}), `profile`, or an icon's name",
        list(&VERBS.iter().map(|(w, _)| *w).collect::<Vec<_>>()),
        list(&NOUNS.iter().map(|(w, _)| *w).collect::<Vec<_>>()),
        list(&COLORS.iter().map(|(w, _)| *w).collect::<Vec<_>>()),
        list(&HALVES.iter().map(|(w, _)| *w).collect::<Vec<_>>()),
        list(&PICKS.iter().map(|(w, _)| *w).collect::<Vec<_>>()),
    )
}

/// Parse a function name into its meaning.
pub fn parse(name: &str) -> Result<Intent, IntentError> {
    let unparsed = |hint: String| IntentError::Unparsed {
        name: name.to_string(),
        hint,
    };
    let tokens: Vec<&str> = name.split('_').filter(|t| !t.is_empty()).collect();
    if tokens.len() < 2 {
        return Err(unparsed(format!(
            "expected at least a verb and a noun; {}",
            vocabulary()
        )));
    }
    let verb = VERBS
        .iter()
        .find(|(w, _)| *w == tokens[0])
        .map(|(_, v)| *v)
        .ok_or_else(|| unparsed(format!("`{}` is not a verb; {}", tokens[0], vocabulary())))?;
    // A position may trail the noun (`find_red_bars_bottom`) as well as
    // precede it (`find_bottom_red_bars`).
    let mut half = None;
    let mut tokens = tokens;
    if tokens.len() > 2
        && let Some((_, h)) = HALVES.iter().find(|(w, _)| *w == tokens[tokens.len() - 1])
    {
        half = Some(*h);
        tokens.pop();
    }
    let last = tokens[tokens.len() - 1];
    let singular = last
        .strip_suffix('s')
        .filter(|s| NOUNS.iter().any(|(w, _)| w == s))
        .unwrap_or(last);
    let noun = NOUNS
        .iter()
        .find(|(w, _)| *w == singular)
        .map(|(_, n)| *n)
        .ok_or_else(|| unparsed(format!("`{last}` is not a noun; {}", vocabulary())))?;
    let mut colors = Vec::new();
    let mut labels = Vec::new();
    let mut pick = None;
    for token in &tokens[1..tokens.len() - 1] {
        if let Some((_, color)) = COLORS.iter().find(|(w, _)| w == token) {
            colors.push(*color);
        } else if let Some((_, h)) = HALVES.iter().find(|(w, _)| w == token) {
            half = Some(*h);
        } else if let Some((_, p)) = PICKS.iter().find(|(w, _)| w == token) {
            pick = Some(*p);
        } else {
            labels.push(token.to_string());
        }
    }
    Ok(Intent {
        name: name.to_string(),
        verb,
        noun,
        colors,
        half,
        pick,
        labels,
    })
}

/// A concrete composition of primitives that carries out an intent.
///
/// Leaves search; the wrapping variants narrow the region, keep one
/// match, count, or add identity across calls.
#[derive(Debug, Clone, PartialEq)]
pub enum Plan {
    /// Run a bundled face cascade.
    Faces {
        profile: bool,
        options: CascadeOptions,
    },
    /// Eyes: found inside frontal faces, so a wall socket is not an eye.
    Eyes { options: CascadeOptions },
    /// Horizontal bars of one colour.
    ColorBars { color: Color },
    /// Connected regions of one colour.
    ColorBlobs { color: Color },
    /// The block of text-like pixels.
    TextBlock,
    /// Every place a picture appears.
    Icon {
        name: String,
        template: Arc<RgbaImage>,
    },
    /// What moved since the previous call.
    Motion,
    /// Fill of the largest bar of one colour, as a percentage.
    BarFill { color: Color },
    /// OCR over the region.
    Ocr,
    /// Search only one half of the region.
    Within { half: Half, inner: Box<Plan> },
    /// Keep one match.
    Keep { pick: Pick, inner: Box<Plan> },
    /// How many the inner plan finds.
    Count { inner: Box<Plan> },
    /// The inner plan's matches with identities kept across calls.
    Track { inner: Box<Plan> },
}

impl Plan {
    pub fn outcome(&self) -> OutcomeKind {
        match self {
            Plan::Faces { .. }
            | Plan::Eyes { .. }
            | Plan::ColorBars { .. }
            | Plan::ColorBlobs { .. }
            | Plan::TextBlock
            | Plan::Icon { .. }
            | Plan::Motion
            | Plan::Track { .. } => OutcomeKind::Matches,
            Plan::BarFill { .. } | Plan::Count { .. } => OutcomeKind::Scalar,
            Plan::Ocr => OutcomeKind::Text,
            Plan::Within { inner, .. } | Plan::Keep { inner, .. } => inner.outcome(),
        }
    }

    /// Whether running the plan keeps something between calls.
    pub fn is_stateful(&self) -> bool {
        match self {
            Plan::Motion | Plan::Track { .. } => true,
            Plan::Within { inner, .. } | Plan::Keep { inner, .. } | Plan::Count { inner } => {
                inner.is_stateful()
            }
            _ => false,
        }
    }
}

/// Choose a plan for an intent, or say what is missing.
pub fn plan(intent: &Intent) -> Result<Plan, IntentError> {
    let unsupported = |missing: String| IntentError::Unsupported {
        name: intent.name.clone(),
        missing,
    };
    let one_color = || -> Result<Color, IntentError> {
        match intent.colors.as_slice() {
            [color] => Ok(*color),
            [] => Err(unsupported("say which colour, e.g. `red`".into())),
            _ => Err(unsupported("one colour at a time".into())),
        }
    };
    let labels: Vec<&str> = intent.labels.iter().map(String::as_str).collect();

    // What is being looked for, before the verb decides what to do with it.
    let subject = match intent.noun {
        Noun::Face => {
            let profile = labels.contains(&"profile");
            let unknown: Vec<&str> = labels.iter().copied().filter(|l| *l != "profile").collect();
            if !unknown.is_empty() {
                return Err(unsupported(format!(
                    "faces can be `profile`, not `{}`",
                    unknown.join("`, `")
                )));
            }
            Plan::Faces {
                profile,
                options: CascadeOptions::default(),
            }
        }
        Noun::Eye => Plan::Eyes {
            options: CascadeOptions::default(),
        },
        Noun::Bar => Plan::ColorBars {
            color: one_color()?,
        },
        Noun::Blob => Plan::ColorBlobs {
            color: one_color()?,
        },
        Noun::Text => Plan::TextBlock,
        Noun::Icon => {
            let Some(name) = labels.first() else {
                return Err(unsupported("say which icon, e.g. `find_boss_icon`".into()));
            };
            match template_named(name) {
                Some(template) => Plan::Icon {
                    name: name.to_string(),
                    template,
                },
                None => {
                    return Err(unsupported(format!(
                        "no picture named `{name}`: register one with syrup::intent::register_template(\"{name}\", &image), or put {name}.png in $SYRUP_TEMPLATES"
                    )));
                }
            }
        }
        Noun::Motion => Plan::Motion,
    };

    let subject = match intent.pick {
        Some(_) if matches!(intent.verb, Verb::Measure | Verb::Read) => {
            return Err(unsupported(format!(
                "`{}` picks one of several matches, but this name yields {}",
                intent.name,
                intent.verb.outcome()
            )));
        }
        Some(pick) => Plan::Keep {
            pick,
            inner: Box::new(subject),
        },
        None => subject,
    };
    let mut plan = match (intent.verb, intent.noun) {
        (Verb::Measure, Noun::Bar) => Plan::BarFill {
            color: one_color()?,
        },
        (Verb::Measure, _) => {
            return Err(unsupported(
                "only bars can be measured (measure_<colour>_bar)".into(),
            ));
        }
        (Verb::Read, Noun::Text) => Plan::Ocr,
        (Verb::Read, _) => return Err(unsupported("only text can be read (read_text)".into())),
        (Verb::Find, _) => subject,
        (Verb::Count, _) => Plan::Count {
            inner: Box::new(subject),
        },
        (Verb::Track, Noun::Motion) => subject,
        (Verb::Track, _) => Plan::Track {
            inner: Box::new(subject),
        },
    };
    if let Some(half) = intent.half {
        plan = Plan::Within {
            half,
            inner: Box::new(plan),
        };
    }
    Ok(plan)
}

/// What a stateful plan keeps between calls.
#[derive(Default)]
pub struct State {
    pub motion: Option<crate::motion::MotionDetector>,
    pub tracker: Option<crate::tracking::ObjectTracker>,
}

impl State {
    pub const fn new() -> Self {
        Self {
            motion: None,
            tracker: None,
        }
    }
}

/// Run a plan in-process; `state` carries what stateful plans remember.
pub fn execute(
    plan: &Plan,
    state: &mut State,
    image: &RgbaImage,
    region: Rect,
) -> Detection<Outcome> {
    match plan {
        Plan::Faces { profile, options } => plans::find_faces(image, region, *profile, options),
        Plan::Eyes { options } => plans::find_eyes(image, region, options),
        Plan::ColorBars { color } => plans::find_bars(image, region, color.hue()),
        Plan::ColorBlobs { color } => plans::find_blobs(image, region, color.hue()),
        Plan::TextBlock => plans::find_text(image, region),
        Plan::Icon { template, .. } => plans::find_icon(image, region, template),
        Plan::Motion => plans::find_motion(state, image, region),
        Plan::BarFill { color } => plans::measure_bar(image, region, color.hue()),
        Plan::Ocr => plans::read_text(image, region),
        Plan::Within { half, inner } => execute(inner, state, image, half.of(region)),
        Plan::Keep { pick, inner } => plans::keep(*pick, execute(inner, state, image, region)),
        Plan::Count { inner } => plans::count(execute(inner, state, image, region)),
        Plan::Track { inner } => {
            let found = execute(inner, state, image, region);
            plans::track(state, found)
        }
    }
}

/// An intent that has been understood and can be run.
pub struct Resolved {
    intent: Intent,
    plan: Plan,
    /// What stateful plans remember between in-process calls.
    state: Mutex<State>,
    /// Read-locked for the duration of a call, so calls run concurrently
    /// and only [`Resolved::compile`] waits for them.
    native: RwLock<Option<native::Library>>,
}

impl fmt::Debug for Resolved {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Resolved")
            .field("intent", &self.intent)
            .field("plan", &self.plan)
            .field("native", &self.is_native())
            .finish()
    }
}

impl Resolved {
    pub fn intent(&self) -> &Intent {
        &self.intent
    }

    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The Rust source of this intent's implementation — what would be
    /// compiled by [`Resolved::compile`].
    pub fn source(&self) -> String {
        codegen::source(&self.intent.name, &self.plan)
    }

    /// Compile the implementation to a shared library (cached) and use it
    /// for every call from now on. Needs `cargo` on the machine.
    pub fn compile(&self) -> Result<std::path::PathBuf, IntentError> {
        let generated = codegen::generate(&self.intent.name, &self.plan);
        let library = native::build_and_load(&self.intent.name, &generated)?;
        let path = library.path().to_path_buf();
        *self.native.write().unwrap_or_else(|e| e.into_inner()) = Some(library);
        Ok(path)
    }

    /// Whether calls go through a loaded shared library.
    pub fn is_native(&self) -> bool {
        self.native
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// Where the loaded shared library lives, if any.
    pub fn library_path(&self) -> Option<std::path::PathBuf> {
        self.native
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .map(|l| l.path().to_path_buf())
    }

    /// Run on `region` of `image` (the whole image when `None`).
    pub fn run(&self, image: &RgbaImage, region: Option<Rect>) -> Detection<Outcome> {
        let region = region.unwrap_or(Rect {
            x: 0,
            y: 0,
            w: image.width(),
            h: image.height(),
        });
        let native = self.native.read().unwrap_or_else(|e| e.into_inner());
        match native.as_ref() {
            Some(library) => library.run(image, region),
            None => {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                execute(&self.plan, &mut state, image, region)
            }
        }
    }
}

/// Understand `name` and prepare to run it in-process.
pub fn resolve(name: &str) -> Result<Resolved, IntentError> {
    let intent = parse(name)?;
    let plan = plan(&intent)?;
    Ok(Resolved {
        intent,
        plan,
        state: Mutex::new(State::new()),
        native: RwLock::new(None),
    })
}

/// [`resolve`], then [`Resolved::compile`]: the function as a shared
/// library, loaded.
pub fn resolve_native(name: &str) -> Result<Resolved, IntentError> {
    let resolved = resolve(name)?;
    resolved.compile()?;
    Ok(resolved)
}

/// Resolve and check that the verb yields `T`, so a mismatch between the
/// declared return type and the name is reported once, up front.
pub fn resolve_as<T: FromOutcome>(name: &str) -> Result<Resolved, IntentError> {
    let resolved = if std::env::var_os("SYRUP_INTENT_MODE").is_some_and(|m| m == "native") {
        resolve_native(name)?
    } else {
        resolve(name)?
    };
    if resolved.plan.outcome() != T::KIND {
        return Err(IntentError::WrongShape {
            name: name.to_string(),
            yields: resolved.plan.outcome(),
            declared: T::KIND,
        });
    }
    Ok(resolved)
}

/// The Rust type an [`Outcome`] variant unpacks into.
pub trait FromOutcome: Sized {
    const KIND: OutcomeKind;
    fn from_outcome(outcome: Outcome) -> Option<Self>;

    fn from_detection(detection: Detection<Outcome>) -> Detection<Self> {
        let Detection {
            value,
            confidence,
            timestamp,
            source,
            reliability,
            failure_reason,
        } = detection;
        Detection {
            value: value.and_then(Self::from_outcome),
            confidence,
            timestamp,
            source,
            reliability,
            failure_reason,
        }
    }
}

impl FromOutcome for Vec<Match> {
    const KIND: OutcomeKind = OutcomeKind::Matches;
    fn from_outcome(outcome: Outcome) -> Option<Self> {
        match outcome {
            Outcome::Matches(m) => Some(m),
            _ => None,
        }
    }
}

impl FromOutcome for f32 {
    const KIND: OutcomeKind = OutcomeKind::Scalar;
    fn from_outcome(outcome: Outcome) -> Option<Self> {
        match outcome {
            Outcome::Scalar(v) => Some(v),
            _ => None,
        }
    }
}

impl FromOutcome for String {
    const KIND: OutcomeKind = OutcomeKind::Text;
    fn from_outcome(outcome: Outcome) -> Option<Self> {
        match outcome {
            Outcome::Text(t) => Some(t),
            _ => None,
        }
    }
}

/// Declare a function by name and let the library implement it.
///
/// ```
/// use syrup::{Detection, Rect};
/// use syrup::image::RgbaImage;
/// use syrup::intent::Match;
///
/// syrup::intent!(fn find_face(image: &RgbaImage) -> Detection<Vec<Match>>);
/// syrup::intent!(pub fn measure_red_bar(image: &RgbaImage, region: Rect) -> Detection<f32>);
/// ```
///
/// The intent is resolved on the first call and kept. If the name cannot
/// be resolved, every call returns `Detection::missing` with the reason;
/// [`resolve`](crate::intent::resolve) gives the typed error up front. Set
/// `SYRUP_INTENT_MODE=native` to have the first call compile and load the
/// implementation as a shared library instead of running it in-process.
///
/// The parameter types are spelled exactly `&RgbaImage` and `Rect` (the
/// macro matches them by name and resolves them to the library's types),
/// and the return type is one of `Vec<Match>`, `f32` or `String`.
#[macro_export]
macro_rules! intent {
    ($(#[$meta:meta])* $vis:vis fn $name:ident ($image:ident : &RgbaImage) -> Detection<$ret:ty>) => {
        $(#[$meta])*
        $vis fn $name($image: &$crate::image::RgbaImage) -> $crate::Detection<$ret> {
            $crate::intent!(@call $name, $ret, $image, None)
        }
    };
    ($(#[$meta:meta])* $vis:vis fn $name:ident ($image:ident : &RgbaImage, $region:ident : Rect) -> Detection<$ret:ty>) => {
        $(#[$meta])*
        $vis fn $name($image: &$crate::image::RgbaImage, $region: $crate::Rect) -> $crate::Detection<$ret> {
            $crate::intent!(@call $name, $ret, $image, Some($region))
        }
    };
    (@call $name:ident, $ret:ty, $image:expr, $region:expr) => {{
        static RESOLVED: ::std::sync::OnceLock<Result<$crate::intent::Resolved, $crate::intent::IntentError>> =
            ::std::sync::OnceLock::new();
        match RESOLVED.get_or_init(|| $crate::intent::resolve_as::<$ret>(stringify!($name))) {
            Ok(resolved) => {
                <$ret as $crate::intent::FromOutcome>::from_detection(resolved.run($image, $region))
            }
            Err(error) => $crate::Detection::missing("intent", error.to_string()),
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_parse_into_their_parts() {
        let intent = parse("find_red_bar").unwrap();
        assert_eq!(intent.verb, Verb::Find);
        assert_eq!(intent.noun, Noun::Bar);
        assert_eq!(intent.colors, vec![Color::Red]);
        assert!(intent.labels.is_empty());

        let intent = parse("find_faces").unwrap();
        assert_eq!(intent.noun, Noun::Face);

        let intent = parse("find_hp_green_bars").unwrap();
        assert_eq!(intent.colors, vec![Color::Green]);
        assert_eq!(intent.labels, vec!["hp".to_string()]);

        assert_eq!(parse("measure_blue_bar").unwrap().verb, Verb::Measure);
        assert_eq!(parse("read_text").unwrap().verb, Verb::Read);

        let intent = parse("count_largest_red_blobs_bottom").unwrap();
        assert_eq!(intent.verb, Verb::Count);
        assert_eq!(intent.pick, Some(Pick::Largest));
        assert_eq!(intent.half, Some(Half::Bottom));
        assert_eq!(intent.colors, vec![Color::Red]);
        let intent = parse("find_profile_face").unwrap();
        assert_eq!(intent.labels, vec!["profile".to_string()]);
        let intent = parse("track_boss_icon").unwrap();
        assert_eq!((intent.verb, intent.noun), (Verb::Track, Noun::Icon));
        assert_eq!(intent.labels, vec!["boss".to_string()]);
    }

    #[test]
    fn unknown_words_are_refused_with_a_hint() {
        for name in ["find_unicorn", "summon_face", "face", "find", "findface"] {
            match parse(name) {
                Err(IntentError::Unparsed { hint, .. }) => {
                    assert!(
                        hint.contains("verbs:"),
                        "{name}: hint should list the vocabulary, got {hint}"
                    )
                }
                other => panic!("{name}: expected Unparsed, got {other:?}"),
            }
        }
        // A trailing s only pluralises known nouns.
        assert!(matches!(
            parse("find_bars"),
            Ok(Intent {
                noun: Noun::Bar,
                ..
            })
        ));
    }

    #[test]
    fn plans_say_what_they_need() {
        assert!(matches!(
            resolve("find_face").unwrap().plan(),
            Plan::Faces { profile: false, .. }
        ));
        assert!(matches!(
            resolve("find_profile_faces").unwrap().plan(),
            Plan::Faces { profile: true, .. }
        ));
        assert!(matches!(
            resolve("find_eyes").unwrap().plan(),
            Plan::Eyes { .. }
        ));
        assert_eq!(
            resolve("find_red_bar").unwrap().plan(),
            &Plan::ColorBars { color: Color::Red }
        );
        assert_eq!(
            resolve("measure_blue_bar").unwrap().plan(),
            &Plan::BarFill { color: Color::Blue }
        );
        assert_eq!(resolve("read_text").unwrap().plan(), &Plan::Ocr);
        assert_eq!(resolve("find_text").unwrap().plan(), &Plan::TextBlock);
        assert_eq!(resolve("find_motion").unwrap().plan(), &Plan::Motion);
        assert_eq!(resolve("track_motion").unwrap().plan(), &Plan::Motion);
        assert_eq!(
            resolve("count_largest_red_blobs_bottom").unwrap().plan(),
            &Plan::Within {
                half: Half::Bottom,
                inner: Box::new(Plan::Count {
                    inner: Box::new(Plan::Keep {
                        pick: Pick::Largest,
                        inner: Box::new(Plan::ColorBlobs { color: Color::Red })
                    })
                })
            }
        );
        assert_eq!(
            resolve("track_green_blobs").unwrap().plan(),
            &Plan::Track {
                inner: Box::new(Plan::ColorBlobs {
                    color: Color::Green
                })
            }
        );

        let missing = |name: &str| match resolve(name) {
            Err(IntentError::Unsupported { missing, .. }) => missing,
            other => panic!("{name}: expected Unsupported, got {other:?}"),
        };
        assert!(missing("find_bar").contains("colour"));
        assert!(missing("find_red_blue_bar").contains("one colour"));
        assert!(missing("find_boss_icon").contains("register_template"));
        assert!(missing("find_icon").contains("which icon"));
        assert!(missing("find_happy_face").contains("profile"));
        assert!(missing("measure_face").contains("bars"));
        assert!(missing("read_face").contains("text"));
        assert!(missing("measure_largest_red_bar").contains("picks one"));
    }

    #[test]
    fn registered_pictures_make_icon_intents_resolvable() {
        let picture = RgbaImage::from_fn(6, 6, |x, y| {
            image::Rgba([x as u8 * 40, y as u8 * 40, 0, 255])
        });
        register_template("resolvetesticon", &picture);
        let resolved = resolve("find_resolvetesticon_icon");
        assert!(
            matches!(resolved.as_ref().map(|r| r.plan()), Ok(Plan::Icon { name, .. }) if name == "resolvetesticon")
        );
    }

    #[test]
    fn halves_split_a_region() {
        let region = Rect {
            x: 10,
            y: 20,
            w: 100,
            h: 50,
        };
        assert_eq!(
            Half::Top.of(region),
            Rect {
                x: 10,
                y: 20,
                w: 100,
                h: 25
            }
        );
        assert_eq!(
            Half::Bottom.of(region),
            Rect {
                x: 10,
                y: 45,
                w: 100,
                h: 25
            }
        );
        assert_eq!(
            Half::Left.of(region),
            Rect {
                x: 10,
                y: 20,
                w: 50,
                h: 50
            }
        );
        assert_eq!(
            Half::Right.of(region),
            Rect {
                x: 60,
                y: 20,
                w: 50,
                h: 50
            }
        );
    }

    #[test]
    fn the_declared_type_must_match_the_verb() {
        match resolve_as::<f32>("find_face") {
            Err(IntentError::WrongShape {
                yields, declared, ..
            }) => {
                assert_eq!(yields, OutcomeKind::Matches);
                assert_eq!(declared, OutcomeKind::Scalar);
            }
            other => panic!("expected WrongShape, got {other:?}"),
        }
        assert!(resolve_as::<Vec<Match>>("find_face").is_ok());
        assert!(resolve_as::<f32>("measure_red_bar").is_ok());
        assert!(resolve_as::<String>("read_text").is_ok());
    }

    #[test]
    fn errors_read_as_sentences() {
        let text = resolve("find_unicorn").unwrap_err().to_string();
        assert!(text.starts_with("`find_unicorn` is not an intent I understand"));
        let text = resolve("find_boss_icon").unwrap_err().to_string();
        assert!(text.contains("understood but cannot be built"));
    }

    crate::intent!(fn count_red_bars(image: &RgbaImage) -> Detection<f32>);
    crate::intent!(fn find_largest_red_blob(image: &RgbaImage) -> Detection<Vec<Match>>);
    crate::intent!(fn find_red_blobs_right(image: &RgbaImage) -> Detection<Vec<Match>>);
    crate::intent!(fn track_red_blobs(image: &RgbaImage) -> Detection<Vec<Match>>);
    crate::intent!(fn find_motion(image: &RgbaImage) -> Detection<Vec<Match>>);

    #[test]
    fn qualifiers_and_counts_compose() {
        let image = frame_with_bar();
        assert_eq!(count_red_bars(&image).value, Some(1.0));
        let largest = find_largest_red_blob(&image).value.unwrap();
        assert_eq!(largest.len(), 1);
        assert_eq!(
            largest[0].bounds,
            Rect {
                x: 10,
                y: 15,
                w: 60,
                h: 10
            }
        );
        // The bar spans x 10..70; the right half of a 120-wide frame starts
        // at 60, so only its tail is in there.
        let right = find_red_blobs_right(&image).value.unwrap();
        assert_eq!(right.len(), 1);
        assert_eq!(
            right[0].bounds,
            Rect {
                x: 60,
                y: 15,
                w: 10,
                h: 10
            }
        );
    }

    #[test]
    fn tracked_matches_keep_their_ids_between_calls() {
        let image = frame_with_bar();
        let first = track_red_blobs(&image).value.unwrap();
        let second = track_red_blobs(&image).value.unwrap();
        assert_eq!(first.len(), 1);
        assert!(first[0].id.is_some());
        assert_eq!(first[0].id, second[0].id, "the same bar keeps its identity");
        assert_eq!(second[0].bounds, first[0].bounds);
    }

    #[test]
    fn motion_needs_a_previous_frame_then_reports_change() {
        let still = frame_with_bar();
        let warm_up = find_motion(&still);
        assert!(
            warm_up.value.is_none(),
            "the first call has nothing to compare with"
        );
        assert!(warm_up.failure_reason.unwrap().contains("warming up"));
        let nothing = find_motion(&still);
        assert_eq!(
            nothing.value.as_ref().map(Vec::len),
            Some(0),
            "an identical frame has no motion"
        );
        let mut moved = still.clone();
        for y in 15..25 {
            for x in 10..70 {
                moved.put_pixel(x + 30, y, image::Rgba([220, 30, 30, 255]));
                moved.put_pixel(x, y, image::Rgba([40, 40, 40, 255]));
            }
        }
        let motion = find_motion(&moved).value.unwrap();
        assert!(!motion.is_empty(), "the bar moved");
        assert!(motion.iter().all(|m| m.id.is_some()));
    }

    crate::intent!(fn find_face(image: &RgbaImage) -> Detection<Vec<Match>>);
    crate::intent!(fn find_red_bar(image: &RgbaImage, region: Rect) -> Detection<Vec<Match>>);
    crate::intent!(fn measure_red_bar(image: &RgbaImage) -> Detection<f32>);
    crate::intent!(fn find_unicorn(image: &RgbaImage) -> Detection<Vec<Match>>);
    crate::intent!(fn measure_face(image: &RgbaImage) -> Detection<Vec<Match>>);

    fn frame_with_bar() -> RgbaImage {
        let mut image = RgbaImage::from_pixel(120, 40, image::Rgba([40, 40, 40, 255]));
        for y in 15..25 {
            for x in 10..70 {
                image.put_pixel(x, y, image::Rgba([220, 30, 30, 255]));
            }
            for x in 70..110 {
                // The empty track: a bluish groove, clearly neither the
                // red fill nor the grey panel around the bar.
                image.put_pixel(x, y, image::Rgba([70, 70, 110, 255]));
            }
        }
        image
    }

    #[test]
    fn declared_functions_run_their_plans() {
        let image = frame_with_bar();
        let bars = find_red_bar(
            &image,
            Rect {
                x: 0,
                y: 0,
                w: 120,
                h: 40,
            },
        );
        let bars = bars.value.expect("searched");
        assert_eq!(bars.len(), 1);
        assert_eq!(
            bars[0].bounds,
            Rect {
                x: 10,
                y: 15,
                w: 60,
                h: 10
            }
        );

        let fill = measure_red_bar(&image);
        let fill = fill.value.expect("measured");
        assert!((fill - 60.0).abs() < 2.0, "fill {fill}");

        let faces = find_face(&image);
        assert_eq!(
            faces.value,
            Some(vec![]),
            "no face in a bar: searched, nothing there"
        );
    }

    #[test]
    fn unresolvable_declarations_explain_themselves_on_every_call() {
        let image = frame_with_bar();
        let result = find_unicorn(&image);
        assert!(result.value.is_none());
        assert!(
            result
                .failure_reason
                .unwrap()
                .contains("not an intent I understand")
        );
        let result = measure_face(&image);
        assert!(result.failure_reason.unwrap().contains("cannot be built"));
    }
}
