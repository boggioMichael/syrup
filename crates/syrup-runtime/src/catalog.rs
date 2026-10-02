//! Targets Syrup can find and how each one is found. Built-in targets are
//! entries here; targets added at run time live in [`crate::custom`]. The
//! grammar, planner and generator are shared by both.

use serde::Serialize;

use crate::abi;
use crate::custom;

/// The noun of a target added at run time. Interned, so it stays `Copy`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct Name(pub(crate) &'static str);

impl Name {
    pub fn as_str(self) -> &'static str {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Target {
    Face,
    Word,
    Region,
    Bar,
    /// What changed since the previous frame; only sessions see frames.
    MovingRegion,
    QrCode,
    TextBlock,
    Panel,
    /// The searched image or region as one item; only measurements use it.
    Image,
    Custom(Name),
}

/// What `measure_*` operations measure, each a fraction in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Quantity {
    /// The share of strong luminance steps that are abrupt, as the core's
    /// assess_text_quality measures it. Below 0.35 text is too blurred to
    /// read reliably.
    Sharpness,
    /// How full a bar is, as the core's measure_bar_fill measures it.
    Fill,
}

impl Quantity {
    pub const ALL: [Quantity; 2] = [Quantity::Sharpness, Quantity::Fill];

    pub fn name(self) -> &'static str {
        match self {
            Quantity::Sharpness => "sharpness",
            Quantity::Fill => "fill",
        }
    }

    pub fn named(word: &str) -> Option<Quantity> {
        Quantity::ALL.into_iter().find(|q| q.name() == word)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    FaceDetection,
    TextRecognition,
    Motion,
    QrDecoding,
    TextBlocks,
    Panels,
    Custom(Name),
}

impl Capability {
    /// Run-time capabilities get an id derived from their name, so a module
    /// compiled in one process calls the same detector in another.
    pub fn abi_id(self) -> u32 {
        match self {
            Capability::FaceDetection => abi::SYRUP_CAP_FACE,
            Capability::TextRecognition => abi::SYRUP_CAP_TEXT,
            Capability::Motion => abi::SYRUP_CAP_MOTION,
            Capability::QrDecoding => abi::SYRUP_CAP_QR,
            Capability::TextBlocks => abi::SYRUP_CAP_TEXT_BLOCKS,
            Capability::Panels => abi::SYRUP_CAP_PANELS,
            Capability::Custom(name) => custom_id(name.as_str()),
        }
    }

    pub fn from_abi_id(id: u32) -> Option<Self> {
        match id {
            abi::SYRUP_CAP_FACE => Some(Capability::FaceDetection),
            abi::SYRUP_CAP_TEXT => Some(Capability::TextRecognition),
            abi::SYRUP_CAP_MOTION => Some(Capability::Motion),
            abi::SYRUP_CAP_QR => Some(Capability::QrDecoding),
            abi::SYRUP_CAP_TEXT_BLOCKS => Some(Capability::TextBlocks),
            abi::SYRUP_CAP_PANELS => Some(Capability::Panels),
            id => custom::entries().into_iter().find_map(|e| match e.finder {
                Finder::Detect(c @ Capability::Custom(_)) if c.abi_id() == id => Some(c),
                _ => None,
            }),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Capability::FaceDetection => "face_detection",
            Capability::TextRecognition => "text_recognition",
            Capability::Motion => "motion",
            Capability::QrDecoding => "qr_decoding",
            Capability::TextBlocks => "text_blocks",
            Capability::Panels => "panels",
            Capability::Custom(name) => name.as_str(),
        }
    }
}

/// FNV-1a with the top bit set, clear of the built-in ids.
pub(crate) fn custom_id(name: &str) -> u32 {
    let hash = name.bytes().fold(0x811c_9dc5u32, |h, b| {
        (h ^ b as u32).wrapping_mul(0x0100_0193)
    });
    hash | 0x8000_0000
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Color {
    Red,
    Orange,
    Yellow,
    Green,
    Cyan,
    Blue,
    Purple,
    Magenta,
}

#[derive(Debug)]
pub struct ColorEntry {
    pub color: Color,
    pub names: &'static [&'static str],
    /// Hue range in degrees, wrapping through 360 when the start is larger.
    pub hue: (u32, u32),
}

/// Neighbouring hue ranges, so every saturated colour has exactly one name.
pub const COLORS: &[ColorEntry] = &[
    ColorEntry {
        color: Color::Red,
        names: &["red"],
        hue: (340, 20),
    },
    ColorEntry {
        color: Color::Orange,
        names: &["orange"],
        hue: (20, 45),
    },
    ColorEntry {
        color: Color::Yellow,
        names: &["yellow"],
        hue: (45, 70),
    },
    ColorEntry {
        color: Color::Green,
        names: &["green"],
        hue: (70, 165),
    },
    ColorEntry {
        color: Color::Cyan,
        names: &["cyan"],
        hue: (165, 195),
    },
    ColorEntry {
        color: Color::Blue,
        names: &["blue"],
        hue: (195, 255),
    },
    ColorEntry {
        color: Color::Purple,
        names: &["purple", "violet"],
        hue: (255, 290),
    },
    ColorEntry {
        color: Color::Magenta,
        names: &["magenta"],
        hue: (290, 340),
    },
];

// Below these a pixel is grey, white or black rather than a colour; the
// core's bar detection uses the same thresholds.
pub const MIN_SATURATION_PCT: u32 = 35;
pub const MIN_VALUE_PCT: u32 = 30;

impl Color {
    pub fn name(self) -> &'static str {
        color(self).names[0]
    }
}

pub fn color(color: Color) -> &'static ColorEntry {
    COLORS
        .iter()
        .find(|c| c.color == color)
        .expect("every colour is in the table")
}

/// How a target is found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finder {
    /// By a detector the host provides.
    Detect(Capability),
    /// By the generated module: runs of one colour, grouped into regions.
    Color {
        min_run: u32,
        min_height: u32,
        max_gap: u32,
        /// Keep regions at least this many times wider than tall.
        min_aspect: Option<u32>,
    },
    /// The whole searched view, as one item.
    Whole,
}

#[derive(Debug)]
pub struct TargetEntry {
    pub target: Target,
    pub label: &'static str,
    // Noun phrases, words joined by `_`.
    pub singular: &'static [&'static str],
    pub plural: &'static [&'static str],
    pub finder: Finder,
    pub default_min_confidence: f32,
    pub keypoints: &'static [&'static str],
    pub description: &'static str,
}

pub const TARGETS: &[TargetEntry] = &[
    TargetEntry {
        target: Target::Face,
        label: "face",
        singular: &["face", "human_face"],
        plural: &["faces", "human_faces"],
        finder: Finder::Detect(Capability::FaceDetection),
        // YuNet stays below ~0.4 on the non-face fixtures and above ~0.85 on faces.
        default_min_confidence: 0.6,
        // YuNet's order; right/left are the subject's.
        keypoints: &[
            "right_eye",
            "left_eye",
            "nose_tip",
            "right_mouth_corner",
            "left_mouth_corner",
        ],
        description: "where human faces are; never who, or how they look",
    },
    TargetEntry {
        target: Target::Word,
        label: "word",
        singular: &["word"],
        plural: &["words"],
        finder: Finder::Detect(Capability::TextRecognition),
        // Tesseract scores legible print above 0.9; below half it is guessing.
        default_min_confidence: 0.5,
        keypoints: &[],
        description: "words of printed text, with what they say",
    },
    TargetEntry {
        target: Target::Region,
        label: "region",
        singular: &["region", "blob"],
        plural: &["regions", "blobs"],
        finder: Finder::Color {
            min_run: 3,
            min_height: 3,
            max_gap: 1,
            min_aspect: None,
        },
        // Confidence is the share of the box's pixels that have the colour.
        default_min_confidence: 0.0,
        keypoints: &[],
        description: "areas of one colour, e.g. red_regions",
    },
    TargetEntry {
        target: Target::Bar,
        label: "bar",
        singular: &["bar"],
        plural: &["bars"],
        // The core's find_color_bar uses runs of at least 8 and 2-row gaps.
        finder: Finder::Color {
            min_run: 8,
            min_height: 2,
            max_gap: 2,
            min_aspect: Some(3),
        },
        default_min_confidence: 0.0,
        keypoints: &[],
        description: "bars of one colour, at least 3 times wider than tall, e.g. red_bars",
    },
    TargetEntry {
        target: Target::QrCode,
        label: "qr_code",
        singular: &["qr_code"],
        plural: &["qr_codes"],
        finder: Finder::Detect(Capability::QrDecoding),
        // Decoded codes score 1; unreadable ones 0.5.
        default_min_confidence: 0.6,
        // rqrr's order, in the code's own orientation.
        keypoints: &["top_left", "top_right", "bottom_right", "bottom_left"],
        description: "QR codes, with what they say",
    },
    TargetEntry {
        target: Target::TextBlock,
        label: "text_block",
        singular: &["text_block"],
        plural: &["text_blocks"],
        finder: Finder::Detect(Capability::TextBlocks),
        // Confidence is the share of the block's pixels that look like text.
        default_min_confidence: 0.0,
        keypoints: &[],
        description: "the block of text-like pixels in the searched region, at most one per region",
    },
    TargetEntry {
        target: Target::Panel,
        label: "panel",
        singular: &["panel"],
        plural: &["panels"],
        finder: Finder::Detect(Capability::Panels),
        // Confidence is the share of the panel's pixels in its colour.
        default_min_confidence: 0.0,
        keypoints: &[],
        description: "the largest panel of the searched region's dominant colour, at most one per region",
    },
    TargetEntry {
        target: Target::MovingRegion,
        label: "moving_region",
        singular: &["moving_region", "moving_blob"],
        plural: &["moving_regions", "moving_blobs"],
        finder: Finder::Detect(Capability::Motion),
        // Confidence is the share of the box's pixels that changed.
        default_min_confidence: 0.0,
        keypoints: &[],
        description: "what changed since the previous frame, in track_ sessions",
    },
    TargetEntry {
        target: Target::Image,
        label: "image",
        // No nouns: `find_image` would mean nothing; measure_sharpness does.
        singular: &[],
        plural: &[],
        finder: Finder::Whole,
        default_min_confidence: 0.0,
        keypoints: &[],
        description: "the searched image or region",
    },
];

/// Built-in targets, then those added at run time.
pub fn all() -> Vec<&'static TargetEntry> {
    TARGETS.iter().chain(custom::entries()).collect()
}

pub fn entry(target: Target) -> &'static TargetEntry {
    all()
        .into_iter()
        .find(|e| e.target == target)
        .expect("targets are only made by the catalog")
}

/// A target by any of its nouns, e.g. `face` or `human_faces`.
pub fn target_named(noun: &str) -> Option<Target> {
    all()
        .into_iter()
        .find(|e| e.singular.contains(&noun) || e.plural.contains(&noun))
        .map(|e| e.target)
}

pub fn color_named(word: &str) -> Option<Color> {
    COLORS
        .iter()
        .find(|c| c.names.contains(&word))
        .map(|c| c.color)
}

pub fn known_targets() -> String {
    all()
        .into_iter()
        .filter(|e| !e.plural.is_empty())
        .map(|e| format!("{} ({})", e.plural[0], e.description))
        .collect::<Vec<_>>()
        .join("; ")
}
