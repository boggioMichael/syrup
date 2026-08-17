//! The shared result vocabulary for every detector built on this library.
//!
//! Instead of returning a bare `Option<T>`, detectors wrap their output in
//! [`Detection<T>`], which always carries a [`Confidence`] score, a capture
//! [`Timestamp`], a short label naming the technique that produced it, a
//! [`Reliability`] estimate, and a human-readable failure reason when
//! detection did not succeed. That makes "detector honesty" visible in the
//! type: callers cannot silently treat "not found" the same as "found but
//! uncertain".

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// A confidence score in the closed range `[0.0, 1.0]`.
///
/// `0.0` means "no evidence", `1.0` means "certain". Detectors should prefer
/// graded estimates over binary 0/1 whenever one is available.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Confidence(f32);

impl Confidence {
    pub const NONE: Confidence = Confidence(0.0);
    pub const CERTAIN: Confidence = Confidence(1.0);

    /// Build a confidence value, clamping to `[0.0, 1.0]`.
    pub fn new(value: f32) -> Self {
        Self(value.clamp(0.0, 1.0))
    }

    pub fn value(self) -> f32 {
        self.0
    }

    /// Combine two independent pieces of evidence (probabilistic OR).
    ///
    /// Used when a detector corroborates one signal with a second,
    /// independent one (e.g. OCR text confirming a color-based match).
    /// Never decreases confidence; saturates at 1.0.
    pub fn combine(self, other: Confidence) -> Confidence {
        Confidence::new(self.0 + other.0 - self.0 * other.0)
    }

    /// Scale confidence down, e.g. to account for a stale observation.
    pub fn decay(self, factor: f32) -> Confidence {
        Confidence::new(self.0 * factor.clamp(0.0, 1.0))
    }

    pub fn is_confident(self, threshold: f32) -> bool {
        self.0 >= threshold
    }
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.0}%", self.0 * 100.0)
    }
}

/// Wall-clock capture timestamp in milliseconds since the Unix epoch, so it
/// is `Copy`, comparable, and cheap to carry on every detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(pub u128);

impl Timestamp {
    /// Capture "now" as a timestamp.
    pub fn now() -> Self {
        let millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        Self(millis)
    }
}

/// A qualitative estimate of how trustworthy the technique behind a
/// detection is, independent of the numeric confidence of this one sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reliability {
    /// Backed by multiple independent signals (e.g. geometry + OCR).
    Corroborated,
    /// A single geometric/color heuristic with no independent confirmation.
    Heuristic,
    /// Derived purely from history/prediction with no direct evidence this
    /// frame (e.g. an object presumed present during brief occlusion).
    Predicted,
    /// Detection failed outright.
    Unreliable,
}

/// The result of running a detector once. Always present, even on failure,
/// so failure metadata (the reason, the source) is not lost.
#[derive(Debug, Clone)]
pub struct Detection<T> {
    pub value: Option<T>,
    pub confidence: Confidence,
    pub timestamp: Timestamp,
    /// Short label naming the detector/technique that produced this value,
    /// e.g. `"motion"` or `"hud"`. Consumers define their own vocabulary.
    pub source: &'static str,
    pub reliability: Reliability,
    pub failure_reason: Option<String>,
}

impl<T> Detection<T> {
    /// Build a successful detection.
    pub fn found(
        value: T,
        confidence: Confidence,
        source: &'static str,
        reliability: Reliability,
    ) -> Self {
        Self {
            value: Some(value),
            confidence,
            timestamp: Timestamp::now(),
            source,
            reliability,
            failure_reason: None,
        }
    }

    /// Build a failed detection with an explanation, so downstream consumers
    /// can distinguish "absent" from "not evaluated".
    pub fn missing(source: &'static str, reason: impl Into<String>) -> Self {
        Self {
            value: None,
            confidence: Confidence::NONE,
            timestamp: Timestamp::now(),
            source,
            reliability: Reliability::Unreliable,
            failure_reason: Some(reason.into()),
        }
    }

    pub fn is_present(&self) -> bool {
        self.value.is_some()
    }

    pub fn as_ref(&self) -> Detection<&T> {
        Detection {
            value: self.value.as_ref(),
            confidence: self.confidence,
            timestamp: self.timestamp,
            source: self.source,
            reliability: self.reliability,
            failure_reason: self.failure_reason.clone(),
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Detection<U> {
        Detection {
            value: self.value.map(f),
            confidence: self.confidence,
            timestamp: self.timestamp,
            source: self.source,
            reliability: self.reliability,
            failure_reason: self.failure_reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_clamps_and_combines() {
        assert_eq!(Confidence::new(1.5).value(), 1.0);
        assert_eq!(Confidence::new(-0.2).value(), 0.0);

        let combined = Confidence::new(0.5).combine(Confidence::new(0.5));
        assert!((combined.value() - 0.75).abs() < 1e-6);
        // Saturation is approximate under f32 arithmetic, not exact.
        let saturated = Confidence::CERTAIN.combine(Confidence::new(0.3));
        assert!((saturated.value() - 1.0).abs() < 1e-6);
        let identity = Confidence::NONE.combine(Confidence::new(0.3));
        assert!((identity.value() - 0.3).abs() < 1e-6);
    }

    #[test]
    fn detection_missing_has_no_value_and_a_reason() {
        let detection: Detection<u32> = Detection::missing("motion", "no frames yet");
        assert!(!detection.is_present());
        assert_eq!(detection.confidence, Confidence::NONE);
        assert_eq!(detection.reliability, Reliability::Unreliable);
        assert!(detection.failure_reason.is_some());
    }

    #[test]
    fn detection_found_round_trips_value() {
        let detection =
            Detection::found(42u32, Confidence::new(0.8), "test", Reliability::Heuristic);
        assert_eq!(detection.value, Some(42));
        assert!(detection.confidence.is_confident(0.5));
    }
}
