//! Comparing sequences of measurements that may run at different speeds:
//! dynamic time warping, and a nearest-example matcher built on it.
//!
//! A word mouthed quickly and the same word mouthed slowly trace the same
//! path through mouth-shape space at different rates. Dynamic time warping
//! (Sakoe & Chiba 1978) finds the alignment between two sequences that
//! minimises the summed distance between aligned samples, so it compares
//! the paths rather than the clocks. The [`Matcher`] keeps labelled
//! example sequences and names the label of the nearest one, refusing
//! when the nearest is not clearly nearer than the runner-up.

/// Euclidean distance between two samples.
fn distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

/// Dynamic-time-warping distance between two sequences of samples,
/// normalised by the longer sequence's length so sequences of different
/// lengths are comparable (normalising by the alignment path would let a
/// long, meandering path dilute its cost). `band` limits how far the alignment
/// may stray from the diagonal, as a fraction of the longer sequence (1.0:
/// unconstrained). Empty sequences are infinitely far from anything.
pub fn dtw(a: &[Vec<f32>], b: &[Vec<f32>], band: f32) -> f32 {
    if a.is_empty() || b.is_empty() {
        return f32::INFINITY;
    }
    let (n, m) = (a.len(), b.len());
    let window = ((n.max(m) as f32 * band.clamp(0.0, 1.0)).ceil() as usize).max(n.abs_diff(m));
    // cost[i][j] = best summed distance aligning a[..i] with b[..j].
    let inf = f32::INFINITY;
    let mut cost = vec![inf; (n + 1) * (m + 1)];
    let at = |i: usize, j: usize| i * (m + 1) + j;
    cost[at(0, 0)] = 0.0;
    for i in 1..=n {
        let lo = i.saturating_sub(window).max(1);
        let hi = (i + window).min(m);
        for j in lo..=hi {
            let d = distance(&a[i - 1], &b[j - 1]);
            let best = cost[at(i - 1, j - 1)]
                .min(cost[at(i - 1, j)])
                .min(cost[at(i, j - 1)]);
            if best.is_finite() {
                cost[at(i, j)] = best + d;
            }
        }
    }
    cost[at(n, m)] / n.max(m) as f32
}

/// A labelled example sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub label: String,
    pub samples: Vec<Vec<f32>>,
}

/// What the matcher decided for a sequence.
#[derive(Debug, Clone, PartialEq)]
pub struct Recognition {
    /// The nearest example's label.
    pub label: String,
    /// Its distance.
    pub distance: f32,
    /// How much nearer it is than the nearest example with another label,
    /// as a fraction of that runner-up's distance (0: a tie; 1: the runner-up
    /// is twice as far). `None` when there is no other label to compare to.
    pub margin: Option<f32>,
}

/// Names sequences after the nearest labelled example.
#[derive(Debug, Clone, Default)]
pub struct Matcher {
    examples: Vec<Example>,
    /// See [`dtw`].
    pub band: f32,
    /// Minimum [`Recognition::margin`] to answer rather than refuse.
    pub min_margin: f32,
}

impl Matcher {
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
            band: 0.3,
            min_margin: 0.15,
        }
    }

    pub fn learn(&mut self, label: &str, samples: Vec<Vec<f32>>) {
        if !samples.is_empty() {
            self.examples.push(Example {
                label: label.to_string(),
                samples,
            });
        }
    }

    pub fn examples(&self) -> &[Example] {
        &self.examples
    }

    pub fn labels(&self) -> Vec<&str> {
        let mut labels: Vec<&str> = self.examples.iter().map(|e| e.label.as_str()).collect();
        labels.sort_unstable();
        labels.dedup();
        labels
    }

    /// The nearest label and how sure the choice is, or `None` with no
    /// examples or an empty sequence.
    pub fn nearest(&self, samples: &[Vec<f32>]) -> Option<Recognition> {
        if samples.is_empty() {
            return None;
        }
        let mut best: Option<(&str, f32)> = None;
        let mut runner_up: Option<f32> = None;
        // Nearest example per label first, then the best two labels.
        let mut per_label: Vec<(&str, f32)> = Vec::new();
        for example in &self.examples {
            let d = dtw(samples, &example.samples, self.band);
            match per_label.iter_mut().find(|(l, _)| *l == example.label) {
                Some(entry) => entry.1 = entry.1.min(d),
                None => per_label.push((&example.label, d)),
            }
        }
        for (label, d) in per_label {
            match best {
                None => best = Some((label, d)),
                Some((_, bd)) if d < bd => {
                    runner_up = Some(bd);
                    best = Some((label, d));
                }
                Some(_) => runner_up = Some(runner_up.map_or(d, |r| r.min(d))),
            }
        }
        let (label, distance) = best?;
        let margin = runner_up.map(|r| if r > 0.0 { (r - distance) / r } else { 0.0 });
        Some(Recognition {
            label: label.to_string(),
            distance,
            margin,
        })
    }

    /// [`Matcher::nearest`], refusing (`None`) when the best label is not
    /// clearly better than the next.
    pub fn recognise(&self, samples: &[Vec<f32>]) -> Option<Recognition> {
        let recognition = self.nearest(samples)?;
        match recognition.margin {
            Some(margin) if margin < self.min_margin => None,
            _ => Some(recognition),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(len: usize, peak: f32) -> Vec<Vec<f32>> {
        (0..len)
            .map(|i| {
                let t = i as f32 / (len - 1).max(1) as f32;
                vec![peak * (std::f32::consts::PI * t).sin()]
            })
            .collect()
    }

    #[test]
    fn identical_sequences_are_at_distance_zero_and_speed_does_not_matter() {
        let slow = ramp(20, 1.0);
        let fast = ramp(8, 1.0);
        assert_eq!(dtw(&slow, &slow, 1.0), 0.0);
        let same_shape = dtw(&slow, &fast, 1.0);
        let other_shape = dtw(&slow, &ramp(20, 0.3), 1.0);
        assert!(same_shape < other_shape, "{same_shape} vs {other_shape}");
        assert_eq!(dtw(&[], &slow, 1.0), f32::INFINITY);
    }

    #[test]
    fn the_matcher_names_the_nearest_label_and_refuses_ties() {
        let mut matcher = Matcher::new();
        matcher.learn("open", ramp(12, 1.0));
        matcher.learn("open", ramp(16, 1.0));
        matcher.learn("half", ramp(12, 0.5));
        matcher.learn("flat", vec![vec![0.0]; 12]);
        assert_eq!(matcher.labels(), vec!["flat", "half", "open"]);

        let heard = matcher.recognise(&ramp(10, 0.95)).expect("clear");
        assert_eq!(heard.label, "open");
        assert!(heard.margin.unwrap() > 0.15);

        let heard = matcher.recognise(&vec![vec![0.02]; 9]).expect("clear");
        assert_eq!(heard.label, "flat");

        let heard = matcher.recognise(&ramp(14, 0.45)).expect("clear");
        assert_eq!(heard.label, "half");

        // Two labels taught the very same movement: nothing can tell them
        // apart, so the matcher says so instead of picking one.
        let mut ambiguous = Matcher::new();
        ambiguous.learn("yes", ramp(12, 1.0));
        ambiguous.learn("no", ramp(12, 1.0));
        let nearest = ambiguous.nearest(&ramp(12, 0.9)).unwrap();
        assert_eq!(nearest.margin, Some(0.0), "{nearest:?}");
        assert!(ambiguous.recognise(&ramp(12, 0.9)).is_none());

        assert!(Matcher::new().recognise(&ramp(5, 1.0)).is_none());
    }
}
