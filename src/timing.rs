//! Frame-rate and moving-average measurement.

use std::time::Duration;

/// Moving average over the most recent samples, kept in a fixed ring buffer:
/// no allocation after [`MovingAverage::new`].
///
/// Until the window has filled, the average is over the samples seen so far,
/// so the first readings are real measurements rather than values diluted by
/// the empty slots.
#[derive(Debug, Clone)]
pub struct MovingAverage {
    window: Vec<f64>,
    /// Slot the next sample goes into.
    pos: usize,
    /// Number of slots holding a sample; reaches `window.len()` and stays.
    filled: usize,
    sum: f64,
}

impl MovingAverage {
    /// An average over the last `size` samples. A size of 0 is treated as 1,
    /// so the average is always defined once a sample has been added.
    pub fn new(size: usize) -> Self {
        Self {
            window: vec![0.0; size.max(1)],
            pos: 0,
            filled: 0,
            sum: 0.0,
        }
    }

    pub fn add(&mut self, v: f64) {
        if self.filled == self.window.len() {
            self.sum -= self.window[self.pos];
        } else {
            self.filled += 1;
        }
        self.window[self.pos] = v;
        self.sum += v;
        self.pos += 1;
        if self.pos == self.window.len() {
            self.pos = 0;
            // Re-add the window once per lap. The running subtract-and-add
            // otherwise accumulates rounding error without bound over a long
            // session, and a single NaN or infinity would poison the sum
            // forever instead of leaving with its sample.
            self.sum = self.window[..self.filled].iter().sum();
        }
    }

    /// Mean of the samples in the window, or 0 before any sample was added.
    pub fn average(&self) -> f64 {
        if self.filled == 0 {
            0.0
        } else {
            self.sum / self.filled as f64
        }
    }

    /// Number of samples currently averaged.
    pub fn len(&self) -> usize {
        self.filled
    }

    pub fn is_empty(&self) -> bool {
        self.filled == 0
    }

    /// Whether the window holds as many samples as it can.
    pub fn is_full(&self) -> bool {
        self.filled == self.window.len()
    }

    /// Maximum number of samples averaged.
    pub fn capacity(&self) -> usize {
        self.window.len()
    }

    /// Forget every sample, keeping the window's size.
    pub fn clear(&mut self) {
        self.pos = 0;
        self.filled = 0;
        self.sum = 0.0;
    }
}

/// FPS counter using a moving average over recent frame times.
#[derive(Debug, Clone)]
pub struct FPSCounter {
    ma: MovingAverage,
}

impl FPSCounter {
    pub fn new(samples: usize) -> Self {
        Self {
            ma: MovingAverage::new(samples),
        }
    }

    /// Add a frame duration (seconds) and return the smoothed FPS.
    ///
    /// Durations that are negative or not finite (a clock hiccup) are
    /// ignored rather than averaged in.
    pub fn add_frame_seconds(&mut self, secs: f64) -> f64 {
        if secs.is_finite() && secs >= 0.0 {
            self.ma.add(secs);
        }
        self.fps()
    }

    /// Add a frame duration and return the smoothed FPS.
    pub fn add_frame(&mut self, duration: Duration) -> f64 {
        self.add_frame_seconds(duration.as_secs_f64())
    }

    /// Current FPS estimate, or 0 before any frame was added.
    pub fn fps(&self) -> f64 {
        let avg = self.ma.average();
        if avg <= 0.0 { 0.0 } else { 1.0 / avg }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moving_avg_works() {
        let mut ma = MovingAverage::new(3);
        ma.add(0.1);
        ma.add(0.1);
        ma.add(0.1);
        assert!((ma.average() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn average_before_the_window_fills_uses_only_real_samples() {
        let mut ma = MovingAverage::new(60);
        assert!(ma.is_empty());
        assert_eq!(ma.average(), 0.0);
        ma.add(0.5);
        assert_eq!(ma.average(), 0.5);
        ma.add(1.5);
        assert_eq!(ma.average(), 1.0);
        assert_eq!(ma.len(), 2);
        assert!(!ma.is_full());
    }

    #[test]
    fn old_samples_leave_the_window() {
        let mut ma = MovingAverage::new(2);
        for v in [100.0, 100.0, 1.0, 3.0] {
            ma.add(v);
        }
        assert_eq!(ma.average(), 2.0);
        assert!(ma.is_full());
        assert_eq!(ma.capacity(), 2);
    }

    #[test]
    fn zero_size_behaves_like_one() {
        let mut ma = MovingAverage::new(0);
        ma.add(4.0);
        ma.add(6.0);
        assert_eq!(ma.average(), 6.0);
        assert_eq!(ma.capacity(), 1);
    }

    #[test]
    fn a_bad_sample_only_affects_the_window_it_is_in() {
        let mut ma = MovingAverage::new(3);
        ma.add(f64::NAN);
        for _ in 0..6 {
            ma.add(2.0);
        }
        assert_eq!(ma.average(), 2.0);
    }

    #[test]
    fn long_runs_do_not_drift() {
        let mut ma = MovingAverage::new(7);
        for i in 0..1_000_000u32 {
            ma.add(f64::from(i % 13) * 0.1 + 1e6);
        }
        for _ in 0..7 {
            ma.add(0.0);
        }
        assert_eq!(ma.average(), 0.0);
    }

    #[test]
    fn clear_forgets_samples() {
        let mut ma = MovingAverage::new(4);
        ma.add(9.0);
        ma.clear();
        assert!(ma.is_empty());
        ma.add(1.0);
        assert_eq!(ma.average(), 1.0);
    }

    #[test]
    fn fps_is_right_from_the_first_frame() {
        let mut fps = FPSCounter::new(60);
        assert_eq!(fps.fps(), 0.0);
        let first = fps.add_frame(Duration::from_millis(20));
        assert!((first - 50.0).abs() < 1e-9, "got {first}");
    }

    #[test]
    fn fps_ignores_impossible_durations() {
        let mut fps = FPSCounter::new(4);
        fps.add_frame_seconds(0.025);
        fps.add_frame_seconds(-1.0);
        fps.add_frame_seconds(f64::NAN);
        fps.add_frame_seconds(f64::INFINITY);
        assert!((fps.fps() - 40.0).abs() < 1e-9);
    }
}
