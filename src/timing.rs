//! Frame-rate and moving-average measurement.

/// Fixed-size moving average over a ring buffer; no allocation after `new`.
#[derive(Debug)]
pub struct MovingAverage {
    window: Vec<f64>,
    pos: usize,
    sum: f64,
    size: usize,
}

impl MovingAverage {
    pub fn new(size: usize) -> Self {
        let window = vec![0.0; size];
        Self {
            window,
            pos: 0,
            sum: 0.0,
            size,
        }
    }

    pub fn add(&mut self, v: f64) {
        self.sum -= self.window[self.pos];
        self.window[self.pos] = v;
        self.sum += v;
        self.pos = (self.pos + 1) % self.size;
    }

    pub fn average(&self) -> f64 {
        self.sum / (self.size as f64)
    }
}

/// FPS counter using a moving average over recent frame times.
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
    pub fn add_frame_seconds(&mut self, secs: f64) -> f64 {
        self.ma.add(secs);
        self.fps()
    }

    /// Current FPS estimate.
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
}
