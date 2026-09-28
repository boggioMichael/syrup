//! Live behavioural cues from a face — the signals "lie detection" folklore
//! points at, measured honestly and labelled for what they are.
//!
//! **This is not a lie detector.** No measurement of a face can tell truth
//! from lies; decades of research put humans and machines near chance.
//! What a camera *can* measure is arousal and behaviour: how often the
//! person blinks, how much the head moves, how much the gaze shifts, how
//! much the mouth moves, how much the face changes frame to frame. This
//! program measures those, compares them with the person's own first ten
//! seconds, and shows how far they have drifted. Whether a drift means
//! stress, discomfort, concentration, a joke, or a lie is not something
//! pixels know — and the overlay says so.
//!
//! Frames come in on stdin, annotated frames go out on stdout (see
//! `examples/common/mod.rs` for the ffmpeg lines):
//!
//! ```sh
//! ffmpeg -v error -f v4l2 -video_size 640x480 -i /dev/video0 -pix_fmt rgba -f rawvideo - \
//!   | cargo run --release --example face_cues -- 640 480 30 \
//!   | ffplay -v error -f rawvideo -pix_fmt rgba -video_size 640x480 -
//! ```
//!
//! Built from: the frontal-face and eye cascades, `face::eye_openness`
//! (blinks), `face::mouth_shape` (mouth activity), the gaze proxy in this
//! file (where the dark iris sits inside the eye box), the face box's own
//! movement, and frame differencing inside the face.

mod common;

use std::collections::VecDeque;

use common::*;
use image::RgbaImage;
use syrup::cascade::CascadeOptions;
use syrup::face;
use syrup::geometry::Rect;
use syrup::motion::motion_mask;
use syrup::threshold::{Channel, channel_image};

/// Seconds of the person's own behaviour to take as the baseline.
const BASELINE_SECONDS: f32 = 10.0;

/// One eye's recent openness, with blink detection against its own level.
struct Eye {
    level: Ema,
    closed_for: u32,
    blinks: VecDeque<u64>,
    /// Horizontal position of the iris inside the box, `0..1`.
    gaze: Ema,
    gaze_shifts: VecDeque<u64>,
}

impl Eye {
    fn new(fps: f32) -> Self {
        Self {
            level: Ema::new(fps * 2.0),
            closed_for: 0,
            blinks: VecDeque::new(),
            gaze: Ema::new(3.0),
            gaze_shifts: VecDeque::new(),
        }
    }

    /// Feed this frame's openness and iris position; true on a blink.
    fn observe(&mut self, openness: f32, iris_x: f32, frame: u64, fps: f32) -> bool {
        let mut blinked = false;
        let baseline = self.level.value;
        if self.level.has_seen(1) && openness < baseline * 0.5 {
            self.closed_for += 1;
        } else {
            // A closure of one to ten frames that reopened is a blink.
            if self.closed_for >= 1 && self.closed_for <= (fps * 0.4).ceil() as u32 {
                self.blinks.push_back(frame);
                blinked = true;
            }
            self.closed_for = 0;
            self.level.push(openness);
        }
        let before = self.gaze.value;
        let now = self.gaze.push(iris_x);
        if self.gaze.has_seen(5) && (now - before).abs() > 0.12 {
            self.gaze_shifts.push_back(frame);
        }
        blinked
    }
}

/// Where the dark iris sits horizontally inside an eye box, `0..1`.
fn iris_position(image: &RgbaImage, eye: Rect) -> f32 {
    let gray = channel_image(image, eye, Channel::Luma);
    let (w, h) = gray.dimensions();
    if w < 3 || h < 3 {
        return 0.5;
    }
    // Column sums of the middle rows; the darkest column is the iris.
    let mut darkest = (u32::MAX, 0u32);
    for x in 0..w {
        let sum: u32 = (h / 4..h * 3 / 4)
            .map(|y| u32::from(gray.get_pixel(x, y).0[0]))
            .sum();
        if sum < darkest.0 {
            darkest = (sum, x);
        }
    }
    (darkest.1 as f32 + 0.5) / w as f32
}

/// Count of events in the last `window` frames, dropping older ones.
fn recent(events: &mut VecDeque<u64>, now: u64, window: u64) -> usize {
    while events.front().is_some_and(|&f| f + window < now) {
        events.pop_front();
    }
    events.len()
}

struct Baseline {
    samples: Vec<[f32; 5]>,
    mean: [f32; 5],
    spread: [f32; 5],
    done: bool,
}

impl Baseline {
    fn new() -> Self {
        Self {
            samples: Vec::new(),
            mean: [0.0; 5],
            spread: [1.0; 5],
            done: false,
        }
    }

    fn push(&mut self, sample: [f32; 5]) {
        self.samples.push(sample);
    }

    fn finish(&mut self) {
        let n = self.samples.len().max(1) as f32;
        for k in 0..5 {
            let mean = self.samples.iter().map(|s| s[k]).sum::<f32>() / n;
            let var = self
                .samples
                .iter()
                .map(|s| (s[k] - mean).powi(2))
                .sum::<f32>()
                / n;
            self.mean[k] = mean;
            // A floor keeps a perfectly still baseline from making every
            // later twitch look enormous.
            self.spread[k] = var.sqrt().max(mean.abs() * 0.25).max(0.02);
        }
        self.done = true;
    }

    /// How many spreads above its baseline each cue is, clipped to 0..3.
    fn drift(&self, sample: [f32; 5]) -> [f32; 5] {
        let mut d = [0.0; 5];
        for k in 0..5 {
            d[k] = ((sample[k] - self.mean[k]) / self.spread[k]).clamp(0.0, 3.0);
        }
        d
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (width, height) = frame_size(&args, 1);
    let fps: f32 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let mut pipe = Pipe::new(width, height);

    // Faces are at least a tenth of the frame tall, and a coarser scale step
    // keeps the cascade well inside a frame period.
    let options = CascadeOptions {
        min_size: height / 10,
        scale_factor: 1.2,
        ..CascadeOptions::default()
    };
    let mut follower = FaceFollower::new(options, (fps * 0.5) as u32);

    let mut eyes = [Eye::new(fps), Eye::new(fps)];
    let mut last_face: Option<Rect> = None;
    let mut previous_face: Option<RgbaImage> = None;
    let mut head = Ema::new(fps * 0.5);
    let mut mouth_level = Ema::new(fps * 0.5);
    let mut mouth_history: VecDeque<f32> = VecDeque::new();
    let mut face_motion = Ema::new(fps * 0.5);
    let mut baseline = Baseline::new();
    let mut arousal = Ema::new(fps);
    let minute = (fps * 60.0) as u64;

    while let Some(mut frame) = pipe.next_frame() {
        let started = std::time::Instant::now();
        let now = pipe.index;
        let mut panel = Panel::new();
        panel.line("BEHAVIOURAL CUES (NOT A LIE DETECTOR)", YELLOW);

        let Some(followed) = follower.observe(&frame) else {
            last_face = None;
            panel.line("NO FACE IN VIEW", GREY);
            panel.draw(&mut frame, 8, height.saturating_sub(80));
            if !pipe.send(&frame) {
                break;
            }
            continue;
        };
        let layout = FollowedFace {
            face: followed.face,
            eyes: followed.eyes.clone(),
            mouth: followed.mouth,
            missed: followed.missed,
            eyes_fresh: followed.eyes_fresh,
        };

        // Head movement: how far the face box's centre moved, in face widths.
        let (cx, cy) = layout.face.center();
        let moved = match last_face {
            Some(previous) => {
                let (px, py) = previous.center();
                ((cx - px).powi(2) + (cy - py).powi(2)).sqrt() / layout.face.w as f32
            }
            None => 0.0,
        };
        head.push(moved);

        // Blinks and gaze, per eye.
        let mut blinked = false;
        for (slot, eye) in layout.eyes.iter().enumerate().take(2) {
            let openness = face::eye_openness(&frame, *eye);
            let iris = iris_position(&frame, *eye);
            if std::env::var_os("FACE_CUES_TRACE").is_some() {
                eprintln!(
                    "frame {now} eye {slot} box {eye:?} fresh={} openness {openness:.3} level {:.3} closed_for {}",
                    layout.eyes_fresh, eyes[slot].level.value, eyes[slot].closed_for
                );
            }
            blinked |= eyes[slot].observe(openness, iris, now, fps);
            outline(
                &mut frame,
                *eye,
                if eyes[slot].closed_for > 0 {
                    ORANGE
                } else {
                    CYAN
                },
                1,
            );
            label(
                &mut frame,
                &format!("EYE {openness:.2} IRIS {iris:.2}"),
                eye.x as i64,
                eye.y as i64 - 12,
                CYAN,
            );
        }
        let blinks_per_minute = eyes
            .iter_mut()
            .map(|e| recent(&mut e.blinks, now, minute))
            .max()
            .unwrap_or(0) as f32
            * 60.0
            / (now.min(minute) as f32 / fps).max(1.0);
        let gaze_shifts = eyes
            .iter_mut()
            .map(|e| recent(&mut e.gaze_shifts, now, (fps * 10.0) as u64))
            .max()
            .unwrap_or(0);

        // Mouth: openness now, and how much it has been changing lately.
        let shape = face::mouth_shape(&frame, layout.mouth);
        mouth_level.push(shape.openness);
        mouth_history.push_back(shape.openness);
        while mouth_history.len() > (fps * 1.5) as usize {
            mouth_history.pop_front();
        }
        let mouth_mean = mouth_history.iter().sum::<f32>() / mouth_history.len().max(1) as f32;
        let mouth_activity = (mouth_history
            .iter()
            .map(|o| (o - mouth_mean).powi(2))
            .sum::<f32>()
            / mouth_history.len().max(1) as f32)
            .sqrt();
        outline(&mut frame, layout.mouth, MAGENTA, 1);
        label(
            &mut frame,
            &format!("MOUTH {:.2}", shape.openness),
            layout.mouth.x as i64,
            (layout.mouth.y + layout.mouth.h) as i64 + 2,
            MAGENTA,
        );

        // Facial motion: the share of face pixels that changed since the
        // previous frame, with the face box held still.
        let crop = image::imageops::crop_imm(
            &frame,
            layout.face.x,
            layout.face.y,
            layout.face.w,
            layout.face.h,
        )
        .to_image();
        let changed = match &previous_face {
            Some(previous) if previous.dimensions() == crop.dimensions() => {
                motion_mask(previous, &crop, 24)
                    .map(|mask| {
                        mask.as_raw().iter().filter(|&&v| v > 0).count() as f32
                            / mask.len().max(1) as f32
                    })
                    .unwrap_or(0.0)
            }
            _ => 0.0,
        };
        previous_face = Some(crop);
        face_motion.push(changed);

        // The five cues, and their drift from this person's own baseline.
        let sample = [
            blinks_per_minute / 60.0,
            head.value,
            gaze_shifts as f32 / 10.0,
            mouth_activity,
            face_motion.value,
        ];
        let seconds = now as f32 / fps;
        let drift = if seconds < BASELINE_SECONDS {
            baseline.push(sample);
            [0.0; 5]
        } else {
            if !baseline.done {
                baseline.finish();
            }
            baseline.drift(sample)
        };
        let index = arousal.push(drift.iter().sum::<f32>() / 15.0 * 100.0);

        outline(&mut frame, layout.face, GREEN, 2);
        if blinked {
            label(
                &mut frame,
                "BLINK",
                layout.face.x as i64,
                layout.face.y as i64 - 12,
                ORANGE,
            );
        }

        let names = [
            "BLINK RATE",
            "HEAD MOTION",
            "GAZE SHIFTS",
            "MOUTH ACTIVITY",
            "FACE MOTION",
        ];
        let values = [
            format!("{blinks_per_minute:4.0}/MIN"),
            format!("{:.3}", head.value),
            format!("{gaze_shifts:2} /10S"),
            format!("{mouth_activity:.3}"),
            format!("{:.3}", face_motion.value),
        ];
        for k in 0..5 {
            let color = if drift[k] > 2.0 {
                RED
            } else if drift[k] > 1.0 {
                ORANGE
            } else {
                GREEN
            };
            panel.line(
                format!(
                    "{:<14} {:>9}  {}",
                    names[k],
                    values[k],
                    "#".repeat(drift[k].round() as usize)
                ),
                color,
            );
        }
        if seconds < BASELINE_SECONDS {
            panel.line(
                format!(
                    "LEARNING YOUR BASELINE: {:.0}S LEFT",
                    BASELINE_SECONDS - seconds
                ),
                GREY,
            );
        } else {
            let verdict = if index > 60.0 {
                "WELL ABOVE START"
            } else if index > 25.0 {
                "ABOVE START"
            } else {
                "AS AT START"
            };
            panel.line(
                format!("AROUSAL {index:3.0}/100  {verdict}"),
                if index > 60.0 { RED } else { WHITE },
            );
        }
        panel.line("AROUSAL IS NOT DECEPTION", GREY);
        panel.line(
            format!(
                "FRAME {now}  {:.0} MS",
                started.elapsed().as_secs_f64() * 1000.0
            ),
            GREY,
        );
        // Bottom-left, under where a face usually is.
        let panel_h = (syrup::draw::text_height(2) + 6) * panel.lines.len() as u32 + 10;
        panel.draw(&mut frame, 8, height.saturating_sub(panel_h + 30));
        gauge(
            &mut frame,
            8,
            height.saturating_sub(20),
            200,
            index / 100.0,
            if index > 60.0 { RED } else { GREEN },
        );

        if now.is_multiple_of((fps as u64).max(1)) {
            eprintln!(
                "t={seconds:5.1}s face={:?} blinks/min={blinks_per_minute:.0} head={:.3} gaze={gaze_shifts} mouth={mouth_activity:.3} motion={:.3} arousal={index:.0}",
                layout.face, head.value, face_motion.value
            );
        }
        last_face = Some(layout.face);
        if !pipe.send(&frame) {
            break;
        }
    }
}
