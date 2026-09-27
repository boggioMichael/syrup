//! Lip reading, honestly scoped: learn a few words from your own mouth,
//! then read them back.
//!
//! Real lip reading — arbitrary sentences from anyone's lips — needs a
//! large learned model; nothing here pretends to that. What a camera and
//! this library can do is measure the mouth frame by frame
//! (`face::mouth_shape`: how open, how wide, how dark the cavity, and
//! `face::mouth_patch`: a small normalised picture of the lips), cut the
//! stream into utterances (the mouth moving, then still), and compare
//! each utterance against examples you recorded, with dynamic time warping
//! so speed does not matter (`sequence::Matcher`). With a handful of
//! examples per word it tells a small vocabulary apart — your words, your
//! lighting, your camera — and says "?" rather than guess when two words
//! are equally close.
//!
//! Record a word (say it a few times, with a pause between):
//!
//! ```sh
//! ffmpeg -v error -f v4l2 -video_size 640x480 -i /dev/video0 -pix_fmt rgba -f rawvideo - \
//!   | cargo run --release --example lip_reading -- learn hello 640 480 30 \
//!   | ffplay -v error -f rawvideo -pix_fmt rgba -video_size 640x480 -
//! ```
//!
//! Then read: the same pipeline with `read 640 480 30`. Examples live in
//! `lips.txt` next to the program (override with `LIPS_FILE`).

mod common;

use std::collections::VecDeque;

use common::*;
use syrup::cascade::CascadeOptions;
use syrup::face;
use syrup::geometry::Rect;
use syrup::sequence::Matcher;

/// The mouth must open at least this much (over its resting level) to
/// start an utterance, and stay this still to end one.
const START_OPENNESS: f32 = 0.18;
const END_OPENNESS: f32 = 0.10;
const END_STILL_SECONDS: f32 = 0.4;
const MIN_UTTERANCE_SECONDS: f32 = 0.2;
const MAX_UTTERANCE_SECONDS: f32 = 4.0;

/// The mouth patch compared between frames: small enough to shrug off
/// where exactly the box landed, large enough to show teeth and rounding.
const PATCH_WIDTH: u32 = 16;
const PATCH_HEIGHT: u32 = 8;

/// Cuts the stream of mouth shapes into utterances.
struct Segmenter {
    fps: f32,
    current: Vec<Vec<f32>>,
    still_for: u32,
    /// Resting openness, learned while the mouth is closed.
    rest: Ema,
}

impl Segmenter {
    fn new(fps: f32) -> Self {
        Self {
            fps,
            current: Vec::new(),
            still_for: 0,
            rest: Ema::new(fps * 3.0),
        }
    }

    fn speaking(&self) -> bool {
        !self.current.is_empty()
    }

    /// Feed one frame's features (openness first); a finished utterance
    /// when one just ended.
    fn push(&mut self, features: Vec<f32>) -> Option<Vec<Vec<f32>>> {
        let openness = features[0];
        let above_rest = openness - self.rest.value;
        if self.current.is_empty() {
            if above_rest > START_OPENNESS && self.rest.has_seen(3) {
                self.current.push(features);
                self.still_for = 0;
            } else {
                self.rest.push(openness);
            }
            return None;
        }
        self.current.push(features);
        if above_rest < END_OPENNESS {
            self.still_for += 1;
        } else {
            self.still_for = 0;
        }
        let too_long = self.current.len() as f32 > MAX_UTTERANCE_SECONDS * self.fps;
        if self.still_for as f32 >= END_STILL_SECONDS * self.fps || too_long {
            let trailing = self.still_for as usize;
            let mut utterance = std::mem::take(&mut self.current);
            utterance.truncate(utterance.len().saturating_sub(trailing).max(1));
            self.still_for = 0;
            if utterance.len() as f32 >= MIN_UTTERANCE_SECONDS * self.fps {
                return Some(utterance);
            }
        }
        None
    }
}

fn lips_file() -> std::path::PathBuf {
    std::env::var_os("LIPS_FILE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| "lips.txt".into())
}

fn load_examples() -> Matcher {
    let mut matcher = Matcher::new();
    let Ok(text) = std::fs::read_to_string(lips_file()) else {
        return matcher;
    };
    for line in text.lines() {
        let mut parts = line.splitn(2, '\t');
        let (Some(label), Some(samples)) = (parts.next(), parts.next()) else {
            continue;
        };
        let samples: Vec<Vec<f32>> = samples
            .split(';')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.split(' ').filter_map(|v| v.parse().ok()).collect())
            .collect();
        matcher.learn(label, samples);
    }
    matcher
}

fn save_example(label: &str, samples: &[Vec<f32>]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(lips_file())?;
    let text: Vec<String> = samples
        .iter()
        .map(|s| {
            s.iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    writeln!(file, "{label}\t{}", text.join(";"))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("read");
    let (word, first_size) = match mode {
        "learn" => (args.get(2).cloned().unwrap_or_else(|| "word".into()), 3),
        _ => (String::new(), 2),
    };
    let (width, height) = frame_size(&args, first_size);
    let fps: f32 = args
        .get(first_size + 2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30.0);
    let mut pipe = Pipe::new(width, height);

    let options = CascadeOptions {
        min_size: height / 10,
        scale_factor: 1.2,
        ..CascadeOptions::default()
    };
    let mut follower = FaceFollower::new(options, (fps * 0.5) as u32);
    let mut segmenter = Segmenter::new(fps);
    let matcher = load_examples();
    if mode != "learn" && matcher.examples().is_empty() {
        eprintln!(
            "no examples in {}: record some first with `lip_reading learn <word> {width} {height}`",
            lips_file().display()
        );
    }
    eprintln!(
        "mode {mode}{}; {} example(s) for {:?}",
        if mode == "learn" {
            format!(" ({word})")
        } else {
            String::new()
        },
        matcher.examples().len(),
        matcher.labels()
    );

    let mut recorded = 0;
    let mut heard: VecDeque<(String, u64)> = VecDeque::new();
    let mut trace: VecDeque<f32> = VecDeque::new();
    while let Some(mut frame) = pipe.next_frame() {
        let now = pipe.index;
        let mut panel = Panel::new();
        panel.line(
            if mode == "learn" {
                format!("LIP READING - LEARNING \"{}\"", word.to_uppercase())
            } else {
                "LIP READING - YOUR WORDS ONLY".into()
            },
            YELLOW,
        );

        if let Some(followed) = follower.observe(&frame) {
            let mouth = followed.mouth;
            let shape = face::mouth_shape(&frame, mouth);
            let mut features = shape.features().to_vec();
            features.extend(face::mouth_patch(&frame, mouth, PATCH_WIDTH, PATCH_HEIGHT));
            trace.push_back(shape.openness);
            while trace.len() > width as usize / 2 {
                trace.pop_front();
            }
            outline(&mut frame, followed.face, GREEN, 1);
            outline(
                &mut frame,
                mouth,
                if segmenter.speaking() { MAGENTA } else { CYAN },
                2,
            );
            label(
                &mut frame,
                &format!("OPEN {:.2} WIDE {:.2}", shape.openness, shape.width),
                mouth.x as i64,
                (mouth.y + mouth.h) as i64 + 2,
                CYAN,
            );

            if let Some(utterance) = segmenter.push(features) {
                let seconds = utterance.len() as f32 / fps;
                if mode == "learn" {
                    match save_example(&word, &utterance) {
                        Ok(()) => {
                            recorded += 1;
                            eprintln!(
                                "recorded utterance {recorded} of {word:?}: {} frames ({seconds:.2} s)",
                                utterance.len()
                            );
                            heard.push_back((format!("SAVED #{recorded}"), now));
                        }
                        Err(error) => eprintln!("could not save: {error}"),
                    }
                } else {
                    match matcher.recognise(&utterance) {
                        Some(found) => {
                            eprintln!(
                                "heard {:?} (distance {:.3}, margin {}) from {} frames",
                                found.label,
                                found.distance,
                                found.margin.map_or("-".into(), |m| format!("{m:.2}")),
                                utterance.len()
                            );
                            heard.push_back((found.label.to_uppercase(), now));
                        }
                        None => {
                            let nearest = matcher.nearest(&utterance);
                            eprintln!(
                                "unsure: {} frames, nearest {:?}",
                                utterance.len(),
                                nearest.map(|n| (n.label, n.distance, n.margin))
                            );
                            heard.push_back(("?".into(), now));
                        }
                    }
                }
            }
            panel.line(
                if segmenter.speaking() {
                    format!("MOUTH MOVING ({} FRAMES)", segmenter.current.len())
                } else {
                    "MOUTH STILL".into()
                },
                if segmenter.speaking() { MAGENTA } else { GREY },
            );
        } else {
            panel.line("NO FACE IN VIEW", GREY);
        }

        // What was heard, for a couple of seconds.
        while heard
            .front()
            .is_some_and(|(_, at)| at + ((fps * 2.5) as u64) < now)
        {
            heard.pop_front();
        }
        if let Some((text, _)) = heard.back() {
            let x = (width / 2).saturating_sub(syrup::draw::text_width(text, 4) / 2);
            fill(
                &mut frame,
                Rect {
                    x: x.saturating_sub(10),
                    y: 30,
                    w: syrup::draw::text_width(text, 4) + 20,
                    h: 44,
                },
                [0, 0, 0],
                0.7,
            );
            common::text(
                &mut frame,
                text,
                x as i64,
                38,
                4,
                if text == "?" { ORANGE } else { WHITE },
            );
        }
        // The openness trace along the bottom.
        for (i, o) in trace.iter().enumerate() {
            let x = 8 + i as u32 * 2;
            let h = (o * 60.0) as u32;
            fill(
                &mut frame,
                Rect {
                    x,
                    y: height.saturating_sub(12 + h),
                    w: 2,
                    h: h.max(1),
                },
                [255, 100, 255],
                0.9,
            );
        }
        if mode == "learn" {
            panel.line(format!("SAY IT, PAUSE, REPEAT.  SAVED: {recorded}"), WHITE);
        } else {
            panel.line(
                format!("KNOWS: {}", matcher.labels().join(", ").to_uppercase()),
                WHITE,
            );
        }
        panel.draw(&mut frame, 8, height.saturating_sub(120));
        if !pipe.send(&frame) {
            break;
        }
    }
}
