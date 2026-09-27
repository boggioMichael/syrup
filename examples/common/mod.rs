//! Shared plumbing for the live examples: raw RGBA frames in on stdin, raw
//! RGBA frames out on stdout, so any camera or file reaches them through
//! ffmpeg, and a small overlay toolkit over `syrup::draw`.
//!
//! Camera in, live view out (Linux):
//!
//! ```sh
//! ffmpeg -v error -f v4l2 -video_size 640x480 -i /dev/video0 -pix_fmt rgba -f rawvideo - \
//!   | cargo run --release --example face_cues -- 640 480 \
//!   | ffplay -v error -f rawvideo -pix_fmt rgba -video_size 640x480 -
//! ```
//!
//! Windows: `-f dshow -i video="Integrated Camera"`; macOS: `-f avfoundation
//! -i "0"`. A video file works the same way with `-i clip.mp4 -r 15`.

#![allow(dead_code)]

use std::io::{Read, Write};

use image::{Rgba, RgbaImage};
use syrup::draw::{draw_rect, draw_text, text_height, text_width};
use syrup::geometry::Rect;

pub const WHITE: Rgba<u8> = Rgba([255, 255, 255, 255]);
pub const GREY: Rgba<u8> = Rgba([170, 170, 170, 255]);
pub const GREEN: Rgba<u8> = Rgba([80, 255, 120, 255]);
pub const CYAN: Rgba<u8> = Rgba([90, 220, 255, 255]);
pub const YELLOW: Rgba<u8> = Rgba([255, 230, 80, 255]);
pub const ORANGE: Rgba<u8> = Rgba([255, 160, 60, 255]);
pub const RED: Rgba<u8> = Rgba([255, 80, 80, 255]);
pub const MAGENTA: Rgba<u8> = Rgba([255, 100, 255, 255]);

/// Frame size from the command line: `<width> <height>` after the mode
/// words, defaulting to 640x480.
pub fn frame_size(args: &[String], first: usize) -> (u32, u32) {
    let width = args.get(first).and_then(|s| s.parse().ok()).unwrap_or(640);
    let height = args
        .get(first + 1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(480);
    (width, height)
}

/// Reads frames from stdin and writes them to stdout.
pub struct Pipe {
    width: u32,
    height: u32,
    buffer: Vec<u8>,
    stdin: std::io::StdinLock<'static>,
    stdout: std::io::StdoutLock<'static>,
    pub index: u64,
}

impl Pipe {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            buffer: vec![0u8; (width * height * 4) as usize],
            stdin: std::io::stdin().lock(),
            stdout: std::io::stdout().lock(),
            index: 0,
        }
    }

    /// The next frame, or `None` at end of input.
    pub fn next_frame(&mut self) -> Option<RgbaImage> {
        self.stdin.read_exact(&mut self.buffer).ok()?;
        self.index += 1;
        RgbaImage::from_raw(self.width, self.height, self.buffer.clone())
    }

    /// Write an annotated frame; false once the viewer has gone away.
    pub fn send(&mut self, frame: &RgbaImage) -> bool {
        self.stdout.write_all(frame.as_raw()).is_ok()
    }
}

pub fn fill(frame: &mut RgbaImage, r: Rect, rgb: [u8; 3], alpha: f32) {
    let (fw, fh) = frame.dimensions();
    for y in r.y..(r.y + r.h).min(fh) {
        for x in r.x..(r.x + r.w).min(fw) {
            let p = frame.get_pixel_mut(x, y);
            for c in 0..3 {
                p[c] = (p[c] as f32 * (1.0 - alpha) + rgb[c] as f32 * alpha) as u8;
            }
        }
    }
}

pub fn text(frame: &mut RgbaImage, s: &str, x: i64, y: i64, scale: u32, color: Rgba<u8>) {
    let (fw, fh) = frame.dimensions();
    draw_text(s, x, y, scale, |px, py| {
        if px >= 0 && py >= 0 && (px as u32) < fw && (py as u32) < fh {
            frame.put_pixel(px as u32, py as u32, color);
        }
    });
}

/// A label with a dark backing, at scale 1.
pub fn label(frame: &mut RgbaImage, s: &str, x: i64, y: i64, color: Rgba<u8>) {
    if x >= 0 && y >= 0 {
        fill(
            frame,
            Rect {
                x: x as u32,
                y: y as u32,
                w: text_width(s, 1) + 4,
                h: text_height(1) + 4,
            },
            [0, 0, 0],
            0.7,
        );
    }
    text(frame, s, x + 2, y + 2, 1, color);
}

pub fn outline(frame: &mut RgbaImage, r: Rect, color: Rgba<u8>, thickness: u32) {
    draw_rect(frame, r.x, r.y, r.w, r.h, color, thickness);
}

/// A horizontal gauge: `value` in `[0, 1]` of `width` pixels.
pub fn gauge(frame: &mut RgbaImage, x: u32, y: u32, width: u32, value: f32, color: Rgba<u8>) {
    let h = 8;
    fill(frame, Rect { x, y, w: width, h }, [30, 30, 30], 0.85);
    let filled = (width as f32 * value.clamp(0.0, 1.0)).round() as u32;
    if filled > 0 {
        fill(
            frame,
            Rect { x, y, w: filled, h },
            [color[0], color[1], color[2]],
            0.95,
        );
    }
    outline(frame, Rect { x, y, w: width, h }, GREY, 1);
}

/// A translucent panel of text lines at scale 2.
pub struct Panel {
    pub lines: Vec<(String, Rgba<u8>)>,
}

impl Panel {
    pub fn new() -> Self {
        Self { lines: Vec::new() }
    }

    pub fn line(&mut self, s: impl Into<String>, color: Rgba<u8>) {
        self.lines.push((s.into(), color));
    }

    pub fn draw(&self, frame: &mut RgbaImage, x: u32, y: u32) {
        let scale = 2;
        let line_h = text_height(scale) + 6;
        let width = self
            .lines
            .iter()
            .map(|(t, _)| text_width(t, scale))
            .max()
            .unwrap_or(0)
            + 16;
        let height = line_h * self.lines.len() as u32 + 10;
        fill(
            frame,
            Rect {
                x,
                y,
                w: width,
                h: height,
            },
            [10, 10, 14],
            0.78,
        );
        outline(
            frame,
            Rect {
                x,
                y,
                w: width,
                h: height,
            },
            Rgba([70, 70, 80, 255]),
            1,
        );
        for (i, (t, color)) in self.lines.iter().enumerate() {
            text(
                frame,
                t,
                x as i64 + 8,
                (y + 6 + i as u32 * line_h) as i64,
                scale,
                *color,
            );
        }
    }
}

/// An exponential moving average with a settable time constant in frames.
#[derive(Debug, Clone, Copy)]
pub struct Ema {
    pub value: f32,
    alpha: f32,
    pushes: u32,
}

impl Ema {
    pub fn new(frames: f32) -> Self {
        Self {
            value: 0.0,
            alpha: 1.0 / frames.max(1.0),
            pushes: 0,
        }
    }

    pub fn push(&mut self, v: f32) -> f32 {
        if self.pushes == 0 {
            self.value = v;
        } else {
            self.value += self.alpha * (v - self.value);
        }
        self.pushes += 1;
        self.value
    }

    /// Whether at least `n` samples have gone in.
    pub fn has_seen(&self, n: u32) -> bool {
        self.pushes >= n
    }
}

/// A face followed across frames: the cascade's box smoothed, and the eye
/// boxes kept through blinks (when the eye cascade sees nothing) by moving
/// them with the face.
pub struct FollowedFace {
    pub face: Rect,
    pub eyes: Vec<Rect>,
    pub mouth: Rect,
    /// Frames since the face cascade last saw the face.
    pub missed: u32,
    /// Whether this frame's eyes came from the cascade (true) or were
    /// carried over.
    pub eyes_fresh: bool,
}

pub struct FaceFollower {
    options: syrup::cascade::CascadeOptions,
    current: Option<FollowedFace>,
    /// Frames a lost face is kept before it is dropped.
    patience: u32,
}

fn blend(a: u32, b: u32, weight: f32) -> u32 {
    (a as f32 * (1.0 - weight) + b as f32 * weight).round() as u32
}

fn smooth(previous: Rect, fresh: Rect, weight: f32) -> Rect {
    Rect {
        x: blend(previous.x, fresh.x, weight),
        y: blend(previous.y, fresh.y, weight),
        w: blend(previous.w, fresh.w, weight),
        h: blend(previous.h, fresh.h, weight),
    }
}

fn overlaps(a: Rect, b: Rect) -> bool {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = (a.x + a.w).min(b.x + b.w);
    let y1 = (a.y + a.h).min(b.y + b.h);
    let inter = x1.saturating_sub(x0) as f32 * y1.saturating_sub(y0) as f32;
    inter > 0.5 * a.area().min(b.area()) as f32
}

impl FaceFollower {
    pub fn new(options: syrup::cascade::CascadeOptions, patience: u32) -> Self {
        Self {
            options,
            current: None,
            patience,
        }
    }

    /// Update with a frame; the followed face, if any.
    pub fn observe(&mut self, image: &RgbaImage) -> Option<&FollowedFace> {
        let (width, height) = image.dimensions();
        let whole = Rect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        };
        let seen = syrup::cascade::Cascade::frontal_face()
            .detect_in(image, whole, &self.options)
            .into_iter()
            .max_by_key(|m| m.bounds.area())
            .map(|m| m.bounds);

        let face = match (seen, &self.current) {
            (Some(fresh), Some(current)) if overlaps(fresh, current.face) => {
                smooth(current.face, fresh, 0.35)
            }
            (Some(fresh), _) => fresh,
            (None, Some(current)) if current.missed < self.patience => current.face,
            (None, _) => {
                self.current = None;
                return None;
            }
        };
        let missed = if seen.is_some() {
            0
        } else {
            self.current.as_ref().map_or(0, |c| c.missed + 1)
        };

        let fresh_eyes = syrup::face::find_eyes(image, face, &self.options);
        let (eyes, eyes_fresh) = match (&self.current, fresh_eyes.len()) {
            (Some(current), n) if n < 2 && current.eyes.len() == 2 => {
                // Blink, or a miss: carry the eyes along with the face.
                let dx = face.x as i64 - current.face.x as i64;
                let dy = face.y as i64 - current.face.y as i64;
                let moved: Vec<Rect> = current
                    .eyes
                    .iter()
                    .map(|e| Rect {
                        x: (e.x as i64 + dx).max(0) as u32,
                        y: (e.y as i64 + dy).max(0) as u32,
                        ..*e
                    })
                    .collect();
                (moved, false)
            }
            (Some(current), 2) if current.eyes.len() == 2 => (
                current
                    .eyes
                    .iter()
                    .zip(&fresh_eyes)
                    .map(|(p, f)| smooth(*p, *f, 0.35))
                    .collect(),
                true,
            ),
            _ => (fresh_eyes, true),
        };
        let (mouth, _) = syrup::face::mouth_box(face, &eyes);
        let mouth = Rect {
            x: mouth.x.min(width.saturating_sub(1)),
            y: mouth.y.min(height.saturating_sub(1)),
            w: mouth
                .w
                .min(width - mouth.x.min(width.saturating_sub(1)))
                .max(1),
            h: mouth
                .h
                .min(height - mouth.y.min(height.saturating_sub(1)))
                .max(1),
        };
        self.current = Some(FollowedFace {
            face,
            eyes,
            mouth,
            missed,
            eyes_fresh,
        });
        self.current.as_ref()
    }

    pub fn current(&self) -> Option<&FollowedFace> {
        self.current.as_ref()
    }
}
