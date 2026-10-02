//! Moving regions, from the core's frame differencing. Each session owns one
//! of these, since it compares a frame with the one before it.

use std::sync::Mutex;

use image::RgbaImage;
use syrup::motion::{MotionConfig, extract_blobs, motion_mask};

use super::{Detections, Provider, ViewRef};
use crate::abi::{SYRUP_MAX_KEYPOINTS, SyrupDetection};
use crate::catalog::Capability;
use crate::contract::ProviderInfo;
use crate::error::Result;
use crate::intent::PixelRect;

pub struct Motion {
    config: MotionConfig,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// The region this frame searches, set by `begin`.
    region: Option<PixelRect>,
    /// The last committed frame's pixels, and the region they came from.
    previous: Option<(Option<PixelRect>, RgbaImage)>,
    /// This frame's pixels, kept once `commit` says the frame succeeded.
    pending: Option<RgbaImage>,
}

impl Motion {
    pub fn new(config: MotionConfig) -> Motion {
        Motion {
            config,
            state: Mutex::default(),
        }
    }

    fn state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Starts a frame. Motion is only measured against a previous frame
    /// that searched the same region.
    pub fn begin(&self, region: Option<PixelRect>) {
        let mut state = self.state();
        state.region = region;
        state.pending = None;
    }

    /// The frame succeeded: it becomes the one the next frame is compared with.
    pub fn commit(&self) {
        let mut state = self.state();
        if let Some(pixels) = state.pending.take() {
            state.previous = Some((state.region, pixels));
        }
    }
}

impl Provider for Motion {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            capability: Capability::Motion.as_str(),
            name: "frame_difference",
            model_sha256: None,
            runtime: "rust",
        }
    }

    /// Nothing moves in the first frame, in a frame whose size differs from
    /// the one before, or after the searched region changed.
    fn detect(&self, view: &ViewRef<'_>) -> Result<Detections> {
        let current = view.to_rgba();
        let mut state = self.state();
        let mask = match &state.previous {
            Some((region, before)) if *region == state.region => {
                motion_mask(before, &current, self.config.diff_threshold)
            }
            _ => None,
        };
        state.pending = Some(current);
        let Some(mask) = mask else {
            return Ok(Detections::default());
        };
        let boxes = extract_blobs(&mask, &self.config)
            .into_iter()
            .map(|r| {
                let mut changed = 0u32;
                for y in r.y..r.y + r.h {
                    for x in r.x..r.x + r.w {
                        changed += (mask.get_pixel(x, y)[0] > 0) as u32;
                    }
                }
                SyrupDetection {
                    x: r.x as f32,
                    y: r.y as f32,
                    w: r.w as f32,
                    h: r.h as f32,
                    score: changed as f32 / r.area() as f32,
                    value: 0.0,
                    n_keypoints: 0,
                    keypoints: [0.0; 2 * SYRUP_MAX_KEYPOINTS],
                    payload: 0,
                }
            })
            .collect();
        Ok(Detections {
            boxes,
            texts: vec![],
        })
    }
}
