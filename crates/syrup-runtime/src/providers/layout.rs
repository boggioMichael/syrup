//! The core's layout helpers: the block of text in a region, and a region's
//! largest panel of uniform colour.

use syrup::color::is_text_pixel;
use syrup::geometry::{Rect, dominant_color_bucket, find_text_block, find_uniform_color_panel};

use super::{Detections, Provider, ViewRef};
use crate::abi::{SYRUP_MAX_KEYPOINTS, SyrupDetection};
use crate::catalog::Capability;
use crate::contract::ProviderInfo;
use crate::error::Result;

/// Colour levels per channel when looking for a panel's colour: 256 / 32,
/// so near-identical shades count as one colour.
pub const PANEL_STEP: u8 = 32;

fn whole(view: &ViewRef<'_>) -> Rect {
    Rect {
        x: 0,
        y: 0,
        w: view.width,
        h: view.height,
    }
}

/// The share of `rect`'s pixels for which `hit` holds.
fn share(image: &image::RgbaImage, rect: Rect, hit: impl Fn(&image::Rgba<u8>) -> bool) -> f32 {
    let mut hits = 0u64;
    for y in rect.y..rect.y + rect.h {
        for x in rect.x..rect.x + rect.w {
            hits += hit(image.get_pixel(x, y)) as u64;
        }
    }
    hits as f32 / (rect.w as u64 * rect.h as u64).max(1) as f32
}

fn detection(rect: Rect, score: f32) -> SyrupDetection {
    SyrupDetection {
        x: rect.x as f32,
        y: rect.y as f32,
        w: rect.w as f32,
        h: rect.h as f32,
        score,
        value: 0.0,
        n_keypoints: 0,
        keypoints: [0.0; 2 * SYRUP_MAX_KEYPOINTS],
        payload: 0,
    }
}

/// `find_text_block`: the padded bounds of the text-like pixels in the view.
pub struct TextBlocks;

impl Provider for TextBlocks {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            capability: Capability::TextBlocks.as_str(),
            name: "syrup::geometry::find_text_block",
            model_sha256: None,
            runtime: "rust",
        }
    }

    fn detect(&self, view: &ViewRef<'_>) -> Result<Detections> {
        let image = view.to_rgba();
        let boxes = find_text_block(&image, whole(view))
            .map(|rect| {
                // The block is padded, so clip it before scoring.
                let rect = Rect {
                    w: rect.w.min(view.width - rect.x),
                    h: rect.h.min(view.height - rect.y),
                    ..rect
                };
                detection(rect, share(&image, rect, is_text_pixel))
            })
            .into_iter()
            .collect();
        Ok(Detections {
            boxes,
            texts: vec![],
        })
    }
}

/// `find_uniform_color_panel` with the view's dominant colour.
pub struct Panels;

impl Provider for Panels {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            capability: Capability::Panels.as_str(),
            name: "syrup::geometry::find_uniform_color_panel",
            model_sha256: None,
            runtime: "rust",
        }
    }

    fn detect(&self, view: &ViewRef<'_>) -> Result<Detections> {
        let image = view.to_rgba();
        let all = whole(view);
        let boxes = dominant_color_bucket(&image, all, PANEL_STEP)
            .and_then(|bucket| {
                let rect = find_uniform_color_panel(&image, all, bucket, PANEL_STEP)?;
                let same = |p: &image::Rgba<u8>| {
                    p[3] >= 200
                        && (p[0] / PANEL_STEP, p[1] / PANEL_STEP, p[2] / PANEL_STEP) == bucket
                };
                Some(detection(rect, share(&image, rect, same)))
            })
            .into_iter()
            .collect();
        Ok(Detections {
            boxes,
            texts: vec![],
        })
    }
}
