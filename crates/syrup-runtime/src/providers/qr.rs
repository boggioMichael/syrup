//! QR codes, located and decoded by rqrr (pure Rust).

use super::{Detections, Provider, ViewRef};
use crate::abi::{SYRUP_MAX_KEYPOINTS, SyrupDetection};
use crate::catalog::Capability;
use crate::contract::ProviderInfo;
use crate::error::Result;

pub struct Qr;

/// Decoded codes score 1. Codes whose finder patterns were found but whose
/// content could not be read score 0.5, below the default threshold, so a
/// caller can still ask for damaged codes with a lower min_confidence.
const UNREADABLE: f32 = 0.5;

impl Provider for Qr {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            capability: Capability::QrDecoding.as_str(),
            name: "rqrr",
            model_sha256: None,
            runtime: "rust",
        }
    }

    fn detect(&self, view: &ViewRef<'_>) -> Result<Detections> {
        let rgba = view.to_rgba();
        // Rec. 709 luma, as image's to_luma8 computes it.
        let mut prepared = rqrr::PreparedImage::prepare_from_greyscale(
            view.width as usize,
            view.height as usize,
            |x, y| {
                let p = rgba.get_pixel(x as u32, y as u32);
                ((2126 * p[0] as u32 + 7152 * p[1] as u32 + 722 * p[2] as u32) / 10000) as u8
            },
        );
        let mut out = Detections::default();
        for grid in prepared.detect_grids() {
            // Corners in the code's own orientation: top left, top right,
            // bottom right, bottom left.
            let corners = grid.bounds;
            let xs = corners.map(|p| p.x as f32);
            let ys = corners.map(|p| p.y as f32);
            let (x0, y0) = (
                xs.iter().copied().fold(f32::MAX, f32::min),
                ys.iter().copied().fold(f32::MAX, f32::min),
            );
            let (x1, y1) = (
                xs.iter().copied().fold(f32::MIN, f32::max),
                ys.iter().copied().fold(f32::MIN, f32::max),
            );
            let mut keypoints = [0.0; 2 * SYRUP_MAX_KEYPOINTS];
            for (i, p) in corners.iter().enumerate() {
                keypoints[2 * i] = p.x as f32;
                keypoints[2 * i + 1] = p.y as f32;
            }
            let (score, text) = match grid.decode() {
                Ok((_, text)) => (1.0, text),
                Err(_) => (UNREADABLE, String::new()),
            };
            out.boxes.push(SyrupDetection {
                x: x0,
                y: y0,
                w: x1 - x0 + 1.0,
                h: y1 - y0 + 1.0,
                score,
                value: 0.0,
                n_keypoints: 4,
                keypoints,
                payload: 0,
            });
            out.texts.push(text);
        }
        Ok(out)
    }
}
