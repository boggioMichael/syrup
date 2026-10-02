//! Words from the core's Tesseract OCR.

use syrup::geometry::Rect;
use syrup::ocr::{self, OcrConfig, OcrError};

use super::{Detections, Provider, ViewRef};
use crate::abi::{SYRUP_MAX_KEYPOINTS, SyrupDetection};
use crate::contract::ProviderInfo;
use crate::error::{ErrorKind, Result, Stage, SyrupError};

pub struct Tesseract;

impl Provider for Tesseract {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            capability: "text_recognition",
            name: "tesseract",
            model_sha256: None,
            runtime: "tesseract subprocess",
        }
    }

    fn detect(&self, view: &ViewRef<'_>) -> Result<Detections> {
        let whole = Rect {
            x: 0,
            y: 0,
            w: view.width,
            h: view.height,
        };
        let recognition =
            ocr::recognize(&view.to_rgba(), whole, &OcrConfig::default()).map_err(|e| match e {
                OcrError::EngineMissing => SyrupError::new(
                    Stage::Execute,
                    ErrorKind::MissingDependency,
                    e.to_string(),
                )
                .with_hint(
                    "install Tesseract (e.g. apt install tesseract-ocr) or set TESSERACT_BIN",
                ),
                e => SyrupError::new(Stage::Execute, ErrorKind::ProviderFailed, e.to_string()),
            })?;
        let mut out = Detections::default();
        for word in recognition.words {
            let b = word.bounds;
            out.boxes.push(SyrupDetection {
                x: b.x as f32,
                y: b.y as f32,
                w: b.w as f32,
                h: b.h as f32,
                score: word.confidence,
                value: 0.0,
                n_keypoints: 0,
                keypoints: [0.0; 2 * SYRUP_MAX_KEYPOINTS],
                payload: 0,
            });
            out.texts.push(word.text);
        }
        Ok(out)
    }
}
