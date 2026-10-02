//! YuNet (OpenCV Zoo, 2023mar, MIT) run through tract. Pre- and
//! post-processing follow OpenCV's `FaceDetectorYN`.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex, OnceLock};

use image::imageops::{self, FilterType};
use sha2::{Digest, Sha256};
use tract_onnx::prelude::*;

use super::{Detections, ModelSource, Provider, ViewRef};
use crate::abi::{SYRUP_MAX_KEYPOINTS, SyrupDetection};
use crate::contract::ProviderInfo;
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::plan::hex;

const BUNDLED: &[u8] = include_bytes!("../../models/face_detection_yunet_2023mar.onnx");
pub const MODEL_SHA256: &str = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4";

// The image is scaled so its longer side is this long, then padded right
// and bottom to a multiple of the largest stride. The network runs at that
// size, prepared once per size.
const CANVAS: usize = 640;
const STRIDES: [usize; 3] = [8, 16, 32];
// Candidates below this score are dropped before NMS.
const SCORE_FLOOR: f32 = 0.1;
const NMS_IOU: f32 = 0.3;

type Model = Arc<TypedRunnableModel>;

pub struct YuNet {
    source: ModelSource,
    graph: OnceLock<Result<InferenceModel>>,
    /// Prepared models by (height, width).
    models: Mutex<HashMap<(usize, usize), Model>>,
}

fn failed(reason: String) -> SyrupError {
    SyrupError::new(Stage::Execute, ErrorKind::ProviderFailed, reason)
}

impl YuNet {
    pub fn new(source: ModelSource) -> YuNet {
        YuNet {
            source,
            graph: OnceLock::new(),
            models: Mutex::default(),
        }
    }

    fn model(&self, height: usize, width: usize) -> Result<Model> {
        let mut models = self.models.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(model) = models.get(&(height, width)) {
            return Ok(model.clone());
        }
        let graph = self
            .graph
            .get_or_init(|| load(&self.source))
            .as_ref()
            .map_err(Clone::clone)?;
        let model = graph
            .clone()
            .with_input_fact(0, f32::fact([1, 3, height, width]).into())
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| failed(format!("cannot prepare the face model: {e:#}")))?;
        models.insert((height, width), model.clone());
        Ok(model)
    }
}

/// The verified model, with the shapes the file declares for intermediate
/// values cleared: they assume a 640x640 input.
fn load(source: &ModelSource) -> Result<InferenceModel> {
    let bytes = match source {
        ModelSource::Bundled => Cow::Borrowed(BUNDLED),
        ModelSource::File(path) => Cow::Owned(fs::read(path).map_err(|e| {
            SyrupError::new(
                Stage::Execute,
                ErrorKind::MissingDependency,
                format!("cannot read the face model at {}: {e}", path.display()),
            )
            .with_hint("unset SYRUP_FACE_MODEL to use the bundled model")
        })?),
    };
    let sha256 = hex(&Sha256::digest(&bytes));
    if sha256 != MODEL_SHA256 {
        return Err(SyrupError::new(
            Stage::Execute,
            ErrorKind::Integrity,
            format!("the face model has sha256 {sha256}, expected YuNet 2023mar ({MODEL_SHA256})"),
        )
        .with_hint("the decoder is specific to this exact model; restore the original file"));
    }
    let mut graph = tract_onnx::onnx()
        .model_for_read(&mut &bytes[..])
        .map_err(|e| failed(format!("cannot read the face model: {e:#}")))?;
    for node in graph.nodes_mut() {
        for output in &mut node.outputs {
            output.fact = InferenceFact::default();
        }
    }
    Ok(graph)
}

impl Provider for YuNet {
    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            capability: "face_detection",
            name: "yunet-2023mar",
            model_sha256: Some(MODEL_SHA256.to_string()),
            runtime: "tract-onnx 0.23",
        }
    }

    fn detect(&self, view: &ViewRef<'_>) -> Result<Detections> {
        let scale = (CANVAS as f32 / view.width as f32).min(CANVAS as f32 / view.height as f32);
        let w = ((view.width as f32 * scale).round() as u32).clamp(1, CANVAS as u32);
        let h = ((view.height as f32 * scale).round() as u32).clamp(1, CANVAS as u32);
        let largest = STRIDES[STRIDES.len() - 1];
        let (height, width) = (
            (h as usize).div_ceil(largest) * largest,
            (w as usize).div_ceil(largest) * largest,
        );
        let model = self.model(height, width)?;
        let resized = imageops::resize(&view.to_rgb(), w, h, FilterType::Triangle);

        // BGR, 0..255, no normalisation.
        let mut input = tract_ndarray::Array4::<f32>::zeros((1, 3, height, width));
        for (x, y, p) in resized.enumerate_pixels() {
            let (x, y) = (x as usize, y as usize);
            input[[0, 0, y, x]] = p[2] as f32;
            input[[0, 1, y, x]] = p[1] as f32;
            input[[0, 2, y, x]] = p[0] as f32;
        }
        let outputs = model
            .run(tvec!(input.into_tensor().into()))
            .map_err(|e| failed(format!("face detection failed: {e:#}")))?;
        if outputs.len() != 12 {
            return Err(failed(format!(
                "the face model produced {} outputs, expected 12",
                outputs.len()
            )));
        }
        let output = |i: usize, per_cell: usize, cells: usize| -> Result<&[f32]> {
            let data = outputs[i]
                .try_as_plain_ram()
                .and_then(|view| view.as_slice::<f32>())
                .map_err(|e| failed(format!("face model output {i}: {e}")))?;
            if data.len() != per_cell * cells {
                return Err(failed(format!(
                    "face model output {i} has {} values, expected {}",
                    data.len(),
                    per_cell * cells
                )));
            }
            Ok(data)
        };

        let mut candidates = vec![];
        for (s, stride) in STRIDES.into_iter().enumerate() {
            let cols = width / stride;
            let cells = cols * (height / stride);
            let cls = output(s, 1, cells)?;
            let obj = output(s + 3, 1, cells)?;
            let bbox = output(s + 6, 4, cells)?;
            let kps = output(s + 9, 10, cells)?;
            for i in 0..cells {
                let score = (cls[i].clamp(0.0, 1.0) * obj[i].clamp(0.0, 1.0)).sqrt();
                if score < SCORE_FLOOR {
                    continue;
                }
                let (c, r, st) = ((i % cols) as f32, (i / cols) as f32, stride as f32);
                let (cx, cy) = ((c + bbox[4 * i]) * st, (r + bbox[4 * i + 1]) * st);
                let (bw, bh) = (bbox[4 * i + 2].exp() * st, bbox[4 * i + 3].exp() * st);
                let mut keypoints = [0.0; 2 * SYRUP_MAX_KEYPOINTS];
                for k in 0..SYRUP_MAX_KEYPOINTS {
                    keypoints[2 * k] = (kps[10 * i + 2 * k] + c) * st;
                    keypoints[2 * k + 1] = (kps[10 * i + 2 * k + 1] + r) * st;
                }
                candidates.push(SyrupDetection {
                    x: cx - bw / 2.0,
                    y: cy - bh / 2.0,
                    w: bw,
                    h: bh,
                    score,
                    value: 0.0,
                    n_keypoints: SYRUP_MAX_KEYPOINTS as u32,
                    keypoints,
                    payload: 0,
                });
            }
        }

        // Back from canvas pixels to view pixels.
        let (sx, sy) = (w as f32 / view.width as f32, h as f32 / view.height as f32);
        let boxes = nms(candidates)
            .into_iter()
            .map(|mut d| {
                (d.x, d.y, d.w, d.h) = (d.x / sx, d.y / sy, d.w / sx, d.h / sy);
                for k in 0..SYRUP_MAX_KEYPOINTS {
                    d.keypoints[2 * k] /= sx;
                    d.keypoints[2 * k + 1] /= sy;
                }
                d
            })
            .collect();
        Ok(Detections {
            boxes,
            texts: vec![],
        })
    }
}

fn iou(a: &SyrupDetection, b: &SyrupDetection) -> f32 {
    let w = (a.x + a.w).min(b.x + b.w) - a.x.max(b.x);
    let h = (a.y + a.h).min(b.y + b.h) - a.y.max(b.y);
    let inter = w.max(0.0) * h.max(0.0);
    let union = a.w * a.h + b.w * b.h - inter;
    if union > 0.0 { inter / union } else { 0.0 }
}

fn nms(mut candidates: Vec<SyrupDetection>) -> Vec<SyrupDetection> {
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<SyrupDetection> = vec![];
    for d in candidates {
        if kept.iter().all(|k| iou(k, &d) <= NMS_IOU) {
            kept.push(d);
        }
    }
    kept
}
