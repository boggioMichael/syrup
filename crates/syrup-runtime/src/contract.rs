//! Inputs and results, as specified in docs/contract.md.

use std::path::Path;

use serde::Serialize;

use crate::abi::SyrupImageView;
use crate::error::{ErrorKind, Result, Stage, SyrupError};
use crate::intent::PixelRect;

pub const MAX_IMAGE_SIDE: u32 = 16_384;

fn bad_image(reason: impl Into<String>) -> SyrupError {
    SyrupError::new(Stage::Input, ErrorKind::BadImage, reason)
}

/// Borrowed 8-bit pixels: row-major, 1 (grey), 3 (RGB) or 4 (RGBA) channels.
#[derive(Debug, Clone, Copy)]
pub struct ImageInput<'a> {
    data: &'a [u8],
    width: u32,
    height: u32,
    stride: usize,
    channels: u32,
}

impl<'a> ImageInput<'a> {
    pub fn new(data: &'a [u8], width: u32, height: u32, channels: u32) -> Result<Self> {
        Self::with_stride(
            data,
            width,
            height,
            width as usize * channels as usize,
            channels,
        )
    }

    pub fn with_stride(
        data: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
        channels: u32,
    ) -> Result<Self> {
        if !matches!(channels, 1 | 3 | 4) {
            return Err(bad_image(format!(
                "images have 1 (grey), 3 (RGB) or 4 (RGBA) channels, not {channels}"
            ))
            .with_hint("convert the image to 8-bit grey, RGB or RGBA first"));
        }
        if width == 0 || height == 0 {
            return Err(bad_image(format!("the image is empty ({width}x{height})")));
        }
        if width > MAX_IMAGE_SIDE || height > MAX_IMAGE_SIDE {
            return Err(bad_image(format!(
                "{width}x{height} exceeds the {MAX_IMAGE_SIDE}-pixel side limit"
            ))
            .with_hint("downscale the image first"));
        }
        let row = width as usize * channels as usize;
        if stride < row {
            return Err(bad_image(format!(
                "stride {stride} is shorter than a row ({row} bytes)"
            )));
        }
        let needed = stride * (height as usize - 1) + row;
        if data.len() < needed {
            return Err(bad_image(format!(
                "{width}x{height}x{channels} with stride {stride} needs {needed} bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            data,
            width,
            height,
            stride,
            channels,
        })
    }

    pub fn from_rgba(image: &'a image::RgbaImage) -> Self {
        Self::new(image.as_raw(), image.width(), image.height(), 4)
            .expect("RgbaImage is well-formed")
    }

    pub fn from_rgb(image: &'a image::RgbImage) -> Self {
        Self::new(image.as_raw(), image.width(), image.height(), 3)
            .expect("RgbImage is well-formed")
    }

    pub fn from_gray(image: &'a image::GrayImage) -> Self {
        Self::new(image.as_raw(), image.width(), image.height(), 1)
            .expect("GrayImage is well-formed")
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn channels(&self) -> u32 {
        self.channels
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    pub(crate) fn as_abi(&self) -> SyrupImageView {
        SyrupImageView {
            data: self.data.as_ptr(),
            width: self.width,
            height: self.height,
            stride: self.stride,
            channels: self.channels,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OwnedImage {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub channels: u32,
}

impl OwnedImage {
    /// Grey stays grey, images with alpha become RGBA, the rest RGB.
    pub fn open(path: &Path) -> Result<OwnedImage> {
        let decoded = image::open(path).map_err(|e| {
            bad_image(format!("cannot decode `{}`: {e}", path.display()))
                .with_detail("path", path.display().to_string())
        })?;
        Ok(Self::from_dynamic(decoded))
    }

    pub fn from_dynamic(decoded: image::DynamicImage) -> OwnedImage {
        use image::DynamicImage as D;
        let (width, height) = (decoded.width(), decoded.height());
        let (data, channels) = match decoded {
            D::ImageLuma8(img) => (img.into_raw(), 1),
            D::ImageLuma16(_) => (decoded.to_luma8().into_raw(), 1),
            D::ImageRgb8(img) => (img.into_raw(), 3),
            img if img.color().has_alpha() => (img.to_rgba8().into_raw(), 4),
            img => (img.to_rgb8().into_raw(), 3),
        };
        OwnedImage {
            data,
            width,
            height,
            channels,
        }
    }

    pub fn as_input(&self) -> Result<ImageInput<'_>> {
        ImageInput::new(&self.data, self.width, self.height, self.channels)
    }

    pub fn to_rgba(&self) -> image::RgbaImage {
        let mut out = image::RgbaImage::new(self.width, self.height);
        let c = self.channels as usize;
        for (i, pixel) in out.pixels_mut().enumerate() {
            let p = &self.data[i * c..i * c + c];
            *pixel = match c {
                1 => image::Rgba([p[0], p[0], p[0], 255]),
                3 => image::Rgba([p[0], p[1], p[2], 255]),
                _ => image::Rgba([p[0], p[1], p[2], p[3]]),
            };
        }
        out
    }
}

/// Input-image pixels, covering `[x, x+w) × [y, y+h)`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BoxF {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Keypoint {
    pub name: &'static str,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Found {
    pub label: &'static str,
    #[serde(rename = "box")]
    pub bbox: BoxF,
    /// Provider score in [0, 1], not a calibrated probability.
    pub confidence: f32,
    pub keypoints: Vec<Keypoint>,
    /// What a word says, for targets that read text.
    pub text: Option<String>,
    /// The measured quantity, in [0, 1], for `measure_*` operations.
    pub value: Option<f32>,
    /// The item's identity across a session's frames, for `track_*` operations.
    pub track: Option<TrackRef>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TrackRef {
    /// The same object keeps its id from frame to frame within a session.
    pub id: u64,
    /// Frames this track has been seen in, this one included.
    pub age_frames: u32,
    /// Movement of the box centre since the previous frame, in pixels.
    pub velocity: (f32, f32),
}

/// How a run got its compiled module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactStatus {
    Compiled,
    LoadedFromDisk,
    InMemory,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ProviderInfo {
    pub capability: &'static str,
    pub name: &'static str,
    pub model_sha256: Option<String>,
    pub runtime: &'static str,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EffectiveParams {
    pub min_confidence: f32,
    pub max_results: Option<u32>,
    pub region: Option<PixelRect>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArtifactRef {
    pub key: String,
    pub path: String,
    pub binary_sha256: String,
    pub rustc: String,
    pub target: String,
    pub generator: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Provenance {
    pub operation: String,
    pub intent: String,
    pub plan_hash: String,
    pub artifact: ArtifactRef,
    pub artifact_status: ArtifactStatus,
    pub providers: Vec<ProviderInfo>,
    pub params: EffectiveParams,
    pub image: (u32, u32, u32),
    pub prepare_ms: f64,
    pub execute_ms: f64,
    /// The frame's number in its session, from 1.
    pub frame: Option<u64>,
}

/// Empty means the operation ran and accepted nothing. Failures are errors.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FindResult {
    pub items: Vec<Found>,
    pub provenance: Provenance,
}

/// Per-run settings; none of them changes what the operation means.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RunParams {
    pub min_confidence: Option<f32>,
    pub max_results: Option<u32>,
    /// Required by `_in_region` operations and rejected by the others.
    pub region: Option<PixelRect>,
}
