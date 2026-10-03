use image::RgbaImage;

use super::CaptureError;

pub enum Window {}

impl Window {
    pub fn capture_frame(&mut self) -> Result<super::FrameSource<'_>, CaptureError> {
        match *self {}
    }

    pub fn capture(&mut self) -> Result<RgbaImage, CaptureError> {
        match *self {}
    }
}

fn unavailable() -> CaptureError {
    CaptureError::Unavailable("window capture needs Windows, macOS, Linux or a BSD".into())
}

pub fn find(_: &str) -> Result<(String, Window), CaptureError> {
    Err(unavailable())
}

pub fn list_windows() -> Result<Vec<String>, CaptureError> {
    Err(unavailable())
}
