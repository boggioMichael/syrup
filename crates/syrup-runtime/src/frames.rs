//! Where a session's frames come from: image files, or a live window.

use std::path::PathBuf;

use syrup::capture::{CaptureError, Window};

use crate::contract::OwnedImage;
use crate::error::{ErrorKind, Result, Stage, SyrupError};

pub trait FrameSource {
    /// The next frame, or `None` when the source has ended.
    fn next_frame(&mut self) -> Result<Option<OwnedImage>>;
}

/// Image files, in the order given.
pub struct ImageFiles {
    paths: std::vec::IntoIter<PathBuf>,
}

impl ImageFiles {
    pub fn new(paths: impl IntoIterator<Item = PathBuf>) -> ImageFiles {
        ImageFiles {
            paths: paths.into_iter().collect::<Vec<_>>().into_iter(),
        }
    }
}

impl FrameSource for ImageFiles {
    fn next_frame(&mut self) -> Result<Option<OwnedImage>> {
        self.paths
            .next()
            .map(|path| OwnedImage::open(&path))
            .transpose()
    }
}

/// A window's contents, captured with the core's `capture` module until
/// the window closes.
pub struct WindowCapture {
    window: Window,
}

fn capture_error(e: CaptureError, query: &str) -> SyrupError {
    let (kind, reason) = match e {
        CaptureError::NotFound => (
            ErrorKind::BadParameter,
            format!("no window title contains {query:?}"),
        ),
        CaptureError::Closed => (ErrorKind::BadParameter, "the window was closed".into()),
        CaptureError::Unavailable(reason) => (ErrorKind::MissingDependency, reason),
        CaptureError::Denied(reason) => (ErrorKind::PermissionDenied, reason),
        CaptureError::Failed(reason) => (ErrorKind::Io, reason),
    };
    let error = SyrupError::new(Stage::Input, kind, reason);
    match kind {
        ErrorKind::BadParameter => {
            error.with_hint("`syrup windows` lists the windows that can be captured")
        }
        _ => error,
    }
}

impl WindowCapture {
    /// The first window whose title contains `query`, ignoring case. On a
    /// Wayland desktop the user picks the window the first time.
    pub fn new(query: &str) -> Result<WindowCapture> {
        Window::find(query)
            .map(|window| WindowCapture { window })
            .map_err(|e| capture_error(e, query))
    }

    /// The window's full title.
    pub fn title(&self) -> &str {
        self.window.title()
    }

    /// Titles of the windows that can be captured without asking the user.
    pub fn windows() -> Result<Vec<String>> {
        syrup::capture::window_titles().map_err(|e| capture_error(e, ""))
    }
}

impl FrameSource for WindowCapture {
    /// Ends when the window closes.
    fn next_frame(&mut self) -> Result<Option<OwnedImage>> {
        match self.window.capture() {
            Ok(image) => Ok(Some(OwnedImage {
                width: image.width(),
                height: image.height(),
                channels: 4,
                data: image.into_raw(),
            })),
            Err(CaptureError::Closed) => Ok(None),
            Err(e) => Err(capture_error(e, self.window.title())),
        }
    }
}
