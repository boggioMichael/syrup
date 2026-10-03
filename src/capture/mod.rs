//! Live capture: a window by title on Windows, macOS and Linux, or the
//! whole screen (Windows).
//!
//! [`Window::find`] opens the first window whose title contains a query
//! (ignoring case) and [`Window::capture`] returns its current contents as
//! RGBA. Each platform uses its own mechanism:
//!
//! - **Windows**: `PrintWindow`, so a covered window is captured as drawn.
//! - **Linux and the BSDs, X11** (including X11 applications on a Wayland
//!   desktop, via XWayland): the Composite extension, so a covered window
//!   is captured as drawn.
//! - **Linux and the BSDs, Wayland**: the desktop's screen-cast portal and
//!   PipeWire. Wayland does not show other applications' window titles, so
//!   the desktop asks the user to pick the window. The choice is remembered
//!   for that query, and later runs capture the same window without asking.
//! - **macOS 14 and later**: ScreenCaptureKit. The process needs the Screen
//!   Recording permission (System Settings > Privacy & Security).
//!
//! [`capture_screen`] copies everything on every monitor, with where that
//! picture sits on the desktop, so that something found in it can be
//! pointed at on the screen (Windows; `None` elsewhere).
//!
//! The one-shot [`capture_window_by_title_info`] and [`list_windows`] are
//! the original entry points, kept for existing callers: they look the
//! window up on every call and answer `None` or an empty list instead of
//! saying why. A [`Window`] keeps its window between frames and reports a
//! [`CaptureError`].

use std::fmt;

use image::RgbaImage;

#[cfg_attr(target_os = "windows", path = "windows.rs")]
#[cfg_attr(target_os = "macos", path = "macos.rs")]
#[cfg_attr(all(unix, not(target_os = "macos")), path = "unix.rs")]
#[cfg_attr(not(any(unix, windows)), path = "unsupported.rs")]
mod platform;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CaptureError {
    /// No window's title contains the query.
    NotFound,
    /// The window has been closed.
    Closed,
    /// The window is minimised: it exists but has no picture to give.
    Minimised,
    /// This system cannot capture windows, e.g. there is no display.
    Unavailable(String),
    /// Capture was refused: a missing permission, or the user declined.
    Denied(String),
    /// The platform reported an error.
    Failed(String),
}

impl fmt::Display for CaptureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CaptureError::NotFound => f.write_str("no window matches"),
            CaptureError::Closed => f.write_str("the window was closed"),
            CaptureError::Minimised => f.write_str("the window is minimised"),
            CaptureError::Unavailable(reason)
            | CaptureError::Denied(reason)
            | CaptureError::Failed(reason) => f.write_str(reason),
        }
    }
}

impl std::error::Error for CaptureError {}

/// An open window, captured on demand.
pub struct Window {
    title: String,
    inner: platform::Window,
}

impl Window {
    /// The first window whose title contains `query`, ignoring case.
    pub fn find(query: &str) -> Result<Window, CaptureError> {
        let (title, inner) = platform::find(query)?;
        Ok(Window { title, inner })
    }

    /// The window's full title.
    pub fn title(&self) -> &str {
        &self.title
    }

    /// The window's current contents, without its frame.
    pub fn capture(&mut self) -> Result<RgbaImage, CaptureError> {
        self.inner.capture()
    }
}

/// Titles of the windows [`Window::find`] can match without asking the
/// user. On Wayland those are X11 applications' windows only.
pub fn window_titles() -> Result<Vec<String>, CaptureError> {
    platform::list_windows()
}

/// Every visible titled window, so a caller can pick one instead of relying
/// on a title heuristic; empty where windows cannot be listed. Kept for
/// existing callers: [`window_titles`] says why a list is unavailable.
pub fn list_windows() -> Vec<String> {
    window_titles().unwrap_or_default()
}

/// Capture the first visible window whose title contains `search_title`
/// (ignoring case), returning the full title and the client-area pixels,
/// or `None` when there is no such window or it cannot be captured.
///
/// Kept for existing callers. It looks the window up on every call; for a
/// stream of frames, find the [`Window`] once and capture it repeatedly.
pub fn capture_window_by_title_info(search_title: &str) -> Option<(String, RgbaImage)> {
    let mut window = Window::find(search_title).ok()?;
    let image = window.capture().ok()?;
    Some((window.title, image))
}

/// The whole screen as captured: the picture, and the desktop position of
/// its top left pixel (negative when a monitor sits left of or above the
/// main one). A pixel at `(x, y)` in the picture is at `(left + x, top + y)`
/// on the desktop, in physical pixels when the process is DPI aware.
#[derive(Debug, Clone)]
pub struct Screen {
    pub left: i32,
    pub top: i32,
    pub image: RgbaImage,
}

/// Everything on every monitor, as the user sees it (windows that exclude
/// themselves from capture excepted). Windows only; `None` elsewhere.
pub fn capture_screen() -> Option<Screen> {
    #[cfg(target_os = "windows")]
    {
        platform::capture_screen()
    }
    #[cfg(not(target_os = "windows"))]
    {
        None
    }
}

/// Does `title` contain `query`, ignoring case?
#[cfg_attr(not(any(unix, windows)), allow(dead_code))]
fn matches(title: &str, query: &str) -> bool {
    title.to_lowercase().contains(&query.to_lowercase())
}
