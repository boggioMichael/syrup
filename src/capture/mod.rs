//! Live capture: a window by title on Windows, macOS and Linux, or the
//! whole screen (Windows).
//!
//! [`Window::find`] opens the first window whose title contains a query
//! (ignoring case) and [`Window::capture`] returns its current contents as
//! RGBA. Each platform uses its own mechanism:
//!
//! - **Windows**: Windows.Graphics.Capture, the compositor's own frames on
//!   the GPU (Windows 10 1903 and later), with only the regions asked for
//!   read back; `PrintWindow`, then a copy of the screen, where that is not
//!   to be had.
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
//! [`CaptureError`]. [`Window::capture_frame`] gives a [`Frame`] that can
//! be read a region at a time, which on the GPU path is what saves the
//! trip to the CPU for the pixels nobody looks at.

use std::fmt;

use image::RgbaImage;

use crate::geometry::Rect;

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

    /// The window's current contents as a [`Frame`], to read whole or a
    /// region at a time. On Windows the frame stays on the GPU until read.
    pub fn capture_frame(&mut self) -> Result<Frame<'_>, CaptureError> {
        Ok(Frame::new(self.inner.capture_frame()?))
    }

    /// Why this window's frames are not coming from the GPU, when they are
    /// not: the platform has no such path, the system or the window rules
    /// it out, it failed too often, or the CPU path was asked for (with
    /// `SYRUP_CAPTURE=cpu` in the environment, or [`Window::without_gpu`]).
    /// `None` while they are — and, on Windows, before the first capture
    /// has tried.
    pub fn gpu_unavailable(&self) -> Option<&str> {
        #[cfg(target_os = "windows")]
        {
            self.inner.gpu_unavailable()
        }
        #[cfg(not(target_os = "windows"))]
        {
            Some("no GPU capture path on this platform")
        }
    }

    /// Frames through the CPU path only, from now on (on Windows, GDI
    /// rather than Windows.Graphics.Capture): for comparing the two, or a
    /// driver the GPU path does not get on with. Elsewhere nothing changes.
    pub fn without_gpu(&mut self) {
        #[cfg(target_os = "windows")]
        self.inner.without_gpu();
    }
}

/// `SYRUP_CAPTURE=cpu` (or `gdi`) in the environment: the CPU path only.
#[cfg(target_os = "windows")]
fn cpu_asked_for() -> bool {
    std::env::var("SYRUP_CAPTURE")
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "cpu" | "gdi"))
        .unwrap_or(false)
}

/// Where a frame's pixels are: on the CPU already, or still on the GPU.
pub enum FrameSource<'a> {
    Cpu(RgbaImage),
    #[cfg(target_os = "windows")]
    Gpu(&'a mut platform::Gpu),
    #[cfg(not(target_os = "windows"))]
    #[doc(hidden)]
    Never(std::marker::PhantomData<&'a ()>),
}

/// One captured frame, read whole or a region at a time.
pub struct Frame<'a> {
    source: FrameSource<'a>,
}

impl<'a> Frame<'a> {
    fn new(source: FrameSource<'a>) -> Self {
        Frame { source }
    }

    /// Width and height.
    pub fn size(&self) -> (u32, u32) {
        match &self.source {
            FrameSource::Cpu(image) => image.dimensions(),
            #[cfg(target_os = "windows")]
            FrameSource::Gpu(gpu) => gpu.size(),
            #[cfg(not(target_os = "windows"))]
            FrameSource::Never(_) => (0, 0),
        }
    }

    /// Whether the pixels are still on the GPU, to be read back a region
    /// at a time.
    pub fn on_gpu(&self) -> bool {
        !matches!(self.source, FrameSource::Cpu(_))
    }

    /// `region` of the frame, clipped to it; `None` when nothing is left.
    pub fn read(&mut self, region: Rect) -> Option<RgbaImage> {
        match &mut self.source {
            FrameSource::Cpu(image) => {
                let (fw, fh) = image.dimensions();
                let x = region.x.min(fw);
                let y = region.y.min(fh);
                let w = region.w.min(fw - x);
                let h = region.h.min(fh - y);
                (w > 0 && h > 0).then(|| image::imageops::crop_imm(image, x, y, w, h).to_image())
            }
            #[cfg(target_os = "windows")]
            FrameSource::Gpu(gpu) => gpu.read(region),
            #[cfg(not(target_os = "windows"))]
            FrameSource::Never(_) => None,
        }
    }

    /// The whole frame. Takes the frame: pixels already on the CPU are
    /// handed over rather than copied.
    pub fn read_all(self) -> Option<RgbaImage> {
        match self.source {
            FrameSource::Cpu(image) => Some(image),
            #[cfg(target_os = "windows")]
            FrameSource::Gpu(gpu) => {
                let (w, h) = gpu.size();
                gpu.read(Rect { x: 0, y: 0, w, h })
            }
            #[cfg(not(target_os = "windows"))]
            FrameSource::Never(_) => None,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// CI opens a window on each system and names it in SYRUP_TEST_WINDOW
    /// (`.github/open-window.sh`); with `--nocapture` the test says which
    /// path the frames took.
    #[test]
    fn a_window_is_captured_whole_and_a_region_at_a_time() {
        let Ok(title) = std::env::var("SYRUP_TEST_WINDOW") else {
            return;
        };
        let mut window = Window::find(&title).expect("the window CI opened");
        let mut frame = window.capture_frame().expect("a frame");
        let (w, h) = frame.size();
        assert!(w > 20 && h > 20, "{w}x{h}");
        let region = frame
            .read(Rect {
                x: 1,
                y: 1,
                w: 16,
                h: 16,
            })
            .expect("a region");
        assert_eq!(region.dimensions(), (16, 16));
        let clipped = frame
            .read(Rect {
                x: w - 4,
                y: h - 4,
                w: 100,
                h: 100,
            })
            .expect("a region clipped to the frame");
        assert_eq!(clipped.dimensions(), (4, 4));
        assert!(
            frame
                .read(Rect {
                    x: w,
                    y: 0,
                    w: 8,
                    h: 8
                })
                .is_none(),
            "a region past the frame is nothing"
        );
        let on_gpu = frame.on_gpu();
        let whole = frame.read_all().expect("the whole frame");
        assert_eq!(whole.dimensions(), (w, h));
        // A region is those pixels of the whole frame, whichever side they
        // were read from.
        let same = image::imageops::crop_imm(&whole, 1, 1, 16, 16).to_image();
        assert_eq!(region.as_raw(), same.as_raw());
        println!(
            "capture of \"{}\": {w}x{h}, {}",
            window.title(),
            match window.gpu_unavailable() {
                None if on_gpu => "frames from the GPU".to_string(),
                None => "frames from the CPU".to_string(),
                Some(why) => format!("frames from the CPU ({why})"),
            }
        );
    }
}
