//! X11 windows are found by title and captured directly. On a Wayland
//! desktop, X11 applications run under XWayland and are found the same
//! way; any other window is shared through the screen-cast portal.

use image::RgbaImage;

use super::CaptureError;

mod pipewire;
mod portal;
mod x11;

pub enum Window {
    X11(Box<x11::Window>),
    Wayland(portal::Window),
}

fn wayland() -> bool {
    std::env::var_os("WAYLAND_DISPLAY").is_some()
}

pub fn find(query: &str) -> Result<(String, Window), CaptureError> {
    let x11 = x11::find(query);
    match x11 {
        Ok((title, window)) => Ok((title, Window::X11(Box::new(window)))),
        Err(CaptureError::NotFound | CaptureError::Unavailable(_)) if wayland() => {
            let window = portal::Window::open(query)?;
            Ok((query.to_string(), Window::Wayland(window)))
        }
        Err(e) => Err(e),
    }
}

pub fn list_windows() -> Result<Vec<String>, CaptureError> {
    match x11::list_windows() {
        Err(CaptureError::Unavailable(_)) if wayland() => Ok(Vec::new()),
        listed => listed,
    }
}

impl Window {
    pub fn capture(&mut self) -> Result<RgbaImage, CaptureError> {
        match self {
            Window::X11(window) => window.capture(),
            Window::Wayland(window) => window.capture(),
        }
    }
}
