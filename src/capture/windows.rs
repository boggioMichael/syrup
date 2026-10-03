//! `PrintWindow` asks the window to draw itself, so a covered window is
//! captured as drawn. Windows that refuse it (hardware-accelerated or
//! protected surfaces) fall back to a copy of the screen.
//!
//! The whole screen is a copy of every monitor, as the user sees it.

use std::ffi::OsString;
use std::os::windows::ffi::OsStringExt;

use image::RgbaImage;
use windows::Win32::Foundation::{HWND, LPARAM, POINT, RECT};
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, ClientToScreen, CreateCompatibleBitmap,
    CreateCompatibleDC, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetDIBits, HBITMAP, HDC,
    ReleaseDC, SRCCOPY, SelectObject,
};
use windows::Win32::Storage::Xps::{PRINT_WINDOW_FLAGS, PrintWindow};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetClientRect, GetSystemMetrics, GetWindowTextLengthW, GetWindowTextW, IsIconic,
    IsWindow, IsWindowVisible, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN, SM_XVIRTUALSCREEN,
    SM_YVIRTUALSCREEN,
};
use windows::core::BOOL;

use super::{CaptureError, Screen, matches};

/// PW_CLIENTONLY | PW_RENDERFULLCONTENT: render just the client area,
/// and include content drawn outside the classic GDI path.
const PW_CLIENTONLY_FULL: u32 = 0x0000_0001 | 0x0000_0002;

pub struct Window {
    hwnd: HWND,
}

// SAFETY: an HWND is a handle valid from any thread.
unsafe impl Send for Window {}

/// The title of a visible window, if it has one.
fn title(hwnd: HWND) -> Option<String> {
    unsafe {
        if !IsWindowVisible(hwnd).as_bool() {
            return None;
        }
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return None;
        }
        let mut buffer = vec![0u16; len as usize + 1];
        let written = GetWindowTextW(hwnd, &mut buffer);
        (written > 0).then(|| {
            OsString::from_wide(&buffer[..written as usize])
                .to_string_lossy()
                .into_owned()
        })
    }
}

/// Every visible, titled window, in Z order.
fn windows() -> Vec<(HWND, String)> {
    unsafe extern "system" fn visit(hwnd: HWND, found: LPARAM) -> BOOL {
        // SAFETY: `found` is the vector below, alive for the synchronous
        // EnumWindows call.
        let found = unsafe { &mut *(found.0 as *mut Vec<(HWND, String)>) };
        if let Some(title) = title(hwnd) {
            found.push((hwnd, title));
        }
        BOOL(1)
    }
    let mut found = Vec::new();
    unsafe {
        let _ = EnumWindows(Some(visit), LPARAM(&mut found as *mut _ as isize));
    }
    found
}

pub fn list_windows() -> Result<Vec<String>, CaptureError> {
    Ok(windows().into_iter().map(|(_, title)| title).collect())
}

pub fn find(query: &str) -> Result<(String, Window), CaptureError> {
    windows()
        .into_iter()
        .find(|(_, title)| matches(title, query))
        .map(|(hwnd, title)| (title, Window { hwnd }))
        .ok_or(CaptureError::NotFound)
}

impl Window {
    pub fn capture(&mut self) -> Result<RgbaImage, CaptureError> {
        if unsafe { !IsWindow(Some(self.hwnd)).as_bool() } {
            return Err(CaptureError::Closed);
        }
        let mut rect = RECT::default();
        let mut origin = POINT::default();
        unsafe {
            GetClientRect(self.hwnd, &mut rect)
                .map_err(|e| CaptureError::Failed(format!("GetClientRect: {e}")))?;
            if !ClientToScreen(self.hwnd, &mut origin).as_bool() {
                return Err(CaptureError::Failed("ClientToScreen failed".into()));
            }
        }
        let (width, height) = (rect.right - rect.left, rect.bottom - rect.top);
        if width <= 0 || height <= 0 {
            // A minimised window's client rectangle collapses to nothing.
            return Err(if unsafe { IsIconic(self.hwnd).as_bool() } {
                CaptureError::Minimised
            } else {
                CaptureError::Failed("the window has no client area".into())
            });
        }

        let screen = unsafe { GetDC(None) };
        let memory = unsafe { CreateCompatibleDC(Some(screen)) };
        let bitmap = unsafe { CreateCompatibleBitmap(screen, width, height) };
        let previous = unsafe { SelectObject(memory, bitmap.into()) };
        let surface = Surface {
            screen,
            memory,
            bitmap,
            origin,
            width,
            height,
        };

        // Drawing the window itself rather than copying the screen means a
        // covered window yields its own pixels, not whatever is on top.
        let printed = unsafe {
            PrintWindow(self.hwnd, memory, PRINT_WINDOW_FLAGS(PW_CLIENTONLY_FULL)).as_bool()
        };
        let mut pixels = if printed { surface.read() } else { None };
        // PrintWindow can succeed yet return a flat surface for GPU-drawn
        // content; the screen copy is right whenever the window is visible.
        if pixels.as_deref().is_none_or(is_blank) && surface.copy_screen() {
            pixels = surface.read();
        }

        unsafe {
            SelectObject(memory, previous);
            let _ = DeleteObject(bitmap.into());
            let _ = DeleteDC(memory);
            ReleaseDC(None, screen);
        }

        let mut pixels =
            pixels.ok_or_else(|| CaptureError::Failed("GetDIBits returned nothing".into()))?;
        for pixel in pixels.as_chunks_mut::<4>().0 {
            pixel.swap(0, 2);
            pixel[3] = 255;
        }
        Ok(RgbaImage::from_raw(width as u32, height as u32, pixels).expect("sized to fit"))
    }
}

/// Everything on every monitor, as the user sees it.
pub fn capture_screen() -> Option<Screen> {
    let (left, top, width, height) = unsafe {
        (
            GetSystemMetrics(SM_XVIRTUALSCREEN),
            GetSystemMetrics(SM_YVIRTUALSCREEN),
            GetSystemMetrics(SM_CXVIRTUALSCREEN),
            GetSystemMetrics(SM_CYVIRTUALSCREEN),
        )
    };
    if width <= 0 || height <= 0 {
        return None;
    }
    let screen = unsafe { GetDC(None) };
    let memory = unsafe { CreateCompatibleDC(Some(screen)) };
    let bitmap = unsafe { CreateCompatibleBitmap(screen, width, height) };
    let previous = unsafe { SelectObject(memory, bitmap.into()) };
    let surface = Surface {
        screen,
        memory,
        bitmap,
        origin: POINT { x: left, y: top },
        width,
        height,
    };
    // A plain copy, without CAPTUREBLT: layered windows drawn over the
    // screen (an overlay pointing at what was found) stay out of it.
    let pixels = if surface.copy_screen() {
        surface.read()
    } else {
        None
    };
    unsafe {
        SelectObject(memory, previous);
        let _ = DeleteObject(bitmap.into());
        let _ = DeleteDC(memory);
        ReleaseDC(None, screen);
    }
    let mut pixels = pixels?;
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
        pixel[3] = 255;
    }
    RgbaImage::from_raw(width as u32, height as u32, pixels).map(|image| Screen {
        left,
        top,
        image,
    })
}

struct Surface {
    screen: HDC,
    memory: HDC,
    bitmap: HBITMAP,
    origin: POINT,
    width: i32,
    height: i32,
}

impl Surface {
    fn copy_screen(&self) -> bool {
        unsafe {
            BitBlt(
                self.memory,
                0,
                0,
                self.width,
                self.height,
                Some(self.screen),
                self.origin.x,
                self.origin.y,
                SRCCOPY,
            )
            .is_ok()
        }
    }

    /// The bitmap as top-down BGRA.
    fn read(&self) -> Option<Vec<u8>> {
        let mut info = BITMAPINFO::default();
        info.bmiHeader.biSize = size_of::<BITMAPINFOHEADER>() as u32;
        info.bmiHeader.biWidth = self.width;
        info.bmiHeader.biHeight = -self.height;
        info.bmiHeader.biPlanes = 1;
        info.bmiHeader.biBitCount = 32;
        info.bmiHeader.biCompression = BI_RGB.0;
        let mut pixels = vec![0u8; self.width as usize * self.height as usize * 4];
        let rows = unsafe {
            GetDIBits(
                self.memory,
                self.bitmap,
                0,
                self.height as u32,
                Some(pixels.as_mut_ptr().cast()),
                &mut info,
                DIB_RGB_COLORS,
            )
        };
        (rows != 0).then_some(pixels)
    }
}

/// Is this BGRA surface one flat colour? Samples rather than scanning every
/// pixel: a real frame varies within a few hundred samples.
fn is_blank(pixels: &[u8]) -> bool {
    const SAMPLES: usize = 512;
    let count = pixels.len() / 4;
    let step = (count / SAMPLES).max(1);
    (0..count)
        .step_by(step)
        .all(|i| pixels[i * 4..i * 4 + 3] == pixels[..3])
}
