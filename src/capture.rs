//! Live window capture (Windows).
//!
//! Captures a window's client area by title substring and returns an RGBA
//! image. On non-Windows platforms every function is a stub that reports
//! "nothing available", so callers can compile portably and fall back to
//! file-based frames.
//!
//! Capture uses `PrintWindow`, which asks the window to draw itself — so it
//! works while the window is occluded — and falls back to a screen copy for
//! windows that refuse (hardware-accelerated or protected surfaces).

#[cfg(target_os = "windows")]
mod windows_capture {
    use std::ffi::OsString;
    use std::os::windows::ffi::OsStringExt;
    use std::ptr;

    use image::RgbaImage;
    use windows::Win32::Foundation::{HWND, LPARAM, POINT, RECT};
    use windows::Win32::Graphics::Gdi::{
        BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, CreateCompatibleBitmap, CreateCompatibleDC,
        DeleteDC, DeleteObject, GetDC, GetDIBits, ReleaseDC, SRCCOPY, SelectObject,
    };
    use windows::Win32::Storage::Xps::{PRINT_WINDOW_FLAGS, PrintWindow};
    use windows::Win32::UI::WindowsAndMessaging::{
        EnumWindows, GetClientRect, GetWindowTextLengthW, GetWindowTextW, IsWindowVisible,
    };
    use windows::core::BOOL;

    /// PW_CLIENTONLY | PW_RENDERFULLCONTENT: render just the client area,
    /// and include content drawn outside the classic GDI path.
    const PW_CLIENTONLY_FULL: u32 = 0x0000_0001 | 0x0000_0002;

    struct WindowSearchState {
        query: String,
        found: HWND,
        title: String,
    }

    struct WindowListState {
        titles: Vec<String>,
    }

    unsafe extern "system" fn enum_windows_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
        // Sound: the state lives on the caller's stack for the entire
        // synchronous EnumWindows call.
        let state = unsafe { &mut *(lparam.0 as *mut WindowSearchState) };
        if unsafe { !IsWindowVisible(hwnd).as_bool() } {
            return BOOL(1);
        }

        let len = unsafe { GetWindowTextLengthW(hwnd) };
        if len <= 0 {
            return BOOL(1);
        }

        let mut buffer = vec![0u16; (len + 1) as usize];
        let written = unsafe { GetWindowTextW(hwnd, &mut buffer) };
        if written <= 0 {
            return BOOL(1);
        }

        let title = OsString::from_wide(&buffer[..written as usize])
            .to_string_lossy()
            .into_owned();
        if title.to_ascii_lowercase().contains(&state.query) {
            state.found = hwnd;
            state.title = title;
            return BOOL(0);
        }

        BOOL(1)
    }

    unsafe extern "system" fn list_windows_proc(hwnd: HWND, lparam: LPARAM) -> BOOL {
        let state = unsafe { &mut *(lparam.0 as *mut WindowListState) };
        if unsafe { !IsWindowVisible(hwnd).as_bool() } {
            return BOOL(1);
        }
        let len = unsafe { GetWindowTextLengthW(hwnd) };
        if len <= 0 {
            return BOOL(1);
        }
        let mut buffer = vec![0u16; (len + 1) as usize];
        let written = unsafe { GetWindowTextW(hwnd, &mut buffer) };
        if written > 0 {
            state.titles.push(
                OsString::from_wide(&buffer[..written as usize])
                    .to_string_lossy()
                    .into_owned(),
            );
        }
        BOOL(1)
    }

    /// Every visible titled window, so a caller can pick one instead of
    /// relying on a title heuristic.
    pub fn list_windows() -> Vec<String> {
        let mut state = WindowListState { titles: Vec::new() };
        unsafe {
            let _ = EnumWindows(
                Some(list_windows_proc),
                LPARAM(&mut state as *mut _ as isize),
            );
        }
        state.titles
    }

    /// Capture the first visible window whose title contains `search_title`
    /// (case-insensitive), returning the full title and the client-area
    /// pixels.
    pub fn capture_window_by_title_info(search_title: &str) -> Option<(String, RgbaImage)> {
        let query = search_title.to_lowercase();
        let mut state = WindowSearchState {
            query,
            found: HWND(ptr::null_mut()),
            title: String::new(),
        };

        unsafe {
            let enumeration = EnumWindows(
                Some(enum_windows_proc),
                LPARAM(&mut state as *mut _ as isize),
            );
            if enumeration.is_err() && state.found.0.is_null() {
                return None;
            }
            if state.found.0.is_null() {
                return None;
            }

            let hwnd = state.found;
            let mut rect = RECT::default();
            if GetClientRect(hwnd, &mut rect).is_err() {
                return None;
            }

            let mut origin = POINT::default();
            if !windows::Win32::Graphics::Gdi::ClientToScreen(hwnd, &mut origin).as_bool() {
                return None;
            }

            let width = rect.right - rect.left;
            let height = rect.bottom - rect.top;
            if width <= 0 || height <= 0 {
                return None;
            }

            let hdc_screen = GetDC(None);
            let hdc_mem = CreateCompatibleDC(Some(hdc_screen));
            let hbitmap = CreateCompatibleBitmap(hdc_screen, width, height);
            let old_obj = SelectObject(hdc_mem, hbitmap.into());

            // Ask the window to draw itself, rather than copying the screen
            // region it occupies: a screen copy returns whatever is visually
            // on top, so an occluded window would yield the wrong pixels.
            let printed = PrintWindow(hwnd, hdc_mem, PRINT_WINDOW_FLAGS(PW_CLIENTONLY_FULL));

            if !printed.as_bool() {
                // Some windows (hardware-accelerated or protected content)
                // refuse PrintWindow; fall back to the screen copy, which
                // works as long as the window is unobstructed.
                if BitBlt(
                    hdc_mem,
                    0,
                    0,
                    width,
                    height,
                    Some(hdc_screen),
                    origin.x,
                    origin.y,
                    SRCCOPY,
                )
                .is_err()
                {
                    let _ = SelectObject(hdc_mem, old_obj);
                    let _ = DeleteObject(hbitmap.into());
                    let _ = DeleteDC(hdc_mem);
                    // The screen DC comes from GetDC and must be released on
                    // every path out of this block, or each failed capture
                    // burns one of the process' 10k GDI handles.
                    let _ = ReleaseDC(None, hdc_screen);
                    return None;
                }
            }

            let mut bmi: BITMAPINFO = std::mem::zeroed();
            bmi.bmiHeader.biSize = std::mem::size_of::<BITMAPINFOHEADER>() as u32;
            bmi.bmiHeader.biWidth = width;
            bmi.bmiHeader.biHeight = -height;
            bmi.bmiHeader.biPlanes = 1;
            bmi.bmiHeader.biBitCount = 32;
            bmi.bmiHeader.biCompression = BI_RGB.0;

            let mut buffer = vec![0u8; (width as usize) * (height as usize) * 4];
            let mut result = GetDIBits(
                hdc_mem,
                hbitmap,
                0,
                height as u32,
                Some(buffer.as_mut_ptr() as *mut _),
                &mut bmi,
                windows::Win32::Graphics::Gdi::DIB_RGB_COLORS,
            );

            // PrintWindow can report success yet return an empty surface for
            // GPU-drawn content: the frame comes back a flat colour, which is
            // indistinguishable from a real capture to everything downstream.
            // Check for it and retry via the screen instead.
            if printed.as_bool()
                && result != 0
                && is_blank(&buffer)
                && BitBlt(
                    hdc_mem,
                    0,
                    0,
                    width,
                    height,
                    Some(hdc_screen),
                    origin.x,
                    origin.y,
                    SRCCOPY,
                )
                .is_ok()
            {
                result = GetDIBits(
                    hdc_mem,
                    hbitmap,
                    0,
                    height as u32,
                    Some(buffer.as_mut_ptr() as *mut _),
                    &mut bmi,
                    windows::Win32::Graphics::Gdi::DIB_RGB_COLORS,
                );
            }

            let _ = SelectObject(hdc_mem, old_obj);
            let _ = DeleteObject(hbitmap.into());
            let _ = DeleteDC(hdc_mem);
            let _ = ReleaseDC(None, hdc_screen);

            if result == 0 {
                return None;
            }

            for chunk in buffer.chunks_exact_mut(4) {
                chunk.swap(0, 2);
                chunk[3] = 255;
            }

            RgbaImage::from_raw(width as u32, height as u32, buffer)
                .map(|image| (state.title.clone(), image))
        }
    }

    /// Is this captured surface effectively featureless?
    ///
    /// Used to spot a PrintWindow call that "succeeded" but returned nothing
    /// drawable. Sampling rather than scanning every pixel keeps this off
    /// the per-frame cost; a real frame varies within any few hundred
    /// samples, so a uniform sample means a uniform image.
    fn is_blank(buffer: &[u8]) -> bool {
        const SAMPLES: usize = 512;
        let pixels = buffer.len() / 4;
        if pixels == 0 {
            return true;
        }
        let stride = (pixels / SAMPLES).max(1);
        let first = &buffer[0..3];
        !(0..pixels)
            .step_by(stride)
            .any(|index| buffer[index * 4..index * 4 + 3] != *first)
    }
}

#[cfg(target_os = "windows")]
pub use windows_capture::{capture_window_by_title_info, list_windows};

#[cfg(not(target_os = "windows"))]
use image::RgbaImage;

#[cfg(not(target_os = "windows"))]
pub fn capture_window_by_title_info(_: &str) -> Option<(String, RgbaImage)> {
    None
}

#[cfg(not(target_os = "windows"))]
pub fn list_windows() -> Vec<String> {
    Vec::new()
}
