//! The coach live on Windows: its voice (the system's own text to speech,
//! SAPI) and its marks — a window of their own laid over the board,
//! see-through, letting every click through to the game, never taking the
//! focus, and left out of screen captures, so the coach never reads its own
//! marks back. The screen itself comes from `syrup::capture::capture_screen`.

use std::ffi::c_void;
use std::sync::mpsc::{Sender, channel};

use windows::Win32::Foundation::{
    COLORREF, HINSTANCE, HWND, LPARAM, LRESULT, POINT, RECT, SIZE, WPARAM,
};
use windows::Win32::Graphics::Dwm::DwmFlush;
use windows::Win32::Graphics::Gdi::{
    AC_SRC_ALPHA, AC_SRC_OVER, ANTIALIASED_QUALITY, BI_RGB, BITMAPINFO, BITMAPINFOHEADER,
    BLENDFUNCTION, CLIP_DEFAULT_PRECIS, CreateCompatibleDC, CreateDIBSection, CreateFontW,
    DEFAULT_CHARSET, DIB_RGB_COLORS, DT_CALCRECT, DT_CENTER, DT_NOPREFIX, DT_WORDBREAK, DeleteDC,
    DeleteObject, DrawTextW, FW_SEMIBOLD, GdiFlush, GetDC, HDC, OUT_DEFAULT_PRECIS, ReleaseDC,
    SelectObject, SetBkMode, SetTextColor, TRANSPARENT,
};
use windows::Win32::Media::Speech::{
    ISpVoice, SPF_ASYNC, SPF_IS_NOT_XML, SPF_PURGEBEFORESPEAK, SpVoice,
};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx,
};
use windows::Win32::System::Console::{GetConsoleWindow, SetConsoleTitleW};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetProcessDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetSystemMetrics, MSG, PM_REMOVE,
    PeekMessageW, RegisterClassExW, SM_CXSCREEN, SM_CYSCREEN, SW_HIDE, SW_SHOWNOACTIVATE,
    SWP_NOACTIVATE, SWP_NOZORDER, SetWindowDisplayAffinity, SetWindowPos, ShowWindow,
    TranslateMessage, ULW_ALPHA, UpdateLayeredWindow, WDA_EXCLUDEFROMCAPTURE, WNDCLASSEXW,
    WS_EX_LAYERED, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_POPUP,
};
use windows::core::{PCWSTR, w};

use mines_coach::coach::Marks;
use mines_coach::overlay::{Layout, paint};

/// Screen pixels are real pixels (not scaled for the display's zoom), so that
/// what is found in a capture is marked exactly there; and the console says
/// what it is, and moves to the bottom right corner, out of the game's way.
pub fn init() {
    unsafe {
        // Fails if the process already has an awareness (from a manifest): fine either way.
        let _ = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
        let _ = SetConsoleTitleW(w!("Minesweeper coach (close this window to stop)"));
        let console = GetConsoleWindow();
        if !console.is_invalid() {
            let (sw, sh) = (GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN));
            let (w, h) = ((sw / 3).clamp(480, 900), (sh / 4).clamp(220, 420));
            let _ = SetWindowPos(
                console,
                None,
                sw - w - 8,
                sh - h - 56,
                w,
                h,
                SWP_NOZORDER | SWP_NOACTIVATE,
            );
        }
    }
}

/// The voice: lines are spoken on a thread of their own, each one cutting off
/// whatever was still being said.
pub struct Voice {
    lines: Sender<String>,
}

impl Voice {
    /// The system's default voice, `rate` from -10 (slow) to 10 (fast); `None` if there is none.
    pub fn start(rate: i32) -> Option<Voice> {
        let (lines, rx) = channel::<String>();
        let (ready, is_ready) = channel::<bool>();
        std::thread::spawn(move || {
            let voice: ISpVoice = unsafe {
                let _ = CoInitializeEx(None, COINIT_MULTITHREADED);
                match CoCreateInstance(&SpVoice, None, CLSCTX_ALL) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("no voice: {e}");
                        let _ = ready.send(false);
                        return;
                    }
                }
            };
            unsafe {
                let _ = voice.SetRate(rate);
            }
            let _ = ready.send(true);
            let flags = (SPF_ASYNC.0 | SPF_PURGEBEFORESPEAK.0 | SPF_IS_NOT_XML.0) as u32;
            // The text being spoken is kept until the next line has cut it off.
            let mut speaking: Vec<u16> = Vec::new();
            for line in rx {
                let text: Vec<u16> = line.encode_utf16().chain(Some(0)).collect();
                unsafe {
                    if let Err(e) = voice.Speak(PCWSTR(text.as_ptr()), flags, None) {
                        eprintln!("could not speak: {e}");
                    }
                }
                let _said_before = std::mem::replace(&mut speaking, text);
            }
            unsafe {
                let _ = voice.WaitUntilDone(10_000);
            }
            drop(speaking);
        });
        is_ready
            .recv()
            .ok()
            .filter(|&ok| ok)
            .map(|_| Voice { lines })
    }

    pub fn say(&self, line: &str) {
        let _ = self.lines.send(line.to_string());
    }
}

unsafe extern "system" fn window_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    unsafe { DefWindowProcW(hwnd, msg, wparam, lparam) }
}

/// The marks' window.
pub struct Overlay {
    hwnd: HWND,
    /// What it shows now: the layout, the marks, and where on the desktop the capture's top left is.
    shown: Option<(Layout, Marks, (i32, i32))>,
    visible: bool,
    /// Whether it keeps out of screen captures (Windows 10 2004 and later).
    pub hidden_from_capture: bool,
}

impl Overlay {
    /// `keep_out_of_captures`: the window asks to be left out of screen captures
    /// (screenshots then show the board without the marks). Without it, or on a
    /// Windows too old for it, the window hides while the coach looks.
    pub fn new(keep_out_of_captures: bool) -> windows::core::Result<Overlay> {
        unsafe {
            let instance: HINSTANCE = GetModuleHandleW(PCWSTR::null())?.into();
            let class = w!("MinesCoachMarks");
            let wc = WNDCLASSEXW {
                cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
                lpfnWndProc: Some(window_proc),
                hInstance: instance,
                lpszClassName: class,
                ..Default::default()
            };
            RegisterClassExW(&wc);
            let hwnd = CreateWindowExW(
                WS_EX_LAYERED
                    | WS_EX_TRANSPARENT
                    | WS_EX_TOPMOST
                    | WS_EX_TOOLWINDOW
                    | WS_EX_NOACTIVATE,
                class,
                w!("Minesweeper coach marks"),
                WS_POPUP,
                0,
                0,
                1,
                1,
                None,
                None,
                Some(instance),
                None,
            )?;
            let hidden_from_capture = keep_out_of_captures
                && SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE).is_ok();
            Ok(Overlay {
                hwnd,
                shown: None,
                visible: false,
                hidden_from_capture,
            })
        }
    }

    /// Lets the window handle what the system sends it.
    pub fn pump(&self) {
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                let _ = DispatchMessageW(&msg);
            }
        }
    }

    /// Out of the way while the screen is captured, when the capture would see it.
    pub fn before_capture(&self) {
        if self.visible && !self.hidden_from_capture {
            unsafe {
                let _ = ShowWindow(self.hwnd, SW_HIDE);
                let _ = DwmFlush();
            }
        }
    }

    pub fn after_capture(&self) {
        if self.visible && !self.hidden_from_capture {
            unsafe {
                let _ = ShowWindow(self.hwnd, SW_SHOWNOACTIVATE);
            }
        }
    }

    pub fn hide(&mut self) {
        if self.visible {
            unsafe {
                let _ = ShowWindow(self.hwnd, SW_HIDE);
            }
            self.visible = false;
        }
    }

    /// Shows `marks` laid out by `layout`, the capture's top left being at `desktop`.
    pub fn show(
        &mut self,
        layout: Layout,
        marks: &Marks,
        desktop: (i32, i32),
    ) -> windows::core::Result<()> {
        if self.visible
            && self
                .shown
                .as_ref()
                .is_some_and(|(l, m, d)| *l == layout && m == marks && *d == desktop)
        {
            return Ok(());
        }
        let img = paint(&layout, marks);
        let (w, h) = (layout.width as i32, layout.height as i32);
        unsafe {
            let screen = GetDC(None);
            let mem = CreateCompatibleDC(Some(screen));
            let bmi = BITMAPINFO {
                bmiHeader: BITMAPINFOHEADER {
                    biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                    biWidth: w,
                    biHeight: -h,
                    biPlanes: 1,
                    biBitCount: 32,
                    biCompression: BI_RGB.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut bits: *mut c_void = std::ptr::null_mut();
            let dib = match CreateDIBSection(Some(mem), &bmi, DIB_RGB_COLORS, &mut bits, None, 0) {
                Ok(d) if !bits.is_null() => d,
                Ok(d) => {
                    let _ = DeleteObject(d.into());
                    let _ = DeleteDC(mem);
                    let _ = ReleaseDC(None, screen);
                    return Err(windows::core::Error::from_win32());
                }
                Err(e) => {
                    let _ = DeleteDC(mem);
                    let _ = ReleaseDC(None, screen);
                    return Err(e);
                }
            };
            let old = SelectObject(mem, dib.into());
            let buf = std::slice::from_raw_parts_mut(bits as *mut u8, (w * h * 4) as usize);
            for (i, p) in img.pixels().enumerate() {
                buf[4 * i..4 * i + 4].copy_from_slice(&[p.0[2], p.0[1], p.0[0], p.0[3]]);
            }
            if !marks.caption.is_empty() {
                write_caption(mem, &layout, &marks.caption);
                // GDI leaves the alpha of what it writes at 0: the words are made opaque,
                // and anything it wrote outside the box is cleared.
                for (i, p) in img.pixels().enumerate() {
                    let px = &mut buf[4 * i..4 * i + 4];
                    if p.0[3] == 0 {
                        px.copy_from_slice(&[0, 0, 0, 0]);
                    } else if px[..3] != [p.0[2], p.0[1], p.0[0]] {
                        px[3] = 255;
                    }
                }
            }
            let blend = BLENDFUNCTION {
                BlendOp: AC_SRC_OVER as u8,
                BlendFlags: 0,
                SourceConstantAlpha: 255,
                AlphaFormat: AC_SRC_ALPHA as u8,
            };
            let at = POINT {
                x: desktop.0 + layout.left,
                y: desktop.1 + layout.top,
            };
            let size = SIZE { cx: w, cy: h };
            let from = POINT { x: 0, y: 0 };
            let done = UpdateLayeredWindow(
                self.hwnd,
                Some(screen),
                Some(&at as *const _),
                Some(&size as *const _),
                Some(mem),
                Some(&from as *const _),
                COLORREF(0),
                Some(&blend as *const _),
                ULW_ALPHA,
            );
            let _ = SelectObject(mem, old);
            let _ = DeleteObject(dib.into());
            let _ = DeleteDC(mem);
            let _ = ReleaseDC(None, screen);
            done?;
            if !self.visible {
                let _ = ShowWindow(self.hwnd, SW_SHOWNOACTIVATE);
                self.visible = true;
            }
        }
        self.shown = Some((layout, marks.clone(), desktop));
        Ok(())
    }
}

/// The words, in white, centred in the caption's box, wrapped to its width.
fn write_caption(dc: HDC, layout: &Layout, caption: &str) {
    let (a, b, c, d) = layout.caption;
    let pad = layout.font / 2;
    unsafe {
        let font = CreateFontW(
            -layout.font,
            0,
            0,
            0,
            FW_SEMIBOLD.0 as i32,
            0,
            0,
            0,
            DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS,
            CLIP_DEFAULT_PRECIS,
            ANTIALIASED_QUALITY,
            0,
            w!("Segoe UI"),
        );
        let old = SelectObject(dc, font.into());
        let _ = SetBkMode(dc, TRANSPARENT);
        let _ = SetTextColor(dc, COLORREF(0x00FF_FFFF));
        let mut text: Vec<u16> = caption.encode_utf16().collect();
        let mut measure = RECT {
            left: a + pad,
            top: b,
            right: c - pad,
            bottom: d,
        };
        let _ = DrawTextW(
            dc,
            &mut text,
            &mut measure,
            DT_CENTER | DT_WORDBREAK | DT_NOPREFIX | DT_CALCRECT,
        );
        let height = measure.bottom - measure.top;
        let top = b + ((d - b) - height).max(0) / 2;
        let mut rect = RECT {
            left: a + pad,
            top,
            right: c - pad,
            bottom: top + height,
        };
        let _ = DrawTextW(
            dc,
            &mut text,
            &mut rect,
            DT_CENTER | DT_WORDBREAK | DT_NOPREFIX,
        );
        let _ = GdiFlush();
        let _ = SelectObject(dc, old);
        let _ = DeleteObject(font.into());
    }
}
