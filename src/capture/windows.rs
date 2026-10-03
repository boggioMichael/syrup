//! Windows: the window's frames straight from the compositor, on the GPU,
//! through Windows.Graphics.Capture (Windows 10 1903 and later); the GDI
//! path — `PrintWindow`, else a copy of the screen — where that is not to
//! be had.
//!
//! ```text
//!   compositor ─▶ Direct3D11CaptureFramePool (2 buffers, BGRA)
//!                      │ a new frame when the window's content changed
//!                      ▼
//!                 the client area of the latest frame, kept on the GPU
//!                      │ a region, copied to a staging texture and mapped
//!                      ▼
//!                 RgbaImage of just that region (or of the whole frame)
//! ```
//!
//! The compositor's frame is the whole window, title bar and all; only the
//! client area is kept, as `PrintWindow` gave it, so a layout learned on
//! one path holds on the other. The frame pool is free-threaded, so no
//! message loop is needed. The cursor is left out, and the capture border
//! is turned off where the system allows it (Windows 11). A window whose
//! content does not change yields no new frame; the last one is served
//! again. When the GPU path
//! fails three times running — a system without the API, a window it
//! cannot capture, a device lost — the GDI path takes over for good, with
//! its device context and bitmap kept between frames rather than made and
//! destroyed each time.
//!
//! The whole screen is a copy of every monitor, as the user sees it.

use std::ffi::OsString;
use std::os::windows::ffi::OsStringExt;
use std::time::{Duration, Instant};

use image::RgbaImage;
use windows::Graphics::Capture::{
    Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureAccess,
    GraphicsCaptureAccessKind, GraphicsCaptureItem, GraphicsCaptureSession,
};
use windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
use windows::Graphics::DirectX::DirectXPixelFormat;
use windows::Graphics::SizeInt32;
use windows::Win32::Foundation::{HMODULE, HWND, LPARAM, POINT, RECT};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_HARDWARE;
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BOX, D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_MAP_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
    D3D11_USAGE_STAGING, D3D11CreateDevice, ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dwm::{DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::IDXGIDevice;
use windows::Win32::Graphics::Gdi::{
    BI_RGB, BITMAPINFO, BITMAPINFOHEADER, BitBlt, ClientToScreen, CreateCompatibleBitmap,
    CreateCompatibleDC, DIB_RGB_COLORS, DeleteDC, DeleteObject, GetDC, GetDIBits, HBITMAP, HDC,
    HGDIOBJ, ReleaseDC, SRCCOPY, SelectObject,
};
use windows::Win32::Storage::Xps::{PRINT_WINDOW_FLAGS, PrintWindow};
use windows::Win32::System::WinRT::Direct3D11::{
    CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::Win32::System::WinRT::{RO_INIT_MULTITHREADED, RoInitialize};
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, SetThreadDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GetClientRect, GetSystemMetrics, GetWindowRect, GetWindowTextLengthW,
    GetWindowTextW, IsIconic, IsWindow, IsWindowVisible, SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN,
    SM_XVIRTUALSCREEN, SM_YVIRTUALSCREEN,
};
use windows::core::{BOOL, Interface};

use super::{CaptureError, FrameSource, Rect, Screen, matches};

/// PW_CLIENTONLY | PW_RENDERFULLCONTENT: render just the client area,
/// and include content drawn outside the classic GDI path.
const PW_CLIENTONLY_FULL: u32 = 0x0000_0001 | 0x0000_0002;

/// How long the first frame of a GPU capture is waited for, and later
/// frames: the compositor sends one only when the window's content changed.
const FIRST_FRAME_WAIT: Duration = Duration::from_millis(400);
const FRAME_WAIT: Duration = Duration::from_millis(40);
/// GPU captures failing in a row before the GDI path takes over for good.
const GPU_FAILURES: u32 = 3;

pub struct Window {
    hwnd: HWND,
    gpu: GpuState,
    gdi: Option<GdiSurface>,
}

enum GpuState {
    Untried,
    Ready(Box<Gpu>),
    /// Why the GPU path is not used, for diagnostics.
    Unavailable(String),
}

// SAFETY: an HWND is a handle valid from any thread; the Direct3D and
// capture objects are used from one thread at a time, through `&mut`.
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
        .map(|(hwnd, title)| {
            (
                title,
                Window {
                    hwnd,
                    gpu: if super::cpu_asked_for() {
                        GpuState::Unavailable("SYRUP_CAPTURE=cpu".into())
                    } else {
                        GpuState::Untried
                    },
                    gdi: None,
                },
            )
        })
        .ok_or(CaptureError::NotFound)
}

/// A window's client area now, in the screen's own pixels: its size, where
/// it sits on the screen, and where it sits within the window's visible
/// bounds — which is the frame the compositor sends, title bar and all.
#[derive(Clone, Copy, Debug)]
struct Client {
    width: i32,
    height: i32,
    origin: POINT,
    /// The client area's offset within the window's visible bounds.
    offset: (u32, u32),
}

impl Window {
    /// The client area now. `Minimised` when it has no area to show.
    fn client(&self) -> Result<Client, CaptureError> {
        if unsafe { !IsWindow(Some(self.hwnd)).as_bool() } {
            return Err(CaptureError::Closed);
        }
        // Measured in physical pixels whatever the process's DPI awareness:
        // the compositor's frame and GDI's copy of the screen are in those.
        let previous =
            unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
        let measured = self.measure();
        unsafe { SetThreadDpiAwarenessContext(previous) };
        let client = measured?;
        if client.width <= 0 || client.height <= 0 {
            // A minimised window's client rectangle collapses to nothing.
            return Err(if unsafe { IsIconic(self.hwnd).as_bool() } {
                CaptureError::Minimised
            } else {
                CaptureError::Failed("the window has no client area".into())
            });
        }
        Ok(client)
    }

    fn measure(&self) -> Result<Client, CaptureError> {
        let mut rect = RECT::default();
        let mut origin = POINT::default();
        let mut bounds = RECT::default();
        unsafe {
            GetClientRect(self.hwnd, &mut rect)
                .map_err(|e| CaptureError::Failed(format!("GetClientRect: {e}")))?;
            if !ClientToScreen(self.hwnd, &mut origin).as_bool() {
                return Err(CaptureError::Failed("ClientToScreen failed".into()));
            }
            // The visible bounds leave out the invisible resize borders the
            // window rectangle includes; the compositor's frame starts at
            // the visible ones.
            if DwmGetWindowAttribute(
                self.hwnd,
                DWMWA_EXTENDED_FRAME_BOUNDS,
                (&raw mut bounds).cast(),
                size_of::<RECT>() as u32,
            )
            .is_err()
            {
                GetWindowRect(self.hwnd, &mut bounds)
                    .map_err(|e| CaptureError::Failed(format!("GetWindowRect: {e}")))?;
            }
        }
        Ok(Client {
            width: rect.right - rect.left,
            height: rect.bottom - rect.top,
            origin,
            offset: (
                (origin.x - bounds.left).max(0) as u32,
                (origin.y - bounds.top).max(0) as u32,
            ),
        })
    }

    /// The window's contents as a frame: on the GPU when the system
    /// captures it there, else as pixels from GDI.
    pub fn capture_frame(&mut self) -> Result<FrameSource<'_>, CaptureError> {
        let client = self.client()?;
        if matches!(self.gpu, GpuState::Untried) {
            self.gpu = match Gpu::new(self.hwnd) {
                Ok(gpu) => GpuState::Ready(Box::new(gpu)),
                Err(why) => GpuState::Unavailable(why),
            };
        }
        // Decided first, borrowed after: a borrow of the GPU state that is
        // returned on one path cannot be live where the other path falls
        // back to GDI.
        let mut given_up = None;
        let fresh = match &mut self.gpu {
            GpuState::Ready(gpu) => match gpu.refresh(&client) {
                Ok(()) => {
                    gpu.failures = 0;
                    true
                }
                Err(why) => {
                    gpu.failures += 1;
                    if gpu.failures >= GPU_FAILURES {
                        given_up = Some(why);
                    }
                    false
                }
            },
            _ => false,
        };
        if let Some(why) = given_up {
            self.gpu = GpuState::Unavailable(format!(
                "the GPU capture failed {GPU_FAILURES} times running: {why}"
            ));
        }
        if fresh {
            return match &mut self.gpu {
                GpuState::Ready(gpu) => Ok(FrameSource::Gpu(gpu)),
                _ => Err(CaptureError::Failed("the GPU frame went missing".into())),
            };
        }
        self.capture_gdi(client.width, client.height, client.origin)
            .map(FrameSource::Cpu)
    }

    /// Frames through GDI only, from now on: for comparing the paths, or a
    /// driver the GPU path does not get on with.
    pub fn without_gpu(&mut self) {
        self.gpu = GpuState::Unavailable("asked for the CPU path".into());
    }

    pub fn capture(&mut self) -> Result<RgbaImage, CaptureError> {
        super::Frame::new(self.capture_frame()?)
            .read_all()
            .ok_or_else(|| CaptureError::Failed("the frame could not be read".into()))
    }

    /// Why frames are not coming from the GPU, when they are not: `None`
    /// while they are (or before the first capture).
    pub fn gpu_unavailable(&self) -> Option<&str> {
        match &self.gpu {
            GpuState::Unavailable(why) => Some(why),
            _ => None,
        }
    }

    /// The GDI path: the window asked to draw itself, else a copy of the
    /// screen where it is; the objects kept between frames of one size.
    fn capture_gdi(
        &mut self,
        width: i32,
        height: i32,
        origin: POINT,
    ) -> Result<RgbaImage, CaptureError> {
        if self
            .gdi
            .as_ref()
            .is_none_or(|s| s.width != width || s.height != height)
        {
            self.gdi = Some(GdiSurface::new(width, height)?);
        }
        let surface = self.gdi.as_mut().expect("made above");
        surface.origin = origin;
        // Drawing the window itself rather than copying the screen means a
        // covered window yields its own pixels, not whatever is on top.
        let printed = unsafe {
            PrintWindow(
                self.hwnd,
                surface.memory,
                PRINT_WINDOW_FLAGS(PW_CLIENTONLY_FULL),
            )
            .as_bool()
        };
        let mut pixels = if printed { surface.read() } else { None };
        // PrintWindow can succeed yet return a flat surface for GPU-drawn
        // content; the screen copy is right whenever the window is visible.
        if pixels.as_deref().is_none_or(is_blank) && surface.copy_screen() {
            pixels = surface.read();
        }
        let mut pixels =
            pixels.ok_or_else(|| CaptureError::Failed("GetDIBits returned nothing".into()))?;
        swizzle(&mut pixels);
        Ok(RgbaImage::from_raw(width as u32, height as u32, pixels).expect("sized to fit"))
    }
}

/// BGRA in place to RGBA, opaque.
fn swizzle(pixels: &mut [u8]) {
    for pixel in pixels.as_chunks_mut::<4>().0 {
        pixel.swap(0, 2);
        pixel[3] = 255;
    }
}

/// Windows.Graphics.Capture of one window: the Direct3D device, the frame
/// pool and session, and the latest frame kept on the GPU.
pub struct Gpu {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    d3d_device: IDirect3DDevice,
    pool: Direct3D11CaptureFramePool,
    _session: GraphicsCaptureSession,
    /// The pool's buffer size; recreated when the window changes size.
    pool_size: SizeInt32,
    /// The latest frame's pixels, on the GPU, and their size.
    latest: Option<(ID3D11Texture2D, u32, u32)>,
    /// Staging textures by size, for reading regions back.
    staging: Vec<(u32, u32, ID3D11Texture2D)>,
    first_frame_by: Option<Instant>,
    failures: u32,
}

impl Gpu {
    fn new(hwnd: HWND) -> Result<Gpu, String> {
        unsafe {
            // Already initialised, or initialised the other way, are fine:
            // the API only needs the runtime up on this thread.
            let _ = RoInitialize(RO_INIT_MULTITHREADED);
        }
        if !GraphicsCaptureSession::IsSupported().map_err(|e| format!("IsSupported: {e}"))? {
            return Err("Windows.Graphics.Capture is not supported here".into());
        }
        let (mut device, mut context) = (None, None);
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_HARDWARE,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .map_err(|e| format!("D3D11CreateDevice: {e}"))?;
        }
        let device = device.ok_or("no Direct3D device")?;
        let context = context.ok_or("no Direct3D context")?;
        let d3d_device = winrt_device(&device)?;
        let interop: IGraphicsCaptureItemInterop =
            windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()
                .map_err(|e| format!("IGraphicsCaptureItemInterop: {e}"))?;
        let item: GraphicsCaptureItem = unsafe {
            interop
                .CreateForWindow(hwnd)
                .map_err(|e| format!("CreateForWindow: {e}"))?
        };
        let pool_size = item.Size().map_err(|e| format!("Size: {e}"))?;
        if pool_size.Width <= 0 || pool_size.Height <= 0 {
            return Err("the window has no size to capture".into());
        }
        let pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            &d3d_device,
            DirectXPixelFormat::B8G8R8A8UIntNormalized,
            2,
            pool_size,
        )
        .map_err(|e| format!("CreateFreeThreaded: {e}"))?;
        let session = pool
            .CreateCaptureSession(&item)
            .map_err(|e| format!("CreateCaptureSession: {e}"))?;
        // Neither the cursor nor the yellow capture border belong in a
        // picture of the game. Older systems lack these settings, and the
        // border needs permission newer ones grant a desktop app without
        // asking; where any of this fails, the capture goes on as is.
        let _ = session.SetIsCursorCaptureEnabled(false);
        if let Ok(request) =
            GraphicsCaptureAccess::RequestAccessAsync(GraphicsCaptureAccessKind::Borderless)
        {
            let _ = request.get();
        }
        let _ = session.SetIsBorderRequired(false);
        session
            .StartCapture()
            .map_err(|e| format!("StartCapture: {e}"))?;
        Ok(Gpu {
            device,
            context,
            d3d_device,
            pool,
            _session: session,
            pool_size,
            latest: None,
            staging: Vec::new(),
            first_frame_by: Some(Instant::now() + FIRST_FRAME_WAIT),
            failures: 0,
        })
    }

    /// The newest frame the compositor has sent, with any older ones in
    /// the pool returned to it.
    fn newest(&self) -> Option<Direct3D11CaptureFrame> {
        let mut newest: Option<Direct3D11CaptureFrame> = None;
        while let Ok(frame) = self.pool.TryGetNextFrame() {
            if let Some(older) = newest.replace(frame) {
                let _ = older.Close();
            }
        }
        newest
    }

    /// Bring `latest` up to date: the newest frame, if one came, with just
    /// the client area copied to a texture of its own so the pool's buffer
    /// goes back at once. With no new frame — the window did not change —
    /// the last one stands, once there is one.
    fn refresh(&mut self, client: &Client) -> Result<(), String> {
        let deadline = match self.first_frame_by.take() {
            Some(first) => first,
            None => Instant::now() + FRAME_WAIT,
        };
        let frame = loop {
            if let Some(frame) = self.newest() {
                break Some(frame);
            }
            if self.latest.is_some() || Instant::now() >= deadline {
                break None;
            }
            std::thread::sleep(Duration::from_millis(2));
        };
        let Some(frame) = frame else {
            return match &self.latest {
                Some(_) => Ok(()),
                None => Err("no frame came from the compositor".into()),
            };
        };
        let result = self.take(&frame, client);
        let _ = frame.Close();
        result
    }

    fn take(&mut self, frame: &Direct3D11CaptureFrame, client: &Client) -> Result<(), String> {
        let content = frame
            .ContentSize()
            .map_err(|e| format!("ContentSize: {e}"))?;
        // The window changed size: the pool follows, from the next frame.
        if (content.Width != self.pool_size.Width || content.Height != self.pool_size.Height)
            && content.Width > 0
            && content.Height > 0
        {
            self.pool
                .Recreate(
                    &self.d3d_device,
                    DirectXPixelFormat::B8G8R8A8UIntNormalized,
                    2,
                    content,
                )
                .map_err(|e| format!("Recreate: {e}"))?;
            self.pool_size = content;
        }
        let surface = frame.Surface().map_err(|e| format!("Surface: {e}"))?;
        let access: IDirect3DDxgiInterfaceAccess = surface
            .cast()
            .map_err(|e| format!("IDirect3DDxgiInterfaceAccess: {e}"))?;
        let texture: ID3D11Texture2D = unsafe {
            access
                .GetInterface()
                .map_err(|e| format!("GetInterface: {e}"))?
        };
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { texture.GetDesc(&mut desc) };
        // The client area within the frame: the frame is the window's
        // visible bounds, title bar and all, and may lag a resize.
        let (x0, y0) = client.offset;
        let frame_w = (content.Width.max(0) as u32).min(desc.Width);
        let frame_h = (content.Height.max(0) as u32).min(desc.Height);
        let w = (client.width.max(0) as u32).min(frame_w.saturating_sub(x0));
        let h = (client.height.max(0) as u32).min(frame_h.saturating_sub(y0));
        if w == 0 || h == 0 {
            return Err("an empty frame".into());
        }
        // A texture of our own for the latest frame, GPU to GPU.
        let keep = match &self.latest {
            Some((t, tw, th)) if *tw == w && *th == h => t.clone(),
            _ => self.texture(w, h, D3D11_USAGE_DEFAULT, 0)?,
        };
        let region = D3D11_BOX {
            left: x0,
            top: y0,
            front: 0,
            right: x0 + w,
            bottom: y0 + h,
            back: 1,
        };
        unsafe {
            self.context
                .CopySubresourceRegion(&keep, 0, 0, 0, 0, &texture, 0, Some(&region));
        }
        self.latest = Some((keep, w, h));
        Ok(())
    }

    /// A BGRA texture of `w`×`h` with the given usage and CPU access.
    fn texture(
        &self,
        w: u32,
        h: u32,
        usage: windows::Win32::Graphics::Direct3D11::D3D11_USAGE,
        cpu_access: u32,
    ) -> Result<ID3D11Texture2D, String> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: w,
            Height: h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: usage,
            BindFlags: 0,
            CPUAccessFlags: cpu_access,
            MiscFlags: 0,
        };
        let mut made = None;
        unsafe {
            self.device
                .CreateTexture2D(&desc, None, Some(&mut made))
                .map_err(|e| format!("CreateTexture2D: {e}"))?;
        }
        made.ok_or_else(|| "no texture".to_string())
    }

    /// The latest frame's size.
    pub fn size(&self) -> (u32, u32) {
        self.latest
            .as_ref()
            .map(|(_, w, h)| (*w, *h))
            .unwrap_or((0, 0))
    }

    /// A staging texture of `w`×`h`, kept for the next read of that size.
    fn staging(&mut self, w: u32, h: u32) -> Result<ID3D11Texture2D, String> {
        if let Some((_, _, t)) = self.staging.iter().find(|(sw, sh, _)| *sw == w && *sh == h) {
            return Ok(t.clone());
        }
        let texture = self.texture(w, h, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ.0 as u32)?;
        // A few sizes come round again and again (the whole frame, a band,
        // a window around a track); more than that would be a leak.
        if self.staging.len() >= 8 {
            self.staging.remove(0);
        }
        self.staging.push((w, h, texture.clone()));
        Ok(texture)
    }

    /// `region` of the latest frame (clipped to it), read back from the
    /// GPU: only those pixels cross to the CPU.
    pub fn read(&mut self, region: Rect) -> Option<RgbaImage> {
        let (latest, fw, fh) = self.latest.clone()?;
        let x0 = region.x.min(fw);
        let y0 = region.y.min(fh);
        let w = region.w.min(fw - x0);
        let h = region.h.min(fh - y0);
        if w == 0 || h == 0 {
            return None;
        }
        let staging = self.staging(w, h).ok()?;
        let source = D3D11_BOX {
            left: x0,
            top: y0,
            front: 0,
            right: x0 + w,
            bottom: y0 + h,
            back: 1,
        };
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        unsafe {
            self.context
                .CopySubresourceRegion(&staging, 0, 0, 0, 0, &latest, 0, Some(&source));
            self.context
                .Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))
                .ok()?;
            let pitch = mapped.RowPitch as usize;
            let row_bytes = (w * 4) as usize;
            for y in 0..h as usize {
                let src = (mapped.pData as *const u8).add(y * pitch);
                let dst = &mut pixels[y * row_bytes..(y + 1) * row_bytes];
                std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), row_bytes);
            }
            self.context.Unmap(&staging, 0);
        }
        swizzle(&mut pixels);
        RgbaImage::from_raw(w, h, pixels)
    }
}

/// The WinRT view of a Direct3D 11 device, which the capture API takes.
fn winrt_device(device: &ID3D11Device) -> Result<IDirect3DDevice, String> {
    let dxgi: IDXGIDevice = device.cast().map_err(|e| format!("IDXGIDevice: {e}"))?;
    unsafe {
        CreateDirect3D11DeviceFromDXGIDevice(&dxgi)
            .map_err(|e| format!("CreateDirect3D11DeviceFromDXGIDevice: {e}"))?
            .cast()
            .map_err(|e| format!("IDirect3DDevice: {e}"))
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
    let mut surface = GdiSurface::new(width, height).ok()?;
    surface.origin = POINT { x: left, y: top };
    // A plain copy, without CAPTUREBLT: layered windows drawn over the
    // screen (an overlay pointing at what was found) stay out of it.
    let pixels = if surface.copy_screen() {
        surface.read()
    } else {
        None
    };
    let mut pixels = pixels?;
    swizzle(&mut pixels);
    RgbaImage::from_raw(width as u32, height as u32, pixels).map(|image| Screen {
        left,
        top,
        image,
    })
}

/// A memory device context with a bitmap of one size, kept between frames.
struct GdiSurface {
    screen: HDC,
    memory: HDC,
    bitmap: HBITMAP,
    previous: HGDIOBJ,
    origin: POINT,
    width: i32,
    height: i32,
}

impl GdiSurface {
    fn new(width: i32, height: i32) -> Result<GdiSurface, CaptureError> {
        unsafe {
            let screen = GetDC(None);
            let memory = CreateCompatibleDC(Some(screen));
            let bitmap = CreateCompatibleBitmap(screen, width, height);
            if memory.is_invalid() || bitmap.is_invalid() {
                let _ = DeleteObject(bitmap.into());
                let _ = DeleteDC(memory);
                ReleaseDC(None, screen);
                return Err(CaptureError::Failed("no GDI surface could be made".into()));
            }
            let previous = SelectObject(memory, bitmap.into());
            Ok(GdiSurface {
                screen,
                memory,
                bitmap,
                previous,
                origin: POINT::default(),
                width,
                height,
            })
        }
    }

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

impl Drop for GdiSurface {
    fn drop(&mut self) {
        unsafe {
            SelectObject(self.memory, self.previous);
            let _ = DeleteObject(self.bitmap.into());
            let _ = DeleteDC(self.memory);
            ReleaseDC(None, self.screen);
        }
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
