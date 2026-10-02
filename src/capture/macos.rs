//! ScreenCaptureKit (macOS 14 and later) captures one window's own
//! contents, so a covered window is captured as drawn. The process needs
//! the Screen Recording permission; the first refusal asks the system to
//! prompt for it.

use std::sync::mpsc;
use std::time::Duration;

use block2::RcBlock;
use image::RgbaImage;
use objc2::rc::Retained;
use objc2::{AllocAnyThread, available};
use objc2_core_graphics::{
    CGBitmapInfo, CGDataProvider, CGImage, CGImageAlphaInfo, CGImageByteOrderInfo, CGMainDisplayID,
    CGRequestScreenCaptureAccess,
};
use objc2_foundation::NSError;
use objc2_screen_capture_kit::{
    SCContentFilter, SCScreenshotManager, SCShareableContent, SCStreamConfiguration, SCWindow,
};

use super::{CaptureError, matches};

/// The error ScreenCaptureKit reports without the Screen Recording permission.
const USER_DECLINED: isize = -3801;
const REPLY: Duration = Duration::from_secs(10);
/// kCGBitmapByteOrderMask.
const BYTE_ORDER_MASK: u32 = 0x7000;

pub struct Window {
    id: u32,
    filter: Retained<SCContentFilter>,
}

// SAFETY: the filter is immutable once made, and ScreenCaptureKit accepts
// it from any thread.
unsafe impl Send for Window {}

/// A value handed out of a completion handler to the waiting thread.
struct Reply<T>(T);
// SAFETY: each reply is moved once, from the handler to the caller.
unsafe impl<T> Send for Reply<T> {}

fn error(e: &NSError) -> CaptureError {
    if e.code() == USER_DECLINED {
        CGRequestScreenCaptureAccess();
        return CaptureError::Denied(
            "window capture needs the Screen Recording permission: allow this application \
             in System Settings > Privacy & Security > Screen Recording, then restart it"
                .into(),
        );
    }
    CaptureError::Failed(format!("ScreenCaptureKit: {}", e.localizedDescription()))
}

fn no_reply() -> CaptureError {
    CaptureError::Failed("ScreenCaptureKit did not reply".into())
}

fn shareable() -> Result<Retained<SCShareableContent>, CaptureError> {
    if !available!(macos = 14.0) {
        return Err(CaptureError::Unavailable(
            "window capture needs macOS 14 or later".into(),
        ));
    }
    // Connects a command-line process to the window server, which
    // ScreenCaptureKit expects to have happened.
    CGMainDisplayID();
    let (send, receive) = mpsc::channel();
    let handler = RcBlock::new(move |content: *mut SCShareableContent, e: *mut NSError| {
        let reply = unsafe {
            match Retained::retain(content) {
                Some(content) => Ok(content),
                None => Err(Retained::retain(e)),
            }
        };
        let _ = send.send(Reply(reply));
    });
    unsafe {
        SCShareableContent::getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler(
            true, true, &handler,
        );
    }
    match receive.recv_timeout(REPLY).map_err(|_| no_reply())?.0 {
        Ok(content) => Ok(content),
        Err(Some(e)) => Err(error(&e)),
        Err(None) => Err(no_reply()),
    }
}

/// Every on-screen window with a title.
fn windows() -> Result<Vec<(String, Retained<SCWindow>)>, CaptureError> {
    let content = shareable()?;
    let windows = unsafe { content.windows() };
    Ok(windows
        .iter()
        .filter_map(|window| {
            let title = unsafe { window.title() }?.to_string();
            (!title.is_empty()).then_some((title, window))
        })
        .collect())
}

pub fn list_windows() -> Result<Vec<String>, CaptureError> {
    Ok(windows()?.into_iter().map(|(title, _)| title).collect())
}

pub fn find(query: &str) -> Result<(String, Window), CaptureError> {
    let (title, window) = windows()?
        .into_iter()
        .find(|(title, _)| matches(title, query))
        .ok_or(CaptureError::NotFound)?;
    let filter = unsafe {
        SCContentFilter::initWithDesktopIndependentWindow(SCContentFilter::alloc(), &window)
    };
    let id = unsafe { window.windowID() };
    Ok((title, Window { id, filter }))
}

impl Window {
    pub fn capture(&mut self) -> Result<RgbaImage, CaptureError> {
        let image = self.screenshot().map_err(|e| {
            let open = windows().is_ok_and(|windows| {
                windows
                    .iter()
                    .any(|(_, w)| unsafe { w.windowID() } == self.id)
            });
            if open { e } else { CaptureError::Closed }
        })?;
        rgba(&image)
    }

    fn screenshot(&self) -> Result<Retained<CGImage>, CaptureError> {
        // The window's current size, in pixels.
        let info = unsafe { SCShareableContent::infoForFilter(&self.filter) };
        let (rect, scale) = unsafe { (info.contentRect(), info.pointPixelScale() as f64) };
        let config = unsafe { SCStreamConfiguration::new() };
        unsafe {
            config.setWidth((rect.size.width * scale).round() as usize);
            config.setHeight((rect.size.height * scale).round() as usize);
            config.setShowsCursor(false);
            config.setIgnoreShadowsSingleWindow(true);
        }
        let (send, receive) = mpsc::channel();
        let handler = RcBlock::new(move |image: *mut CGImage, e: *mut NSError| {
            let reply = unsafe {
                match std::ptr::NonNull::new(image) {
                    Some(image) => Ok(Retained::retain(image.as_ptr()).expect("not null")),
                    None => Err(Retained::retain(e)),
                }
            };
            let _ = send.send(Reply(reply));
        });
        unsafe {
            SCScreenshotManager::captureImageWithFilter_configuration_completionHandler(
                &self.filter,
                &config,
                Some(&handler),
            );
        }
        match receive.recv_timeout(REPLY).map_err(|_| no_reply())?.0 {
            Ok(image) => Ok(image),
            Err(Some(e)) => Err(error(&e)),
            Err(None) => Err(no_reply()),
        }
    }
}

/// The image's pixels as opaque RGBA.
fn rgba(image: &CGImage) -> Result<RgbaImage, CaptureError> {
    let (width, height) = (CGImage::width(Some(image)), CGImage::height(Some(image)));
    let stride = CGImage::bytes_per_row(Some(image));
    let info = CGImage::bitmap_info(Some(image));
    let order = info.0 & BYTE_ORDER_MASK;
    let alpha = CGImageAlphaInfo(info.0 & CGBitmapInfo::AlphaInfoMask.0);
    let alpha_first = [
        CGImageAlphaInfo::PremultipliedFirst,
        CGImageAlphaInfo::First,
        CGImageAlphaInfo::NoneSkipFirst,
    ]
    .contains(&alpha);
    // Which byte of each pixel holds red, green and blue.
    let (r, g, b) = match (order, alpha_first) {
        (o, true) if o == CGImageByteOrderInfo::Order32Little.0 => (2, 1, 0),
        (o, false) if o == CGImageByteOrderInfo::Order32Big.0 || o == 0 => (0, 1, 2),
        _ => {
            return Err(CaptureError::Failed(format!(
                "unexpected pixel layout (bitmap info {:#x})",
                info.0
            )));
        }
    };
    if CGImage::bits_per_pixel(Some(image)) != 32 || stride < width * 4 {
        return Err(CaptureError::Failed("unexpected pixel size".into()));
    }
    let data = CGDataProvider::data(CGImage::data_provider(Some(image)).as_deref())
        .ok_or_else(|| CaptureError::Failed("the image has no pixel data".into()))?
        .to_vec();
    let mut pixels = Vec::with_capacity(width * height * 4);
    for row in data.chunks(stride).take(height) {
        for p in row[..width * 4].as_chunks::<4>().0 {
            pixels.extend_from_slice(&[p[r], p[g], p[b], 255]);
        }
    }
    RgbaImage::from_raw(width as u32, height as u32, pixels)
        .ok_or_else(|| CaptureError::Failed("the image is shorter than its size".into()))
}
