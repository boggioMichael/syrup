//! X11 capture through the Composite extension: the server keeps each
//! redirected window's contents in its own pixmap, so a covered window is
//! read as drawn. Without the extension the window is read from the screen.

use image::RgbaImage;
use x11rb::connection::{Connection, RequestConnection};
use x11rb::errors::{ConnectionError, ReplyError};
use x11rb::protocol::ErrorKind;
use x11rb::protocol::composite::{self, ConnectionExt as _, Redirect};
use x11rb::protocol::xproto::{
    AtomEnum, ConnectionExt as _, ImageFormat, ImageOrder, MapState, Visualtype, Window as WindowId,
};
use x11rb::rust_connection::RustConnection;

use crate::capture::{CaptureError, matches};

pub struct Window {
    conn: RustConnection,
    id: WindowId,
    composite: bool,
}

fn failed(e: impl std::fmt::Display) -> CaptureError {
    CaptureError::Failed(format!("X11: {e}"))
}

/// A request on a window that may have closed or been minimised since.
fn on_window(e: ReplyError) -> CaptureError {
    match e {
        ReplyError::X11Error(e)
            if matches!(e.error_kind, ErrorKind::Window | ErrorKind::Drawable) =>
        {
            CaptureError::Closed
        }
        ReplyError::X11Error(e) if e.error_kind == ErrorKind::Match => {
            failed("the window is not shown; is it minimised?")
        }
        e => failed(e),
    }
}

impl From<ConnectionError> for CaptureError {
    fn from(e: ConnectionError) -> CaptureError {
        failed(e)
    }
}

fn connect() -> Result<(RustConnection, WindowId), CaptureError> {
    if std::env::var_os("DISPLAY").is_none() {
        return Err(CaptureError::Unavailable(
            "no X11 display (DISPLAY is not set)".into(),
        ));
    }
    let (conn, screen) = x11rb::connect(None)
        .map_err(|e| CaptureError::Unavailable(format!("cannot connect to X11: {e}")))?;
    let root = conn.setup().roots[screen].root;
    Ok((conn, root))
}

fn atom(conn: &RustConnection, name: &str) -> Result<u32, CaptureError> {
    Ok(conn
        .intern_atom(false, name.as_bytes())?
        .reply()
        .map_err(failed)?
        .atom)
}

/// The windows the window manager lists, then the root's children: those
/// cover unmanaged windows and a list a departed window manager left
/// behind. Listed windows that no longer exist have no title.
fn candidates(conn: &RustConnection, root: WindowId) -> Result<Vec<WindowId>, CaptureError> {
    let client_list = atom(conn, "_NET_CLIENT_LIST")?;
    let listed = conn
        .get_property(false, root, client_list, AtomEnum::WINDOW, 0, u32::MAX)?
        .reply()
        .map_err(failed)?;
    let mut ids: Vec<WindowId> = listed.value32().into_iter().flatten().collect();
    for child in conn.query_tree(root)?.reply().map_err(failed)?.children {
        if !ids.contains(&child) {
            ids.push(child);
        }
    }
    Ok(ids)
}

/// The window's title, if it is mapped and has one.
fn title(conn: &RustConnection, id: WindowId) -> Result<Option<String>, CaptureError> {
    let Ok(attributes) = conn.get_window_attributes(id)?.reply() else {
        return Ok(None);
    };
    if attributes.map_state != MapState::VIEWABLE {
        return Ok(None);
    }
    let (net_name, utf8) = (atom(conn, "_NET_WM_NAME")?, atom(conn, "UTF8_STRING")?);
    for (property, kind) in [
        (net_name, utf8),
        (AtomEnum::WM_NAME.into(), AtomEnum::ANY.into()),
    ] {
        let Ok(reply) = conn
            .get_property(false, id, property, kind, 0, 1024)?
            .reply()
        else {
            return Ok(None);
        };
        if !reply.value.is_empty() {
            return Ok(Some(String::from_utf8_lossy(&reply.value).into_owned()));
        }
    }
    Ok(None)
}

fn titled(conn: &RustConnection, root: WindowId) -> Result<Vec<(WindowId, String)>, CaptureError> {
    let mut found = Vec::new();
    for id in candidates(conn, root)? {
        if let Some(title) = title(conn, id)? {
            found.push((id, title));
        }
    }
    Ok(found)
}

pub fn list_windows() -> Result<Vec<String>, CaptureError> {
    let (conn, root) = connect()?;
    Ok(titled(&conn, root)?.into_iter().map(|(_, t)| t).collect())
}

pub fn find(query: &str) -> Result<(String, Window), CaptureError> {
    let (conn, root) = connect()?;
    let (id, title) = titled(&conn, root)?
        .into_iter()
        .find(|(_, title)| matches(title, query))
        .ok_or(CaptureError::NotFound)?;
    let composite = conn
        .extension_information(composite::X11_EXTENSION_NAME)?
        .is_some()
        && conn.composite_query_version(0, 2)?.reply().is_ok();
    if composite {
        // Automatic redirection keeps the window on screen as before; it
        // lasts as long as this connection.
        conn.composite_redirect_window(id, Redirect::AUTOMATIC)?
            .check()
            .map_err(on_window)?;
    }
    Ok((
        title,
        Window {
            conn,
            id,
            composite,
        },
    ))
}

impl Window {
    pub fn capture(&mut self) -> Result<RgbaImage, CaptureError> {
        let conn = &self.conn;
        let geometry = conn.get_geometry(self.id)?.reply().map_err(on_window)?;
        let visual = conn
            .get_window_attributes(self.id)?
            .reply()
            .map_err(on_window)?
            .visual;
        let (width, height) = (geometry.width, geometry.height);

        let image = if self.composite {
            let pixmap = conn.generate_id().map_err(failed)?;
            conn.composite_name_window_pixmap(self.id, pixmap)?
                .check()
                .map_err(on_window)?;
            let image = conn
                .get_image(ImageFormat::Z_PIXMAP, pixmap, 0, 0, width, height, !0)?
                .reply();
            conn.free_pixmap(pixmap)?;
            image
        } else {
            conn.get_image(ImageFormat::Z_PIXMAP, self.id, 0, 0, width, height, !0)?
                .reply()
        }
        .map_err(on_window)?;

        let format = conn
            .setup()
            .pixmap_formats
            .iter()
            .find(|f| f.depth == image.depth)
            .filter(|f| f.bits_per_pixel == 32)
            .ok_or_else(|| failed(format!("depth {} is not 32 bits per pixel", image.depth)))?;
        let visual = self
            .visual(visual)
            .ok_or_else(|| failed("the window's visual is not on any screen"))?;
        let stride = (width as usize * 4).next_multiple_of(format.scanline_pad as usize / 8);
        let big_endian = conn.setup().image_byte_order == ImageOrder::MSB_FIRST;
        let channel = |mask: u32| {
            let shift = mask.trailing_zeros();
            move |pixel: u32| ((pixel & mask) >> shift) as u8
        };
        let (red, green, blue) = (
            channel(visual.red_mask),
            channel(visual.green_mask),
            channel(visual.blue_mask),
        );

        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
        for row in image.data.chunks(stride).take(height as usize) {
            for bytes in row[..width as usize * 4].as_chunks::<4>().0 {
                let pixel = if big_endian {
                    u32::from_be_bytes(*bytes)
                } else {
                    u32::from_le_bytes(*bytes)
                };
                rgba.extend_from_slice(&[red(pixel), green(pixel), blue(pixel), 255]);
            }
        }
        RgbaImage::from_raw(width.into(), height.into(), rgba)
            .ok_or_else(|| failed("the server returned a short image"))
    }

    fn visual(&self, id: u32) -> Option<Visualtype> {
        self.conn
            .setup()
            .roots
            .iter()
            .flat_map(|screen| &screen.allowed_depths)
            .flat_map(|depth| &depth.visuals)
            .find(|visual| visual.visual_id == id)
            .copied()
    }
}
