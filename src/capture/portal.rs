//! A window shared through the XDG desktop portal's ScreenCast interface.
//! The desktop asks the user which window to share; the portal's restore
//! token for that choice is kept per query, so later runs share the same
//! window without asking (desktops whose portal supports persistence).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use image::RgbaImage;
use zbus::blocking::{Connection, Proxy};
use zbus::zvariant::{ObjectPath, OwnedFd, OwnedObjectPath, OwnedValue, Value};

use super::pipewire::Stream;
use crate::capture::CaptureError;

const DESTINATION: &str = "org.freedesktop.portal.Desktop";
const PATH: &str = "/org/freedesktop/portal/desktop";
const SCREEN_CAST: &str = "org.freedesktop.portal.ScreenCast";
const SOURCE_WINDOW: u32 = 2;
const CURSOR_HIDDEN: u32 = 1;
const PERSIST_UNTIL_REVOKED: u32 = 2;

pub struct Window {
    conn: Connection,
    session: OwnedObjectPath,
    stream: Stream,
}

fn failed(e: impl std::fmt::Display) -> CaptureError {
    CaptureError::Failed(format!("screen-cast portal: {e}"))
}

impl Window {
    pub fn open(query: &str) -> Result<Window, CaptureError> {
        let conn = Connection::session()
            .map_err(|e| CaptureError::Unavailable(format!("no D-Bus session bus: {e}")))?;
        let portal = Proxy::new(&conn, DESTINATION, PATH, SCREEN_CAST).map_err(failed)?;
        let sources: u32 = portal.get_property("AvailableSourceTypes").map_err(|e| {
            CaptureError::Unavailable(format!("the desktop has no screen-cast portal: {e}"))
        })?;
        if sources & SOURCE_WINDOW == 0 {
            return Err(CaptureError::Unavailable(
                "this desktop's screen-cast portal shares whole screens only, not windows".into(),
            ));
        }
        let requests = Requests::new(&conn)?;
        let token = requests.token();
        let mut created = requests.call(&token, || {
            portal.call(
                "CreateSession",
                &(HashMap::from([
                    ("handle_token", Value::from(token.as_str())),
                    ("session_handle_token", Value::from(token.as_str())),
                ]),),
            )
        })?;
        let session = created
            .remove("session_handle")
            .and_then(|handle| String::try_from(handle).ok())
            .and_then(|handle| OwnedObjectPath::try_from(handle).ok())
            .ok_or_else(|| failed("CreateSession returned no session"))?;
        match share(&portal, &requests, &session, query) {
            Ok(stream) => Ok(Window {
                conn,
                session,
                stream,
            }),
            Err(e) => {
                close(&conn, &session);
                Err(e)
            }
        }
    }

    pub fn capture(&mut self) -> Result<RgbaImage, CaptureError> {
        self.stream.capture()
    }
}

impl Drop for Window {
    fn drop(&mut self) {
        close(&self.conn, &self.session);
    }
}

/// Asks the desktop for a window (or restores the one chosen for `query`
/// before) and connects to its stream.
fn share(
    portal: &Proxy,
    requests: &Requests,
    session: &OwnedObjectPath,
    query: &str,
) -> Result<Stream, CaptureError> {
    let version: u32 = portal.get_property("version").map_err(failed)?;
    let cursors: u32 = portal.get_property("AvailableCursorModes").unwrap_or(0);
    let remembered = restore_file(query);
    let restore = remembered
        .as_ref()
        .and_then(|path| std::fs::read_to_string(path).ok());

    let token = requests.token();
    let mut select = HashMap::from([
        ("handle_token", Value::from(token.as_str())),
        ("types", Value::from(SOURCE_WINDOW)),
        ("multiple", Value::from(false)),
    ]);
    if cursors & CURSOR_HIDDEN != 0 {
        select.insert("cursor_mode", Value::from(CURSOR_HIDDEN));
    }
    if version >= 4 {
        select.insert("persist_mode", Value::from(PERSIST_UNTIL_REVOKED));
        if let Some(restore) = &restore {
            select.insert("restore_token", Value::from(restore.as_str()));
        }
    }
    requests.call(&token, || {
        portal.call("SelectSources", &(ObjectPath::from(session), select))
    })?;

    let token = requests.token();
    let mut started = requests.call(&token, || {
        portal.call(
            "Start",
            &(
                ObjectPath::from(session),
                "",
                HashMap::from([("handle_token", Value::from(token.as_str()))]),
            ),
        )
    })?;
    let streams: Vec<(u32, HashMap<String, OwnedValue>)> = started
        .remove("streams")
        .and_then(|streams| streams.try_into().ok())
        .ok_or_else(|| failed("Start returned no streams"))?;
    let (node, _) = streams
        .first()
        .ok_or_else(|| CaptureError::Denied("no window was shared".into()))?;
    let token = started
        .remove("restore_token")
        .and_then(|token| String::try_from(token).ok());
    if let (Some(path), Some(token)) = (remembered, token) {
        // Losing the token only means asking again next time.
        let _ = std::fs::create_dir_all(path.parent().expect("a file in a directory"))
            .and_then(|()| std::fs::write(&path, token));
    }

    let fd: OwnedFd = portal
        .call(
            "OpenPipeWireRemote",
            &(ObjectPath::from(session), HashMap::<&str, Value>::new()),
        )
        .map_err(failed)?;
    Stream::connect(fd.into(), *node)
}

fn close(conn: &Connection, session: &OwnedObjectPath) {
    if let Ok(session) = Proxy::new(
        conn,
        DESTINATION,
        session.as_ref(),
        "org.freedesktop.portal.Session",
    ) {
        let _: zbus::Result<()> = session.call("Close", &());
    }
}

/// Where the restore token for `query` is kept:
/// `$XDG_STATE_HOME/syrup/screencast/<query>`.
fn restore_file(query: &str) -> Option<PathBuf> {
    let state = std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".local/state"))
        })?;
    let name: String = query
        .to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .take(128)
        .collect();
    Some(state.join("syrup/screencast").join(name))
}

/// Portal calls answer through a Request object whose path is known in
/// advance from the caller's bus name and a token.
struct Requests<'a> {
    conn: &'a Connection,
    sender: String,
}

impl<'a> Requests<'a> {
    fn new(conn: &'a Connection) -> Result<Requests<'a>, CaptureError> {
        let name = conn
            .unique_name()
            .ok_or_else(|| failed("the bus gave no unique name"))?;
        Ok(Requests {
            conn,
            sender: name.trim_start_matches(':').replace('.', "_"),
        })
    }

    fn token(&self) -> String {
        static NEXT: AtomicU32 = AtomicU32::new(0);
        format!(
            "syrup{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        )
    }

    /// Makes the call, then waits for its Response.
    fn call(
        &self,
        token: &str,
        call: impl FnOnce() -> zbus::Result<OwnedObjectPath>,
    ) -> Result<HashMap<String, OwnedValue>, CaptureError> {
        let path = format!("{PATH}/request/{}/{token}", self.sender);
        let request = Proxy::new(
            self.conn,
            DESTINATION,
            path.as_str(),
            "org.freedesktop.portal.Request",
        )
        .map_err(failed)?;
        let mut responses = request.receive_signal("Response").map_err(failed)?;
        call().map_err(failed)?;
        let response = responses
            .next()
            .ok_or_else(|| failed("the portal closed the request"))?;
        let (code, results): (u32, HashMap<String, OwnedValue>) =
            response.body().deserialize().map_err(failed)?;
        match code {
            0 => Ok(results),
            1 => Err(CaptureError::Denied("the window was not shared".into())),
            _ => Err(failed("the request failed")),
        }
    }
}
