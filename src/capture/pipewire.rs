//! Frames from a PipeWire video stream. libpipewire is loaded when a stream
//! is first opened, so neither building nor importing Syrup needs it.
//! Only the few calls below are used; the constants and layouts are those
//! of `pipewire/stream.h` and `spa/` in PipeWire 0.3 and 1.x.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::os::fd::{IntoRawFd, OwnedFd};
use std::ptr;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Duration;

use image::RgbaImage;
use libloading::Library;

use crate::capture::CaptureError;

const TYPE_ID: u32 = 3;
const TYPE_RECTANGLE: u32 = 10;
const TYPE_FRACTION: u32 = 11;
const TYPE_OBJECT: u32 = 15;
const TYPE_CHOICE: u32 = 19;
const OBJECT_FORMAT: u32 = 0x40003;
const PARAM_ENUM_FORMAT: u32 = 3;
const PARAM_FORMAT: u32 = 4;
const CHOICE_RANGE: u32 = 1;
const CHOICE_ENUM: u32 = 3;
const FORMAT_MEDIA_TYPE: u32 = 1;
const FORMAT_MEDIA_SUBTYPE: u32 = 2;
const FORMAT_VIDEO_FORMAT: u32 = 0x20001;
const FORMAT_VIDEO_SIZE: u32 = 0x20003;
const FORMAT_VIDEO_FRAMERATE: u32 = 0x20004;
const MEDIA_VIDEO: u32 = 2;
const SUBTYPE_RAW: u32 = 1;
const VIDEO_RGBX: u32 = 7;
const VIDEO_BGRX: u32 = 8;
const VIDEO_RGBA: u32 = 11;
const VIDEO_BGRA: u32 = 12;
const DIRECTION_INPUT: c_int = 0;
const FLAG_AUTOCONNECT: c_int = 1 << 0;
const FLAG_MAP_BUFFERS: c_int = 1 << 2;
const STATE_ERROR: c_int = -1;
const STATE_UNCONNECTED: c_int = 0;
const STATE_STREAMING: c_int = 3;
const CHUNK_CORRUPTED: i32 = 1;
const EVENTS_VERSION: u32 = 2;

/// How long to wait for the first frame of a newly shared window.
const FIRST_FRAME: Duration = Duration::from_secs(10);
/// How long a stream may pause, e.g. to renegotiate after a resize, before
/// its window counts as closed. Closing a shared window unlinks its stream,
/// which pauses it.
const PAUSE: Duration = Duration::from_secs(2);

type Ptr = *mut c_void;

struct Api {
    _library: Library,
    thread_loop_new: unsafe extern "C" fn(*const c_char, *const c_void) -> Ptr,
    thread_loop_get_loop: unsafe extern "C" fn(Ptr) -> Ptr,
    thread_loop_start: unsafe extern "C" fn(Ptr) -> c_int,
    thread_loop_stop: unsafe extern "C" fn(Ptr),
    thread_loop_destroy: unsafe extern "C" fn(Ptr),
    thread_loop_lock: unsafe extern "C" fn(Ptr),
    thread_loop_unlock: unsafe extern "C" fn(Ptr),
    context_new: unsafe extern "C" fn(Ptr, Ptr, usize) -> Ptr,
    context_destroy: unsafe extern "C" fn(Ptr),
    context_connect_fd: unsafe extern "C" fn(Ptr, c_int, Ptr, usize) -> Ptr,
    core_disconnect: unsafe extern "C" fn(Ptr) -> c_int,
    properties_new_string: unsafe extern "C" fn(*const c_char) -> Ptr,
    stream_new: unsafe extern "C" fn(Ptr, *const c_char, Ptr) -> Ptr,
    stream_add_listener: unsafe extern "C" fn(Ptr, Ptr, *const Events, Ptr),
    stream_connect: unsafe extern "C" fn(Ptr, c_int, u32, c_int, *mut *const c_void, u32) -> c_int,
    stream_destroy: unsafe extern "C" fn(Ptr),
    stream_dequeue_buffer: unsafe extern "C" fn(Ptr) -> *mut PwBuffer,
    stream_queue_buffer: unsafe extern "C" fn(Ptr, *mut PwBuffer) -> c_int,
}

fn api() -> Result<&'static Api, CaptureError> {
    static API: OnceLock<Result<Api, String>> = OnceLock::new();
    API.get_or_init(|| unsafe { load() })
        .as_ref()
        .map_err(|e| CaptureError::Unavailable(e.clone()))
}

unsafe fn load() -> Result<Api, String> {
    let library = unsafe { Library::new("libpipewire-0.3.so.0") }
        .map_err(|e| format!("PipeWire is not installed: {e}"))?;
    macro_rules! symbol {
        ($name:literal) => {
            *unsafe { library.get(concat!($name, "\0").as_bytes()) }
                .map_err(|e| format!("libpipewire has no {}: {e}", $name))?
        };
    }
    let init: unsafe extern "C" fn(*mut c_int, *mut *mut *mut c_char) = symbol!("pw_init");
    unsafe { init(ptr::null_mut(), ptr::null_mut()) };
    Ok(Api {
        thread_loop_new: symbol!("pw_thread_loop_new"),
        thread_loop_get_loop: symbol!("pw_thread_loop_get_loop"),
        thread_loop_start: symbol!("pw_thread_loop_start"),
        thread_loop_stop: symbol!("pw_thread_loop_stop"),
        thread_loop_destroy: symbol!("pw_thread_loop_destroy"),
        thread_loop_lock: symbol!("pw_thread_loop_lock"),
        thread_loop_unlock: symbol!("pw_thread_loop_unlock"),
        context_new: symbol!("pw_context_new"),
        context_destroy: symbol!("pw_context_destroy"),
        context_connect_fd: symbol!("pw_context_connect_fd"),
        core_disconnect: symbol!("pw_core_disconnect"),
        properties_new_string: symbol!("pw_properties_new_string"),
        stream_new: symbol!("pw_stream_new"),
        stream_add_listener: symbol!("pw_stream_add_listener"),
        stream_connect: symbol!("pw_stream_connect"),
        stream_destroy: symbol!("pw_stream_destroy"),
        stream_dequeue_buffer: symbol!("pw_stream_dequeue_buffer"),
        stream_queue_buffer: symbol!("pw_stream_queue_buffer"),
        _library: library,
    })
}

/// `struct pw_stream_events`; unused callbacks stay null.
#[repr(C)]
struct Events {
    version: u32,
    destroy: Option<unsafe extern "C" fn()>,
    state_changed: Option<unsafe extern "C" fn(Ptr, c_int, c_int, *const c_char)>,
    control_info: Option<unsafe extern "C" fn()>,
    io_changed: Option<unsafe extern "C" fn()>,
    param_changed: Option<unsafe extern "C" fn(Ptr, u32, *const u8)>,
    add_buffer: Option<unsafe extern "C" fn()>,
    remove_buffer: Option<unsafe extern "C" fn()>,
    process: Option<unsafe extern "C" fn(Ptr)>,
    drained: Option<unsafe extern "C" fn()>,
    command: Option<unsafe extern "C" fn()>,
    trigger_done: Option<unsafe extern "C" fn()>,
}

static EVENTS: Events = Events {
    version: EVENTS_VERSION,
    destroy: None,
    state_changed: Some(on_state_changed),
    control_info: None,
    io_changed: None,
    param_changed: Some(on_param_changed),
    add_buffer: None,
    remove_buffer: None,
    process: Some(on_process),
    drained: None,
    command: None,
    trigger_done: None,
};

/// `struct pw_buffer`, as far as it is read.
#[repr(C)]
struct PwBuffer {
    buffer: *const SpaBuffer,
}

#[repr(C)]
struct SpaBuffer {
    n_metas: u32,
    n_datas: u32,
    metas: *const c_void,
    datas: *const SpaData,
}

#[repr(C)]
struct SpaData {
    kind: u32,
    flags: u32,
    fd: i64,
    mapoffset: u32,
    maxsize: u32,
    data: *const u8,
    chunk: *const SpaChunk,
}

#[repr(C)]
struct SpaChunk {
    offset: u32,
    size: u32,
    stride: i32,
    flags: i32,
}

#[derive(Default)]
struct State {
    /// Negotiated (video format, width, height).
    format: Option<(u32, u32, u32)>,
    frame: Option<RgbaImage>,
    streaming: bool,
    /// Why the stream stopped, once it has.
    ended: Option<String>,
}

/// What the callbacks share with the caller.
struct Shared {
    api: &'static Api,
    stream: Ptr,
    state: Mutex<State>,
    changed: Condvar,
}

pub struct Stream {
    api: &'static Api,
    thread_loop: Ptr,
    context: Ptr,
    core: Ptr,
    shared: Box<Shared>,
    /// `struct spa_hook`, owned by the stream while it lives.
    hook: Box<[usize; 6]>,
    /// Whether a frame has been returned: from then on, a stream that
    /// stops means the window closed.
    seen: bool,
}

// SAFETY: the PipeWire objects are only touched with the thread loop's lock
// held, or after the loop has stopped.
unsafe impl Send for Stream {}

fn failed(e: impl std::fmt::Display) -> CaptureError {
    CaptureError::Failed(format!("PipeWire: {e}"))
}

impl Stream {
    /// Connects to `node` over the remote `fd` the portal opened.
    pub fn connect(fd: OwnedFd, node: u32) -> Result<Stream, CaptureError> {
        let api = api()?;
        unsafe {
            let thread_loop = (api.thread_loop_new)(c"syrup-capture".as_ptr(), ptr::null());
            if thread_loop.is_null() {
                return Err(failed("cannot create a thread loop"));
            }
            let context =
                (api.context_new)((api.thread_loop_get_loop)(thread_loop), ptr::null_mut(), 0);
            if context.is_null() {
                (api.thread_loop_destroy)(thread_loop);
                return Err(failed("cannot create a context"));
            }
            let mut stream = Stream {
                api,
                thread_loop,
                context,
                core: ptr::null_mut(),
                shared: Box::new(Shared {
                    api,
                    stream: ptr::null_mut(),
                    state: Mutex::default(),
                    changed: Condvar::new(),
                }),
                hook: Box::new([0; 6]),
                seen: false,
            };
            if (api.thread_loop_start)(thread_loop) < 0 {
                return Err(failed("cannot start the thread loop"));
            }
            (api.thread_loop_lock)(thread_loop);
            let connected = stream.start(fd, node);
            (api.thread_loop_unlock)(thread_loop);
            connected.map(|()| stream)
        }
    }

    /// With the loop locked.
    unsafe fn start(&mut self, fd: OwnedFd, node: u32) -> Result<(), CaptureError> {
        let api = self.api;
        unsafe {
            self.core =
                (api.context_connect_fd)(self.context, fd.into_raw_fd(), ptr::null_mut(), 0);
            if self.core.is_null() {
                return Err(failed("cannot connect to the portal's PipeWire remote"));
            }
            let properties = (api.properties_new_string)(
                c"media.type=Video media.category=Capture media.role=Screen".as_ptr(),
            );
            let stream = (api.stream_new)(self.core, c"syrup".as_ptr(), properties);
            if stream.is_null() {
                return Err(failed("cannot create a stream"));
            }
            self.shared.stream = stream;
            (api.stream_add_listener)(
                stream,
                self.hook.as_mut_ptr().cast(),
                &EVENTS,
                (&raw const *self.shared).cast_mut().cast(),
            );
            let formats = enum_format();
            let mut params = [formats.as_ptr().cast::<c_void>()];
            let status = (api.stream_connect)(
                stream,
                DIRECTION_INPUT,
                node,
                FLAG_AUTOCONNECT | FLAG_MAP_BUFFERS,
                params.as_mut_ptr(),
                1,
            );
            if status < 0 {
                return Err(failed(format!("cannot connect to node {node} ({status})")));
            }
        }
        Ok(())
    }

    /// The latest frame; waits for the first one.
    pub fn capture(&mut self) -> Result<RgbaImage, CaptureError> {
        let state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        let (state, _) = if self.seen {
            self.shared
                .changed
                .wait_timeout_while(state, PAUSE, |s| !s.streaming && s.ended.is_none())
        } else {
            self.shared
                .changed
                .wait_timeout_while(state, FIRST_FRAME, |s| {
                    s.frame.is_none() && s.ended.is_none()
                })
        }
        .unwrap_or_else(|e| e.into_inner());
        match (&state.ended, &state.frame) {
            _ if self.seen && (state.ended.is_some() || !state.streaming) => {
                Err(CaptureError::Closed)
            }
            (Some(reason), _) => Err(failed(reason)),
            (None, Some(frame)) => {
                self.seen = true;
                Ok(frame.clone())
            }
            (None, None) => Err(failed("no frame arrived from the shared window")),
        }
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        let api = self.api;
        unsafe {
            (api.thread_loop_lock)(self.thread_loop);
            if !self.shared.stream.is_null() {
                (api.stream_destroy)(self.shared.stream);
            }
            if !self.core.is_null() {
                (api.core_disconnect)(self.core);
            }
            (api.thread_loop_unlock)(self.thread_loop);
            (api.thread_loop_stop)(self.thread_loop);
            (api.context_destroy)(self.context);
            (api.thread_loop_destroy)(self.thread_loop);
        }
    }
}

unsafe extern "C" fn on_state_changed(data: Ptr, _old: c_int, state: c_int, error: *const c_char) {
    let shared = unsafe { &*(data as *const Shared) };
    let mut s = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    s.streaming = state == STATE_STREAMING;
    if state == STATE_ERROR || state == STATE_UNCONNECTED {
        let reason = if error.is_null() {
            "the stream disconnected".to_string()
        } else {
            unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned()
        };
        s.ended.get_or_insert(reason);
    }
    shared.changed.notify_all();
}

unsafe extern "C" fn on_param_changed(data: Ptr, id: u32, param: *const u8) {
    if id != PARAM_FORMAT || param.is_null() {
        return;
    }
    let shared = unsafe { &*(data as *const Shared) };
    let size = u32::from_ne_bytes(unsafe { *param.cast::<[u8; 4]>() }) as usize;
    let pod = unsafe { std::slice::from_raw_parts(param, 8 + size) };
    let mut s = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    s.format = parse_format(pod);
}

unsafe extern "C" fn on_process(data: Ptr) {
    let shared = unsafe { &*(data as *const Shared) };
    let api = shared.api;
    // Only the newest buffer matters; older ones go straight back.
    let mut newest: *mut PwBuffer = ptr::null_mut();
    loop {
        let buffer = unsafe { (api.stream_dequeue_buffer)(shared.stream) };
        if buffer.is_null() {
            break;
        }
        if !newest.is_null() {
            unsafe { (api.stream_queue_buffer)(shared.stream, newest) };
        }
        newest = buffer;
    }
    if newest.is_null() {
        return;
    }
    let mut s = shared.state.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(format) = s.format
        && let Some(frame) = unsafe { read(newest, format) }
    {
        s.frame = Some(frame);
        shared.changed.notify_all();
    }
    drop(s);
    unsafe { (api.stream_queue_buffer)(shared.stream, newest) };
}

/// The buffer's first plane as RGBA, unless it holds no new picture.
unsafe fn read(
    buffer: *mut PwBuffer,
    (format, width, height): (u32, u32, u32),
) -> Option<RgbaImage> {
    let buffer = unsafe { &*(*buffer).buffer };
    if buffer.n_datas == 0 {
        return None;
    }
    let data = unsafe { &*buffer.datas };
    if data.data.is_null() || data.chunk.is_null() {
        return None;
    }
    let chunk = unsafe { &*data.chunk };
    if chunk.size == 0 || chunk.flags & CHUNK_CORRUPTED != 0 {
        return None;
    }
    let (width, height) = (width as usize, height as usize);
    let stride = if chunk.stride > 0 {
        chunk.stride as usize
    } else {
        width * 4
    };
    let offset = (chunk.offset % data.maxsize.max(1)) as usize;
    if height == 0 || offset + stride * (height - 1) + width * 4 > data.maxsize as usize {
        return None;
    }
    let bytes = unsafe {
        std::slice::from_raw_parts(data.data.add(offset), data.maxsize as usize - offset)
    };
    let swap = match format {
        VIDEO_BGRX | VIDEO_BGRA => true,
        VIDEO_RGBX | VIDEO_RGBA => false,
        _ => return None,
    };
    let mut rgba = Vec::with_capacity(width * height * 4);
    for row in bytes.chunks(stride).take(height) {
        for &[first, green, third, _] in row[..width * 4].as_chunks::<4>().0 {
            let (red, blue) = if swap { (third, first) } else { (first, third) };
            rgba.extend_from_slice(&[red, green, blue, 255]);
        }
    }
    RgbaImage::from_raw(width as u32, height as u32, rgba)
}

/// Appends a pod: its size and type, the body, and padding to 8 bytes.
fn pod(out: &mut Vec<u8>, kind: u32, body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_ne_bytes());
    out.extend_from_slice(&kind.to_ne_bytes());
    out.extend_from_slice(body);
    out.resize(out.len().next_multiple_of(8), 0);
}

/// Appends an object property holding a choice of `values`, each `size`
/// bytes of `kind`; the first value is the default.
fn choice(out: &mut Vec<u8>, key: u32, choice: u32, kind: u32, size: u32, values: &[u32]) {
    out.extend_from_slice(&key.to_ne_bytes());
    out.extend_from_slice(&0u32.to_ne_bytes());
    let mut body = Vec::new();
    for word in [choice, 0, size, kind] {
        body.extend_from_slice(&word.to_ne_bytes());
    }
    for value in values {
        body.extend_from_slice(&value.to_ne_bytes());
    }
    pod(out, TYPE_CHOICE, &body);
}

/// Appends an object property holding one id.
fn id(out: &mut Vec<u8>, key: u32, value: u32) {
    out.extend_from_slice(&key.to_ne_bytes());
    out.extend_from_slice(&0u32.to_ne_bytes());
    pod(out, TYPE_ID, &value.to_ne_bytes());
}

/// The formats accepted: raw video in 32-bit RGB orders, any size and rate.
/// No DMA-BUF modifiers are offered, so frames arrive in shared memory.
fn enum_format() -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&OBJECT_FORMAT.to_ne_bytes());
    body.extend_from_slice(&PARAM_ENUM_FORMAT.to_ne_bytes());
    id(&mut body, FORMAT_MEDIA_TYPE, MEDIA_VIDEO);
    id(&mut body, FORMAT_MEDIA_SUBTYPE, SUBTYPE_RAW);
    choice(
        &mut body,
        FORMAT_VIDEO_FORMAT,
        CHOICE_ENUM,
        TYPE_ID,
        4,
        &[VIDEO_BGRX, VIDEO_BGRX, VIDEO_BGRA, VIDEO_RGBX, VIDEO_RGBA],
    );
    choice(
        &mut body,
        FORMAT_VIDEO_SIZE,
        CHOICE_RANGE,
        TYPE_RECTANGLE,
        8,
        &[1920, 1080, 1, 1, 16384, 16384],
    );
    choice(
        &mut body,
        FORMAT_VIDEO_FRAMERATE,
        CHOICE_RANGE,
        TYPE_FRACTION,
        8,
        &[30, 1, 0, 1, 1000, 1],
    );
    let mut object = Vec::new();
    pod(&mut object, TYPE_OBJECT, &body);
    object
}

/// (video format, width, height) from a negotiated Format object.
fn parse_format(pod: &[u8]) -> Option<(u32, u32, u32)> {
    let word = |at: usize| Some(u32::from_ne_bytes(pod.get(at..at + 4)?.try_into().ok()?));
    if word(4)? != TYPE_OBJECT || word(8)? != OBJECT_FORMAT {
        return None;
    }
    let (mut format, mut size) = (None, None);
    let mut at = 16;
    while at + 16 <= pod.len() {
        let (key, value_size, kind) = (word(at)?, word(at + 8)? as usize, word(at + 12)?);
        // A choice's first value follows its 16-byte header, after which
        // both read the same way.
        let (value, kind) = if kind == TYPE_CHOICE {
            (at + 32, word(at + 28)?)
        } else {
            (at + 16, kind)
        };
        match (key, kind) {
            (FORMAT_VIDEO_FORMAT, TYPE_ID) => format = word(value),
            (FORMAT_VIDEO_SIZE, TYPE_RECTANGLE) => size = Some((word(value)?, word(value + 4)?)),
            _ => {}
        }
        at += 16 + value_size.next_multiple_of(8);
    }
    let (width, height) = size?;
    Some((format?, width, height))
}
