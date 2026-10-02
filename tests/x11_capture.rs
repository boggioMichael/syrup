//! X11 capture on windows this test creates: a red window half covered by
//! a blue one must still be captured red. Needs an X server (CI runs Xvfb);
//! without one the test passes vacuously unless SYRUP_EXPECT_WINDOW is set.

#![cfg(all(unix, not(target_os = "macos")))]

use syrup::capture::{CaptureError, Window};
use x11rb::connection::Connection;
use x11rb::protocol::xproto::{
    AtomEnum, ConnectionExt, CreateWindowAux, PropMode, Window as WindowId, WindowClass,
};
use x11rb::rust_connection::RustConnection;
use x11rb::wrapper::ConnectionExt as _;

const RED: u32 = 0xd0_20_20;
const BLUE: u32 = 0x20_20_d0;

fn open(conn: &RustConnection, title: &str, x: i16, y: i16, colour: u32) -> WindowId {
    let root = conn.setup().roots[0].root;
    let id = conn.generate_id().unwrap();
    conn.create_window(
        x11rb::COPY_DEPTH_FROM_PARENT,
        id,
        root,
        x,
        y,
        320,
        200,
        0,
        WindowClass::INPUT_OUTPUT,
        0,
        &CreateWindowAux::new().background_pixel(colour),
    )
    .unwrap();
    conn.change_property8(
        PropMode::REPLACE,
        id,
        AtomEnum::WM_NAME,
        AtomEnum::STRING,
        title.as_bytes(),
    )
    .unwrap();
    conn.map_window(id).unwrap();
    id
}

#[test]
fn covered_windows_are_captured_as_drawn() {
    let Ok((conn, _)) = x11rb::connect(None) else {
        assert!(
            std::env::var_os("SYRUP_EXPECT_WINDOW").is_none(),
            "SYRUP_EXPECT_WINDOW is set but there is no X server"
        );
        return;
    };
    let title = format!("Syrup X11 test {}", std::process::id());
    let red = open(&conn, &title, 10, 10, RED);
    open(&conn, "Syrup X11 cover", 60, 40, BLUE);
    conn.sync().unwrap();

    // A window manager maps windows when it gets to them.
    let listed = (0..50).any(|_| {
        std::thread::sleep(std::time::Duration::from_millis(100));
        syrup::capture::window_titles().unwrap().contains(&title)
    });
    assert!(listed, "{title} never appeared");
    let mut window = Window::find(&title.to_uppercase()).unwrap();
    assert_eq!(window.title(), title);
    let frame = window.capture().unwrap();
    assert_eq!(frame.dimensions(), (320, 200));
    // (200, 150) lies under the blue window.
    for (x, y) in [(5, 5), (200, 150)] {
        assert_eq!(
            frame.get_pixel(x, y).0,
            [0xd0, 0x20, 0x20, 255],
            "at ({x}, {y})"
        );
    }

    conn.destroy_window(red).unwrap();
    conn.sync().unwrap();
    assert_eq!(window.capture().unwrap_err(), CaptureError::Closed);
    assert_eq!(
        Window::find("no window is called this 7f3a").err(),
        Some(CaptureError::NotFound)
    );
}
