//! Wayland capture through the screen-cast portal and PipeWire. Run by
//! tests/portal/run.sh, which starts PipeWire, a video source standing in
//! for the shared window, and tests/portal/screencast.py standing in for
//! the desktop's portal; without it the test has nothing to talk to.

#![cfg(all(unix, not(target_os = "macos")))]

use syrup::capture::{CaptureError, Window};

#[test]
fn windows_are_shared_through_the_portal() {
    let Some(log) = std::env::var_os("SYRUP_TEST_PORTAL") else {
        return;
    };
    let source: u32 = std::env::var("SYRUP_TEST_PORTAL_SOURCE")
        .unwrap()
        .parse()
        .unwrap();

    for _ in 0..2 {
        let mut window = Window::find("Some App").unwrap();
        assert_eq!(window.title(), "Some App");
        for _ in 0..3 {
            let frame = window.capture().unwrap();
            assert_eq!(frame.dimensions(), (320, 200));
            // The source paints 0x20c040.
            assert_eq!(frame.get_pixel(160, 100).0, [0x20, 0xc0, 0x40, 255]);
        }
    }

    // The first run is asked to remember the window; the second restores it.
    let log = std::fs::read_to_string(log).unwrap();
    let calls: Vec<&str> = log.lines().collect();
    assert_eq!(calls.len(), 2);
    assert!(calls[0].contains(r#""types": "2""#) && calls[0].contains(r#""persist_mode": "2""#));
    assert!(!calls[0].contains("restore_token"));
    assert!(calls[1].contains(r#""restore_token": "token-1""#));

    let mut window = Window::find("Some App").unwrap();
    window.capture().unwrap();
    std::process::Command::new("kill")
        .arg(source.to_string())
        .status()
        .unwrap();
    // The stream notices shortly after; until then the last frame stands.
    let closed = (0..50).any(|_| {
        std::thread::sleep(std::time::Duration::from_millis(100));
        matches!(window.capture(), Err(CaptureError::Closed))
    });
    assert!(closed, "the window never closed");
}
