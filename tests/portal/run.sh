#!/bin/sh
# Runs tests/wayland_capture.rs as on a Wayland desktop: a private D-Bus
# session and PipeWire server, a GStreamer source as the shared window, and
# screencast.py answering for the desktop's portal.
# Needs dbus, pipewire, wireplumber, gstreamer1.0-pipewire, python3-dbus
# and python3-gi.
set -eu

root=$(cd "$(dirname "$0")/../.." && pwd)
dir=$(mktemp -d)
pids=""
trap 'kill $pids 2>/dev/null || true; rm -rf "$dir"' EXIT

export XDG_RUNTIME_DIR="$dir/run" XDG_STATE_HOME="$dir/state" WAYLAND_DISPLAY=wayland-test
unset DISPLAY
mkdir -m 700 "$XDG_RUNTIME_DIR"

eval "$(dbus-launch --sh-syntax)"
pids="$DBUS_SESSION_BUS_PID"

wait_for() {
    for _ in $(seq 50); do
        if eval "$1" >/dev/null 2>&1; then return 0; fi
        sleep 0.1
    done
    echo "timed out waiting for: $1" >&2
    exit 1
}

pipewire & pids="$pids $!"
wait_for 'test -S "$XDG_RUNTIME_DIR/pipewire-0"'
wireplumber & pids="$pids $!"

gst-launch-1.0 -q videotestsrc pattern=solid-color foreground-color=0xff20c040 is-live=true \
    ! video/x-raw,format=BGRx,width=320,height=200,framerate=30/1 \
    ! pipewiresink mode=provide stream-properties="p,media.class=Video/Source,node.name=syrup-shared-window" &
source=$!
pids="$pids $source"
node() {
    pw-cli ls Node | grep -B4 'node.name = "syrup-shared-window"' | sed -n 's/^\s*id \([0-9]*\),.*/\1/p'
}
wait_for 'test -n "$(node)"'

python3 "$root/tests/portal/screencast.py" "$(node)" "$dir/portal.log" & pids="$pids $!"
wait_for 'dbus-send --session --print-reply --dest=org.freedesktop.portal.Desktop /org/freedesktop/portal/desktop org.freedesktop.DBus.Peer.Ping'

cd "$root"
SYRUP_TEST_PORTAL="$dir/portal.log" SYRUP_TEST_PORTAL_SOURCE="$source" \
    cargo test -p syrup --test wayland_capture -- --nocapture
