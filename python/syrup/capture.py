"""Live frames from a window on screen, for sessions.

    session = syrup.ops.track_moving_regions.session()
    for frame in syrup.capture.window("Notepad"):
        print(session(frame))

Windows, macOS 14+ and Linux (X11 and Wayland) are supported. On macOS the
process needs the Screen Recording permission. On a Wayland desktop the
desktop asks which window to share the first time a title is used, and
remembers the choice.
"""

from . import Image, _call, _native


def windows():
    """Titles of the windows that can be captured without asking the user;
    on Wayland, only X11 applications' windows are listed."""
    return _call(_native.list_windows)


def window(title, *, frames=None):
    """Frames of the first window whose title contains `title` (ignoring
    case), as syrup.Image, until it closes or `frames` have been taken.
    Raises InputError if no window matches, DependencyError where capture
    is unavailable, and InputError with kind "permission_denied" when it is
    refused."""
    source = _call(_native.Window, title)
    taken = 0
    while frames is None or taken < frames:
        frame = _call(source.capture)
        if frame is None:
            return
        yield Image(*frame)
        taken += 1
