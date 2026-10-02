"""The core library's in-process intents, through its C API.

    import syrup
    faces = syrup.find_face(image)          # image: numpy HxWx3/HxWx4 uint8, or PIL.Image
    for face in faces.value:
        print(face.bounds, face.score)
    print(syrup.measure_red_bar(image).value)

These are the intents of the core's `syrup::intent` module (cascade faces,
eyes and profile faces, icons, bars, blobs, text, motion), run in-process by
the native library and reached as `syrup.<name>` or `syrup.legacy.<name>`.
`syrup.ops.<name>` is the compiled-operations engine (syrup-runtime); the two
are separate vocabularies with separate result types.

Every attribute that looks like an intent (``verb_qualifiers_noun``) is
resolved through the native library's C API (``src/capi.rs``) the first time
it is used and cached. Names outside the vocabulary raise ``IntentError`` with
the vocabulary in the message; results carry ``confidence``, ``reliability``
and a ``failure_reason`` when the search could not run.

The native library is found through ``SYRUP_LIBRARY``, next to this package, or
in a ``target/release`` / ``target/debug`` directory above it (``cargo build
--release`` in the repository produces it).
"""

from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .errors import BuildError, IntentError

__all__ = ["Detection", "Match", "IntentError", "Intent", "register_template", "library_path", "version"]

_c_char_p = ctypes.POINTER(ctypes.c_char)


class _FrameView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("stride", ctypes.c_uint32),
    ]


class _RegionView(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint32), ("y", ctypes.c_uint32), ("w", ctypes.c_uint32), ("h", ctypes.c_uint32)]


class _MatchView(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint32),
        ("y", ctypes.c_uint32),
        ("w", ctypes.c_uint32),
        ("h", ctypes.c_uint32),
        ("score", ctypes.c_float),
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
        ("id", ctypes.c_uint64),
    ]


class _ResultView(ctypes.Structure):
    _fields_ = [
        ("status", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("confidence", ctypes.c_float),
        ("reliability", ctypes.c_uint32),
        ("matches", ctypes.POINTER(_MatchView)),
        ("match_count", ctypes.c_size_t),
        ("scalar", ctypes.c_float),
        ("text", _c_char_p),
        ("reason", _c_char_p),
    ]


_STATUS_FOUND = 0
_KIND_MATCHES, _KIND_SCALAR, _KIND_TEXT = 0, 1, 2
_RELIABILITY = {0: "corroborated", 1: "heuristic", 2: "predicted", 3: "unreliable"}


@dataclass(frozen=True)
class Match:
    bounds: tuple  # (x, y, w, h)
    score: float
    centre: tuple  # (cx, cy)
    id: Optional[int] = None


@dataclass(frozen=True)
class Detection:
    value: Any  # list[Match], float, str, or None
    confidence: float
    reliability: str
    failure_reason: Optional[str]

    def __bool__(self) -> bool:
        return self.value is not None


def _candidate_paths():
    names = {"linux": "libsyrup.so", "darwin": "libsyrup.dylib", "win32": "syrup.dll"}
    name = names.get(sys.platform, "libsyrup.so")
    if os.environ.get("SYRUP_LIBRARY"):
        yield Path(os.environ["SYRUP_LIBRARY"])
    here = Path(__file__).resolve().parent
    yield here / name
    for ancestor in [here, *here.parents][:4]:
        yield ancestor / "target" / "release" / name
        yield ancestor / "target" / "debug" / name


_lib = None
_path = None


def _library() -> ctypes.CDLL:
    global _lib, _path
    if _lib is not None:
        return _lib
    for path in _candidate_paths():
        if path.is_file():
            lib = ctypes.CDLL(str(path))
            lib.syrup_version.restype = ctypes.c_char_p
            lib.syrup_resolve.argtypes = [ctypes.c_char_p, ctypes.POINTER(_c_char_p)]
            lib.syrup_resolve.restype = ctypes.c_void_p
            lib.syrup_run.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_FrameView),
                ctypes.POINTER(_RegionView),
                ctypes.POINTER(_ResultView),
            ]
            lib.syrup_run.restype = ctypes.c_int32
            lib.syrup_source.argtypes = [ctypes.c_void_p]
            lib.syrup_source.restype = _c_char_p
            lib.syrup_compile.argtypes = [ctypes.c_void_p, ctypes.POINTER(_c_char_p)]
            lib.syrup_compile.restype = _c_char_p
            lib.syrup_register_template.argtypes = [ctypes.c_char_p, ctypes.POINTER(_FrameView)]
            lib.syrup_register_template.restype = ctypes.c_int32
            lib.syrup_result_free.argtypes = [ctypes.POINTER(_ResultView)]
            lib.syrup_release.argtypes = [ctypes.c_void_p]
            lib.syrup_string_free.argtypes = [_c_char_p]
            _lib, _path = lib, path
            return lib
    raise OSError(
        "the syrup native library was not found; build it with `cargo build --release` "
        "or point SYRUP_LIBRARY at it"
    )


def library_path() -> Optional[Path]:
    """Where the native library was loaded from."""
    _library()
    return _path


def version() -> str:
    return _library().syrup_version().decode()


def _take_string(pointer) -> Optional[str]:
    """Copy a library-owned string and free it."""
    if not pointer:
        return None
    text = ctypes.cast(pointer, ctypes.c_char_p).value.decode("utf-8", "replace")
    _library().syrup_string_free(pointer)
    return text


def _as_rgba(image) -> tuple:
    """(bytes buffer, width, height) of an image as tightly packed RGBA8."""
    try:
        import numpy as np  # noqa: F401
    except ImportError:  # pragma: no cover
        np = None
    if hasattr(image, "convert") and hasattr(image, "tobytes"):  # PIL
        rgba = image.convert("RGBA")
        return rgba.tobytes(), rgba.width, rgba.height
    if np is not None and isinstance(image, np.ndarray):
        array = np.ascontiguousarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] not in (3, 4):
            raise TypeError("expected an HxWx3 or HxWx4 uint8 array")
        if array.shape[2] == 3:
            alpha = np.full(array.shape[:2] + (1,), 255, dtype=np.uint8)
            array = np.ascontiguousarray(np.concatenate([array, alpha], axis=2))
        height, width = array.shape[:2]
        return array.tobytes(), width, height
    raise TypeError("expected a PIL image or a numpy HxWx3/HxWx4 uint8 array")


def _frame(image):
    data, width, height = _as_rgba(image)
    buffer = ctypes.create_string_buffer(data, len(data))
    view = _FrameView(
        ctypes.cast(buffer, ctypes.POINTER(ctypes.c_uint8)),
        width,
        height,
        width * 4,
    )
    return view, buffer  # keep the buffer alive alongside the view


class Intent:
    """A resolved intent; call it with an image (and optionally a region)."""

    def __init__(self, name: str):
        lib = _library()
        error = _c_char_p()
        handle = lib.syrup_resolve(name.encode(), ctypes.byref(error))
        if not handle:
            raise IntentError("resolve", "unsupported", _take_string(error) or f"`{name}` could not be resolved", name)
        self.name = name
        self._handle = handle

    def __repr__(self) -> str:
        return f"<syrup intent {self.name}>"

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle and _lib is not None:
            _lib.syrup_release(handle)
            self._handle = None

    def __call__(self, image, region: Optional[tuple] = None) -> Detection:
        lib = _library()
        view, _keep = _frame(image)
        region_view = _RegionView(*region) if region else None
        out = _ResultView()
        code = lib.syrup_run(
            self._handle,
            ctypes.byref(view),
            ctypes.byref(region_view) if region_view else None,
            ctypes.byref(out),
        )
        if code != 0:
            raise ValueError(f"`{self.name}` rejected the frame (code {code})")
        try:
            reason = ctypes.cast(out.reason, ctypes.c_char_p).value.decode("utf-8", "replace") if out.reason else None
            value: Any = None
            if out.status == _STATUS_FOUND:
                if out.kind == _KIND_MATCHES:
                    value = [
                        Match(
                            (m.x, m.y, m.w, m.h),
                            m.score,
                            (m.cx, m.cy),
                            m.id or None,
                        )
                        for m in (out.matches[i] for i in range(out.match_count))
                    ]
                elif out.kind == _KIND_SCALAR:
                    value = out.scalar
                else:
                    value = ctypes.cast(out.text, ctypes.c_char_p).value.decode("utf-8", "replace") if out.text else ""
            return Detection(value, out.confidence, _RELIABILITY.get(out.reliability, "unreliable"), reason)
        finally:
            lib.syrup_result_free(ctypes.byref(out))

    @property
    def source(self) -> str:
        """The Rust source the library generated for this intent."""
        return _take_string(_library().syrup_source(self._handle)) or ""

    def compile(self) -> str:
        """Build the intent as its own shared library (needs cargo) and use it; returns the path."""
        error = _c_char_p()
        path = _library().syrup_compile(self._handle, ctypes.byref(error))
        if not path:
            raise BuildError("compile", "compiler_failed", _take_string(error) or "compilation failed", self.name)
        return _take_string(path) or ""


def register_template(name: str, image) -> None:
    """Make `image` the picture behind ``find_<name>_icon``."""
    view, _keep = _frame(image)
    if _library().syrup_register_template(name.encode(), ctypes.byref(view)) != 0:
        raise ValueError("could not register the template")


_VERBS = ("find_", "track_", "count_", "measure_", "read_")
_intents: dict = {}


def __getattr__(name: str):
    if name.startswith(_VERBS):
        intent = _intents.get(name)
        if intent is None:
            intent = _intents[name] = Intent(name)
        return intent
    raise AttributeError(f"module 'syrup.legacy' has no attribute {name!r}")
