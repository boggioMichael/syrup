"""Computer-vision operations you name instead of write.

    from syrup.ops import find_face

    faces = find_face("photo.jpg")
    for face in faces:
        print(face.box, face.confidence)

The name is parsed into an intent, compiled to a native module the first
time it runs, and reused afterwards. See crates/syrup-runtime/docs/contract.md
for the grammar, the result contract and every way an operation can fail.

`syrup.find_face` and other names on the package itself are the core
library's in-process intents (`syrup.legacy`), kept for existing code; they
need only the core library (`cargo build --release`), not this package's
compiled extension.
"""

import json
import operator
import os

from .errors import (
    BuildError,
    DependencyError,
    ExecutionError,
    InputError,
    IntentError,
    LoadError,
    PlanError,
    SyrupError,
    ValidationError,
    from_native,
)
from .results import Box, FindResult, Found, Provenance, Track, from_json


class _NativeMissing:
    """Stands in for the compiled extension in a checkout where it was not
    built: the in-process intents still work, and every use of the compiled
    operations says how to get them."""

    class NativeError(Exception):
        pass

    def __init__(self, error):
        self._error = error

    def __getattr__(self, name):
        def missing(*args, **kwargs):
            raise DependencyError(
                "load",
                "missing_dependency",
                "syrup's compiled extension (syrup._native) is not installed",
                hint="pip install ./python, or `maturin develop` in python/",
                details={"import_error": str(self._error)},
            )

        return missing


try:
    from . import _native
except ImportError as _error:  # a checkout where only the core library was built
    _native = _NativeMissing(_error)

from .legacy import Detection, Match, library_path, register_template, version  # noqa: E402

__all__ = [
    "Box",
    "BuildError",
    "DependencyError",
    "Detection",
    "ExecutionError",
    "FindResult",
    "Found",
    "Image",
    "InputError",
    "IntentError",
    "LoadError",
    "Match",
    "Operation",
    "PlanError",
    "Provenance",
    "Session",
    "SyrupError",
    "Track",
    "ValidationError",
    "add_target",
    "bundle",
    "cache_dir",
    "capture",
    "define",
    "legacy",
    "library_path",
    "ops",
    "register_template",
    "resolve",
    "version",
]


def _call(fn, *args):
    try:
        return fn(*args)
    except _native.NativeError as e:
        raise from_native(e.args[0]) from None


class Image:
    """8-bit pixels decoded by Syrup, row-major, `channels` bytes per pixel."""

    def __init__(self, data, width, height, channels):
        self.data, self.width, self.height, self.channels = data, width, height, channels

    @classmethod
    def open(cls, path):
        return cls(*_call(_native.decode_image, os.fspath(path)))

    def __repr__(self):
        return f"<syrup.Image {self.width}x{self.height}x{self.channels}>"


def _bad_image(reason, hint=None):
    return InputError("input", "bad_image", reason, hint=hint)


def _pixels(image):
    if isinstance(image, Image):
        return image.data, image.width, image.height, image.channels
    if isinstance(image, (str, os.PathLike)):
        return _call(_native.decode_image, os.fspath(image))
    if hasattr(image, "mode") and hasattr(image, "tobytes"):
        if image.mode == "P":
            image = image.convert("RGB")
        channels = {"L": 1, "RGB": 3, "RGBA": 4}.get(image.mode)
        if channels is None:
            raise _bad_image(f"PIL mode {image.mode!r} is not 8-bit L, RGB or RGBA", "convert it, e.g. image.convert('RGB')")
        return image.tobytes(), image.size[0], image.size[1], channels
    if hasattr(image, "__array_interface__"):
        import numpy as np

        array = np.asarray(image)
        if array.dtype != np.uint8:
            raise _bad_image(f"arrays must be uint8, got {array.dtype}", "scale to 0-255 and use .astype(np.uint8)")
        if array.ndim == 2:
            array = array[:, :, None]
        if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
            raise _bad_image(f"arrays must be (H, W), (H, W, 1), (H, W, 3) or (H, W, 4), got {array.shape}")
        height, width, channels = array.shape
        return array.tobytes(), width, height, channels
    raise _bad_image(
        f"cannot read an image from {type(image).__name__}",
        "pass a path, a NumPy uint8 array, a PIL image or a syrup.Image",
    )


def _ints(values, count, message, operation):
    try:
        values = tuple(operator.index(v) for v in values)
    except TypeError:
        values = ()
    if len(values) != count or min(values) < 0:
        raise InputError("input", "bad_parameter", message, operation)
    return values


class Operation:
    """A resolved operation. Call it with an image to run it."""

    def __init__(self, native):
        self._native = native

    @property
    def name(self):
        return self._native.name

    @property
    def intent(self):
        return self._native.intent

    @property
    def plan_hash(self):
        return self._native.plan_hash

    @property
    def source(self):
        """The Rust source Syrup generates for this operation."""
        return self._native.source()

    def explain(self):
        return self._native.explain()

    def prepare(self):
        """Compile (or load) the native module now instead of on first call."""
        return json.loads(_call(self._native.prepare))

    def session(self, *, max_distance=None, grace_frames=None):
        """Follow a track_* operation's items across frames: call the session
        with each frame in order, and each result carries `track.id`."""
        return Session(self, _call(self._native.session, max_distance, grace_frames))

    def __call__(self, image, *, min_confidence=None, max_results=None, region=None):
        return from_json(_call(self._native.run, *_arguments(self.name, image, min_confidence, max_results, region)))

    def __repr__(self):
        return f"<syrup.Operation {self.name}: {self.intent}>"


class Session:
    """Frames in, tracked results out. See Operation.session."""

    def __init__(self, operation, native):
        self.operation, self._native = operation, native

    def __call__(self, frame, *, min_confidence=None, max_results=None, region=None):
        arguments = _arguments(self.operation.name, frame, min_confidence, max_results, region)
        return from_json(_call(self._native.update, *arguments))

    def __repr__(self):
        return f"<syrup.Session {self.operation.name}>"


def _arguments(name, image, min_confidence, max_results, region):
    pixels, width, height, channels = _pixels(image)
    if region is not None:
        region = _ints(region, 4, "region must be four non-negative ints (x, y, w, h)", name)
    if max_results is not None:
        (max_results,) = _ints([max_results], 1, "max_results must be a non-negative int", name)
    return pixels, width, height, channels, min_confidence, max_results, region


def resolve(name):
    """The operation a name means, or IntentError saying why there is none."""
    return Operation(_call(_native.resolve, name))


def define(
    name, *, find, color=None, region=None, order=None, limit=None, min_area_pct=None, max_area_pct=None, measure=None
):
    """Give a name outside the grammar an explicit meaning.

    `color` is required for regions and bars ("red", "blue", ...). `region`
    is a region name such as "top_half", "region" for a per-call region, or
    fractions (x, y, w, h) of the image. `order` is one of "confidence",
    "size", "area_asc", "left_to_right", "right_to_left", "top_to_bottom" or
    "bottom_to_top". `measure` ("sharpness" or "fill") makes it a
    measurement: each result carries the quantity as `value`.
    """
    spec = {
        "find": find,
        "color": color,
        "region": list(region) if isinstance(region, (tuple, list)) else region,
        "order": order,
        "limit": limit,
        "min_area_pct": min_area_pct,
        "max_area_pct": max_area_pct,
        "measure": measure,
    }
    spec = {k: v for k, v in spec.items() if v is not None}
    return Operation(_call(_native.define, name, json.dumps(spec)))


def add_target(noun, detect, *, plural=None, min_confidence=0.5, provider="python", model_sha256=None):
    """Find `noun` with your own detector, e.g. a model from any Python library.

    `detect(image)` gets the searched pixels as a uint8 NumPy array (H, W, C)
    and returns boxes in those pixels as (x, y, w, h, score) or
    (x, y, w, h, score, text), with score in [0, 1]. Names can then use the
    noun like any other: syrup.ops.find_largest_<noun>_in_center. Regions,
    ordering and limits run in the compiled module around the detector.
    `provider` and `model_sha256` are what provenance reports for it.
    """
    import numpy as np

    def call(pixels, width, height, channels):
        image = np.frombuffer(pixels, np.uint8).reshape(height, width, channels)
        return [_detection(box) for box in detect(image)]

    _call(_native.add_target, noun, plural or noun + "s", float(min_confidence), call, provider, model_sha256)


def _detection(box):
    if len(box) not in (5, 6):
        raise ValueError(f"a detection is (x, y, w, h, score) or (x, y, w, h, score, text), got {box!r}")
    x, y, w, h, score = (float(v) for v in box[:5])
    return x, y, w, h, score, str(box[5]) if len(box) == 6 else None


def bundle(path, *names):
    """Compile `names` and copy them into the bundle directory `path`.

    A machine without a Rust compiler runs them with SYRUP_MODE=frozen and
    SYRUP_CACHE_DIR=path. Bundles are per platform; adding to an existing
    bundle keeps what it holds.
    """
    return json.loads(_call(_native.bundle, os.fspath(path), list(names)))


def cache_dir():
    return _call(_native.cache_dir)


from . import capture, legacy, ops  # noqa: E402  (they need the names above)

_LEGACY_VERBS = ("find_", "track_", "count_", "measure_", "read_")


def __getattr__(name):
    """`syrup.find_face` and the like: the core library's in-process intents
    (syrup.legacy). The compiled operations are `syrup.ops.<name>`."""
    if name.startswith(_LEGACY_VERBS):
        return getattr(legacy, name)
    raise AttributeError(f"module 'syrup' has no attribute {name!r}")
