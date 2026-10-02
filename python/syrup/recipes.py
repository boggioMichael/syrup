"""Ready-made targets from popular detection models.

    import syrup.recipes
    syrup.recipes.yolo()                       # person/people, cup/cups, ...
    cups = syrup.ops.find_cups_left_to_right("table.jpg")

Each recipe registers nouns with syrup.add_target, so names compose with
regions, selectors, measurements and sessions like built-in targets. Model
files are downloaded once into Syrup's cache and checked against a pinned
SHA-256. The libraries themselves are optional: install ultralytics for
yolo(), mediapipe for the mediapipe_* recipes (on Linux mediapipe also
needs libEGL, e.g. apt install libegl1).

Registering a noun again replaces its detector, so when two recipes know
the same class, the last one registered finds it.
"""

import hashlib
import os
import urllib.request

import numpy as np

from . import add_target, cache_dir
from .errors import DependencyError, IntentError

YOLO_MODELS = {
    "yolo11n.pt": (
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    ),
}

MEDIAPIPE_MODELS = {
    "efficientdet_lite0.tflite": (
        "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
        "4b59100025bea1235a84c1038879a6cccc9f6c49f5e41144e91e74d99e780993",
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
}

# Plurals the suffix rules below get wrong.
_PLURALS = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis", "scissors": "scissors"}


def noun(label):
    """A model's class label as a Syrup noun and its plural: "traffic light"
    becomes ("traffic_light", "traffic_lights")."""
    singular = "_".join(label.lower().replace("-", " ").split())
    *head, last = singular.split("_")
    if last in _PLURALS:
        plural = _PLURALS[last]
    elif last.endswith(("s", "sh", "ch", "x", "z")):
        plural = last + "es"
    elif last.endswith("y") and last[-2:-1] not in "aeiou":
        plural = last[:-1] + "ies"
    else:
        plural = last + "s"
    return singular, "_".join([*head, plural])


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_file(name, catalog):
    """The path of a pinned model, downloading it into the cache once."""
    url, sha256 = catalog[name]
    path = os.path.join(cache_dir(), "models", name)
    if not os.path.exists(path) or _sha256(path) != sha256:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        partial = path + ".partial"
        urllib.request.urlretrieve(url, partial)
        if _sha256(partial) != sha256:
            os.remove(partial)
            raise DependencyError("execute", "integrity", f"{url} did not have sha256 {sha256}")
        os.replace(partial, path)
    return path


def _rgb(image):
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return np.ascontiguousarray(image[:, :, :3])


def _register(labels, detector_for, *, classes, rename, min_confidence, provider, sha256):
    """Adds a noun per label; returns {noun: label} and the labels skipped
    because their name already means something to Syrup."""
    added, skipped = {}, {}
    for label in labels:
        if classes is not None and label not in classes:
            continue
        singular, plural = noun(rename.get(label, label))
        try:
            add_target(singular, detector_for(label), plural=plural, min_confidence=min_confidence, provider=provider, model_sha256=sha256)
        except IntentError as e:
            skipped[label] = e.reason
            continue
        added[singular] = label
    return added, skipped


def yolo(model="yolo11n.pt", *, classes=None, rename=None, min_confidence=0.25):
    """Every class an Ultralytics YOLO model detects (COCO's 80 for the
    default yolo11n.pt), or only `classes`. `model` is a name from
    YOLO_MODELS or a path to your own weights. Returns (added, skipped):
    {noun: class} and {class: why}; COCO's "orange" is skipped because it is
    a colour unless renamed, e.g. rename={"orange": "citrus"}."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise DependencyError("execute", "missing_dependency", f"yolo() needs ultralytics: {e}", hint="pip install ultralytics") from None
    path = model_file(model, YOLO_MODELS) if model in YOLO_MODELS else model
    net = YOLO(path)
    ids = {name: i for i, name in net.names.items()}

    def detector_for(label):
        def detect(image):
            # Ultralytics reads NumPy arrays as BGR.
            result = net.predict(_rgb(image)[:, :, ::-1], verbose=False, conf=min_confidence, classes=[ids[label]])[0]
            return [
                (float(x - w / 2), float(y - h / 2), float(w), float(h), float(score))
                for (x, y, w, h), score in zip(result.boxes.xywh.tolist(), result.boxes.conf.tolist())
            ]

        return detect

    return _register(
        ids,
        detector_for,
        classes=classes,
        rename=rename or {},
        min_confidence=min_confidence,
        provider=f"ultralytics {os.path.basename(path)}",
        sha256=_sha256(path),
    )


def _mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as e:
        raise DependencyError("execute", "missing_dependency", f"this recipe needs mediapipe: {e}", hint="pip install mediapipe") from None
    return mp, BaseOptions, vision


# EfficientDet-Lite0's labels: COCO's, as MediaPipe names them.
EFFICIENTDET_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def mediapipe_objects(*, classes=None, rename=None, min_confidence=0.3):
    """COCO objects found by MediaPipe's EfficientDet-Lite0 object detector,
    registered like yolo() (same nouns, same return value)."""
    mp, BaseOptions, vision = _mediapipe()
    path = model_file("efficientdet_lite0.tflite", MEDIAPIPE_MODELS)
    options = vision.ObjectDetectorOptions(base_options=BaseOptions(model_asset_path=path), score_threshold=min_confidence)
    detector = vision.ObjectDetector.create_from_options(options)

    def detector_for(label):
        def detect(image):
            frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=_rgb(image))
            return [
                (b.origin_x, b.origin_y, b.width, b.height, d.categories[0].score)
                for d in detector.detect(frame).detections
                for b in [d.bounding_box]
                if d.categories[0].category_name == label
            ]

        return detect

    return _register(
        EFFICIENTDET_LABELS,
        detector_for,
        classes=classes,
        rename=rename or {},
        min_confidence=min_confidence,
        provider="mediapipe efficientdet_lite0",
        sha256=MEDIAPIPE_MODELS["efficientdet_lite0.tflite"][1],
    )


def mediapipe_poses(*, max_poses=4, min_confidence=0.5):
    """`pose`/`poses`: people found by MediaPipe's pose landmarker. A box
    spans the landmarks inside the image; its confidence is their mean
    visibility."""
    mp, BaseOptions, vision = _mediapipe()
    path = model_file("pose_landmarker_lite.task", MEDIAPIPE_MODELS)
    options = vision.PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=path), num_poses=max_poses)
    landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(image):
        height, width = image.shape[:2]
        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=_rgb(image))
        boxes = []
        for pose in landmarker.detect(frame).pose_landmarks:
            xs = [min(max(p.x, 0.0), 1.0) * width for p in pose]
            ys = [min(max(p.y, 0.0), 1.0) * height for p in pose]
            visibility = sum(p.visibility or 0.0 for p in pose) / len(pose)
            boxes.append((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys), min(max(visibility, 0.0), 1.0)))
        return boxes

    add_target("pose", detect, plural="poses", min_confidence=min_confidence, provider="mediapipe pose_landmarker_lite", model_sha256=MEDIAPIPE_MODELS["pose_landmarker_lite.task"][1])
    return {"pose": "pose"}, {}
