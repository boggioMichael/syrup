"""Syrup runs YuNet through tract with its own pre- and post-processing.
This checks it against OpenCV's FaceDetectorYN on the same model and the
same 640x640 canvas, so a decoding or scaling slip cannot pass unnoticed."""

import numpy as np
import pytest
from PIL import Image as PILImage

import syrup
from conftest import FIXTURES, MODEL
from test_syrup import iou, two_astronauts

cv2 = pytest.importorskip("cv2")

CANVAS = 640


def opencv_faces(rgb, min_confidence=0.6):
    height, width = rgb.shape[:2]
    scale = min(CANVAS / width, CANVAS / height)
    w, h = round(width * scale), round(height * scale)
    filter = cv2.INTER_LINEAR if scale >= 1 else cv2.INTER_AREA
    canvas = np.zeros((CANVAS, CANVAS, 3), np.uint8)
    canvas[:h, :w] = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (w, h), interpolation=filter)
    detector = cv2.FaceDetectorYN.create(str(MODEL), "", (CANVAS, CANVAS), min_confidence, 0.3, 5000)
    _, faces = detector.detect(canvas)
    sx, sy = w / width, h / height
    return [] if faces is None else [(syrup.Box(f[0] / sx, f[1] / sy, f[2] / sx, f[3] / sy), float(f[14])) for f in faces]


@pytest.mark.parametrize("name", ["astronaut", "two_astronauts", "coffee", "chelsea"])
def test_matches_opencv(name):
    if name == "two_astronauts":
        rgb = two_astronauts()
    else:
        rgb = np.asarray(PILImage.open(FIXTURES / f"{name}.jpg").convert("RGB"))
    expected = sorted(opencv_faces(rgb), key=lambda f: f[0].x)
    found = sorted(syrup.resolve("find_faces")(rgb), key=lambda f: f.box.x)
    assert len(found) == len(expected)
    for face, (box, score) in zip(found, expected):
        assert iou(face.box, box) > 0.9, (face.box, box)
        assert abs(face.confidence - score) < 0.05
