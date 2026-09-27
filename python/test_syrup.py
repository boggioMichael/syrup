"""Checks the Python package against the native library on the fixtures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import syrup  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def test_find_face_on_the_astronaut():
    image = Image.open(FIXTURES / "astronaut_320.jpg")
    faces = syrup.find_face(image)
    assert faces.value is not None and len(faces.value) == 1, faces
    (face,) = faces.value
    x, y, w, h = face.bounds
    assert 100 <= x <= 120 and 30 <= y <= 50 and 50 <= w <= 70
    assert face.score > 0.9
    assert faces.reliability == "corroborated"


def test_numpy_arrays_and_regions():
    array = np.asarray(Image.open(FIXTURES / "astronaut_320.jpg").convert("RGB"))
    eyes = syrup.find_eyes(array)
    assert eyes.value is not None and len(eyes.value) == 2, eyes
    nothing = syrup.find_face(array, region=(0, 200, 320, 120))
    assert nothing.value == [] and "no frontal face" in nothing.failure_reason


def test_bars_and_counts():
    frame = np.full((40, 200, 3), (30, 30, 34), dtype=np.uint8)
    frame[14:26, 20:128] = (215, 35, 35)
    frame[14:26, 128:180] = (70, 70, 110)
    fill = syrup.measure_red_bar(frame)
    assert fill.value is not None and abs(fill.value - 67.5) < 1.0, fill
    assert syrup.count_red_bars(frame).value == 1.0
    (bar,) = syrup.find_largest_red_blob(frame).value
    assert bar.bounds == (20, 14, 108, 12)


def test_refusals_explain_themselves():
    try:
        syrup.find_unicorn
    except syrup.IntentError as error:
        assert "not an intent I understand" in str(error)
    else:
        raise AssertionError("find_unicorn should be refused")
    try:
        syrup.find_boss_icon
    except syrup.IntentError as error:
        assert "register_template" in str(error)
    else:
        raise AssertionError("an icon without a picture should be refused")


def test_icons_and_source():
    icon = np.zeros((12, 12, 3), dtype=np.uint8)
    icon[:, :, 0] = np.arange(12, dtype=np.uint8)[:, None] * 20
    icon[:, :, 1] = np.arange(12, dtype=np.uint8)[None, :] * 20
    frame = np.full((90, 160, 3), 90, dtype=np.uint8)
    frame[30:42, 20:32] = icon
    frame[50:62, 120:132] = icon
    syrup.register_template("pytesticon", icon)
    found = syrup.find_pytesticon_icon(frame)
    assert found.value is not None and sorted(m.bounds[:2] for m in found.value) == [(20, 30), (120, 50)], found
    assert "plans::find_icon" in syrup.find_pytesticon_icon.source


if __name__ == "__main__":
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            test()
            print("ok", name)
    print("library:", syrup.library_path(), "version", syrup.version())
