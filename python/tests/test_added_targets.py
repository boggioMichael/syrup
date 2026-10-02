"""A detector from another library becomes a Syrup target: OpenCV's QR
code reader, as the noun "tag", here; it could be any model with a Python
API. (QR codes themselves are built in, as qr_code.)"""

import numpy as np
import pytest

import syrup

cv2 = pytest.importorskip("cv2")


def qr_codes(image):
    found, texts, corners, _ = cv2.QRCodeDetector().detectAndDecodeMulti(np.ascontiguousarray(image))
    if not found:
        return []
    return [(*cv2.boundingRect(c.astype(np.float32)), 1.0, text) for c, text in zip(corners, texts)]


def page():
    """White 640x480 with QR codes saying "top", "left" and "right"."""
    canvas = np.full((480, 640, 3), 255, np.uint8)
    for text, (x, y) in {"top": (250, 20), "left": (40, 300), "right": (420, 300)}.items():
        code = cv2.resize(cv2.QRCodeEncoder.create().encode(text), None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
        h, w = code.shape
        canvas[y : y + h, x : x + w] = code[:, :, None]
    return canvas


@pytest.fixture(scope="module", autouse=True)
def tag_target():
    syrup.add_target("tag", qr_codes, min_confidence=0.0)


def test_operations_compose_around_a_python_detector():
    image = page()
    found = syrup.ops.find_tags_left_to_right(image)
    assert [f.text for f in found] == ["left", "top", "right"]
    assert found.provenance.providers == [{"capability": "tag", "name": "python", "model_sha256": None, "runtime": "python"}]

    lower = syrup.ops.find_tags_in_bottom_half_left_to_right(image)
    assert [f.text for f in lower] == ["left", "right"]
    # The detector saw only the bottom half; the box is back in page pixels.
    left = lower[0].box
    assert 40 <= left.x < 60 and 300 <= left.y < 320, left

    (rightmost,) = syrup.ops.find_rightmost_tag(image)
    assert rightmost.text == "right"
    assert not syrup.ops.find_tags(np.full((100, 100, 3), 255, np.uint8))


def test_added_nouns_must_not_change_what_names_mean():
    with pytest.raises(syrup.IntentError) as e:
        syrup.add_target("largest_box", qr_codes)
    assert e.value.kind == "conflicting"
    with pytest.raises(syrup.IntentError):
        syrup.add_target("face", qr_codes)
    with pytest.raises(syrup.IntentError):
        syrup.add_target("qr_code", qr_codes)
    with pytest.raises(syrup.IntentError):
        syrup.ops.find_widgets


def test_detector_failures_are_execution_errors():
    def broken(image):
        raise RuntimeError("model file missing")

    def sloppy(image):
        return [(1, 2, 3)]

    syrup.add_target("gadget", broken)
    with pytest.raises(syrup.ExecutionError) as e:
        syrup.ops.find_gadgets(page())
    assert e.value.kind == "provider_failed" and "model file missing" in str(e.value)

    syrup.add_target("gadget", sloppy)
    with pytest.raises(syrup.ExecutionError) as e:
        syrup.ops.find_gadgets(page())
    assert "(x, y, w, h, score)" in str(e.value)
