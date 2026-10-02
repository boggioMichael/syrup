import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image as PILImage

import syrup
from conftest import FIXTURES

ASTRONAUT = FIXTURES / "astronaut.jpg"
FACE = syrup.Box(179, 65, 91, 109)


def iou(a, b):
    w = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
    h = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
    inter = max(w, 0) * max(h, 0)
    return inter / (a.w * a.h + b.w * b.h - inter)


def shifted(box, dx, dy):
    return syrup.Box(box.x + dx, box.y + dy, box.w, box.h)


def two_astronauts():
    one = np.asarray(PILImage.open(ASTRONAUT))
    canvas = np.full((1024, 1024, 3), 90, np.uint8)
    canvas[:512, :512] = one
    canvas[512:, 512:] = one
    return canvas


def test_find_face_finds_the_astronaut():
    from syrup.ops import find_face

    faces = find_face(ASTRONAUT)
    assert isinstance(faces, syrup.FindResult)
    assert len(faces) == 1
    face = faces[0]
    assert face.label == "face"
    assert face.confidence > 0.85
    assert iou(face.box, FACE) > 0.8
    assert list(face.keypoints) == ["right_eye", "left_eye", "nose_tip", "right_mouth_corner", "left_mouth_corner"]
    assert faces.provenance.operation == "find_face"
    assert faces.provenance.providers[0]["name"] == "yunet-2023mar"


def test_every_input_form_gives_the_same_answer():
    from syrup.ops import find_face

    assert list(find_face(str(ASTRONAUT))) == list(find_face(syrup.Image.open(ASTRONAUT)))
    # PIL decodes JPEG with slightly different rounding, so compare within it.
    pil = PILImage.open(ASTRONAUT)
    expected = list(find_face(pil))
    for image in (np.asarray(pil), np.asarray(pil.convert("RGBA"))):
        assert list(find_face(image)) == expected
    grey = find_face(np.asarray(pil.convert("L")))
    assert len(grey) == 1 and iou(grey[0].box, FACE) > 0.8


@pytest.mark.parametrize(
    "image",
    [np.zeros((8, 8), np.float32), np.zeros((8, 8, 2), np.uint8), np.zeros((0, 8, 3), np.uint8), "no/such/file.png", 42],
)
def test_bad_images_are_input_errors(image):
    with pytest.raises(syrup.InputError) as e:
        syrup.resolve("find_face")(image)
    assert e.value.stage == "input"


def test_no_face_is_an_empty_result_not_an_error():
    faces = syrup.resolve("find_faces")(FIXTURES / "coffee.jpg")
    assert not faces
    assert len(faces) == 0
    assert faces.provenance.plan_hash


def test_names_resolve_at_import_and_refusals_say_why():
    with pytest.raises(syrup.IntentError) as e:
        from syrup.ops import find_best_face  # noqa: F401
    assert (e.value.stage, e.value.kind) == ("resolve", "ambiguous")
    assert "best" in str(e.value)

    with pytest.raises(syrup.IntentError) as e:
        syrup.ops.find_smiling_faces
    assert e.value.kind == "unsupported"

    with pytest.raises(AttributeError):
        syrup.ops._anything


def test_compositions_restore_coordinates():
    image = two_astronauts()
    top, bottom = FACE, shifted(FACE, 512, 512)
    ops = syrup.ops

    found = ops.find_faces_left_to_right(image)
    assert len(found) == 2
    assert all(iou(f.box, e) > 0.7 for f, e in zip(found, (top, bottom)))
    (lower,) = ops.find_faces_in_bottom_half(image)
    assert iou(lower.box, bottom) > 0.7
    (region,) = ops.find_faces_in_region(image, region=(512, 512, 512, 512))
    assert iou(region.box, bottom) > 0.7
    assert len(ops.find_faces_by_size(image, max_results=1)) == 1
    assert not ops.find_faces_in_top_right(image)

    for name, kwargs in [
        ("find_faces_in_region", {}),
        ("find_faces_in_region", {"region": (1.5, 0, 2, 2)}),
        ("find_faces", {"min_confidence": 2.0}),
        ("find_faces", {"max_results": -1}),
        ("find_faces", {"region": (0, 0, 2, 2)}),
    ]:
        with pytest.raises(syrup.InputError):
            getattr(ops, name)(image, **kwargs)


def test_define_gives_other_names_a_meaning():
    op = syrup.define("lower_faces", find="faces", region=(0, 0.5, 1, 0.5))
    from syrup.ops import lower_faces

    assert lower_faces.plan_hash == op.plan_hash == syrup.resolve("find_faces_in_bottom_half").plan_hash
    assert not lower_faces(ASTRONAUT)

    with pytest.raises(syrup.IntentError) as e:
        syrup.define("find_face", find="face", region="top_half")
    assert e.value.kind == "conflicting"
    with pytest.raises(syrup.IntentError):
        syrup.define("unicorns", find="unicorns")
    with pytest.raises(syrup.IntentError):
        syrup.define("faces_in_nowhere", find="face", region=(0.5, 0, 0.6, 1))


def test_the_generated_source_is_inspectable():
    op = syrup.resolve("find_2_largest_faces_in_center")
    assert "fn run(" in op.source
    assert "select_fixed(&v0, (1, 4), (1, 4), (3, 4), (3, 4))" in op.source
    assert "detect face_detection" in op.explain()


def python(code, **env):
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_a_new_process_reuses_the_compiled_module(tmp_path):
    code = f"import syrup; print(syrup.resolve('find_face')({str(ASTRONAUT)!r}).provenance.artifact_status)"
    assert python(code, SYRUP_CACHE_DIR=str(tmp_path)) == "compiled"
    assert python(code, SYRUP_CACHE_DIR=str(tmp_path)) == "loaded_from_disk"


def test_failures_name_their_stage(tmp_path):
    code = """
import syrup
try:
    syrup.resolve("find_face").prepare()
except syrup.SyrupError as e:
    print(type(e).__name__, e.stage, e.kind)
"""
    missing = python(code, SYRUP_CACHE_DIR=str(tmp_path / "a"), SYRUP_RUSTC="/nonexistent/rustc")
    assert missing == "DependencyError compile missing_dependency"
    frozen = python(code, SYRUP_CACHE_DIR=str(tmp_path / "b"), SYRUP_MODE="frozen")
    assert frozen == "LoadError load not_prepared"


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="Tesseract is not installed")
def test_words_carry_their_text():
    words = syrup.ops.find_words_left_to_right(FIXTURES / "words.png")
    assert [w.text for w in words if w.box.y > 150] == ["HELLO", "WORLD"]
    assert all(w.label == "word" and w.confidence > 0.9 for w in words)
    assert syrup.ops.find_face(ASTRONAUT)[0].text is None


def test_colour_regions_are_found_in_arrays():
    image = np.full((60, 80, 3), 120, np.uint8)
    image[40:46, 10:50] = (220, 30, 30)
    image[5:20, 60:75] = (30, 60, 220)
    (bar,) = syrup.ops.find_red_bars(image)
    assert (bar.box, bar.label, bar.confidence) == (syrup.Box(10, 40, 40, 6), "bar", 1.0)
    (blue,) = syrup.ops.find_blue_regions_in_top_half(image)
    assert blue.box == syrup.Box(60, 5, 15, 15)
    assert not syrup.ops.find_green_regions(image)
    with pytest.raises(syrup.IntentError) as e:
        syrup.ops.find_regions
    assert e.value.kind == "ambiguous"
    assert syrup.define("alarm_bars", find="bars", color="red").plan_hash == syrup.resolve("find_red_bars").plan_hash


def test_bundles_run_frozen_without_a_compiler(tmp_path):
    index = syrup.bundle(tmp_path / "bundle", "find_largest_face")
    assert list(index["operations"]) == ["find_largest_face"]
    script = (
        "import syrup;"
        f"r = syrup.ops.find_biggest_face({str(FIXTURES / 'astronaut.jpg')!r});"
        "print(len(r), r.provenance.artifact_status)"
    )
    env = dict(os.environ, SYRUP_MODE="frozen", SYRUP_CACHE_DIR=str(tmp_path / "bundle"), SYRUP_RUSTC="/nonexistent")
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["1", "loaded_from_disk"]


def test_measurements_carry_a_value():
    page = np.full((120, 300, 3), 30, np.uint8)
    page[100:110, 20:220] = (70, 70, 78)  # the bar's track
    page[100:110, 20:70] = (210, 40, 40)  # a quarter full
    (bar,) = syrup.ops.measure_fill_of_red_bars(page)
    assert abs(bar.value - 0.25) < 0.02 and bar.box.y == 100

    crisp = PILImage.open(FIXTURES / "words.png").convert("RGB")
    readability = syrup.define("readability", find="image", measure="sharpness")
    (whole,) = readability(crisp)
    assert whole.label == "image" and whole.value >= 0.35
    assert syrup.ops.measure_sharpness(crisp)[0].value == whole.value
    assert syrup.ops.find_red_bars(page)[0].value is None
    with pytest.raises(syrup.IntentError) as e:
        syrup.ops.measure_fill_of_faces
    assert e.value.kind == "unsupported"


def test_sessions_follow_objects_across_frames():
    session = syrup.ops.track_largest_face.session()
    photo = PILImage.open(ASTRONAUT)
    tracks = []
    for step in range(3):
        (face,) = session(photo.crop((8 * step, 0, 8 * step + 400, 400)))
        tracks.append(face.track)
    assert len({t.id for t in tracks}) == 1
    assert abs(tracks[-1].velocity[0] + 8) < 2
    with pytest.raises(syrup.InputError):
        syrup.ops.track_faces(photo)
    with pytest.raises(syrup.InputError):
        syrup.ops.find_faces.session()


def test_text_blocks():
    page = PILImage.open(FIXTURES / "words.png")
    (block,) = syrup.ops.find_text_blocks(page)
    assert block.label == "text_block" and block.box.w > 100


@pytest.mark.skipif("SYRUP_TEST_WINDOW" not in os.environ, reason="CI opens a window to capture")
def test_live_windows_feed_sessions():
    title = os.environ["SYRUP_TEST_WINDOW"]
    assert any(title in t for t in syrup.capture.windows())
    with pytest.raises(syrup.InputError):
        next(syrup.capture.window("no window is called this 7f3a"))

    session = syrup.ops.track_moving_regions.session()
    frames = list(syrup.capture.window(title, frames=2))
    assert len(frames) == 2 and frames[0].width > 0 and frames[0].height > 0
    results = [session(frame) for frame in frames]
    assert results[1].provenance.frame == 2
