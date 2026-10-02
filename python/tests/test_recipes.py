"""The recipes on real photographs: a cup of coffee, a cat, and an astronaut.
They download their models once, so they need the network the first time."""

import pytest
from PIL import Image as PILImage

import syrup
import syrup.recipes as recipes
from conftest import FIXTURES

COFFEE, CAT, ASTRONAUT = (PILImage.open(FIXTURES / name) for name in ("coffee.jpg", "chelsea.jpg", "astronaut.jpg"))


def overlaps(box, x, y, w, h):
    ix = min(box.x + box.w, x + w) - max(box.x, x)
    iy = min(box.y + box.h, y + h) - max(box.y, y)
    return max(ix, 0) * max(iy, 0) / (box.w * box.h) > 0.5


def test_nouns_and_plurals():
    assert recipes.noun("traffic light") == ("traffic_light", "traffic_lights")
    assert recipes.noun("person") == ("person", "people")
    assert recipes.noun("wine glass") == ("wine_glass", "wine_glasses")
    assert recipes.noun("teddy bear") == ("teddy_bear", "teddy_bears")
    assert recipes.noun("skis") == ("skis", "skis")


def test_yolo_objects_compose_like_built_in_targets():
    pytest.importorskip("ultralytics")
    added, skipped = recipes.yolo()
    assert {"person", "cup", "cat", "dining_table"} <= set(added)
    assert "orange" in skipped, "a colour cannot be a noun"

    (cup,) = syrup.ops.find_cups(COFFEE)
    assert overlaps(cup.box, 161, 14, 253, 293) and cup.confidence > 0.5
    (cat,) = syrup.ops.find_largest_cat(CAT)
    assert cat.box.w > 300
    people = syrup.ops.find_people_in_left_half(ASTRONAUT)
    assert len(people) == 1 and people[0].box.x + people[0].box.w <= 256
    assert not syrup.ops.find_cups(ASTRONAUT), "no cup in the astronaut photo"

    result = syrup.ops.find_cats(CAT)
    assert result.provenance.providers[0]["name"] == "ultralytics yolo11n.pt"
    assert result.provenance.providers[0]["model_sha256"] == recipes.YOLO_MODELS["yolo11n.pt"][1]

    added, _ = recipes.yolo(rename={"orange": "citrus"}, classes=["orange"])
    assert added == {"citrus": "orange"}


def test_mediapipe_objects_and_poses():
    pytest.importorskip("mediapipe")
    try:
        added, _ = recipes.mediapipe_objects(classes=["cup", "cat", "person"])
    except OSError as e:
        pytest.skip(f"mediapipe cannot load its native library: {e}")
    assert set(added) == {"cup", "cat", "person"}

    (cup,) = syrup.ops.find_most_confident_cup(COFFEE)
    assert overlaps(cup.box, 161, 14, 253, 293)
    assert syrup.ops.find_cats(CAT)[0].confidence > 0.5
    assert syrup.ops.find_people(ASTRONAUT).provenance.providers[0]["name"] == "mediapipe efficientdet_lite0"

    recipes.mediapipe_poses()
    (pose,) = syrup.ops.find_poses(ASTRONAUT)
    person = syrup.ops.find_people(ASTRONAUT)[0]
    assert overlaps(pose.box, person.box.x, person.box.y, person.box.w, person.box.h)
