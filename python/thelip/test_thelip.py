"""Tests that need no weights: the numpy layers against naive references,
the CTC decoding, the GRID name decoding, the spelling corrector, and the
mouth locator on the astronaut. With a LipNet checkout present the weight
file is parsed too.

    python3 -m pytest python/thelip     # or: python3 python/thelip/test_thelip.py
"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lipnet_np import BLANK, LETTERS, Spell, conv3d, greedy_decode, grid_sentence, gru, hard_sigmoid, max_pool_ab  # noqa: E402
from the_lip import LIPNET_DIR, WEIGHTS, streaming_decodes  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "fixtures")


def test_hard_sigmoid_is_keras_hard_sigmoid():
    x = np.array([-10.0, -2.5, 0.0, 2.5, 10.0])
    assert np.allclose(hard_sigmoid(x), [0.0, 0.0, 0.5, 1.0, 1.0])


def test_conv3d_matches_a_naive_loop():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((5, 8, 6, 2)).astype(np.float32)
    kernel = rng.standard_normal((3, 3, 3, 2, 4)).astype(np.float32)
    bias = rng.standard_normal(4).astype(np.float32)
    pad, stride = (1, 2, 1), (1, 2, 1)
    out = conv3d(x, kernel, bias, pad, stride)
    xp = np.pad(x, ((1, 1), (2, 2), (1, 1), (0, 0)))
    expected = np.zeros_like(out)
    for t in range(out.shape[0]):
        for a in range(out.shape[1]):
            for b in range(out.shape[2]):
                window = xp[t : t + 3, a * 2 : a * 2 + 3, b : b + 3]
                for f in range(4):
                    expected[t, a, b, f] = (window * kernel[..., f]).sum() + bias[f]
    assert out.shape == (5, 5, 6, 4)
    assert np.allclose(out, expected, atol=1e-4)


def test_max_pool_halves_the_spatial_axes():
    x = np.arange(2 * 4 * 6 * 1, dtype=np.float32).reshape(2, 4, 6, 1)
    pooled = max_pool_ab(x)
    assert pooled.shape == (2, 2, 3, 1)
    assert pooled[0, 0, 0, 0] == x[0, :2, :2, 0].max()
    assert pooled[1, 1, 2, 0] == x[1, 2:4, 4:6, 0].max()


def test_gru_reversed_is_the_forward_pass_on_the_reversed_sequence():
    rng = np.random.default_rng(2)
    units, dim, steps = 5, 3, 7
    x = rng.standard_normal((steps, dim)).astype(np.float32)
    kernel = rng.standard_normal((dim, 3 * units)).astype(np.float32) * 0.5
    recurrent = rng.standard_normal((units, 3 * units)).astype(np.float32) * 0.5
    bias = rng.standard_normal(3 * units).astype(np.float32) * 0.1
    forward_on_reversed = gru(x[::-1], kernel, recurrent, bias)[::-1]
    assert np.allclose(gru(x, kernel, recurrent, bias, reverse=True), forward_on_reversed)
    # Zero weights: the state never leaves zero.
    assert np.all(gru(x, np.zeros_like(kernel), np.zeros_like(recurrent), np.zeros_like(bias)) == 0)
    # The state is bounded by tanh, whatever the input.
    out = gru(x * 100, kernel, recurrent, bias)
    assert np.all(np.abs(out) <= 1.0)


def _one_hot_path(chars):
    probs = np.full((len(chars), 28), 0.01, dtype=np.float32)
    for t, c in enumerate(chars):
        probs[t, c] = 0.9
    return probs


def test_greedy_decode_collapses_repeats_and_drops_blanks():
    a, b, c = LETTERS.index("a"), LETTERS.index("b"), LETTERS.index("c")
    path = [a, a, BLANK, b, b, BLANK, 26, c, BLANK, BLANK]
    assert greedy_decode(_one_hot_path(path)) == "ab c"
    # A blank between two identical letters keeps both.
    assert greedy_decode(_one_hot_path([a, BLANK, a])) == "aa"
    assert greedy_decode(_one_hot_path([BLANK] * 4)) == ""


def test_grid_file_names_decode_to_sentences():
    assert grid_sentence("sbwe5n") == "set blue with e five now"
    assert grid_sentence("id2_vcd_swwp2s") == "set white with p two soon"
    assert grid_sentence("lbbc2a") == "lay blue by c two again"
    assert grid_sentence("pwij3p") == "place white in j three please"


def test_spelling_snaps_to_the_dictionary():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("set blue with e five now\nbin red by k seven again\n")
        path = f.name
    try:
        spell = Spell(path)
        assert spell.sentence("set blu with e fiv now") == "set blue with e five now"
        assert spell.sentence("bin red by k sevn again") == "bin red by k seven again"
        assert spell.correction("now") == "now"
    finally:
        os.unlink(path)


def test_streaming_decodes_hold_back_the_newest_frames():
    class Fake:
        def features(self, crops):
            return np.zeros((len(crops), 4), dtype=np.float32)

        def probabilities(self, features):
            # 'a' for the first half of what has been seen, 'b' after.
            n = len(features)
            path = [LETTERS.index("a")] * (n // 2) + [LETTERS.index("b")] * (n - n // 2)
            return _one_hot_path(path)

    crops = np.zeros((20, 100, 50, 3), dtype=np.float32)
    decodes = streaming_decodes(Fake(), crops)
    assert len(decodes) == 20
    assert decodes[:8] == [""] * 8  # nothing shown until 8 frames have settled
    assert decodes[-1] == "ab"


def test_mouth_is_located_below_and_between_the_astronauts_eyes():
    from PIL import Image

    from mouth import MouthLocator, mouth_crops

    frame = np.asarray(Image.open(os.path.join(FIXTURES, "astronaut_320.jpg")).convert("RGB"))
    locator = MouthLocator()
    centre, corners = locator.locate(frame)
    assert centre is not None, "syrup should find both eyes on the astronaut"
    left, right = locator.eyes
    assert left[0] < centre[0] < right[0]
    assert centre[1] > max(left[1], right[1]) + 0.8 * np.linalg.norm(right - left)
    assert corners[0] < centre[0] < corners[1]

    crops, boxes, ratio = mouth_crops([frame, frame])
    assert crops.shape == (2, 100, 50, 3) and crops.dtype == np.float32
    assert 0.0 <= crops.min() and crops.max() <= 1.0
    assert boxes[0] is not None and ratio > 0


def test_the_weights_parse_when_a_checkout_is_present():
    path = os.path.join(LIPNET_DIR, WEIGHTS)
    if not os.path.exists(path):
        import pytest

        pytest.skip("no LipNet checkout")
    from minih5 import H5

    arrays = H5(path).arrays()
    assert arrays["/conv1/conv1/kernel:0"].shape == (3, 5, 5, 3, 32)
    assert arrays["/bidirectional_1/bidirectional_1/kernel:0"].shape == (1728, 768)
    assert arrays["/dense1/dense1/kernel:0"].shape == (512, 28)


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print(f"ok   {name}")
            except Exception as error:  # noqa: BLE001
                if type(error).__name__ == "Skipped":
                    print(f"skip {name}")
                    continue
                failures += 1
                print(f"FAIL {name}: {error!r}")
    sys.exit(1 if failures else 0)
