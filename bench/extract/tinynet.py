"""The tiny-network baseline of PREREGISTRATION.md, for glyph reading.

    python3 bench/extract/tinynet.py train --data DATA --out MODEL_DIR
    bench/extract/tinynet glyphs --data DATA --split S --threads 1 --calls 5 \
        --warmup 3 --out FILE.json --model MODEL_DIR/glyphs-int8.onnx

The network reads the same 16 x 16 cells as the OpenCV pipeline of the
OpenCV benchmark (Otsu, connected components, `glyph_cells`), so the two differ
only in the classifier: one nearest neighbour there, a one-hidden-layer MLP
here, quantised to INT8 and run single-threaded in ONNX Runtime.

Training sees what the other readers see: the two training lines (every
character once, at each size). It augments them with generic changes only
(rescaling, a darker or lighter background, pixel noise); it does not draw the
benchmark's red bars or its noise model. The hidden size is the smallest of
16, 32, 64 and 128 whose exact-string accuracy on the dev cases is within one
point of the best, as the pre-registration's rule for the tiny network.

The ONNX file is written by hand (this environment has no `onnx` package): an
integer graph of QuantizeLinear, MatMulInteger and a float epilogue per layer,
then ArgMax. The run command checks that ONNX Runtime's answers equal a NumPy
re-computation of the same integer arithmetic before it times anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vs-opencv"))
from opencv_side import binding_overhead_ns, glyph_cells, timed, usage_now  # noqa: E402

SEED = 20261012
HIDDEN = (16, 32, 64, 128)
VARIANTS = 300  # augmented copies of each training line


# ------------------------------------------------------------ ONNX by hand

def _varint(n: int) -> bytes:
    n &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _key(field: int, wire: int) -> bytes:
    return _varint((field << 3) | wire)


def _int(field: int, v: int) -> bytes:
    return _key(field, 0) + _varint(v)


def _bytes(field: int, b: bytes) -> bytes:
    return _key(field, 2) + _varint(len(b)) + b


def _str(field: int, s: str) -> bytes:
    return _bytes(field, s.encode())


DTYPE = {np.float32: 1, np.uint8: 2, np.int8: 3, np.int32: 6, np.int64: 7}


def _tensor(name: str, a: np.ndarray) -> bytes:
    """TensorProto: dims (1), data_type (2), name (8), raw_data (9)."""
    a = np.ascontiguousarray(a)
    return (b"".join(_int(1, d) for d in a.shape) + _int(2, DTYPE[a.dtype.type]) + _str(8, name)
            + _bytes(9, a.tobytes()))


def _value_info(name: str, elem: int, dims: list) -> bytes:
    """ValueInfoProto: name (1), TypeProto (2) -> tensor_type (1) -> elem_type (1), shape (2)."""
    shape = b"".join(_bytes(1, _int(1, d) if isinstance(d, int) else _str(2, d)) for d in dims)
    return _str(1, name) + _bytes(2, _bytes(1, _int(1, elem) + _bytes(2, shape)))


def _attr_int(name: str, v: int) -> bytes:
    """AttributeProto: name (1), i (3), type (20) = INT (2)."""
    return _str(1, name) + _int(3, v) + _int(20, 2)


def _node(op: str, inputs: list[str], outputs: list[str], attrs=()) -> bytes:
    """NodeProto: input (1), output (2), name (3), op_type (4), attribute (5)."""
    return (b"".join(_str(1, i) for i in inputs) + b"".join(_str(2, o) for o in outputs)
            + _str(3, f"{op}_{outputs[0]}") + _str(4, op) + b"".join(_bytes(5, a) for a in attrs))


def _model(nodes, initialisers, inputs, outputs) -> bytes:
    """ModelProto: ir_version (1), producer_name (2), graph (7), opset_import (8)."""
    graph = (b"".join(_bytes(1, n) for n in nodes) + _str(2, "tinynet")
             + b"".join(_bytes(5, t) for t in initialisers)
             + b"".join(_bytes(11, i) for i in inputs) + b"".join(_bytes(12, o) for o in outputs))
    return _int(1, 8) + _str(2, "syrup bench/extract") + _bytes(7, graph) + _bytes(8, _str(1, "") + _int(2, 13))


# ------------------------------------------------------------ quantisation

def quantise(mlp, calibration: np.ndarray) -> dict:
    """Per-tensor symmetric INT8 weights; inputs and hidden activations as
    UINT8 with zero point 0 (both are non-negative); float biases added after
    rescaling the INT32 sums."""
    w1, w2 = mlp.coefs_
    b1, b2 = mlp.intercepts_
    q = {"x_scale": np.float32(1 / 255)}
    q["w1_scale"] = np.float32(np.abs(w1).max() / 127)
    q["w1"] = np.clip(np.round(w1 / q["w1_scale"]), -127, 127).astype(np.int8)
    q["b1"] = b1.astype(np.float32)
    hidden = np.maximum(calibration @ w1 + b1, 0)
    q["h_scale"] = np.float32(max(float(hidden.max()), 1e-6) / 255)
    q["w2_scale"] = np.float32(np.abs(w2).max() / 127)
    q["w2"] = np.clip(np.round(w2 / q["w2_scale"]), -127, 127).astype(np.int8)
    q["b2"] = b2.astype(np.float32)
    return q


def integer_forward(q: dict, x: np.ndarray) -> np.ndarray:
    """What the ONNX graph computes, in NumPy (round half to even, as QuantizeLinear)."""
    xq = np.clip(np.rint(x / q["x_scale"]), 0, 255).astype(np.int32)
    z1 = (xq @ q["w1"].astype(np.int32)).astype(np.float32) * np.float32(q["x_scale"] * q["w1_scale"]) + q["b1"]
    hq = np.clip(np.rint(np.maximum(z1, 0) / q["h_scale"]), 0, 255).astype(np.int32)
    z2 = (hq @ q["w2"].astype(np.int32)).astype(np.float32) * np.float32(q["h_scale"] * q["w2_scale"]) + q["b2"]
    return z2.argmax(axis=1)


def onnx_bytes(q: dict) -> bytes:
    hidden, classes = q["w2"].shape
    f32 = np.float32
    inits = [
        _tensor("x_scale", np.array(q["x_scale"], f32)), _tensor("x_zp", np.array(0, np.uint8)),
        _tensor("w1", q["w1"]), _tensor("w1_zp", np.array(0, np.int8)),
        _tensor("s1", np.array(q["x_scale"] * q["w1_scale"], f32)), _tensor("b1", q["b1"]),
        _tensor("h_scale", np.array(q["h_scale"], f32)), _tensor("h_zp", np.array(0, np.uint8)),
        _tensor("w2", q["w2"]), _tensor("w2_zp", np.array(0, np.int8)),
        _tensor("s2", np.array(q["h_scale"] * q["w2_scale"], f32)), _tensor("b2", q["b2"]),
    ]
    nodes = [
        _node("QuantizeLinear", ["x", "x_scale", "x_zp"], ["xq"]),
        _node("MatMulInteger", ["xq", "w1", "x_zp", "w1_zp"], ["a1"]),
        _node("Cast", ["a1"], ["a1f"], [_attr_int("to", 1)]),
        _node("Mul", ["a1f", "s1"], ["m1"]),
        _node("Add", ["m1", "b1"], ["z1"]),
        _node("Relu", ["z1"], ["h"]),
        _node("QuantizeLinear", ["h", "h_scale", "h_zp"], ["hq"]),
        _node("MatMulInteger", ["hq", "w2", "h_zp", "w2_zp"], ["a2"]),
        _node("Cast", ["a2"], ["a2f"], [_attr_int("to", 1)]),
        _node("Mul", ["a2f", "s2"], ["m2"]),
        _node("Add", ["m2", "b2"], ["logits"]),
        _node("ArgMax", ["logits"], ["index"], [_attr_int("axis", 1), _attr_int("keepdims", 0)]),
    ]
    return _model(nodes, inits, [_value_info("x", 1, ["n", int(q["w1"].shape[0])])],
                  [_value_info("index", 7, ["n"])])


# ------------------------------------------------------------ training

def augment(line: np.ndarray, rng) -> np.ndarray:
    scale = rng.uniform(0.85, 1.2)
    out = cv2.resize(line, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out += rng.uniform(-20, 40)                       # darker or lighter panel and text
    out += rng.normal(0, rng.uniform(0, 10), out.shape)  # pixel noise
    return np.clip(out, 0, 255).round().astype(np.uint8)


def read_with(classify, chars: str, bgr) -> str | None:
    cells = glyph_cells(bgr)
    if not cells:
        return None
    return "".join(chars[i] for i in classify(np.array(cells, np.float32)))


def train(a) -> int:
    from sklearn.neural_network import MLPClassifier
    with open(os.path.join(a.data, "glyphs", "dev", "cases.json")) as f:
        spec = json.load(f)
    chars = spec["chars"]
    rng = np.random.default_rng(SEED)
    xs, ys, kept, skipped = [], [], 0, 0
    for path in spec["train"].values():
        line = cv2.imread(os.path.join(a.data, path), cv2.IMREAD_COLOR)
        for v in range(VARIANTS + 1):
            image = line if v == 0 else augment(line, rng)
            cells = glyph_cells(image)
            if len(cells) != len(chars):
                skipped += 1
                continue
            kept += 1
            xs.extend(cells)
            ys.extend(range(len(chars)))
    x, y = np.array(xs, np.float32), np.array(ys)
    dev = [(cv2.imread(os.path.join(a.data, c["image"]), cv2.IMREAD_COLOR), c["text"]) for c in spec["cases"]]

    def dev_accuracy(classify):
        return sum(read_with(classify, chars, img) == text for img, text in dev) / len(dev)

    tried = []
    for h in HIDDEN:
        mlp = MLPClassifier(hidden_layer_sizes=(h,), max_iter=400, random_state=SEED).fit(x, y)
        tried.append({"hidden": h, "dev_exact": dev_accuracy(mlp.predict), "model": mlp})
    best = max(t["dev_exact"] for t in tried)
    chosen = next(t for t in tried if t["dev_exact"] >= best - 0.01)
    q = quantise(chosen["model"], x)
    int8_dev = dev_accuracy(lambda cells: integer_forward(q, cells))
    os.makedirs(a.out, exist_ok=True)
    blob = onnx_bytes(q)
    path = os.path.join(a.out, "glyphs-int8.onnx")
    with open(path, "wb") as f:
        f.write(blob)
    np.savez(os.path.join(a.out, "glyphs-int8.npz"), **{k: np.asarray(v) for k, v in q.items()})
    report = {
        "chars": chars, "seed": SEED, "variants_per_line": VARIANTS,
        "training_lines_kept": kept, "training_lines_skipped": skipped, "training_cells": int(len(y)),
        "tried": [{"hidden": t["hidden"], "dev_exact": t["dev_exact"]} for t in tried],
        "chosen_hidden": chosen["hidden"], "float_dev_exact": chosen["dev_exact"], "int8_dev_exact": int8_dev,
        "parameters": int(sum(w.size for w in chosen["model"].coefs_) + sum(b.size for b in chosen["model"].intercepts_)),
        "onnx_sha256": hashlib.sha256(blob).hexdigest(), "onnx_bytes": len(blob),
    }
    with open(os.path.join(a.out, "glyphs-int8.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))
    return 0


# ------------------------------------------------------------ the measured reader

def glyphs(a) -> int:
    import onnxruntime as ort
    directory = os.path.join(a.data, "glyphs", a.split)
    with open(os.path.join(directory, "cases.json")) as f:
        spec = json.load(f)
    chars = spec["chars"]
    meta_path = os.path.splitext(a.model)[0] + ".json"
    with open(meta_path) as f:
        meta = json.load(f)

    def load():
        o = ort.SessionOptions()
        o.intra_op_num_threads = a.threads
        o.inter_op_num_threads = 1
        o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Idle threads that spin between calls burn a core without changing
        # the call's time (measured: 10.1 vs 10.2 us); they would be charged
        # to this implementation's CPU time.
        o.add_session_config_entry("session.intra_op.allow_spinning", "0")
        o.add_session_config_entry("session.inter_op.allow_spinning", "0")
        return ort.InferenceSession(a.model, o, providers=["CPUExecutionProvider"])

    # The segmentation is OpenCV's: held to the same threads as the OpenCV side.
    cv2.setNumThreads(a.threads)
    cv2.setUseOptimized(True)
    base_user, base_sys, base_rss = usage_now()
    started = time.perf_counter()
    session, load_ns = timed(load)
    q = dict(np.load(os.path.splitext(a.model)[0] + ".npz"))
    probe = np.random.default_rng(SEED).random((512, q["w1"].shape[0])).astype(np.float32)
    if not np.array_equal(session.run(["index"], {"x": probe})[0], integer_forward(q, probe)):
        raise SystemExit("ONNX Runtime and the NumPy integer re-computation disagree")

    def classify(cells):
        return session.run(["index"], {"x": cells})[0]

    items, first_call = [], None
    for case in spec["cases"]:
        image = cv2.imread(os.path.join(a.data, case["image"]), cv2.IMREAD_COLOR)
        text = None
        for _ in range(a.warmup):
            text, t = timed(lambda: read_with(classify, chars, image))
            if first_call is None:
                first_call = t
        times = []
        for _ in range(a.calls):
            text, t = timed(lambda: read_with(classify, chars, image))
            times.append(t)
        output = {"text": text} if text is not None else {"refused": "no glyphs"}
        items.append({"input": case["image"], "times_ns": times, "output": output})
    wall = time.perf_counter() - started
    user, system, rss = usage_now()
    out = {
        "impl": "tinynet", "version": f"onnxruntime {ort.__version__}, model {meta['onnx_sha256'][:12]}",
        "task": "glyphs", "split": a.split, "threads": a.threads, "calls": a.calls, "warmup": a.warmup,
        "init": {"load_ns": load_ns, "first_call_ns": first_call or 0, "binding_overhead_ns": binding_overhead_ns(),
                 "learned": [{"file": "glyphs-int8.onnx", "learned": meta["training_cells"]}]},
        "process": {"wall_s": wall, "cpu_user_s": user - base_user, "cpu_sys_s": system - base_sys,
                    "max_rss_kb": rss, "rss_after_imports_kb": base_rss},
        "items": items,
    }
    with open(a.out, "w") as f:
        json.dump(out, f)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    g = sub.add_parser("glyphs")
    g.add_argument("--data", required=True)
    g.add_argument("--split", default="dev")
    g.add_argument("--threads", type=int, default=1)
    g.add_argument("--calls", type=int, default=5)
    g.add_argument("--warmup", type=int, default=3)
    g.add_argument("--out", required=True)
    g.add_argument("--model", required=True)
    a = ap.parse_args()
    return train(a) if a.cmd == "train" else glyphs(a)


if __name__ == "__main__":
    sys.exit(main())
