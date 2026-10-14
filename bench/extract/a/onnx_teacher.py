"""The teacher in ONNX Runtime, FP32 and INT8, for its cost (PILOT-A.md:
"For timing, INT8 in ONNX Runtime, one thread").

    onnx_teacher.py build --teacher NET.npz --lines L.npz --labels EXP-LABELS.json
                          --bank BANK.npz --port PORT_DIR --out DIR
    onnx_teacher.py time  --model DIR/teacher-int8.onnx --port PORT_DIR --out FILE.json
    onnx_teacher.py bench --dir DIR --port PORT_DIR [--rounds 5]

`build` writes teacher-fp32.onnx (five Conv nodes, as the NumPy layers) and
teacher-int8.onnx (QLinearConv: per-channel symmetric INT8 weights, UINT8
activations whose zero point 0 makes each convolution's ReLU part of it, the
input and the last layer's scores asymmetric UINT8), the scales calibrated
on training lines only. It checks that FP32 gives the NumPy scores back and
reports what each model reads on the port's test lines (`export.py`).
`time` reads each test line in its own `session.run` (3 warm-up calls, then 5
timed one by one; the line's time is the median), one thread, the
shakedown's session options; the line arrives normalised (`to_input`), so
this is the network's own cost, which favours the teacher against the Rust
readers, whose times include the normalising.

The ONNX files are written by hand (no `onnx` package here), with the
protobuf helpers of ../tinynet.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import expsynth  # noqa: E402
import linenet  # noqa: E402
import teach  # noqa: E402
from tinynet import _bytes, _int, _model, _node, _str, _tensor, _value_info  # noqa: E402


def _attr_ints(name: str, values) -> bytes:
    """AttributeProto: name (1), ints (8, one entry each), type (20) = INTS (7)."""
    return _str(1, name) + b"".join(_int(8, int(v)) for v in values) + _int(20, 7)


def conv_attrs(c):
    return [_attr_ints("kernel_shape", [c.kh, c.kw]), _attr_ints("pads", [c.ph, c.pw, c.ph, c.pw]),
            _attr_ints("strides", [1, 1])]


def nchw_weights(c):
    """(kh*kw*cin, cout) -> (cout, cin, kh, kw)."""
    return np.ascontiguousarray(c.w.reshape(c.kh, c.kw, c.cin, c.cout).transpose(3, 2, 0, 1), np.float32)


def convs(net):
    return [l for l in net.layers if isinstance(l, linenet.Conv)]


def io():
    return ([_value_info("x", 1, [1, 3, linenet.HEIGHT, linenet.WIDTH])],
            [_value_info("logits", 1, [1, linenet.NCLASS, 1, linenet.WIDTH // 2])])


def fp32_model(net) -> bytes:
    cs = convs(net)
    inits, nodes, x = [], [], "x"
    for i, c in enumerate(cs):
        inits += [_tensor(f"w{i}", nchw_weights(c)), _tensor(f"b{i}", c.b.astype(np.float32))]
        y = "logits" if i == len(cs) - 1 else f"c{i}"
        nodes.append(_node("Conv", [x, f"w{i}", f"b{i}"], [y], conv_attrs(c)))
        x = y
        if i < len(cs) - 1:
            nodes.append(_node("Relu", [x], [f"r{i}"]))
            x = f"r{i}"
        if i == 1:
            nodes.append(_node("MaxPool", [x], ["p1"], [_attr_ints("kernel_shape", [2, 2]), _attr_ints("strides", [2, 2])]))
            x = "p1"
    inputs, outputs = io()
    return _model(nodes, inits, inputs, outputs)


def calibrate(net, x):
    """Each quantised tensor's range on calibration lines: the input, the five
    ReLU'd convolution outputs (the last one's scores, unclipped)."""
    taps = []
    for s in range(0, len(x), 64):
        t = []
        net.forward(x[s:s + 64], keep=False, taps=t)
        if not taps:
            taps = [[] for _ in t]
        for i, v in enumerate(t):
            taps[i].append((float(v.min()), float(v.max())))
    rng = lambda i: (min(a for a, _ in taps[i]), max(b for _, b in taps[i]))
    # layer outputs: conv0 relu0 conv1 relu1 pool conv2 relu2 conv3 relu3 conv4
    return {"x": (float(x.min()), float(x.max())), "a0": rng(1), "a1": rng(3), "a2": rng(6), "a3": rng(8), "a4": rng(9)}


def asym(lo, hi):
    scale = (hi - lo) / 255.0
    zp = int(np.clip(round(-lo / scale), 0, 255))
    return np.float32(scale), np.uint8(zp)


def int8_model(net, ranges) -> tuple[bytes, dict]:
    cs = convs(net)
    xs, xz = asym(*ranges["x"])
    scales = [xs] + [np.float32(ranges[f"a{i}"][1] / 255.0) for i in range(4)]
    s_out, z_out = asym(*ranges["a4"])
    zps = [xz] + [np.uint8(0)] * 4
    inits = [_tensor("xs", np.array(xs, np.float32)), _tensor("xz", np.array(xz, np.uint8)),
             _tensor("z0", np.array(0, np.uint8)), _tensor("s_out", np.array(s_out, np.float32)),
             _tensor("z_out", np.array(z_out, np.uint8))]
    for i in range(1, 5):
        inits.append(_tensor(f"s{i}", np.array(scales[i], np.float32)))
    nodes = [_node("QuantizeLinear", ["x", "xs", "xz"], ["q0"])]
    x, info = "q0", {"input scale": float(xs), "input zero point": int(xz)}
    for i, c in enumerate(cs):
        w = nchw_weights(c)
        ws = (np.abs(w).reshape(c.cout, -1).max(1) / 127.0).astype(np.float32)
        ws[ws == 0] = 1.0
        wq = np.clip(np.round(w / ws[:, None, None, None]), -127, 127).astype(np.int8)
        s_in = scales[i]
        bq = np.round(c.b / (s_in * ws)).astype(np.int32)
        inits += [_tensor(f"w{i}", wq), _tensor(f"ws{i}", ws), _tensor(f"wz{i}", np.zeros(c.cout, np.int8)),
                  _tensor(f"b{i}", bq)]
        last = i == len(cs) - 1
        ys, yz = ("s_out", "z_out") if last else (f"s{i + 1}", "z0")
        xz_name = "xz" if i == 0 else "z0"
        xs_name = "xs" if i == 0 else f"s{i}"
        y = "q_out" if last else f"q{i + 1}"
        nodes.append(_node("QLinearConv", [x, xs_name, xz_name, f"w{i}", f"ws{i}", f"wz{i}", ys, yz, f"b{i}"], [y],
                           conv_attrs(c)))
        x = y
        if i == 1:
            nodes.append(_node("MaxPool", [x], ["qp"], [_attr_ints("kernel_shape", [2, 2]), _attr_ints("strides", [2, 2])]))
            x = "qp"
    nodes.append(_node("DequantizeLinear", [x, "s_out", "z_out"], ["logits"]))
    inputs, outputs = io()
    return _model(nodes, inits, inputs, outputs), info


def session(path, threads=1):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = threads
    o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    o.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def port_lines(port):
    meta = dict(l.split(" ", 1) for l in open(os.path.join(port, "meta.txt")).read().splitlines())
    n, h, w = (int(v) for v in meta["lines"].split())
    px = np.fromfile(os.path.join(port, "lines.u8"), np.uint8).reshape(n, h, w, 3)
    truth = open(os.path.join(port, "truth.txt")).read().splitlines()
    kind = open(os.path.join(port, "kind.txt")).read().splitlines()
    x = np.ascontiguousarray(linenet.to_input(px).transpose(0, 3, 1, 2))
    return x, truth, kind


def read_all(sess, x):
    return linenet.greedy(np.concatenate([sess.run(["logits"], {"x": x[i:i + 1]})[0][:, :, 0, :].transpose(0, 2, 1)
                                          for i in range(len(x))]))


def build(a):
    os.makedirs(a.out, exist_ok=True)
    net = linenet.LineNet.load(a.teacher)
    # calibration: training lines only (made from the training glyphs, and
    # real frames of the training runs)
    bank = expsynth.Bank(a.bank, "train")
    mx, _ = expsynth.make(bank, 600, np.random.default_rng(21), places=[])
    rx, _, _ = teach.real_frames(a.lines, a.labels, "train")
    pick = np.random.default_rng(22).choice(len(rx), min(400, len(rx)), replace=False)
    cal = linenet.to_input(np.concatenate([mx, rx[np.sort(pick)]]))
    ranges = calibrate(net, cal)
    fp32 = fp32_model(net)
    int8, info = int8_model(net, ranges)
    out = {"calibration ranges": ranges, **info}
    for name, blob in (("fp32", fp32), ("int8", int8)):
        path = os.path.join(a.out, f"teacher-{name}.onnx")
        open(path, "wb").write(blob)
        out[name] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    x, truth, kind = port_lines(a.port)
    ref = open(os.path.join(a.port, "py-teacher.txt")).read().splitlines()
    nhwc = np.ascontiguousarray(x.transpose(0, 2, 3, 1))
    want = linenet.logits_of(net, nhwc[:64])
    got = session(os.path.join(a.out, "teacher-fp32.onnx")).run(["logits"], {"x": x[:1]})[0][0, :, 0, :].T
    out["fp32: max |score - NumPy's| on a line"] = float(np.abs(got - want[0]).max())
    for name in ("fp32", "int8"):
        s = session(os.path.join(a.out, f"teacher-{name}.onnx"))
        texts = read_all(s, x)
        out[name]["same as NumPy's teacher"] = f"{sum(t == r for t, r in zip(texts, ref))}/{len(ref)}"
        for k in ("made", "real"):
            idx = [i for i, kk in enumerate(kind) if kk == k]
            out[name][f"exact, {k}"] = f"{sum(texts[i] == truth[i] for i in idx)}/{len(idx)}"
    json.dump(out, open(os.path.join(a.out, "build.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


def time_cmd(a):
    s = session(a.model, 1)
    x, _, kind = port_lines(a.port)
    per = {"made": [], "real": []}
    for i in range(len(x)):
        xi = x[i:i + 1]
        for _ in range(3):
            s.run(["logits"], {"x": xi})
        ns = []
        for _ in range(5):
            t0 = time.perf_counter_ns()
            s.run(["logits"], {"x": xi})
            ns.append(time.perf_counter_ns() - t0)
        per[kind[i]].append(int(np.median(ns)))
    json.dump(per, open(a.out, "w"))


def bench(a):
    """`time` for each model in a process of its own, `--rounds` rounds in
    rotated order; p50 and p95 over the lines of each kind, the median over
    the rounds."""
    import subprocess
    names = ["fp32", "int8"]
    per = {n: {"made": [], "real": []} for n in names}
    for i in range(a.rounds):
        for n in names[i % 2:] + names[:i % 2]:
            out = os.path.join(a.dir, f"time-{n}-{i}.json")
            env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            subprocess.run([sys.executable, __file__, "time", "--model", os.path.join(a.dir, f"teacher-{n}.onnx"),
                            "--port", a.port, "--out", out], check=True, env=env)
            got = json.load(open(out))
            for k in ("made", "real"):
                per[n][k].append(got[k])
            print(f"round {i + 1} {n}: p50 made {np.percentile(got['made'], 50) / 1e3:.1f} us", flush=True)
    import onnxruntime as ort
    summary = {"onnxruntime": ort.__version__, "rounds": a.rounds}
    for n, kinds in per.items():
        summary[n] = {k: {"p50 us": round(float(np.median([np.percentile(r, 50) for r in runs])) / 1e3, 2),
                          "p95 us": round(float(np.median([np.percentile(r, 95) for r in runs])) / 1e3, 2)}
                      for k, runs in kinds.items()}
    json.dump(summary, open(os.path.join(a.dir, "bench.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    bb = sub.add_parser("bench")
    for arg in ("--dir", "--port"):
        bb.add_argument(arg, required=True)
    bb.add_argument("--rounds", type=int, default=5)
    b = sub.add_parser("build")
    for arg in ("--teacher", "--lines", "--labels", "--bank", "--port", "--out"):
        b.add_argument(arg, required=True)
    t = sub.add_parser("time")
    for arg in ("--model", "--port", "--out"):
        t.add_argument(arg, required=True)
    a = p.parse_args()
    {"build": build, "time": time_cmd, "bench": bench}[a.cmd](a)


if __name__ == "__main__":
    main()
