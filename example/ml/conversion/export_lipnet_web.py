"""LipNet for the browser: one binary of int8 weights (batch norm folded
into the convolutions, one scale per tensor) plus a JSON header, for the
plain-JavaScript engine in example/web-live. 4.6 MB instead of 18.

    python3 export_lipnet_web.py --out lipnet-web.bin [--check]

`--check` runs the numpy network with the folded, quantised weights on the
GRID sample clips and prints the words right (64/66 with the float
weights; the int8 export reads the same 64).

Also writes, for the engine's own test, the crops and the reference
probabilities of one clip (`--reference sbwe5n`).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THELIP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "python", "thelip"))
sys.path.insert(0, THELIP)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "lipreader")))
from lipnet_np import LipNet, Spell, greedy_decode, grid_sentence  # noqa: E402
from lipreader.vsr.lipnet import DICTIONARY, WEIGHTS, lipnet_dir  # noqa: E402


def folded(net: LipNet) -> dict:
    """Batch norm folded into each convolution: w' = w * g/sqrt(v+eps), b' = (b - m) * g/sqrt(v+eps) + beta."""
    out = {}
    for i, layer in enumerate(net.conv, 1):
        scale = layer["gamma"] / np.sqrt(layer["var"] + 1e-3)
        out[f"conv{i}_kernel"] = (layer["kernel"] * scale).astype(np.float32)          # (kt, kw, kh, cin, cout) * (cout,)
        out[f"conv{i}_bias"] = ((layer["bias"] - layer["mean"]) * scale + layer["beta"]).astype(np.float32)
    for i, layer in enumerate(net.gru, 1):
        for d in ("fw", "bw"):
            k, r, b = layer[d]
            out[f"gru{i}_{d}_kernel"], out[f"gru{i}_{d}_recurrent"], out[f"gru{i}_{d}_bias"] = k, r, b
    out["dense_kernel"], out["dense_bias"] = net.dense
    return out


def quantise(a: np.ndarray):
    scale = float(np.abs(a).max()) / 127.0 or 1.0
    q = np.round(a / scale).clip(-127, 127).astype(np.int8)
    return q, scale


def write(tensors: dict, path: str) -> dict:
    header = {"format": "lipnet-web-int8", "tensors": []}
    blob = bytearray()
    for name, a in tensors.items():
        q, scale = quantise(a)
        header["tensors"].append({"name": name, "shape": list(a.shape), "scale": scale, "offset": len(blob), "size": q.size})
        blob += q.tobytes()
    head = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(b"LIPW")
        f.write(struct.pack("<I", len(head)))
        f.write(head)
        f.write(blob)
    return header


def net_with(net: LipNet, tensors: dict) -> LipNet:
    """The numpy network running on the folded, quantised tensors (BN becomes identity)."""
    import copy

    q = copy.deepcopy(net)
    deq = {}
    for name, a in tensors.items():
        qa, scale = quantise(a)
        deq[name] = qa.astype(np.float32) * scale
    for i, layer in enumerate(q.conv, 1):
        layer["kernel"], layer["bias"] = deq[f"conv{i}_kernel"], deq[f"conv{i}_bias"]
        layer["gamma"] = np.ones_like(layer["gamma"]); layer["beta"] = np.zeros_like(layer["beta"])
        layer["mean"] = np.zeros_like(layer["mean"]); layer["var"] = np.ones_like(layer["var"]) - 1e-3
    for i, layer in enumerate(q.gru, 1):
        for d in ("fw", "bw"):
            layer[d] = (deq[f"gru{i}_{d}_kernel"], deq[f"gru{i}_{d}_recurrent"], deq[f"gru{i}_{d}_bias"])
    q.dense = (deq["dense_kernel"], deq["dense_bias"])
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lipnet", default=lipnet_dir())
    ap.add_argument("--out", default="lipnet-web.bin")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--reference", default=None, help="GRID clip code: write <out>.<code>.crops.f32 and .probs.f32")
    args = ap.parse_args()
    if not args.lipnet:
        sys.exit("no LipNet checkout (example/ml/models/get_lipnet.sh)")
    net = LipNet(os.path.join(args.lipnet, WEIGHTS))
    tensors = folded(net)
    header = write(tensors, args.out)
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}: {len(header['tensors'])} tensors, {size / 1e6:.1f} MB")
    q = net_with(net, tensors)
    if args.check or args.reference:
        import glob

        from mouth import mouth_crops, read_frames

        spell = Spell(os.path.join(args.lipnet, DICTIONARY))
        samples = sorted(glob.glob(f"{args.lipnet}/evaluation/samples/GRID/*.mpg")) + sorted(glob.glob(f"{args.lipnet}/evaluation/samples/*.mpg"))
        right = total = 0
        for path in samples:
            code = os.path.basename(path)[:-4]
            crops, _, _ = mouth_crops(read_frames(path)[0])
            probs = q.predict(crops)
            if args.reference and code == args.reference:
                base = os.path.splitext(args.out)[0]
                crops.astype(np.float32).tofile(f"{base}.{code}.crops.f32")
                probs.astype(np.float32).tofile(f"{base}.{code}.probs.f32")
                print(f"reference {code}: crops {crops.shape}, probs {probs.shape}, decode {spell.sentence(greedy_decode(probs))!r}")
            if args.check:
                text = spell.sentence(greedy_decode(probs))
                truth = grid_sentence(code)
                right += sum(1 for a, b in zip(truth.split(), text.split()) if a == b)
                total += len(truth.split())
        if args.check:
            print(f"int8 + folded BN: {right}/{total} words on the GRID sample clips")


if __name__ == "__main__":
    main()
