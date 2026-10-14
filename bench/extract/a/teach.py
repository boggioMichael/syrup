"""Train the teacher: the line reader, on made EXP lines and the video's own.

    teach.py --lines L.npz --labels EXP-LABELS.json --bank BANK.npz --out NET.npz

Each epoch draws fresh lines from the bank's training glyphs (any EXP value,
so the network has to read rather than recognise the video's 15 values) and
adds the real frames of the training runs. After each epoch it reads the
validation runs' real frames and a fixed set of made lines from the
validation glyphs, and keeps the network that read the real ones best.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expsynth  # noqa: E402
import linenet  # noqa: E402


def real_frames(lines_path, labels_path, split):
    exp = np.load(lines_path)["exp"]
    lo, hi = expsynth.SPLITS[split]
    xs, texts, segs = [], [], []
    for item in json.load(open(labels_path)):
        if lo <= item["segment"] < hi:
            a, b = item["frames"]
            xs.append(exp[a:b])
            texts += [item["label"]] * (b - a)
            segs += [item["segment"]] * (b - a)
    return np.concatenate(xs), texts, np.array(segs)


def exact(net, x, texts):
    got = linenet.predict(net, x)
    return float(np.mean([a == b for a, b in zip(got, texts)])), got


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lines", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--made", type=int, default=12000, help="made lines per epoch")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init", default=None, help="start from this network's weights")
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    train_bank = expsynth.Bank(args.bank, "train")
    val_bank = expsynth.Bank(args.bank, "val")
    rx, rt, _ = real_frames(args.lines, args.labels, "train")
    vx, vt, _ = real_frames(args.lines, args.labels, "val")
    mx, mt = expsynth.make(val_bank, 1000, np.random.default_rng(12345))
    real_x, val_x, made_x = linenet.to_input(rx), linenet.to_input(vx), linenet.to_input(mx)
    print(json.dumps({"real train frames": len(rx), "real val frames": len(vx), "made val lines": len(mx)}), flush=True)
    net = linenet.LineNet.load(args.init) if args.init else linenet.LineNet(seed=args.seed)
    print(json.dumps({"parameters": net.count()}), flush=True)
    best, opt = -1.0, None
    log = open(args.out.replace(".npz", ".log"), "w")
    for epoch in range(args.epochs):
        t0 = time.time()
        sx, st = expsynth.make(train_bank, args.made, rng)
        x = np.concatenate([linenet.to_input(sx), real_x])
        texts = st + rt
        opt = linenet.train(net, x, texts, 1, rng, lr=args.lr, schedule=(epoch, args.epochs), opt=opt)
        real_acc, _ = exact(net, val_x, vt)
        made_acc, _ = exact(net, made_x, mt)
        msg = {"epoch": epoch + 1, "val real exact": round(real_acc, 4), "val made exact": round(made_acc, 4),
               "seconds": round(time.time() - t0, 1)}
        print(json.dumps(msg), flush=True)
        log.write(json.dumps(msg) + "\n")
        log.flush()
        if real_acc + made_acc > best:
            best = real_acc + made_acc
            net.save(args.out, {"epoch": epoch + 1, **msg})


if __name__ == "__main__":
    main()
