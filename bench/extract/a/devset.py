"""Task A dev set from the video's runs: labels, suggestions, splits.

    devset.py train   --lines L.npz --runs R.json --labels LAB.json --out NET.npz [--epochs N]
    devset.py suggest --lines L.npz --runs R.json --net NET.npz --out SUG.json
    devset.py sheet   --lines L.npz --runs R.json --suggest SUG.json --field mp --out DIR

A run's label is read by a person (here: Claude, looking at the run's mean
picture enlarged) and is never taken from a reader under test. A network
trained on a few such labels only *suggests* labels for the other runs; a
suggestion becomes a label when it is checked on a sheet, as DATA.md's
protocol asks of every line.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import linenet  # noqa: E402

FIELDS = ("hp", "mp", "exp")


def load(lines_path, runs_path):
    z = np.load(lines_path)
    data = {k: z[k] for k in z.files}  # decompressed once: z[k] decompresses on every access
    runs = json.load(open(runs_path))
    return data, runs


def gather(data, runs, labels, fields=FIELDS):
    """Frames of the labelled runs: x, texts, (field, run, frame) keys."""
    xs, texts, keys = [], [], []
    for f in fields:
        for run, text in labels.get(f, {}).items():
            a, b = runs[f][int(run)]
            for i in range(a, b):
                xs.append(data[f][i].copy())
                texts.append(text)
                keys.append((f, int(run), i))
    lines = [linenet.to_input(x[None])[0] for x in xs]
    return np.stack(lines), texts, keys


def train_cmd(args):
    data, runs = load(args.lines, args.runs)
    labels = json.load(open(args.labels))
    x, texts, _ = gather(data, runs, labels)
    print(json.dumps({"frames": len(x)}), flush=True)
    net = linenet.LineNet(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    linenet.train(net, x, texts, args.epochs, rng, lr=args.lr)
    net.save(args.out, {"labels": args.labels, "epochs": args.epochs, "frames": len(x)})


def suggest_cmd(args):
    data, runs = load(args.lines, args.runs)
    net = linenet.LineNet.load(args.net)
    out = {}
    for f in FIELDS:
        x = linenet.to_input(data[f])
        logits = linenet.logits_of(net, x)
        reads = [linenet.decode_path(p) for p in logits.argmax(-1)]
        per_run = []
        for k, (a, b) in enumerate(runs[f]):
            votes = {}
            for i in range(a, b):
                votes[reads[i]] = votes.get(reads[i], 0) + 1
            best = max(votes, key=votes.get)
            # the run's mean picture, read too
            mean = data[f][a:b].astype(np.float32).mean(0).round().astype(np.uint8)
            mean_read = linenet.predict(net, linenet.to_input(mean[None]))[0]
            per_run.append({"run": k, "frames": b - a, "vote": best, "share": votes[best] / (b - a),
                            "mean": mean_read, "readings": len(votes)})
        out[f] = per_run
        agree = np.mean([r["vote"] == r["mean"] and r["share"] == 1.0 for r in per_run])
        print(f, "runs", len(per_run), "unanimous and equal to the mean's reading", round(float(agree), 3))
    json.dump(out, open(args.out, "w"), indent=0)


def sheet_cmd(args):
    import cv2
    data, runs = load(args.lines, args.runs)
    sug = json.load(open(args.suggest))[args.field]
    os.makedirs(args.out, exist_ok=True)
    xs = (40, 140) if args.field != "exp" else (55, 200)
    picks = [r for r in sug if args.min_frames <= r["frames"]]
    if args.only:
        picks = [r for r in picks if r["vote"] == args.only]
    rows = []
    for r in picks:
        a, b = runs[args.field][r["run"]]
        img = data[args.field][a:b].astype(np.float32).mean(0)[:, xs[0]:xs[1]]
        lum = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        lo, hi = np.percentile(lum, 2), np.percentile(lum, 99.5)
        g = np.clip((lum - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
        z = cv2.resize(g, None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((z.shape[0] + 4, z.shape[1] + 330), np.uint8)
        canvas[: z.shape[0], : z.shape[1]] = z
        tag = f"{r['run']}: {r['vote']}" + ("" if r["share"] == 1 else f" ({r['share']:.0%})")
        cv2.putText(canvas, tag, (z.shape[1] + 6, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, 255, 2)
        rows.append(canvas)
    per = args.per_sheet
    for s in range(0, len(rows), per):
        cv2.imwrite(os.path.join(args.out, f"{args.field}-{s // per:03d}.png"), np.vstack(rows[s:s + per]))
    print(len(rows), "rows,", (len(rows) + per - 1) // per, "sheets")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    for a in ("--lines", "--runs", "--labels", "--out"):
        t.add_argument(a, required=True)
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("suggest")
    for a in ("--lines", "--runs", "--net", "--out"):
        s.add_argument(a, required=True)
    h = sub.add_parser("sheet")
    for a in ("--lines", "--runs", "--suggest", "--field", "--out"):
        h.add_argument(a, required=True)
    h.add_argument("--min-frames", type=int, default=1)
    h.add_argument("--per-sheet", type=int, default=24)
    h.add_argument("--only", default=None)
    args = p.parse_args()
    {"train": train_cmd, "suggest": suggest_cmd, "sheet": sheet_cmd}[args.cmd](args)


if __name__ == "__main__":
    main()
