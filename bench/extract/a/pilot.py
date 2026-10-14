"""Pilot A: explicit readers built from the teacher, measured.

    pilot.py --lines L.npz --labels EXP-LABELS.json --bank BANK.npz --teacher NET.npz --out DIR

1. Lines to learn from: made lines from the training glyphs and real frames
   of the training runs. The teacher reads them; nothing else labels them.
2. The explicit reader of `extract.py`, built three ways:
   - white box: places from the teacher's insides (`extract.align_white`);
   - black box: places from the lines and the teacher's strings alone
     (`extract.align_black`);
   - oracle: the renderer's own places, for made lines (a bound, not a reader
     anyone could build).
3. A reader refuses a line whose reading is not an EXP value (`EXP`: digits
   in threes with commas, a bracket, a percentage with two or three decimals,
   the closing bracket), and one whose confidence is under its level, fitted
   on the validation lines (real frames of the validation runs and made lines
   from the validation glyphs): the lowest level at which at most 1% of the
   lines it answers there are wrong. (The first pilot took the level above
   every wrong read and had no format rule: one confident mistake — a dropped
   '.', say, read with every glyph clear — set it so high that the readers
   answered almost nothing.) The teacher is shown with the same format rule.
4. Everything is measured on the test lines (real frames of the test runs and
   made lines from the test glyphs, which no reader saw), with the time each
   reader takes per line.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expsynth  # noqa: E402
import extract  # noqa: E402
import linenet  # noqa: E402
import teach  # noqa: E402

SIZES = {"train": (1500, 800), "val": (600, None), "test": (1000, None)}  # made lines, real frames


def data(args):
    out = {}
    for split, (made_n, real_n) in SIZES.items():
        bank = expsynth.Bank(args.bank, split)
        places = []
        mx, mt = expsynth.make(bank, made_n, np.random.default_rng({"train": 1, "val": 2, "test": 3}[split]), places=places)
        rx, rt, segs = teach.real_frames(args.lines, args.labels, split)
        if real_n and len(rx) > real_n:
            pick = np.sort(np.random.default_rng(4).choice(len(rx), real_n, replace=False))
            rx, rt, segs = rx[pick], [rt[i] for i in pick], segs[pick]
        out[split] = {"made": (mx, mt, places), "real": (rx, rt, segs)}
    return out


def reads(reader, lines):
    g = extract.gray(lines)
    return [reader.read(gl) for gl in g]


MAX_WRONG = 0.01  # of the lines a reader answers, on validation
EXP = re.compile(r"^\d{1,3}(,\d{3})*\[\d{1,3}\.\d{2,3}%\]$")


def answerable(found):
    """Each reading's confidence, or -inf for one that is not an EXP value."""
    return [(t, c if EXP.match(t) else -np.inf) for t, c in found]


def fit_level(found, truths, max_wrong=MAX_WRONG):
    """The lowest confidence level at which, of the validation lines it
    answers, at most `max_wrong` are wrong (inf: none)."""
    pairs = sorted(((c, t == x) for (t, c), x in zip(found, truths)), key=lambda p: -p[0])
    level, wrong = np.inf, 0
    for i, (c, ok) in enumerate(pairs, 1):
        wrong += not ok
        if wrong <= max_wrong * i:
            level = c
    return level


def score(found, truths, level, teacher):
    kept = [bool(np.isfinite(c) and c >= level) for _, c in answerable(found)]
    correct = sum(k and t == x for k, (t, _), x in zip(kept, found, truths))
    wrong = sum(k and t != x for k, (t, _), x in zip(kept, found, truths))
    agree = sum(k and t == r for k, (t, _), r in zip(kept, found, teacher))
    n = len(truths)
    exact = sum(t == x for (t, _), x in zip(found, truths))
    return {"lines": n, "exact before refusing": round(exact / n, 4), "correct": correct, "wrong": wrong, "refused": n - sum(kept),
            "coverage": round(sum(kept) / n, 4), "agrees with the teacher where it answers": round(agree / max(1, sum(kept)), 4)}


def timed(fn, items, rounds=3):
    best = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for it in items:
            fn(it)
        best.append((time.perf_counter() - t0) / len(items))
    return round(min(best) * 1000, 3)


def main():
    p = argparse.ArgumentParser()
    for a in ("--lines", "--labels", "--bank", "--teacher", "--out"):
        p.add_argument(a, required=True)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    net = linenet.LineNet.load(args.teacher)
    d = data(args)
    report = {"teacher parameters": net.count()}
    # 1. the lines to learn from, read by the teacher
    tr_lines = np.concatenate([d["train"]["made"][0], d["train"]["real"][0]])
    tr_teacher = linenet.predict(net, linenet.to_input(tr_lines))
    truth_tr = d["train"]["made"][1] + d["train"]["real"][1]
    report["teacher on its lines to learn from, exact"] = round(float(np.mean([a == b for a, b in zip(tr_teacher, truth_tr)])), 4)
    # 2. the readers
    readers, built = {}, {}
    t0 = time.time()
    white_places, info = extract.align_white_refined(net, tr_lines)
    readers["white box"] = extract.build(tr_lines, white_places)
    built["white box"] = {"seconds": round(time.time() - t0, 1), **{k: v for k, v in info.items()}}
    t0 = time.time()
    black_places, info = extract.align_black(tr_lines, tr_teacher)
    readers["black box"] = extract.build(tr_lines, black_places)
    built["black box"] = {"seconds": round(time.time() - t0, 1), "rounds": info["rounds"], "lines it could start from": info["started_from_lines"]}
    mx, mt, mplaces = d["train"]["made"]
    t0 = time.time()
    readers["oracle (renderer's places)"] = extract.build(mx, extract.align_truth(mt, mplaces))
    built["oracle (renderer's places)"] = {"seconds": round(time.time() - t0, 1)}
    report["built"] = built
    # 3. refusal levels on validation
    vlines = np.concatenate([d["val"]["made"][0], d["val"]["real"][0]])
    vtruth = d["val"]["made"][1] + d["val"]["real"][1]
    levels = {}
    for name, r in readers.items():
        found = reads(r, vlines)
        levels[name] = fit_level(answerable(found), vtruth)
        report[f"validation: {name}"] = {"exact before refusing": round(float(np.mean([t == x for (t, _), x in zip(found, vtruth)])), 4),
                                         "refusal level": None if not np.isfinite(levels[name]) else round(float(levels[name]), 4)}
    # 4. the test lines
    for kind in ("made", "real"):
        lines, truths = d["test"][kind][0], d["test"][kind][1]
        teacher = linenet.predict(net, linenet.to_input(lines))
        report[f"test {kind}: teacher"] = {"lines": len(truths), "correct": int(sum(a == b for a, b in zip(teacher, truths))),
                                           "wrong": int(sum(a != b for a, b in zip(teacher, truths))),
                                           "with the format rule": score([(t, 0.0) for t in teacher], truths, -np.inf, teacher)}
        for name, r in readers.items():
            report[f"test {kind}: {name}"] = score(reads(r, lines), truths, levels[name], teacher)
    # 5. time per line (this container, one thread, Python for the explicit readers)
    sample = d["test"]["made"][0][:100]
    g = extract.gray(sample)
    x = linenet.to_input(sample)
    report["ms per line"] = {"teacher (NumPy, a line at a time)": timed(lambda i: net.forward(x[i:i + 1], keep=False), range(len(x))),
                             **{f"{name} (Python)": timed(r.read, list(g)) for name, r in readers.items()}}
    report["numbers in each reader"] = {name: r.size() for name, r in readers.items()}
    report["numbers in the teacher"] = net.count()
    json.dump(report, open(os.path.join(args.out, "pilot.json"), "w"), indent=1)
    for name, r in readers.items():
        np.savez(os.path.join(args.out, name.split(" (")[0].replace(" ", "-") + ".npz"),
                 background=r.background, weights=r.weights, **{f"t{k}": v for k, v in r.templates.items()})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
