"""Gate G1 of PREREGISTRATION.md, on two separate measurements of the same
implementations: do they give the same outputs, does p50 move by less than 5%,
and is each p95 known to within ±10%?

    python3 bench/extract/repeat_check.py RESULTS_A RESULTS_B [--task glyphs]

Each RESULTS directory is what `bench/vs-opencv/run.py measure` wrote. Times
are pooled over every timed call of every round. The p95 interval is a 95%
bootstrap over inputs (10,000 resamples), within RESULTS_A.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

BOOTSTRAP = 10_000
RNG_SEED = 20261012


def load(results: str, task: str) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(results, "raw", f"{task}.*.r*.json"))):
        impl = os.path.basename(path)[len(task) + 1:].rsplit(".r", 1)[0]
        with open(path) as f:
            runs.setdefault(impl, []).append(json.load(f))
    return runs


def pooled(runs: list[dict]) -> tuple[np.ndarray, list[np.ndarray]]:
    per_input: dict[str, list[int]] = {}
    for r in runs:
        for item in r["items"]:
            per_input.setdefault(item["input"], []).extend(item["times_ns"])
    clusters = [np.array(v) / 1e6 for _, v in sorted(per_input.items())]
    return np.concatenate(clusters), clusters


def p95_interval(clusters: list[np.ndarray], rng) -> list[float]:
    n = len(clusters)
    stats = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, n, n)
        stats.append(np.percentile(np.concatenate([clusters[i] for i in pick]), 95))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--task", default="glyphs")
    ap.add_argument("--out")
    args = ap.parse_args()
    ra, rb = load(args.a, args.task), load(args.b, args.task)
    rng = np.random.default_rng(RNG_SEED)
    rows = {}
    for impl in sorted(set(ra) & set(rb)):
        out_a = {i["input"]: i["output"] for i in ra[impl][0]["items"]}
        out_b = {i["input"]: i["output"] for i in rb[impl][0]["items"]}
        ta, ca = pooled(ra[impl])
        tb, _ = pooled(rb[impl])
        p50a, p50b = float(np.percentile(ta, 50)), float(np.percentile(tb, 50))
        p95a, p95b = float(np.percentile(ta, 95)), float(np.percentile(tb, 95))
        lo, hi = p95_interval(ca, rng)
        half = (hi - lo) / 2 / p95a
        rows[impl] = {
            "same_outputs": out_a == out_b,
            "p50_ms": [p50a, p50b], "p50_change": p50b / p50a - 1,
            "p95_ms": [p95a, p95b], "p95_change": p95b / p95a - 1,
            "p95_ci95_a": [lo, hi], "p95_half_width": half,
            "g1": bool(out_a == out_b and abs(p50b / p50a - 1) < 0.05 and half < 0.10),
        }
    lines = [f"Task `{args.task}`: run A `{args.a}`, run B `{args.b}`.", "",
             "| implementation | same outputs | p50 A → B (ms) | p50 change | p95 A → B (ms) | p95 interval in A | ± of p95 | G1 |",
             "|---|---|---|---|---|---|---|---|"]
    for impl, r in rows.items():
        lines.append(f"| {impl} | {'yes' if r['same_outputs'] else 'no'} | {r['p50_ms'][0]:.3f} → {r['p50_ms'][1]:.3f} "
                     f"| {r['p50_change'] * 100:+.1f}% | {r['p95_ms'][0]:.3f} → {r['p95_ms'][1]:.3f} "
                     f"| {r['p95_ci95_a'][0]:.3f}–{r['p95_ci95_a'][1]:.3f} | {r['p95_half_width'] * 100:.1f}% "
                     f"| {'passes' if r['g1'] else 'fails'} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=1)
        with open(os.path.splitext(args.out)[0] + ".md", "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
