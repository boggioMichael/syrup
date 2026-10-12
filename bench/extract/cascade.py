"""Explicit code first, a fallback behind it: the system measures of
PREREGISTRATION.md, from the per-item results `bench/vs-opencv/run.py measure`
keeps in RESULTS/raw.

    python3 bench/extract/cascade.py --data DATA --results RESULTS \
        --explicit syrup-after --fallback opencv --fallback tinynet [--task glyphs]

For every item: the explicit implementation answers or refuses; when it
refuses, the fallback answers. Reported, with 95% bootstrap intervals over
items (10,000 resamples):

- coverage: items the explicit implementation answers, of all items
- system correct / wrong / refused: of all items
- expected cost per item: the explicit cost, plus the fallback's cost on the
  items the explicit implementation refused (each item's cost is the median of
  all its timed calls, over all rounds, in that implementation's own process)
- the fallback alone, and the ratio of its mean cost to the system's

Outputs come from the first round, as in `run.py summarise`; the script stops
if an implementation's outputs differ between rounds.
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


def load_runs(results: str, task: str) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(os.path.join(results, "raw", f"{task}.*.r*.json"))):
        impl = os.path.basename(path)[len(task) + 1:].rsplit(".r", 1)[0]
        with open(path) as f:
            runs.setdefault(impl, []).append(json.load(f))
    return runs


def glyph_outcome(output: dict, case: dict) -> str:
    """The rule of `run.py summarise`: whitespace is ignored, anything else
    must match the label exactly."""
    if "refused" in output:
        return "refused"
    return "correct" if "".join(output["text"].split()) == case["text"] else "wrong"


def per_item(runs: list[dict], cases: dict[str, dict]) -> tuple[list[str], np.ndarray, list[str]]:
    first = {item["input"]: item["output"] for item in runs[0]["items"]}
    for r in runs[1:]:
        if {item["input"]: item["output"] for item in r["items"]} != first:
            raise SystemExit(f"{runs[0]['impl']}: outputs differ between rounds")
    names = sorted(first)
    times: dict[str, list[int]] = {n: [] for n in names}
    for r in runs:
        for item in r["items"]:
            times[item["input"]].extend(item["times_ns"])
    cost_ms = np.array([np.median(times[n]) / 1e6 for n in names])
    outcomes = [glyph_outcome(first[n], cases[n]) for n in names]
    return names, cost_ms, outcomes


def interval(values: np.ndarray, rng: np.random.Generator, stat=np.mean) -> list[float]:
    n = len(values)
    picks = rng.integers(0, n, size=(BOOTSTRAP, n))
    stats = stat(values[picks], axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def ratio_interval(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Mean of `a` over mean of `b`, the same items resampled for both."""
    n = len(a)
    picks = rng.integers(0, n, size=(BOOTSTRAP, n))
    r = a[picks].mean(axis=1) / b[picks].mean(axis=1)
    lo, hi = np.percentile(r, [2.5, 97.5])
    return [float(lo), float(hi)]


def system(explicit, fallback, rng) -> dict:
    names_e, cost_e, out_e = explicit
    names_f, cost_f, out_f = fallback
    if names_e != names_f:
        raise SystemExit("the two implementations saw different items")
    refused = np.array([o == "refused" for o in out_e])
    sys_out = [f if r else e for e, f, r in zip(out_e, out_f, refused)]
    sys_cost = cost_e + refused * cost_f
    n = len(names_e)

    def share(outs, kind):
        v = np.array([o == kind for o in outs], float)
        return {"count": int(v.sum()), "rate": float(v.mean()), "ci95": interval(v, rng)}

    gain = (np.array([o == "correct" for o in sys_out], float) - np.array([o == "correct" for o in out_f], float))
    return {
        "items": n,
        "system_minus_fallback_correct": {"points": float(gain.mean()), "ci95": interval(gain, rng)},
        "coverage": share([("refused" if r else "answered") for r in refused], "answered"),
        "explicit": {k: share(out_e, k) for k in ("correct", "wrong", "refused")},
        "system": {k: share(sys_out, k) for k in ("correct", "wrong", "refused")},
        "fallback_alone": {k: share(out_f, k) for k in ("correct", "wrong", "refused")},
        "cost_ms": {
            "explicit_mean": float(cost_e.mean()),
            "fallback_alone_mean": float(cost_f.mean()), "fallback_alone_ci95": interval(cost_f, rng),
            "system_mean": float(sys_cost.mean()), "system_ci95": interval(sys_cost, rng),
            "fallback_alone_over_system": float(cost_f.mean() / sys_cost.mean()),
            "fallback_alone_over_system_ci95": ratio_interval(cost_f, sys_cost, rng),
        },
    }


def markdown(report: dict) -> str:
    def pct(s):
        return f"{s['rate'] * 100:.1f}% [{s['ci95'][0] * 100:.1f}–{s['ci95'][1] * 100:.1f}]"
    lines = [f"Task `{report['task']}`, split `{report['split']}`, {report['items']} items. "
             "95% bootstrap intervals over items.", "",
             "| explicit → fallback | coverage | system correct | system wrong | fallback alone: correct | fallback alone: wrong "
             "| correct gain, points (paired) | system ms/item | fallback alone ms/item | fallback ÷ system cost |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for pair, r in report["pairs"].items():
        c = r["cost_ms"]
        g = r["system_minus_fallback_correct"]
        lines.append(f"| {pair} | {pct(r['coverage'])} | {pct(r['system']['correct'])} | {pct(r['system']['wrong'])} "
                     f"| {pct(r['fallback_alone']['correct'])} | {pct(r['fallback_alone']['wrong'])} "
                     f"| {g['points'] * 100:+.1f} [{g['ci95'][0] * 100:+.1f}, {g['ci95'][1] * 100:+.1f}] "
                     f"| {c['system_mean']:.3f} [{c['system_ci95'][0]:.3f}–{c['system_ci95'][1]:.3f}] "
                     f"| {c['fallback_alone_mean']:.3f} | {c['fallback_alone_over_system']:.2f} "
                     f"[{c['fallback_alone_over_system_ci95'][0]:.2f}–{c['fallback_alone_over_system_ci95'][1]:.2f}] |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--task", default="glyphs", choices=["glyphs"])
    ap.add_argument("--explicit", required=True)
    ap.add_argument("--fallback", action="append", required=True)
    ap.add_argument("--out", help="JSON file to write; also writes the .md beside it")
    a = ap.parse_args()
    with open(os.path.join(a.results, "environment.json")) as f:
        split = json.load(f)["split"]
    with open(os.path.join(a.data, a.task, split, "cases.json")) as f:
        cases = {c["image"]: c for c in json.load(f)["cases"]}
    runs = load_runs(a.results, a.task)
    rng = np.random.default_rng(RNG_SEED)
    explicit = per_item(runs[a.explicit], cases)
    report = {"task": a.task, "split": split, "items": len(explicit[0]), "explicit": a.explicit, "pairs": {}}
    for fb in a.fallback:
        report["pairs"][f"{a.explicit} → {fb}"] = system(explicit, per_item(runs[fb], cases), rng)
    text = markdown(report)
    print(text)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=1)
        with open(os.path.splitext(a.out)[0] + ".md", "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
