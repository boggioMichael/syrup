"""Summarise the lip-reading runs (F1-pipeline in PREREGISTRATION.md).

    python3 bench/vs-opencv/lipreader_summary.py RESULTS/lipreader

Reads `<detector>.r<round>.json` (the output of `lipreader evaluate --json`)
and writes `summary.json` and `summary.md` beside them: per fixture, the
median and range of ms per frame over the runs, the ratio to OpenCV, and
whether every quality column is the same as with the detector before.
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

QUALITY = ["wer", "cer", "words_hyp", "attribution_accuracy", "track_consistency", "tracks", "people",
           "speaking_precision", "speaking_recall", "cut_recall"]


def quality(row):
    return {k: row.get(k) for k in QUALITY} | {"per_person": row["detail"]["per_person"]}


def main():
    directory = sys.argv[1]
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(directory, "*.r*.json"))):
        impl = os.path.basename(path).rsplit(".r", 1)[0]
        with open(path) as f:
            runs[impl].append({row["name"]: row for row in json.load(f)})
    impls = sorted(runs)
    fixtures = sorted(runs[impls[0]][0])
    table, summary = [], {"fixtures": {}}
    for name in fixtures:
        entry = {}
        for impl in impls:
            ms = [r[name]["latency_ms_per_frame"] for r in runs[impl]]
            detect = [r[name]["detail"]["stages"]["detect"] for r in runs[impl]]
            entry[impl] = {"ms_per_frame": ms, "median": statistics.median(ms), "range": [min(ms), max(ms)],
                           "detect_s_median": statistics.median(detect),
                           "quality_same_every_run": all(quality(r[name]) == quality(runs[impl][0][name]) for r in runs[impl])}
        if "syrup-before" in runs:
            base = quality(runs["syrup-before"][0][name])
            for impl in impls:
                entry[impl]["quality_same_as_syrup_before"] = quality(runs[impl][0][name]) == base
        if "opencv" in runs:
            for impl in impls:
                entry[impl]["over_opencv"] = entry[impl]["median"] / entry["opencv"]["median"]
        summary["fixtures"][name] = entry
    lines = ["# Lip-reading pipeline (F1-pipeline)", "",
             f"{len(runs[impls[0]])} runs per detector, rotated order; ms per frame of the whole pipeline "
             "(detection, tracking, mouth crops, LipNet), median [range]; × = over OpenCV's median.", "",
             "| fixture | " + " | ".join(impls) + " | quality columns as with syrup-before |",
             "|---|" + "---|" * len(impls) + "---|"]
    for name, entry in summary["fixtures"].items():
        cells = []
        for impl in impls:
            e = entry[impl]
            ratio = f" ({e['over_opencv']:.2f}×)" if "over_opencv" in e and impl != "opencv" else ""
            cells.append(f"{e['median']:.1f} [{e['range'][0]:.1f}–{e['range'][1]:.1f}]{ratio}")
        same = ", ".join(f"{i}: {'yes' if entry[i].get('quality_same_as_syrup_before') else 'NO'}"
                         for i in impls if i.startswith("syrup-after"))
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {same} |")
    lines += ["", "Quality columns (first run; every run gave the same: "
              + ", ".join(f"{i} {'yes' if all(summary['fixtures'][n][i]['quality_same_every_run'] for n in fixtures) else 'NO'}" for i in impls)
              + ")", "", "| fixture | detector | WER | CER | attribution | track consistency | speaking P | speaking R |",
              "|---|---|---|---|---|---|---|---|"]
    for name in fixtures:
        for impl in impls:
            r = runs[impl][0][name]
            lines.append(f"| {name} | {impl} | {r['wer']} | {r['cer']} | {r['attribution_accuracy']} | "
                         f"{r['track_consistency']} | {r['speaking_precision']} | {r['speaking_recall']} |")
    with open(os.path.join(directory, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(directory, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
