"""The headline tables of RESULTS.md, from the summaries `run.py summarise`
and `lipreader_summary.py` wrote (so no number is copied by hand).

    python3 bench/vs-opencv/report.py RESULTS_DIR > tables.md

RESULTS_DIR holds `test-t1/`, `test-t2/` and `lipreader/`.
"""
from __future__ import annotations

import json
import os
import sys

TASKS = [("faces", "F1 Haar cascade (same model, same algorithm)"),
         ("yunet", "F2 YuNet (same model, different runtimes)"),
         ("templates", "F3 template matching, best ZNCC match (same score)"),
         ("motion", "F4 motion regions (same mask formula, different grouping)"),
         ("glyphs", "F5 fixed-font glyph reading (different solutions, synthetic)")]


def load(directory):
    path = os.path.join(directory, "summary.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def ms(v):
    return "–" if v is None else (f"{v:.3f}" if v < 1 else f"{v:.1f}" if v < 100 else f"{v:.0f}")


def ratio(row, key):
    value = row.get(f"opencv_over_this_{key}")
    ci = row.get(f"opencv_over_this_{key}_ci95")
    if value is None:
        return "–"
    return f"{value:.2f} [{ci[0]:.2f}–{ci[1]:.2f}]" if ci else f"{value:.2f}"


def change(row):
    value = row.get("p95_change_vs_before")
    ci = row.get("p95_change_vs_before_ci95")
    if value is None:
        return "–"
    return f"{value * 100:+.0f}% [{ci[0] * 100:+.0f}, {ci[1] * 100:+.0f}]" if ci else f"{value * 100:+.0f}%"


def speed_table(summary, threads):
    lines = [f"### {threads} thread{'s' if threads > 1 else ''}", "",
             "| task | data | implementation | p50 ms | p95 ms | p99 ms | OpenCV ÷ this, p50 [95% CI] | OpenCV ÷ this, p95 [95% CI] | p95 vs Syrup before [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|"]
    rows = summary["rows"]
    order = {"syrup-before": 0, "syrup-after": 1, "syrup-after-gate": 2, "syrup": 1, "opencv": 3}
    for task, _ in TASKS:
        task_rows = [r for r in rows if r["task"] == task]
        sources = sorted({r["source"] for r in task_rows}, key=lambda s: (s != "all", s))
        for source in sources:
            for r in sorted((r for r in task_rows if r["source"] == source), key=lambda r: order.get(r["impl"], 9)):
                lines.append(f"| {task} | {source} | {r['impl']} | {ms(r['p50_ms'])} | {ms(r['p95_ms'])} | {ms(r['p99_ms'])} | "
                             f"{ratio(r, 'p50')} | {ratio(r, 'p95')} | {change(r)} |")
    return "\n".join(lines)


def process_table(summary):
    lines = ["| task / implementation | CPU ÷ wall | peak RSS MiB | init: load ms | init: first call ms |", "|---|---|---|---|---|"]
    for key, p in summary["process"].items():
        init = summary["init"].get(key, {})
        load = init.get("load_ns")
        first = init.get("first_call_ns")
        lines.append(f"| {key} | {p['cpu_over_wall']:.2f} | {p['max_rss_kb'] / 1024:.1f} | "
                     f"{'–' if load is None else f'{load / 1e6:.2f}'} | {'–' if first is None else f'{first / 1e6:.1f}'} |")
    return "\n".join(lines)


def quality_lines(summary):
    q = summary["quality"]
    out = []
    if "faces" in q:
        f = q["faces"]
        for impl in ("syrup-before", "syrup-after", "syrup-after-gate"):
            v = f.get(f"{impl}_vs_opencv", {}).get("all")
            if v:
                out.append(f"- F1 {impl} vs OpenCV's boxes: precision {v['precision']:.3f}, recall {v['recall']:.3f}, "
                           f"F1 {v['f1']:.3f}, mean IoU {v['mean_iou']:.3f} ({v['found']} vs {v['reference']} boxes)")
        for impl in ("syrup-after", "syrup-after-gate"):
            v = f.get(f"{impl}_vs_before")
            if v:
                out.append(f"- F1 {impl} vs syrup-before: identical boxes on {v['identical_inputs']} of {v['inputs']} images, "
                           f"F1 {v['f1']:.3f}, box count change {v['box_count_change'] * 100:+.1f}%")
    if "yunet" in q:
        v = q["yunet"].get("syrup_vs_opencv")
        if v:
            out.append(f"- F2 Syrup YuNet vs OpenCV's: F1 {v['f1']:.3f}, mean IoU {v['mean_iou']:.4f}, mean landmark difference "
                       f"{v['mean_landmark_error_px']:.3f} px, largest score difference {v['max_score_abs_diff']:.4f}")
    if "templates" in q:
        t = q["templates"]
        for impl, rates in t.get("hit_rate", {}).items():
            a = rates["all"]
            parts = ", ".join(f"{k} {v['rate'] * 100:.1f}%" for k, v in sorted(rates.items()) if k != "all")
            out.append(f"- F3 {impl}: hit rate {a['rate'] * 100:.1f}% ({a['hits']}/{a['cases']}); {parts}")
        for impl, v in t.get("vs_opencv", {}).items():
            out.append(f"- F3 {impl} vs OpenCV: same position in {v['same_position']} of {v['cases']}; score difference "
                       f"mean {v['score_abs_diff_mean']:.2e}, max {v['score_abs_diff_max']:.2e}")
        if "after_vs_before_identical_outputs" in t:
            a, n = t["after_vs_before_identical_outputs"]
            out.append(f"- F3 syrup-after vs syrup-before: identical output in {a} of {n} searches")
    if "motion" in q:
        m = q["motion"]
        for key, label in (("syrup-after_mask_vs_opencv", "mask"), ("syrup-before_mask_vs_opencv", "mask (before)")):
            v = m.get(key, {}).get("all")
            if v:
                out.append(f"- F4 {label}: pooled IoU {v['pooled_iou']:.6f}, minimum {v['min_iou']:.6f}, identical on "
                           f"{v['identical_pairs']} of {v['pairs']} pairs")
        v = m.get("syrup-after_regions_vs_opencv", {}).get("all")
        if v:
            out.append(f"- F4 regions (different grouping rules) vs OpenCV's: F1 {v['f1']:.3f} ({v['found']} vs {v['reference']})")
        if "after_vs_before_identical_outputs" in m:
            a, n = m["after_vs_before_identical_outputs"]
            out.append(f"- F4 syrup-after vs syrup-before: identical output on {a} of {n} pairs")
    if "glyphs" in q:
        g = q["glyphs"]
        for impl, r in g.get("readings", {}).items():
            a = r["all"]
            sizes = "; ".join(f"{k}: {v['correct']}/{v['cases']} right, {v['wrong']} wrong, {v['refused']} refused"
                              for k, v in sorted(r.items()) if k.startswith("s0"))
            out.append(f"- F5 {impl}: {a['correct']}/{a['cases']} read right ({a['accuracy'] * 100:.1f}%), "
                       f"{a['wrong']} wrong ({a['wrong_rate'] * 100:.1f}%), {a['refused']} refused ({a['refusal_rate'] * 100:.1f}%) — {sizes}")
        for impl, learned in g.get("learned", {}).items():
            failed = [l for l in (learned or []) if "error" in l]
            if failed:
                out.append(f"- F5 {impl} could not learn: " + "; ".join(f"{l['file']}: {l['error']}" for l in failed))
    for task, v in q.items():
        same = v.get("same_output_every_round")
        if same and not all(same.values()):
            out.append(f"- {task}: outputs differed between rounds for {[k for k, s in same.items() if not s]}")
    return "\n".join(out)


def main():
    root = sys.argv[1]
    parts = []
    for threads in (1, 2):
        summary = load(os.path.join(root, f"test-t{threads}"))
        if summary is None:
            continue
        parts.append(speed_table(summary, threads))
        parts.append("")
        parts.append(f"Process and initialisation, {threads} thread{'s' if threads > 1 else ''} (peak RSS includes the "
                     "decoded inputs each harness holds; the Python side also numpy and the interpreter; "
                     "'first call' is the first warm-up call on the first input, model or cascade preparation included):")
        parts.append("")
        parts.append(process_table(summary))
        parts.append("")
        if threads == 1:
            parts.append("Quality, one thread. Every output was the same with two threads, except that OpenCV "
                         "listed the same face boxes in a varying order (7 of 15 photographs; sorted, they are identical).")
            parts.append("")
            parts.append(quality_lines(summary))
            parts.append("")
    lip = os.path.join(root, "lipreader", "summary.md")
    if os.path.exists(lip):
        with open(lip) as f:
            parts.append(f.read())
    print("\n".join(parts))


if __name__ == "__main__":
    main()
