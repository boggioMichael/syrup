"""Run the Syrup vs OpenCV comparison and summarise it (PREREGISTRATION.md).

    python3 bench/vs-opencv/run.py measure --data DATA --split test --threads 1 \
        --impl syrup-before=BIN --impl syrup-after=BIN --impl opencv \
        --tasks faces,templates,motion --rounds 5 --cascades DIR --out RESULTS
    python3 bench/vs-opencv/run.py summarise --data DATA --out RESULTS

`measure` runs every implementation over every input of every task, once
per round, each run in its own process, in an order rotated round by round,
and keeps each process's JSON in RESULTS/raw. It then runs each
implementation once more, untimed, to dump the motion masks for the mask
comparison. `summarise` reads RESULTS/raw and writes RESULTS/summary.json
and RESULTS/summary.md.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP = 2000
BLOCK = 10  # consecutive frame pairs resampled together (motion)
RNG_SEED = 20261012


# ---------------------------------------------------------------- measure

def command(impl: str, binary: str | None, task: str, a, out: str, extra=()):
    flags = ["--data", a.data, "--split", a.split, "--threads", str(a.threads),
             "--calls", str(a.calls), "--warmup", str(a.warmup), "--out", out, *extra]
    if impl == "opencv":
        return [sys.executable, os.path.join(HERE, "opencv_side.py"), task, *flags, "--cascades", a.cascades]
    args = dict(spec.split("=", 1) for spec in a.args or [])
    if task == "yunet":  # its own binary, one task
        return [binary, *flags]
    return [binary, task, *flags, *args.get(impl, "").split()]


def thread_env(threads: int) -> dict:
    """The environment of a measured process: numpy's BLAS (imported by the
    OpenCV side) held to the same thread count, or its idle worker threads
    spin and count as CPU time."""
    env = dict(os.environ)
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = str(threads)
    return env


def environment(a, impls):
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
        except OSError:
            return ""
    import cv2
    cpu = ""
    with open("/proc/cpuinfo") as f:
        info = f.read()
    for line in info.splitlines():
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
    flags = next((l.split(":", 1)[1].split() for l in info.splitlines() if l.startswith("flags")), [])
    build = cv2.getBuildInformation()
    keep = ("Version control", "Baseline", "Dispatched", "Parallel framework", "C++ Compiler")
    manifest = os.path.join(HERE, "results", "inputs.json")
    return {
        "date": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "command": sys.argv,
        "cpu": cpu, "cpus": os.cpu_count(),
        "simd": [f for f in flags if f.startswith(("avx", "sse4", "fma", "bmi"))],
        "kernel": platform.release(), "os": run(["sh", "-c", ". /etc/os-release && echo $PRETTY_NAME"]),
        "memory_kb": int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemTotal"))),
        "rustc": run(["rustc", "--version"]), "python": sys.version.split()[0],
        "numpy": np.__version__, "opencv": cv2.__version__,
        "opencv_build": [l.strip() for l in build.splitlines() if l.strip().startswith(keep)],
        "impls": {name: (binary or "opencv_side.py") for name, binary in impls},
        "binaries_sha256": {name: hashlib.sha256(open(b, "rb").read()).hexdigest() for name, b in impls if b},
        "inputs_manifest_sha256": hashlib.sha256(open(manifest, "rb").read()).hexdigest() if os.path.exists(manifest) else None,
        "threads": a.threads, "calls": a.calls, "warmup": a.warmup, "rounds": a.rounds, "split": a.split,
    }


def measure(a):
    impls = []
    for spec in a.impl:
        name, _, binary = spec.partition("=")
        impls.append((name, os.path.abspath(binary) if binary else None))
    raw = os.path.join(a.out, "raw")
    os.makedirs(raw, exist_ok=True)
    env_path = os.path.join(a.out, "environment.json")
    env = environment(a, impls)
    if os.path.exists(env_path):
        with open(env_path) as f:
            before = json.load(f)
        before.setdefault("commands", [before.get("command")]).append(sys.argv)
        before["impls"].update(env["impls"])
        before["binaries_sha256"].update(env["binaries_sha256"])
        env = before
    else:
        env["commands"] = [sys.argv]
    with open(env_path, "w") as f:
        json.dump(env, f, indent=1)
    tasks = a.tasks.split(",")
    for r in range(a.rounds):
        order = impls[r % len(impls):] + impls[:r % len(impls)]
        for task in tasks:
            for name, binary in order:
                out = os.path.join(raw, f"{task}.{name}.r{r}.json")
                if os.path.exists(out) and not a.again:
                    continue
                cmd = command(name, binary, task, a, out)
                print(f"round {r} {task:9s} {name:13s}", flush=True)
                subprocess.run(cmd, check=True, env=thread_env(a.threads))
    if "motion" in tasks:
        for name, binary in impls:
            dump = os.path.join(a.out, "masks", name)
            out = os.path.join(raw, f"motion.{name}.masks.json")
            if os.path.exists(out) and not a.again:
                continue
            subprocess.run(command(name, binary, "motion", a, out, ["--dump", dump]), check=True,
                           env=thread_env(a.threads))


# -------------------------------------------------------------- summarise

def iou(a, b):
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    w = min(ax + aw, bx + bw) - max(ax, bx)
    h = min(ay + ah, by + bh) - max(ay, by)
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / (aw * ah + bw * bh - inter)


def match_boxes(found, reference, at=0.5):
    """Greedy one-to-one matching by IoU: (true positives, ious of them)."""
    pairs = sorted(((iou(f, r), i, j) for i, f in enumerate(found) for j, r in enumerate(reference)), reverse=True)
    used_f, used_r, ious = set(), set(), []
    for v, i, j in pairs:
        if v < at:
            break
        if i in used_f or j in used_r:
            continue
        used_f.add(i)
        used_r.add(j)
        ious.append(v)
    return len(ious), ious


def prf(tp, n_found, n_ref):
    p = tp / n_found if n_found else 1.0
    r = tp / n_ref if n_ref else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def load_raw(out):
    runs = defaultdict(list)  # (task, impl) -> [json per round]
    masks = {}
    for path in sorted(glob.glob(os.path.join(out, "raw", "*.json"))):
        task, impl, tag = os.path.basename(path)[:-5].split(".")
        with open(path) as f:
            data = json.load(f)
        if tag == "masks":
            masks[impl] = data
        else:
            runs[(task, impl)].append(data)
    return runs, masks


def manifest_sources(data_dir):
    with open(os.path.join(HERE, "results", "inputs.json")) as f:
        entries = json.load(f)["files"]
    return {e["file"]: e["source"] for e in entries}


def source_of(task, item_input, sources, cases):
    """The data source an item belongs to, and the cluster (independent
    unit) it belongs to for the bootstrap."""
    if task == "faces":
        src = sources.get(item_input, "?")
        dataset = src.split(":")[0]
        if dataset == "grid":
            return src.split(" ")[0], src.split(" ")[0]  # cluster: the video
        return dataset, item_input  # cluster: the photograph
    if task == "templates":
        index = int(item_input.split("|")[0])
        case = cases[index]
        src = case["source"].split(" ")[0]
        dataset = src.split(":")[0]
        if "#" in src:
            dataset += ":recording"
        elif dataset == "ms":
            dataset += ":still"
        return dataset, case["image"].replace(".perturbed", "")
    if task == "glyphs":
        case = cases[item_input]
        return f"{case['size']}/{case['background']}", item_input
    if task == "yunet":
        src = sources.get(item_input, "?")
        dataset = src.split(":")[0]
        if dataset == "grid":
            return src.split(" ")[0], src.split(" ")[0]
        return dataset, item_input
    video, pair = item_input.split("/")
    return video, f"{video}/{int(pair) // BLOCK}"


def percentiles(values):
    if len(values) == 0:
        return [None, None, None]
    return [float(v) for v in np.percentile(values, [50, 95, 99])]


def bootstrap_ratio(clusters_a, clusters_b, stat, rng):
    """95% interval of stat(b)/stat(a) resampling clusters (shared keys)."""
    keys = sorted(set(clusters_a) & set(clusters_b))
    if len(keys) < 2:
        return None
    ratios = []
    for _ in range(BOOTSTRAP):
        pick = rng.integers(0, len(keys), len(keys))
        va = np.concatenate([clusters_a[keys[i]] for i in pick])
        vb = np.concatenate([clusters_b[keys[i]] for i in pick])
        ratios.append(stat(vb) / stat(va))
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarise(a):
    runs, masks = load_raw(a.out)
    sources = manifest_sources(a.data)
    rng = np.random.default_rng(RNG_SEED)
    summary = {"rows": [], "quality": {}, "process": {}, "init": {}}
    with open(os.path.join(a.out, "environment.json")) as f:
        env = json.load(f)
    split = env["split"]
    def task_cases(task):
        path = os.path.join(a.data, task, split, "cases.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        if task == "glyphs":
            return {c["image"]: c for c in data["cases"]}
        return data
    tasks = sorted({t for t, _ in runs})
    for task in tasks:
        cases = task_cases(task)
        impls = sorted(i for t, i in runs if t == task)
        per = {}  # impl -> source -> {"times": [], "clusters": {key: []}, "rounds": [[...] per round]}
        for impl in impls:
            per[impl] = defaultdict(lambda: {"times": [], "clusters": defaultdict(list), "rounds": defaultdict(list)})
            for r, run in enumerate(runs[(task, impl)]):
                for item in run["items"]:
                    src, cluster = source_of(task, item["input"], sources, cases)
                    keys = [src, "all"]
                    if task == "templates":
                        side = cases[int(item["input"].split("|")[0])]["side"]
                        keys.append(f"side {side}")
                    for key in keys:
                        bucket = per[impl][key]
                        bucket["times"].extend(item["times_ns"])
                        bucket["clusters"][cluster].extend(item["times_ns"])
                        bucket["rounds"][r].extend(item["times_ns"])
            summary["process"][f"{task}/{impl}"] = {
                "cpu_over_wall": float(np.median([(x["process"]["cpu_user_s"] + x["process"]["cpu_sys_s"]) / x["process"]["wall_s"] for x in runs[(task, impl)]])),
                "max_rss_kb": int(max(x["process"]["max_rss_kb"] for x in runs[(task, impl)])),
                "rss_after_imports_kb": runs[(task, impl)][0]["process"].get("rss_after_imports_kb"),
                "rounds": len(runs[(task, impl)]),
            }
            init = defaultdict(list)
            for x in runs[(task, impl)]:
                for k, v in x["init"].items():
                    init[k].append(v)
            summary["init"][f"{task}/{impl}"] = {k: float(np.median(v)) for k, v in init.items()
                                                 if all(isinstance(x, (int, float)) for x in v)}
        reference = "opencv" if "opencv" in impls else None
        for src in sorted({s for impl in impls for s in per[impl]}, key=lambda s: (s != "all", s)):
            for impl in impls:
                b = per[impl].get(src)
                if not b or not b["times"]:
                    continue
                ms = np.array(b["times"]) / 1e6
                p50, p95, p99 = percentiles(ms)
                round_p95 = [float(np.percentile(np.array(v) / 1e6, 95)) for _, v in sorted(b["rounds"].items())]
                round_p50 = [float(np.percentile(np.array(v) / 1e6, 50)) for _, v in sorted(b["rounds"].items())]
                row = {"task": task, "source": src, "impl": impl, "calls": len(ms), "inputs": len(b["clusters"]),
                       "p50_ms": p50, "p95_ms": p95, "p99_ms": p99,
                       "p50_round_range": [min(round_p50), max(round_p50)],
                       "p95_round_range": [min(round_p95), max(round_p95)],
                       "throughput_per_s": float(len(ms) / ms.sum() * 1e3)}
                if reference and impl != reference and src in per[reference]:
                    ref = per[reference][src]
                    ca = {k: np.array(v) / 1e6 for k, v in b["clusters"].items()}
                    cr = {k: np.array(v) / 1e6 for k, v in ref["clusters"].items()}
                    row["opencv_over_this_p50"] = float(np.percentile(np.array(ref["times"]) / 1e6, 50) / p50)
                    row["opencv_over_this_p95"] = float(np.percentile(np.array(ref["times"]) / 1e6, 95) / p95)
                    row["opencv_over_this_p50_ci95"] = bootstrap_ratio(ca, cr, lambda v: np.percentile(v, 50), rng)
                    row["opencv_over_this_p95_ci95"] = bootstrap_ratio(ca, cr, lambda v: np.percentile(v, 95), rng)
                if impl.startswith("syrup-after") and "syrup-before" in per and src in per["syrup-before"]:
                    before = per["syrup-before"][src]
                    cb = {k: np.array(v) / 1e6 for k, v in before["clusters"].items()}
                    ca = {k: np.array(v) / 1e6 for k, v in b["clusters"].items()}
                    bp95 = float(np.percentile(np.array(before["times"]) / 1e6, 95))
                    row["p95_change_vs_before"] = p95 / bp95 - 1
                    ci = bootstrap_ratio(cb, ca, lambda v: np.percentile(v, 95), rng)
                    row["p95_change_vs_before_ci95"] = [ci[0] - 1, ci[1] - 1] if ci else None
                summary["rows"].append(row)
        summary["quality"][task] = quality(task, runs, masks, impls, a, cases, sources, split)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(markdown(summary, env))
    print(open(os.path.join(a.out, "summary.md")).read())


def outputs(run):
    return {item["input"]: item["output"] for item in run["items"]}


def quality(task, runs, masks, impls, a, cases, sources, split):
    q = {}
    first = {impl: outputs(runs[(task, impl)][0]) for impl in impls}
    # Determinism: every round gave the same outputs.
    q["same_output_every_round"] = {impl: all(outputs(r) == first[impl] for r in runs[(task, impl)]) for impl in impls}
    if task == "faces":
        for impl in impls:
            counts = sum(len(o["boxes"]) for o in first[impl].values())
            q.setdefault("boxes", {})[impl] = counts
        if "opencv" in impls:
            for impl in impls:
                if impl == "opencv":
                    continue
                per_src = defaultdict(lambda: [0, 0, 0, []])
                for name, o in first[impl].items():
                    src, _ = source_of(task, name, sources, cases)
                    ref = first["opencv"][name]["boxes"]
                    tp, ious = match_boxes(o["boxes"], ref)
                    for key in (src, "all"):
                        acc = per_src[key]
                        acc[0] += tp
                        acc[1] += len(o["boxes"])
                        acc[2] += len(ref)
                        acc[3] += ious
                q[f"{impl}_vs_opencv"] = {
                    k: dict(zip(("precision", "recall", "f1"), prf(v[0], v[1], v[2])),
                            mean_iou=float(np.mean(v[3])) if v[3] else None, found=v[1], reference=v[2])
                    for k, v in per_src.items()}
        for after in (i for i in impls if i.startswith("syrup-after")):
            if "syrup-before" not in impls:
                break
            tp = found = ref = 0
            identical = 0
            for name, o in first[after].items():
                b = first["syrup-before"][name]["boxes"]
                t, _ = match_boxes(o["boxes"], b)
                tp += t
                found += len(o["boxes"])
                ref += len(b)
                identical += o["boxes"] == b
            q[f"{after}_vs_before"] = dict(zip(("precision", "recall", "f1"), prf(tp, found, ref)),
                                           identical_inputs=identical, inputs=len(first[after]),
                                           box_count_change=(found - ref) / ref if ref else 0.0)
    if task == "glyphs":
        for impl in impls:
            per = defaultdict(lambda: {"correct": 0, "wrong": 0, "refused": 0, "cases": 0})
            for name, o in first[impl].items():
                c = cases[name]
                outcome = "refused" if "refused" in o else (
                    "correct" if "".join(o["text"].split()) == c["text"] else "wrong")
                for key in ("all", c["size"], c["background"]):
                    per[key][outcome] += 1
                    per[key]["cases"] += 1
            q.setdefault("readings", {})[impl] = {k: {**v, "accuracy": v["correct"] / v["cases"],
                                                      "wrong_rate": v["wrong"] / v["cases"],
                                                      "refusal_rate": v["refused"] / v["cases"]}
                                                  for k, v in per.items()}
            q.setdefault("learned", {})[impl] = runs[(task, impl)][0]["init"].get("learned")
    if task == "yunet" and "opencv" in impls:
        for impl in impls:
            if impl == "opencv":
                continue
            tp = nf = nr = 0
            ious, landmark, score = [], [], []
            for name, o in first[impl].items():
                ref = first["opencv"][name]["boxes"]
                t, iv = match_boxes(o["boxes"], ref)
                tp += t
                nf += len(o["boxes"])
                nr += len(ref)
                ious += iv
                for f in o["boxes"]:
                    best = max(ref, key=lambda r: iou(f, r), default=None)
                    if best is not None and iou(f, best) >= 0.5:
                        pts = np.array(f[5:15]).reshape(5, 2) - np.array(best[5:15]).reshape(5, 2)
                        landmark.append(float(np.linalg.norm(pts, axis=1).mean()))
                        score.append(abs(f[4] - best[4]))
            q[f"{impl}_vs_opencv"] = dict(zip(("precision", "recall", "f1"), prf(tp, nf, nr)),
                                          mean_iou=float(np.mean(ious)) if ious else None,
                                          mean_landmark_error_px=float(np.mean(landmark)) if landmark else None,
                                          max_score_abs_diff=float(np.max(score)) if score else None,
                                          found=nf, reference=nr)
    if task == "templates":
        for impl in impls:
            per = defaultdict(lambda: [0, 0, 0])
            for name, o in first[impl].items():
                c = cases[int(name.split("|")[0])]
                hit = "x" in o and abs(o["x"] - c["x"]) <= 1 and abs(o["y"] - c["y"]) <= 1
                for key in ("all", c["variant"], f"side {c['side']}", source_of(task, name, sources, cases)[0]):
                    per[key][0] += hit
                    per[key][1] += 1
                    per[key][2] += "refused" in o
            q.setdefault("hit_rate", {})[impl] = {k: {"hits": v[0], "cases": v[1], "rate": v[0] / v[1], "refused": v[2]}
                                                  for k, v in per.items()}
            if impl != "opencv" and "opencv" in impls:
                diffs = [abs(o["score"] - first["opencv"][n]["score"]) for n, o in first[impl].items() if "score" in o]
                same = sum(1 for n, o in first[impl].items() if "x" in o and (o["x"], o["y"]) == (first["opencv"][n]["x"], first["opencv"][n]["y"]))
                q.setdefault("vs_opencv", {})[impl] = {"same_position": same, "cases": len(first[impl]),
                                                       "score_abs_diff_mean": float(np.mean(diffs)) if diffs else None,
                                                       "score_abs_diff_max": float(np.max(diffs)) if diffs else None}
    if task == "motion":
        import cv2
        if "opencv" in impls:
            for impl in impls:
                if impl == "opencv":
                    continue
                per = defaultdict(lambda: {"inter": 0, "union": 0, "identical": 0, "pairs": 0, "min_iou": 1.0})
                for video in sorted(os.listdir(os.path.join(a.out, "masks", "opencv"))):
                    for name in sorted(os.listdir(os.path.join(a.out, "masks", "opencv", video))):
                        m_o = cv2.imread(os.path.join(a.out, "masks", "opencv", video, name), cv2.IMREAD_GRAYSCALE) > 0
                        m_s = cv2.imread(os.path.join(a.out, "masks", impl, video, name), cv2.IMREAD_GRAYSCALE) > 0
                        inter, union = int((m_o & m_s).sum()), int((m_o | m_s).sum())
                        v = 1.0 if union == 0 else inter / union
                        for key in (video, "all"):
                            acc = per[key]
                            acc["inter"] += inter
                            acc["union"] += union
                            acc["identical"] += bool((m_o == m_s).all())
                            acc["pairs"] += 1
                            acc["min_iou"] = min(acc["min_iou"], v)
                q[f"{impl}_mask_vs_opencv"] = {k: {"pooled_iou": (v["inter"] / v["union"]) if v["union"] else 1.0,
                                                   "min_iou": v["min_iou"], "identical_pairs": v["identical"],
                                                   "pairs": v["pairs"]} for k, v in per.items()}
                per_r = defaultdict(lambda: [0, 0, 0])
                for name, o in first[impl].items():
                    ref = first["opencv"][name]["regions"]
                    tp, _ = match_boxes(o["regions"], ref)
                    for key in (name.split("/")[0], "all"):
                        per_r[key][0] += tp
                        per_r[key][1] += len(o["regions"])
                        per_r[key][2] += len(ref)
                q[f"{impl}_regions_vs_opencv"] = {k: dict(zip(("precision", "recall", "f1"), prf(*v)), found=v[1], reference=v[2])
                                                  for k, v in per_r.items()}
        if "syrup-before" in impls and "syrup-after" in impls:
            same = sum(first["syrup-after"][n] == first["syrup-before"][n] for n in first["syrup-after"])
            q["after_vs_before_identical_outputs"] = [same, len(first["syrup-after"])]
    if task == "templates" and "syrup-before" in impls and "syrup-after" in impls:
        same = sum(first["syrup-after"][n] == first["syrup-before"][n] for n in first["syrup-after"])
        q["after_vs_before_identical_outputs"] = [same, len(first["syrup-after"])]
    return q


def fmt(v, digits=2):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def markdown(summary, env):
    lines = [f"# Syrup vs OpenCV — {env['split']} data, {env['threads']} thread(s)", "",
             f"{env['date']} · {env['cpu']} ({env['cpus']} CPUs) · {env['os']} · {env['rustc']} · OpenCV {env['opencv']} · "
             f"{env['rounds']} rounds × ({env['warmup']} warm-up + {env['calls']} timed calls per image; videos once per round)", "",
             "Times in ms per call (library call only, warm). `OpenCV/this` > 1: this implementation is faster than OpenCV; "
             "the interval is a 95% bootstrap over inputs (photographs, source images, or blocks of 10 consecutive frame pairs).", "",
             "| task | source | impl | calls | p50 | p95 | p99 | p95 range over rounds | calls/s | OpenCV/this p50 [95% CI] | OpenCV/this p95 [95% CI] | p95 vs before [95% CI] |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        ci50 = r.get("opencv_over_this_p50_ci95")
        ci95 = r.get("opencv_over_this_p95_ci95")
        chg = r.get("p95_change_vs_before")
        chg_ci = r.get("p95_change_vs_before_ci95")
        lines.append("| " + " | ".join([
            r["task"], r["source"], r["impl"], str(r["calls"]), fmt(r["p50_ms"], 3), fmt(r["p95_ms"], 3), fmt(r["p99_ms"], 3),
            f"{fmt(r['p95_round_range'][0], 3)}–{fmt(r['p95_round_range'][1], 3)}", fmt(r["throughput_per_s"], 1),
            (f"{fmt(r['opencv_over_this_p50'])} [{fmt(ci50[0])}–{fmt(ci50[1])}]" if ci50 else fmt(r.get("opencv_over_this_p50"))),
            (f"{fmt(r['opencv_over_this_p95'])} [{fmt(ci95[0])}–{fmt(ci95[1])}]" if ci95 else fmt(r.get("opencv_over_this_p95"))),
            (f"{chg * 100:+.1f}% [{chg_ci[0] * 100:+.1f}%, {chg_ci[1] * 100:+.1f}%]" if chg is not None and chg_ci else
             (f"{chg * 100:+.1f}%" if chg is not None else "–")),
        ]) + " |")
    lines += ["", "## Process (median over rounds)", "", "| task/impl | CPU time / wall | peak RSS (MiB) | after imports (MiB) |", "|---|---|---|---|"]
    for k, v in summary["process"].items():
        after = v.get("rss_after_imports_kb")
        lines.append(f"| {k} | {v['cpu_over_wall']:.2f} | {v['max_rss_kb'] / 1024:.1f} | {after / 1024:.1f} |" if after else
                     f"| {k} | {v['cpu_over_wall']:.2f} | {v['max_rss_kb'] / 1024:.1f} | – |")
    lines += ["", "## Initialisation (median over rounds, ms)", "", "| task/impl | load | first call |", "|---|---|---|"]
    for k, v in summary["init"].items():
        lines.append(f"| {k} | {fmt(v.get('load_ns', 0) / 1e6, 3) if 'load_ns' in v else '–'} | {fmt(v.get('first_call_ns', 0) / 1e6, 3)} |")
    lines += ["", "## Quality", "", "```json", json.dumps(summary["quality"], indent=1), "```", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--data", required=True)
    m.add_argument("--split", default="test")
    m.add_argument("--threads", type=int, default=1)
    m.add_argument("--calls", type=int, default=5)
    m.add_argument("--warmup", type=int, default=3)
    m.add_argument("--rounds", type=int, default=5)
    m.add_argument("--impl", action="append", required=True, help="name=binary (opencv: no binary)")
    m.add_argument("--args", action="append", help="name=extra arguments for that implementation")
    m.add_argument("--tasks", default="faces,templates,motion")
    m.add_argument("--cascades", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--again", action="store_true", help="re-run rounds already on disk")
    s = sub.add_parser("summarise")
    s.add_argument("--data", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "measure":
        measure(a)
    else:
        summarise(a)


if __name__ == "__main__":
    sys.exit(main())
