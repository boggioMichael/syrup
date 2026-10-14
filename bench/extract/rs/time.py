"""What each reader costs a line, the shakedown's way (SHAKEDOWN.md).

    time.py BIN DIR OUT.json [ROUNDS [pre]]

ROUNDS rounds (default 5); in each, every reader runs in a process of its
own (`BIN time DIR READER`: 3 warm-up and 5 timed calls a line, the line's
time the median of the 5), in an order rotated from round to round. The
summary gives p50 and p95 over the lines of each kind, per round, and the
median of the rounds' p50s and p95s. With `pre`, the explicit readers alone,
each line normalised before the clock starts (as ONNX Runtime is handed it).
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys

import numpy as np

READERS = ["teacher", "white-box", "black-box"]


def main():
    binary, data, out = sys.argv[1:4]
    rounds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    pre = len(sys.argv) > 5 and sys.argv[5] == "pre"
    readers = READERS[1:] if pre else READERS
    per = {r: {"made": [], "real": []} for r in readers}
    for i in range(rounds):
        order = readers[i % len(readers):] + readers[: i % len(readers)]
        for reader in order:
            env = dict(os.environ, RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")
            cmd = [binary, "time", data, reader] + (["3", "5", "pre"] if pre else [])
            res = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)
            times = {"made": [], "real": []}
            for line in res.stdout.split("\n"):
                if line:
                    kind, ns = line.split()
                    times[kind].append(int(ns))
            for kind, ns in times.items():
                per[reader][kind].append(ns)
            print(f"round {i + 1} {reader}: p50 made {np.percentile(times['made'], 50) / 1e3:.1f} us, "
                  f"real {np.percentile(times['real'], 50) / 1e3:.1f} us", flush=True)
    summary = {"machine": platform.processor() or platform.machine(), "rounds": rounds, "normalised before the clock": pre,
               "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), "?")}
    for reader, kinds in per.items():
        summary[reader] = {}
        for kind, runs in kinds.items():
            p50 = [float(np.percentile(r, 50)) / 1e3 for r in runs]
            p95 = [float(np.percentile(r, 95)) / 1e3 for r in runs]
            summary[reader][kind] = {"p50 us": round(float(np.median(p50)), 2), "p95 us": round(float(np.median(p95)), 2),
                                     "p50 us per round": [round(v, 2) for v in p50]}
    for kind in ("made", "real"):
        if "teacher" not in summary:
            break
        t = summary["teacher"][kind]["p50 us"]
        for reader in READERS[1:]:
            summary[reader][kind]["teacher / this, p50"] = round(t / summary[reader][kind]["p50 us"], 1)
    json.dump(summary, open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
