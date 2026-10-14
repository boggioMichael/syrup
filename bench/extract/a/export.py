"""The pilot's readers and test lines, for the Rust port (../rs).

    export.py --lines L.npz --labels EXP-LABELS.json --bank BANK.npz
              --teacher NET.npz --readers DIR --out OUT

Writes into OUT (outside the repository: the lines are game graphics):

- meta.txt: how many lines, each explicit reader's classes and widths, the
  teacher's layers (one `key values...` per line, for a parser in std alone);
- lines.u8: the test lines, (n, 15, 210, 3) RGB; kind.txt (made or real)
  and truth.txt, one per line;
- teacher.f32: each convolution's weights, (kh*kw*cin, cout), then its bias;
- <reader>.f32: the band's colour (15), each class's picture (15 x width,
  in meta.txt's order), the reading templates ((15*9 + 1) x 17);
- py-teacher.txt and py-<reader>.tsv: what the Python code reads on every
  line (and its confidence), which the port must give back.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import expsynth  # noqa: E402
import extract  # noqa: E402
import linenet  # noqa: E402
import teach  # noqa: E402

READERS = ("white-box", "black-box")


def load_reader(path):
    z = np.load(path)
    templates = {int(k[1:]): z[k] for k in z.files if k.startswith("t")}
    return extract.Reader(templates, z["background"], z["weights"])


def main():
    p = argparse.ArgumentParser()
    for a in ("--lines", "--labels", "--bank", "--teacher", "--readers", "--out"):
        p.add_argument(a, required=True)
    p.add_argument("--made", type=int, default=1000)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    bank = expsynth.Bank(args.bank, "test")
    mx, mt = expsynth.make(bank, args.made, np.random.default_rng(3), places=[])
    rx, rt, _ = teach.real_frames(args.lines, args.labels, "test")
    lines = np.concatenate([mx, rx[:, :, : mx.shape[2]]]).astype(np.uint8)
    truth = mt + rt
    kinds = ["made"] * len(mt) + ["real"] * len(rt)
    lines.tofile(os.path.join(args.out, "lines.u8"))
    open(os.path.join(args.out, "truth.txt"), "w").write("\n".join(truth) + "\n")
    open(os.path.join(args.out, "kind.txt"), "w").write("\n".join(kinds) + "\n")
    meta = [f"lines {len(lines)} {lines.shape[1]} {lines.shape[2]}", f"made {len(mt)}"]

    net = linenet.LineNet.load(args.teacher)
    convs = [l for l in net.layers if isinstance(l, linenet.Conv)]
    with open(os.path.join(args.out, "teacher.f32"), "wb") as f:
        for c in convs:
            meta.append(f"layer {c.cin} {c.cout} {c.kh} {c.kw} {c.ph} {c.pw}")
            f.write(np.ascontiguousarray(c.w, np.float32).tobytes())
            f.write(np.ascontiguousarray(c.b, np.float32).tobytes())
    pred = linenet.predict(net, linenet.to_input(lines))
    open(os.path.join(args.out, "py-teacher.txt"), "w").write("\n".join(pred) + "\n")

    g = extract.gray(lines)
    for name in READERS:
        r = load_reader(os.path.join(args.readers, f"{name}.npz"))
        meta.append(f"reader {name} {len(r.templates)} {r.penalty} {extract.HW}")
        with open(os.path.join(args.out, f"{name}.f32"), "wb") as f:
            f.write(np.ascontiguousarray(r.background, np.float32).tobytes())
            for k, t in r.templates.items():
                meta.append(f"class {name} {k} {t.shape[1]}")
                f.write(np.ascontiguousarray(t, np.float32).tobytes())
            f.write(np.ascontiguousarray(r.weights, np.float32).tobytes())
        found = [r.read(gl) for gl in g]
        open(os.path.join(args.out, f"py-{name}.tsv"), "w").write(
            "\n".join(f"{t}\t{c!r}" for t, c in found) + "\n")
    open(os.path.join(args.out, "meta.txt"), "w").write("\n".join(meta) + "\n")
    print(f"{len(lines)} lines ({len(mt)} made, {len(rt)} real) to {args.out}")


if __name__ == "__main__":
    main()
