"""Run LipNet on the GRID sample clips and compare with the sentences their
file names encode: word accuracy, per clip and overall.

    python3 evaluate.py [--lipnet DIR] [--weights FILE]
"""
import argparse
import glob
import os
import time

from lipnet_np import LipNet, Spell, greedy_decode, grid_sentence
from mouth import mouth_crops, read_frames
from the_lip import DICTIONARY, LIPNET_DIR, WEIGHTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lipnet", default=LIPNET_DIR)
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()
    weights = args.weights or os.path.join(args.lipnet, WEIGHTS)
    t = time.time()
    net = LipNet(weights)
    spell = Spell(os.path.join(args.lipnet, DICTIONARY))
    print(f"weights loaded in {time.time() - t:.1f}s")

    samples = sorted(glob.glob(f"{args.lipnet}/evaluation/samples/GRID/*.mpg"))
    samples += sorted(glob.glob(f"{args.lipnet}/evaluation/samples/*.mpg"))
    words_total = words_ok = 0
    for path in samples:
        code = os.path.splitext(os.path.basename(path))[0]
        truth = grid_sentence(code)
        frames, _ = read_frames(path)
        t = time.time()
        crops, _, ratio = mouth_crops(frames)
        t_crop = time.time() - t
        t = time.time()
        raw = greedy_decode(net.predict(crops))
        t_net = time.time() - t
        fixed = spell.sentence(raw)
        tw, fw = truth.split(), fixed.split()
        ok = sum(1 for a, b in zip(tw, fw) if a == b)
        words_total += len(tw)
        words_ok += ok
        print(f"{code:16s} spoken: {truth:32s} raw: {raw:34s} corrected: {fixed:32s} {ok}/{len(tw)}"
              f"  (mouth {t_crop:.1f}s, net {t_net:.1f}s, ratio {ratio:.2f})")
    print(f"word accuracy: {words_ok}/{words_total} = {100 * words_ok / max(words_total, 1):.1f}%")


if __name__ == "__main__":
    main()
