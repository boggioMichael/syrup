"""Builds the-lip-live.html: one self-contained page with the engine, the
worker and the int8 weights (base64) inlined.

    python3 ../ml/conversion/export_lipnet_web.py --out lipnet-web.bin
    python3 build.py [--weights lipnet-web.bin] [--out the-lip-live.html]
"""
import argparse
import base64
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "lipnet-web.bin"))
    ap.add_argument("--out", default=os.path.join(HERE, "the-lip-live.html"))
    args = ap.parse_args()
    with open(os.path.join(HERE, "lipnet.js")) as f:
        engine = re.sub(r"^export (const|function|class) ", r"\1 ", f.read(), flags=re.M)
    with open(os.path.join(HERE, "worker.js")) as f:
        worker = f.read()
    with open(args.weights, "rb") as f:
        weights = base64.b64encode(f.read()).decode()
    with open(os.path.join(HERE, "src", "page.html")) as f:
        page = f.read()
    worker_src = engine + "\n" + worker
    assert "</script>" not in worker_src
    page = page.replace("__WORKER__", worker_src).replace("__WEIGHTS_B64__", weights)
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
