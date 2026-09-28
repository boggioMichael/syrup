"""Builds thelip.syrup.html: one self-contained page with the engine, the
worker and the int8 weights (base64) inlined, and the face finder (faces.js
in its own worker, BlazeFace's float16 weights from blazeface.bin).

    python3 ../ml/conversion/export_thelip_weights.py --out thelip-weights.bin
    python3 tools/export_blazeface.py blaze_face_short_range.tflite   # blazeface.bin (in the repository)
    python3 build.py [--weights thelip-weights.bin] [--out thelip.syrup.html]
"""
import argparse
import base64
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(HERE, "thelip-weights.bin"))
    ap.add_argument("--out", default=os.path.join(HERE, "thelip.syrup.html"))
    ap.add_argument("--standalone", action="store_true",
                    help="a complete HTML document (for hosting); without it, page content only, for a claude.ai artifact")
    args = ap.parse_args()
    with open(os.path.join(HERE, "thelip.js")) as f:
        engine = re.sub(r"^export (const|function|class) ", r"\1 ", f.read(), flags=re.M)
    with open(os.path.join(HERE, "worker.js")) as f:
        worker = f.read()
    with open(args.weights, "rb") as f:
        weights = base64.b64encode(f.read()).decode()
    with open(os.path.join(HERE, "src", "page.html")) as f:
        page = f.read()
    worker_src = engine + "\n" + worker
    assert "</script>" not in worker_src
    with open(os.path.join(HERE, "faces.js")) as f:
        faces = re.sub(r"^export (const|function|class) ", r"\1 ", f.read(), flags=re.M)
    with open(os.path.join(HERE, "face-worker.js")) as f:
        face_worker = faces + "\n" + f.read()
    assert "</script>" not in face_worker
    with open(os.path.join(HERE, "blazeface.bin"), "rb") as f:
        face_weights = base64.b64encode(f.read()).decode()
    page = (page.replace("__WORKER__", worker_src).replace("__WEIGHTS_B64__", weights)
            .replace("__FACE_WORKER__", face_worker).replace("__FACE_WEIGHTS_B64__", face_weights))
    if args.standalone:
        page = ('<!doctype html>\n<html lang="en">\n<head>\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
                # A camera needs a secure page: a plain-http visit moves to https before the page loads.
                '<script>if(location.protocol==="http:"&&!/^(localhost|127\\.0\\.0\\.1|\\[::1\\])$/.test(location.hostname))'
                'location.replace("https://"+location.host+location.pathname+location.search+location.hash)</script>\n'
                '<style>[hidden]{display:none!important} :root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}</style>\n'
                + page + "\n</html>\n")
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {args.out}: {os.path.getsize(args.out) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
