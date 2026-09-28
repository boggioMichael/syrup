"""Set up and run thelip-server, then print the link (and a QR code) that
points thelip.ai at it.

    python run.py                 # download what is missing, start, tunnel, print the link
    python run.py --no-tunnel     # serve on this machine only (http://127.0.0.1:8791)
    python run.py --fake          # no model download, fixed answers (tests)

What it downloads, once, next to this file:
    chaplin/                the Chaplin pipeline (Amanvir Parhar, MIT; pinned commit)
    chaplin/benchmarks/...  the LRS3 model (1 GB) and its language model (215 MB)
                            from Hugging Face (Amanvir/LRS3_V_WER19.1, Amanvir/lm_en_subword)
    cloudflared(.exe)       Cloudflare's tunnel client, for an https address the phone can reach

The server itself is server.py; this script starts it as a child process,
waits for it to answer /health, starts the tunnel, and writes the link to
link.txt. Ctrl+C stops everything.
"""
from __future__ import annotations

import argparse
import io
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
# Downloads and the Python environment live under THELIP_HOME (thelip-server.cmd
# sets it to C:\Users\Public\thelip-server): PyTorch's DLLs fail to initialise
# from a path with non-ASCII characters, such as a Hebrew user name. Logs and
# the link stay next to this file.
WORK = os.environ.get("THELIP_HOME") or HERE
CHAPLIN_DIR = os.path.join(WORK, "chaplin")
FACE_MODEL = os.path.join(WORK, "blaze_face_short_range.tflite")
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
CHAPLIN_COMMIT = "7aee1f8fca776ce4f63690063310b53573b7d804"
CHAPLIN_ZIP = f"https://github.com/amanvirparhar/chaplin/archive/{CHAPLIN_COMMIT}.zip"
MODELS = [
    ("benchmarks/LRS3/models/LRS3_V_WER19.1/model.json", "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json", 1_000),
    ("benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth", "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth", 900_000_000),
    ("benchmarks/LRS3/language_models/lm_en_subword/model.json", "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json", 500),
    ("benchmarks/LRS3/language_models/lm_en_subword/model.pth", "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth", 150_000_000),
]
SITE = "https://thelip.ai/"


def say(*a):
    print(*a, flush=True)


def download(url: str, dest: str, min_size: int = 0) -> None:
    if os.path.isfile(dest) and os.path.getsize(dest) >= min_size:
        return
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    say(f"downloading {url}")
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "thelip-server/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, last = 0, 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if time.time() - last > 2:
                last = time.time()
                say(f"  {got / 1e6:.0f} MB" + (f" of {total / 1e6:.0f} MB" if total else ""))
    if os.path.getsize(tmp) < min_size:
        os.remove(tmp)
        raise SystemExit(f"{url}: got a file of {os.path.getsize(tmp) if os.path.exists(tmp) else 0} bytes; expected at least {min_size}")
    os.replace(tmp, dest)


def ensure_chaplin() -> None:
    if os.path.isdir(os.path.join(CHAPLIN_DIR, "pipelines")):
        return
    old = os.path.join(HERE, "chaplin")
    if WORK != HERE and os.path.isdir(os.path.join(old, "pipelines")):
        say(f"moving {old} to {CHAPLIN_DIR}")
        os.makedirs(WORK, exist_ok=True)
        shutil.move(old, CHAPLIN_DIR)
        return
    say("fetching the Chaplin pipeline (MIT; Imperial College preprocessing, Apache-2.0)")
    req = urllib.request.Request(CHAPLIN_ZIP, headers={"User-Agent": "thelip-server/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        root = z.namelist()[0].split("/")[0]
        os.makedirs(WORK, exist_ok=True)
        z.extractall(WORK)
    shutil.move(os.path.join(WORK, root), CHAPLIN_DIR)


def ensure_models() -> None:
    for rel, url, min_size in MODELS:
        download(url, os.path.join(CHAPLIN_DIR, rel), min_size)
    download(FACE_MODEL_URL, FACE_MODEL, 100_000)


def cloudflared_path() -> str:
    system, machine = platform.system(), platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if system == "Windows":
        name, asset = "cloudflared.exe", "cloudflared-windows-amd64.exe"
    elif system == "Linux":
        name, asset = "cloudflared", "cloudflared-linux-arm64" if arm else "cloudflared-linux-amd64"
    elif system == "Darwin":
        found = shutil.which("cloudflared")
        if found:
            return found
        raise SystemExit("on macOS install the tunnel client with: brew install cloudflared")
    else:
        raise SystemExit(f"no cloudflared build known for {system}")
    path = os.path.join(WORK, name)
    old = os.path.join(HERE, name)
    if not os.path.isfile(path) and WORK != HERE and os.path.isfile(old):
        shutil.move(old, path)
    if not os.path.isfile(path):
        download(f"https://github.com/cloudflare/cloudflared/releases/latest/download/{asset}", path, 10_000_000)
        if system != "Windows":
            os.chmod(path, 0o755)
    return path


def wait_health(port: int, timeout: float) -> dict:
    import json

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                return json.load(r)
        except Exception:
            time.sleep(1)
    raise SystemExit("the server did not come up; see server.log")


def show_link(link: str) -> None:
    say("")
    say("Open this on the phone (it is the site with the server's address in it):")
    say(f"  {link}")
    say("")
    try:
        import qrcode

        qr = qrcode.QRCode(border=1)
        qr.add_data(link)
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
        qr.print_ascii(invert=True)
    except Exception as e:  # noqa: BLE001
        say(f"(no QR code: {e})")
    with open(os.path.join(HERE, "link.txt"), "w") as f:
        f.write(link + "\n")
    say(f"(also in {os.path.join(HERE, 'link.txt')})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--no-lm", action="store_true")
    ap.add_argument("--no-tunnel", action="store_true", help="serve on this machine only")
    ap.add_argument("--token", default=os.environ.get("THELIP_TOKEN") or None)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    if not args.fake:
        ensure_chaplin()
        ensure_models()
    tunnel_bin = None if args.no_tunnel else cloudflared_path()

    log = open(os.path.join(HERE, "server.log"), "a", buffering=1)
    cmd = [sys.executable, os.path.join(HERE, "server.py"), "--port", str(args.port), "--beam", str(args.beam)]
    if args.no_lm:
        cmd.append("--no-lm")
    if args.token:
        cmd += ["--token", args.token]
    if args.fake:
        cmd.append("--fake")
    say("starting the server" + ("" if args.fake else " (loading the model takes a minute the first time)"))
    # UTF-8 mode: Chaplin opens its token list without an encoding, and a
    # Windows with a Hebrew (or any non-Latin) locale would decode it as cp1255.
    env = dict(os.environ, THELIP_HOME=WORK, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    tunnel = None
    try:
        health = wait_health(args.port, timeout=900)
        say(f"server up: {health.get('model')} on {health.get('device')}")
        origin = f"http://127.0.0.1:{args.port}"
        if tunnel_bin:
            say("opening the tunnel")
            tunnel = subprocess.Popen([tunnel_bin, "tunnel", "--url", origin, "--no-autoupdate"],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            deadline = time.time() + 120
            public = None
            while time.time() < deadline and tunnel.poll() is None:
                line = tunnel.stdout.readline()
                if not line:
                    break
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m:
                    public = m.group(0)
                    break
            if not public:
                raise SystemExit("the tunnel gave no address; run again, or use --no-tunnel for this machine only")
            origin = public
            # Keep draining the tunnel's log so its pipe never fills.
            import threading

            tlog = open(os.path.join(HERE, "tunnel.log"), "a", buffering=1)
            threading.Thread(target=lambda: [tlog.write(l) for l in iter(tunnel.stdout.readline, "")], daemon=True).start()
        link = f"{SITE}?server={origin}" + (f"&token={args.token}" if args.token else "")
        show_link(link)
        say("Ctrl+C stops the server.")
        while server.poll() is None:
            time.sleep(1)
        raise SystemExit(f"the server stopped (exit {server.returncode}); see server.log")
    except KeyboardInterrupt:
        say("stopping")
    finally:
        for p in (tunnel, server):
            if p and p.poll() is None:
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()


if __name__ == "__main__":
    main()
