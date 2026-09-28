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
from typing import Optional

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
# A Cloudflare named tunnel (a fixed address such as https://api.thelip.ai)
# instead of a throwaway one: the tunnel's token in tunnel-token.txt, the
# address it routes in public-url.txt; both next to this file, both ignored by git.
TOKEN_FILE = os.path.join(HERE, "tunnel-token.txt")
PUBLIC_FILE = os.path.join(HERE, "public-url.txt")


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


def try_health(url: str, timeout: float = 3) -> Optional[dict]:
    import json

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "thelip-server/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.load(r)
            return body if isinstance(body, dict) and body.get("ok") else None
    except Exception:  # noqa: BLE001
        return None


def wait_health(port: int, timeout: float) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        health = try_health(f"http://127.0.0.1:{port}/health")
        if health:
            return health
        time.sleep(1)
    raise SystemExit("the server did not come up; see server.log")


class Tunnel:
    """One cloudflared quick tunnel to the local server. Every line it prints
    goes to tunnel.log; the public address is taken from those lines; the
    tunnel counts as open only once its /health answers from the outside,
    since a fresh quick tunnel can sit on Cloudflare's error 1033 for a while
    or never route at all, in which case it is started again."""

    def __init__(self, binary: str, origin: str, token: Optional[str] = None, fixed: Optional[str] = None):
        self.binary, self.origin = binary, origin
        self.token, self.fixed = token, fixed
        self.proc: Optional[subprocess.Popen] = None
        self.public: Optional[str] = None
        self.log = open(os.path.join(HERE, "tunnel.log"), "a", buffering=1)

    def _start(self) -> None:
        import threading

        self.public = self.fixed if self.token else None
        cmd = [self.binary, "tunnel", "run", "--token", self.token] if self.token else [self.binary, "tunnel", "--url", self.origin]
        self.proc = subprocess.Popen(cmd + ["--no-autoupdate"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")

        def drain():
            for line in iter(self.proc.stdout.readline, ""):
                self.log.write(line)
                m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
                if m and not self.public:
                    self.public = m.group(0)

        threading.Thread(target=drain, daemon=True).start()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def open(self, attempts: int = 3) -> str:
        for attempt in range(1, attempts + 1):
            self._start()
            deadline = time.time() + 60
            while time.time() < deadline and not self.public and self.proc.poll() is None:
                time.sleep(0.5)
            if self.public:
                say(f"tunnel address: {self.public}; checking that it answers")
                deadline = time.time() + 90
                while time.time() < deadline and self.proc.poll() is None:
                    if try_health(self.public + "/health", timeout=8):
                        return self.public
                    time.sleep(3)
            say(f"the tunnel did not answer (attempt {attempt} of {attempts}); starting it again")
            self.stop()
        raise SystemExit("the tunnel never answered; run again, or use --no-tunnel for this machine only")

    def alive(self) -> bool:
        return bool(self.proc and self.proc.poll() is None and self.public and try_health(self.public + "/health", timeout=8))


def publish_pointer(url: Optional[str]) -> bool:
    """Tell thelip.ai where the server is: docs/server.json on the site's
    gh-pages branch, which the page reads when it opens. Best effort: needs
    git and push rights on this machine; says so when it cannot."""
    git = shutil.which("git")
    if not git:
        say("(git is not on this machine, so thelip.ai cannot be told the address; the link below still works)")
        return False
    root = os.path.normpath(os.path.join(HERE, "..", ".."))
    try:
        remote = subprocess.check_output([git, "-C", root, "remote", "get-url", "origin"], text=True, stderr=subprocess.STDOUT).strip()
    except (subprocess.CalledProcessError, OSError):
        say("(this folder is not a git checkout of syrup, so thelip.ai cannot be told the address; the link below still works)")
        return False
    pages = os.path.join(WORK, "pages")
    try:
        if not os.path.isdir(os.path.join(pages, ".git")):
            subprocess.check_output([git, "clone", "-q", "--branch", "gh-pages", "--single-branch", "--depth", "1", remote, pages], stderr=subprocess.STDOUT, text=True)
        else:
            subprocess.check_output([git, "-C", pages, "fetch", "-q", "--depth", "1", "origin", "gh-pages"], stderr=subprocess.STDOUT, text=True)
            subprocess.check_output([git, "-C", pages, "reset", "-q", "--hard", "origin/gh-pages"], stderr=subprocess.STDOUT, text=True)
        import json

        with open(os.path.join(pages, "server.json"), "w", encoding="utf-8") as f:
            json.dump({"url": url, "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
        subprocess.check_output([git, "-C", pages, "add", "server.json"], stderr=subprocess.STDOUT, text=True)
        if subprocess.call([git, "-C", pages, "diff", "--cached", "--quiet"]) == 0:
            return True  # already says so
        subprocess.check_output([git, "-C", pages, "-c", "user.name=thelip-server", "-c", "user.email=thelip-server@thelip.ai",
                                 "commit", "-q", "-m", f"server: {url or 'off'}"], stderr=subprocess.STDOUT, text=True)
        subprocess.check_output([git, "-C", pages, "push", "-q", "origin", "HEAD:gh-pages"], stderr=subprocess.STDOUT, text=True)
        say(f"thelip.ai told: server {'at ' + url if url else 'off'} (live there in about a minute)")
        return True
    except (subprocess.CalledProcessError, OSError) as e:
        detail = getattr(e, "output", None) or str(e)
        say(f"(could not tell thelip.ai the address — git said: {str(detail).strip()[:300]}; the link below still works)")
        return False


def pointer_on_site() -> Optional[str]:
    import json

    try:
        with urllib.request.urlopen(f"{SITE}server.json?t={int(time.time())}", timeout=10) as r:
            return json.load(r).get("url")
    except Exception:  # noqa: BLE001
        return None


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
    ap.add_argument("--no-publish", action="store_true", help="do not write the address to thelip.ai's server.json")
    ap.add_argument("--token", default=os.environ.get("THELIP_TOKEN") or None)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    if not args.fake:
        ensure_chaplin()
        ensure_models()
    tunnel_bin = None if args.no_tunnel else cloudflared_path()

    origin = f"http://127.0.0.1:{args.port}"
    server = None
    if try_health(origin + "/health"):
        say(f"a thelip-server is already running on this computer at {origin}; using it (close its window to stop it)")
    else:
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
        public = origin
        if tunnel_bin:
            token = open(TOKEN_FILE, encoding="utf-8").read().strip() if os.path.isfile(TOKEN_FILE) else None
            fixed = open(PUBLIC_FILE, encoding="utf-8").read().strip().rstrip("/") if os.path.isfile(PUBLIC_FILE) else "https://api.thelip.ai"
            say("opening the tunnel" + (f" to {fixed} (named tunnel)" if token else " (a throwaway address; see README for a fixed one)"))
            tunnel = Tunnel(tunnel_bin, origin, token=token, fixed=fixed if token else None)
            public = tunnel.open()
        published = False if args.token or args.no_tunnel or args.no_publish else publish_pointer(public)
        link = f"{SITE}" if published else f"{SITE}?server={public}" + (f"&token={args.token}" if args.token else "")
        show_link(link)
        say("Ctrl+C stops the server.")
        misses, ticks = 0, 0
        while server is None or server.poll() is None:
            time.sleep(30)
            ticks += 1
            if tunnel is None:
                continue
            if tunnel.alive():
                misses = 0
                if published and ticks % 20 == 0 and pointer_on_site() != public:
                    publish_pointer(public)
                continue
            misses += 1
            if misses >= 3:
                say("the tunnel stopped answering; opening a new one")
                tunnel.stop()
                public = tunnel.open()
                published = False if args.token or args.no_publish else publish_pointer(public)
                link = f"{SITE}" if published else f"{SITE}?server={public}" + (f"&token={args.token}" if args.token else "")
                show_link(link)
                misses = 0
        raise SystemExit(f"the server stopped (exit {server.returncode}); see server.log")
    except KeyboardInterrupt:
        say("stopping")
    finally:
        if tunnel:
            tunnel.stop()
            if not (args.token or args.no_publish) and tunnel.public and not tunnel.token:
                publish_pointer(None)
        if server and server.poll() is None:
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
