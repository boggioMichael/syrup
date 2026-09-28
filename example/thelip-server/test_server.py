"""thelip-server without the model: the routes, the multipart contract, CORS,
the fps resampling and the error answers, against `--fake`.

    python3 test_server.py
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8797


def jpeg(w=64, h=48, shade=128) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (shade, shade, shade)).save(buf, "JPEG")
    return buf.getvalue()


def multipart(fields, files):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    for k, v in fields:
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    for k, name, data in files:
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{name}\"\r\nContent-Type: image/jpeg\r\n\r\n".encode())
        body.write(data)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", body.getvalue()


def call(method, path, data=None, ctype=None, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method=method)
    if ctype:
        req.add_header("Content-Type", ctype)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers), json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), json.loads(e.read() or b"null")


def main() -> int:
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py"), "--fake", "--port", str(PORT), "--token", "t0k"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(("ok   " if cond else "FAIL ") + name + (f": {detail}" if detail and not cond else ""))
        failures += 0 if cond else 1

    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        status, headers, body = call("GET", "/health", headers={"Origin": "https://thelip.ai"})
        check("health answers", status == 200 and body["ok"] is True and body["fake"] is True, body)
        check("CORS allows thelip.ai", headers.get("access-control-allow-origin") == "*", headers)

        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/read", method="OPTIONS")
        req.add_header("Origin", "https://thelip.ai")
        req.add_header("Access-Control-Request-Method", "POST")
        req.add_header("Access-Control-Request-Headers", "authorization")
        with urllib.request.urlopen(req, timeout=10) as r:
            pre = dict(r.headers)
        check("preflight allows POST with Authorization",
              "POST" in pre.get("access-control-allow-methods", "") and "authorization" in pre.get("access-control-allow-headers", "").lower(), pre)

        ctype, data = multipart([("fps", "25")], [("frames", f"f{i:04d}.jpg", jpeg()) for i in range(30)])
        status, _, body = call("POST", "/read", data, ctype)
        check("token required", status == 401, body)
        auth = {"Authorization": "Bearer t0k"}
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("30 frames at 25 fps read", status == 200 and body["text"] == "FAKE READING OF 30 FRAMES" and body["frames"] == 30, body)

        ctype, data = multipart([("fps", "30")], [("frames", f"f{i:04d}.jpg", jpeg()) for i in range(30)])
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("30 fps resampled to 25 frames", status == 200 and body["frames"] == 25, body)

        ctype, data = multipart([("fps", "25")], [])
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("no frames is 400", status == 400, body)

        ctype, data = multipart([("fps", "25")], [("frames", "a.jpg", jpeg(64, 48)), ("frames", "b.jpg", jpeg(32, 24))])
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("mixed sizes is 400", status == 400, body)
    finally:
        proc.terminate()
        try:
            out = proc.communicate(timeout=10)[0]
        except subprocess.TimeoutExpired:
            proc.kill()
            out = proc.communicate()[0]
    if failures:
        print(out)
    print("ALL PASSED" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
