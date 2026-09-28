"""A stand-in for YouTube's watch page for the end-to-end test: the same
selectors the extension's adapter looks for (#movie_player, a
video.html5-main-video, .ytp-right-controls), a plain <video> that plays a
fixture, served with HTTP range support so the browser can seek. The
fixtures are served as WebM (VP9/Opus): a Chromium build without the
proprietary codecs cannot play H.264.

    python3 fake_youtube.py --port 8791 --media <dir with <name>.webm>
"""
import argparse
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{name} - fake YouTube</title>
<style>body{{margin:0;background:#111;color:#eee;font-family:sans-serif}} #movie_player{{position:relative;width:720px;height:288px;background:#000}}
.html5-video-container{{position:absolute;inset:0}} video{{width:100%;height:100%}}
.ytp-chrome-bottom{{position:absolute;left:0;right:0;bottom:0;height:36px;background:rgba(0,0,0,.5)}}
.ytp-right-controls{{position:absolute;right:8px;top:4px;display:flex;gap:6px}} .ytp-button{{background:none;border:1px solid #888;color:#eee;height:28px}}</style></head>
<body>
<div id="movie_player" class="html5-video-player">
  <div class="html5-video-container"><video class="video-stream html5-main-video" src="/media/{name}.webm" preload="auto" playsinline></video></div>
  <div class="ytp-chrome-bottom"><div class="ytp-chrome-controls"><div class="ytp-right-controls"><button class="ytp-button">CC</button></div></div></div>
</div>
<h1 style="font-size:16px;padding:12px">{name}</h1>
</body></html>
"""


class Handler(SimpleHTTPRequestHandler):
    media_dir = "."

    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/watch":
            name = parse_qs(url.query).get("v", ["one_speaker"])[0]
            body = PAGE.format(name=re.sub(r"[^a-z0-9_]", "", name)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if url.path.startswith("/media/"):
            path = os.path.join(self.media_dir, os.path.basename(url.path))
            if not os.path.isfile(path):
                self.send_error(404)
                return
            size = os.path.getsize(path)
            start, end = 0, size - 1
            status = 200
            rng = self.headers.get("Range")
            if rng and rng.startswith("bytes="):
                a, b = rng[6:].split("-")
                start = int(a) if a else max(0, size - int(b))
                end = int(b) if b and a else size - 1
                status = 206
            self.send_response(status)
            self.send_header("Content-Type", "video/webm" if path.endswith(".webm") else "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(end - start + 1))
            return
        self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8791)
    ap.add_argument("--media", default=".")
    args = ap.parse_args()
    Handler.media_dir = os.path.abspath(args.media)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"fake YouTube on http://127.0.0.1:{args.port}/watch?v=<fixture>", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
