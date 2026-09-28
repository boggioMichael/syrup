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
DATA = os.path.join(HERE, "test-data")


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


def wav(seconds=1.0, rate=16000, channels=1) -> bytes:
    """A WAV file of a 440 Hz tone."""
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate)
        n = int(seconds * rate)
        w.writeframes(b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * i / rate))) * channels for i in range(n)))
    return buf.getvalue()


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


def fake_export(models_dir: str) -> None:
    """What example/thelip-train leaves in models/ for Hebrew, minus the weights."""
    folder = os.path.join(models_dir, "he-thelip-v1")
    os.makedirs(folder, exist_ok=True)
    open(os.path.join(folder, "model.pth"), "wb").write(b"not a real model")
    json.dump({"name": "he-thelip-v1", "language": "he", "results": {"v1": {"overall": 0.61, "unit": "word", "by_source": {"phone": {"n": 40, "rate": 0.58}}}}},
              open(os.path.join(folder, "info.json"), "w"))


def main() -> int:
    import shutil

    shutil.rmtree(DATA, ignore_errors=True)
    models = os.path.join(HERE, "test-models")
    shutil.rmtree(models, ignore_errors=True)
    fake_export(models)
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py"), "--fake", "--port", str(PORT), "--token", "t0k"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, THELIP_DATA=DATA, THELIP_MODELS=models))
    failures = 0

    def check(name, cond, detail=""):
        nonlocal failures
        print(("ok   " if cond else "FAIL ") + name + (f": {detail}" if detail and not cond else ""))
        failures += 0 if cond else 1

    # The worker protocol, without torch: the server's client end against trained_worker.py --fake.
    sys.argv.append("--fake")
    sys.path.insert(0, HERE)
    import numpy as np
    import server

    worker = server.TrainedModel("nowhere", "cpu", fake=True)
    check("a trained model's worker starts and answers over the pipe", worker.info.get("ready") and worker.infer(np.zeros((30, 96, 96), np.uint8)) == "FAKE TRAINED 30 FRAMES")
    worker.proc.kill()
    worker.proc.wait()
    check("a worker that died is started again", worker.infer(np.zeros((7, 96, 96), np.uint8)) == "FAKE TRAINED 7 FRAMES")
    worker.stop()
    check("stop ends the worker", worker.proc is None)
    try:
        worker.infer(np.zeros((3, 100, 100), np.uint8))
        check("wrong crop size is refused", False)
    except ValueError:
        check("wrong crop size is refused", True)
    worker.stop()

    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        status, headers, body = call("GET", "/health", headers={"Origin": "https://thelip.ai"})
        check("health answers", status == 200 and body["ok"] is True and body["fake"] is True, body)
        check("health carries the server version run.py compares", body.get("version") == server.SERVER_VERSION, body.get("version"))
        check("health says the server hears, with ivrit.ai's Whisper for Hebrew", body.get("hears") is True and body.get("whisper", {}).get("he", "").startswith("ivrit-ai/"), body.get("whisper"))
        langs = {l["code"]: l for l in body.get("languages", [])}
        check("health lists the languages, Hebrew first after English", [l["code"] for l in body["languages"]][:2] == ["en", "he"] and langs["ar"]["available"] is False and "status" in langs["ar"], body.get("languages"))
        check("the runnable ones carry their quality and licence", langs["es"]["available"] and "44.5%" in langs["es"]["quality"] and "non-commercial" in langs["es"]["licence"], langs.get("es"))
        check("a trained export makes its language available, with its measured quality",
              langs["he"]["available"] is True and langs["he"]["trained"] == "he-thelip-v1" and "61.0%" in langs["he"]["quality"] and "status" not in langs["he"], langs.get("he"))
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
        check("30 frames at 25 fps read", status == 200 and body["text"] == "FAKE EN READING OF 30 FRAMES" and body["frames"] == 30 and body["language"] == "en", body)
        ctype, data = multipart([("fps", "25"), ("language", "es")], [("frames", f"f{i:04d}.jpg", jpeg()) for i in range(10)])
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("a language goes with the read", status == 200 and body["text"] == "FAKE ES READING OF 10 FRAMES" and body["language"] == "es", body)
        ctype, data = multipart([("fps", "25"), ("language", "he")], [("frames", f"f{i:04d}.jpg", jpeg()) for i in range(8)])
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("a language with only a trained model reads", status == 200 and body["language"] == "he" and body["text"] == "FAKE HE READING OF 8 FRAMES", body)
        ctype, data = multipart([("fps", "25"), ("language", "ar")], [("frames", "a.jpg", jpeg())])
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("a language without a model is 503 with the reason", status == 503 and "Arabic" in body["error"], body)
        ctype, data = multipart([("fps", "25"), ("language", "xx")], [("frames", "a.jpg", jpeg())])
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("an unknown language is 400", status == 400, body)

        ctype, data = multipart([("fps", "30")], [("frames", f"f{i:04d}.jpg", jpeg()) for i in range(30)])
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("30 fps resampled to 25 frames", status == 200 and body["frames"] == 25, body)

        # Samples are kept only when asked; feedback joins them.
        ctype, data = multipart([("fps", "25"), ("improve", "1"), ("source", "camera")], [("frames", f"f{i:04d}.jpg", jpeg()) for i in range(12)])
        status, _, body = call("POST", "/read", data, ctype, auth)
        sample = body.get("id")
        check("improve=1 keeps a sample and returns its id", status == 200 and isinstance(sample, str) and len(sample) == 32, body)
        folder = os.path.join(DATA, sample or "none")
        check("the sample holds the crops and the reading", os.path.isfile(os.path.join(folder, "crops.npy")) and json.load(open(os.path.join(folder, "meta.json")))["raw"] == "FAKE EN READING OF 12 FRAMES", os.listdir(folder) if os.path.isdir(folder) else "no folder")
        import numpy as np

        crops = np.load(os.path.join(folder, "crops.npy"))
        check("crops are (T, 96, 96) uint8", crops.shape == (12, 96, 96) and crops.dtype == np.uint8, str(crops.shape))
        status, _, body = call("POST", "/feedback", json.dumps({"id": sample, "raw": "FAKE EN READING OF 12 FRAMES", "corrected": "hello there"}).encode(), "application/json", auth)
        meta = json.load(open(os.path.join(folder, "meta.json")))
        check("feedback records the correction", status == 200 and meta.get("corrected") == "hello there" and meta.get("confirmed") is False, meta)
        status, _, body = call("POST", "/feedback", json.dumps({"id": "../evil", "raw": "", "corrected": ""}).encode(), "application/json", auth)
        check("feedback refuses a bad id", status == 404, body)
        ctype, data = multipart([("fps", "25")], [("frames", f"f{i:04d}.jpg", jpeg()) for i in range(5)])
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("without improve nothing is kept", status == 200 and body.get("id") is None and len(os.listdir(DATA)) == 1, body)

        # The sound of an utterance labels the kept sample, whichever request lands first.
        ctype, data = multipart([("language", "en"), ("utt", "aabbccdd0011"), ("improve", "1")], [("audio", "u.wav", wav(1.0))])
        data = data.replace(b"Content-Type: image/jpeg", b"Content-Type: audio/wav")
        status, _, body = call("POST", "/hear", data, ctype, auth)
        check("the sound is heard (fake) before the frames arrive", status == 200 and body["heard"] == "FAKE HEARD 1.0s" and body["id"] is None and body["seconds"] == 1.0, body)
        ctype, data = multipart([("fps", "25"), ("improve", "1"), ("utt", "aabbccdd0011")], [("frames", f"f{i:04d}.jpg", jpeg()) for i in range(10)])
        status, _, body = call("POST", "/read", data, ctype, auth)
        meta = json.load(open(os.path.join(DATA, body["id"], "meta.json")))
        check("the read that follows keeps the sample with what was heard", meta.get("heard") == "FAKE HEARD 1.0s" and meta.get("heard_confidence") == 0.9 and meta.get("utt") == "aabbccdd0011", meta)
        ctype, data = multipart([("fps", "25"), ("improve", "1"), ("utt", "0011aabbccdd")], [("frames", f"f{i:04d}.jpg", jpeg()) for i in range(10)])
        status, _, body = call("POST", "/read", data, ctype, auth)
        sid = body["id"]
        ctype, data = multipart([("language", "en"), ("utt", "0011aabbccdd"), ("improve", "1")], [("audio", "u.wav", wav(0.5, rate=48000, channels=2))])
        data = data.replace(b"Content-Type: image/jpeg", b"Content-Type: audio/wav")
        status, _, body = call("POST", "/hear", data, ctype, auth)
        meta = json.load(open(os.path.join(DATA, sid, "meta.json")))
        check("the sound that follows a read joins its sample (48 kHz stereo resampled)", status == 200 and body["id"] == sid and body["heard"] == "FAKE HEARD 0.5s" and meta.get("heard") == "FAKE HEARD 0.5s", (body, meta))
        ctype, data = multipart([("language", "en"), ("utt", "0011aabbccdd")], [("audio", "u.wav", wav(0.5))])
        data = data.replace(b"Content-Type: image/jpeg", b"Content-Type: audio/wav")
        status, _, body = call("POST", "/hear", data, ctype, auth)
        check("without improve the sound is heard and nothing is written", status == 200 and body["id"] is None, body)
        ctype, data = multipart([("language", "en")], [("audio", "u.wav", b"not a wav at all")])
        status, _, body = call("POST", "/hear", data, ctype, auth)
        check("a file that is not a wav is 400", status == 400, body)

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
    shutil.rmtree(DATA, ignore_errors=True)
    shutil.rmtree(models, ignore_errors=True)
    if failures:
        print(out)
    print("ALL PASSED" if not failures else f"{failures} FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
