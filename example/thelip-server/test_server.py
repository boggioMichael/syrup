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


def jpeg2(left, right, w=96, h=40) -> bytes:
    """Two halves of different shades: two "faces" for the fake server."""
    from PIL import Image

    im = Image.new("RGB", (w, h), (right, right, right))
    im.paste(Image.new("RGB", (w // 2, h), (left, left, left)), (0, 0))
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return buf.getvalue()


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


def wav(seconds=1.0, rate=16000, channels=1, amplitude=12000) -> bytes:
    """A WAV file of a 440 Hz tone (amplitude 0: digital silence)."""
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate)
        n = int(seconds * rate)
        w.writeframes(b"".join(struct.pack("<h", int(amplitude * math.sin(2 * math.pi * 440 * i / rate))) * channels for i in range(n)))
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
        raw = e.read() or b"null"
        try:
            return e.code, dict(e.headers), json.loads(raw)
        except ValueError:
            return e.code, dict(e.headers), {"error": raw.decode("utf-8", "replace")[:300]}


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

    # Following several faces through a stretch of frames (faces.py).
    sys.path.insert(0, HERE)
    import numpy as np
    from faces import activity, track_faces

    kp = lambda x, y: np.array([[x - 10, y - 10], [x + 10, y - 10], [x, y], [x, y + 15]])  # noqa: E731
    dets = []
    for t in range(20):
        frame = [((100 + t, 50, 80, 100), kp(140 + t, 100)), ((400 - t, 60, 50, 60), kp(425 - t, 90))]
        if t in (7,):
            frame = frame[:1]                     # the small face missed once
        if t < 5:
            frame.append(((250, 10, 20, 20), kp(260, 20)))   # a fleeting third face
        dets.append(frame)
    tracks = track_faces(dets, 640, 360)
    check("two faces followed through the frames, the largest first, a fleeting one dropped",
          len(tracks) == 2 and tracks[0]["box"][2] > tracks[1]["box"][2] and abs(tracks[0]["box"][0] - (109.5 / 640)) < 0.01, [t["box"] for t in tracks])
    check("a frame where a face was missed is left for the alignment to fill", tracks[1]["landmarks"][7] is None and tracks[1]["landmarks"][8] is not None)
    still = np.tile(np.random.default_rng(1).normal(size=(1, 16)), (30, 1))
    moving = np.random.default_rng(2).normal(size=(30, 16))
    check("activity: a still mouth near 0, a moving one well above", activity(still) < 1e-6 and activity(moving) > 0.5, (activity(still), activity(moving)))

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
    # ... and the speech worker's, the same way.
    ear = server.Transcriber("medium", "cpu", fake_worker=True)
    check("the speech worker starts and hears over the pipe", ear.hear(wav(1.0), "en")["heard"] == "FAKE HEARD 1.0s" and ear.proc is not None)
    ear.proc.kill(); ear.proc.wait()
    check("a speech worker that died is started again", ear.hear(wav(0.5), "he")["heard"] == "FAKE HEARD 0.5s")
    try:
        ear.hear(b"not a wav", "en"); check("a bad wav is refused before the worker", False)
    except Exception:  # noqa: BLE001
        check("a bad wav is refused before the worker", True)
    ear.stop()
    check("stop ends the speech worker", ear.proc is None)

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
        ctype, data = multipart([("language", "en")], [("audio", "u.wav", wav(1.5, amplitude=0))])
        data = data.replace(b"Content-Type: image/jpeg", b"Content-Type: audio/wav")
        status, _, body = call("POST", "/hear", data, ctype, auth)
        check("silence is reported as silence, with its level, and not transcribed", status == 200 and body["silent"] is True and body["heard"] == "" and body["level_db"] <= -100, body)
        ctype, data = multipart([("language", "en")], [("audio", "u.wav", b"not a wav at all")])
        status, _, body = call("POST", "/hear", data, ctype, auth)
        check("a file that is not a wav is 400", status == 400, body)

        # A language with no model, read from the phrases this reader kept with the microphone on.
        P = "f" * 32
        ramp = lambda n, a, b: [("frames", f"f{i:04d}.jpg", jpeg(shade=int(a + (b - a) * i / max(1, n - 1)))) for i in range(n)]  # noqa: E731
        ctype, data = multipart([("fps", "25"), ("language", "ar"), ("profile", P)], ramp(30, 40, 220))
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("Arabic (no model at all) with a profile but nothing learned yet answers 200 with nothing", status == 200 and body["text"] == "" and body["how"] == "none", body)
        for utt, frames, seconds in (("aa11aa11aa11", ramp(30, 40, 220), 1.0), ("bb22bb22bb22", ramp(30, 220, 40), 2.0), ("cc33cc33cc33", ramp(28, 50, 210), 1.0)):
            ctype, data = multipart([("fps", "25"), ("language", "ar"), ("profile", P), ("improve", "1"), ("utt", utt)], frames)
            status, _, body = call("POST", "/read", data, ctype, auth)
            ctype, data = multipart([("language", "ar"), ("utt", utt), ("improve", "1")], [("audio", "u.wav", wav(seconds))])
            data = data.replace(b"Content-Type: image/jpeg", b"Content-Type: audio/wav")
            call("POST", "/hear", data, ctype, auth)
        ctype, data = multipart([("fps", "25"), ("language", "ar"), ("profile", P)], ramp(26, 45, 215))
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("a phrase said before with the microphone on is read from the lips alone", status == 200 and body["how"] == "learned" and body["text"] == "FAKE HEARD 1.0s", body)
        ctype, data = multipart([("fps", "25"), ("language", "ar"), ("profile", P)], ramp(34, 210, 50))
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("the other phrase too", status == 200 and body["how"] == "learned" and body["text"] == "FAKE HEARD 2.0s", body)
        ctype, data = multipart([("fps", "25"), ("language", "ar"), ("profile", P)], [("frames", f"f{i:04d}.jpg", jpeg(shade=10)) for i in range(30)])
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("a movement unlike any phrase is not read as one", status == 200 and body["how"] == "unmatched" and body["text"] == "", body)
        status, _, body = call("GET", f"/phrases?profile={P}&language=ar")
        check("the reader's phrases are listed with their example counts", status == 200 and [(p["text"], p["examples"]) for p in body["phrases"]] == [("FAKE HEARD 1.0s", 2), ("FAKE HEARD 2.0s", 1)], body)
        ctype, data = multipart([("fps", "25"), ("language", "ar")], ramp(30, 40, 220))
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("without a profile Arabic is still 503", status == 503, body)

        # Two faces: the one whose mouth moves is read, the still one is not.
        frames = [("frames", f"f{i:04d}.jpg", jpeg2(60 + (i % 5) * 35, 120)) for i in range(30)]
        ctype, data = multipart([("fps", "25")], frames)
        status, _, body = call("POST", "/read", data, ctype, auth)
        faces = body.get("faces") or []
        check("several faces: each with its box and whether it spoke; the speaking one read and main",
              status == 200 and len(faces) == 2 and faces[0]["speaking"] and faces[0]["main"] and faces[0]["text"] == "FAKE EN FACE 1 READING OF 30 FRAMES"
              and not faces[1]["speaking"] and faces[1]["text"] is None and body["text"] == faces[0]["text"] and len(faces[0]["box"]) == 4, body)
        frames = [("frames", f"f{i:04d}.jpg", jpeg2(60 + (i % 5) * 35, 40 + (i % 3) * 60)) for i in range(30)]
        ctype, data = multipart([("fps", "25")], frames)
        status, _, body = call("POST", "/read", data, ctype, auth)
        check("two people speaking at once are both read", status == 200 and [f["text"] for f in body.get("faces", [])] == ["FAKE EN FACE 1 READING OF 30 FRAMES", "FAKE EN FACE 2 READING OF 30 FRAMES"], body)

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
