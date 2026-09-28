"""The API end to end with Starlette's test client: a job from an upload,
polling, export in every format, deletion, auth, rate limiting, a streamed
session of JPEG frames, and the honest health table.
"""
import io
import json
import os
import sys
import time
import unittest

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from starlette.testclient import TestClient  # noqa: E402

from lipreader_api import Config, create_app  # noqa: E402

FIXTURES = os.path.normpath(os.path.join(HERE, "..", "..", "lipreader", "tests", "fixtures"))
ONE = os.path.join(FIXTURES, "one_speaker.mp4")
TWO = os.path.join(FIXTURES, "two_alternating.mp4")
have_fixtures = os.path.exists(ONE) and os.environ.get("LIPREADER_QUICK") != "1"


def wait_done(client, job_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/jobs/{job_id}", params={"wait": 5})
        job = r.json()
        if job["status"] in ("done", "failed"):
            return job
    raise AssertionError("job did not finish")


class HealthAndAuth(unittest.TestCase):
    def test_health_lists_languages_honestly(self):
        client = TestClient(create_app(Config(token=None)))
        r = client.get("/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["processing"], "local")
        table = {l["code"]: l for l in body["languages"]}
        self.assertFalse(table["he"]["visual"]["available"])
        self.assertIn("audio", table["he"])
        self.assertIn("upload", body["fetchers"])

    def test_token_required_when_configured(self):
        client = TestClient(create_app(Config(token="s3cret")))
        self.assertEqual(client.get("/health").status_code, 401)
        self.assertEqual(client.get("/health", headers={"Authorization": "Bearer nope"}).status_code, 401)
        self.assertEqual(client.get("/health", headers={"Authorization": "Bearer s3cret"}).status_code, 200)

    def test_rate_limit(self):
        client = TestClient(create_app(Config(token=None, rate_limit_per_minute=2)))
        for _ in range(2):
            r = client.post("/jobs", json={"url": "ftp://nowhere"})
            self.assertEqual(r.status_code, 202)  # accepted (it will fail in the worker), and counted
        r = client.post("/jobs", json={"url": "ftp://nowhere"})
        self.assertEqual(r.status_code, 429)
        self.assertIn("Retry-After", r.headers)

    def test_bad_requests(self):
        client = TestClient(create_app(Config(token=None)))
        self.assertEqual(client.post("/jobs", json={}).status_code, 400)
        self.assertEqual(client.post("/jobs", json={"url": "http://x", "options": {"mode": "psychic"}}).status_code, 400)
        self.assertEqual(client.get("/jobs/nope").status_code, 404)
        self.assertEqual(client.delete("/jobs/nope").status_code, 404)
        self.assertEqual(client.get("/nowhere").status_code, 404)

    def test_local_files_are_off_by_default(self):
        client = TestClient(create_app(Config(token=None, allow_local_files=False)))
        r = client.post("/jobs", json={"url": "file:///etc/hostname"})
        self.assertEqual(r.status_code, 202)
        job = wait_done(client, r.json()["id"], timeout=30)
        self.assertEqual(job["status"], "failed")
        self.assertIn("not allowed", job["error"])


@unittest.skipUnless(have_fixtures, "fixtures missing or LIPREADER_QUICK=1")
class Jobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app(Config(token=None))
        cls.client = TestClient(cls.app)

    def test_upload_job_runs_exports_and_deletes(self):
        with open(TWO, "rb") as f:
            r = self.client.post("/jobs", files={"file": ("two.mp4", f, "video/mp4")},
                                 data={"options": json.dumps({"mode": "visual", "language": "en"})})
        self.assertEqual(r.status_code, 202)
        job = r.json()
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["processing"], "local")
        job = wait_done(self.client, job["id"])
        self.assertEqual(job["status"], "done", job.get("error"))
        result = job["result"]
        self.assertEqual(len(result["tracks"]), 2)
        self.assertEqual(len(result["segments"]), 2)
        self.assertIn("probabilistic", result["notice"])
        # The upload is gone from disk once read.
        self.assertFalse(any(name.endswith(".upload") for name in os.listdir(self.app.state.workdir)))
        for fmt in ("srt", "vtt", "json", "txt"):
            e = self.client.get(f"/jobs/{job['id']}/export", params={"format": fmt})
            self.assertEqual(e.status_code, 200, fmt)
            self.assertIn("attachment", e.headers["content-disposition"])
            self.assertTrue(e.text.strip(), fmt)
        srt_one = self.client.get(f"/jobs/{job['id']}/export", params={"format": "srt", "track": result["tracks"][0]["trackId"]}).text
        self.assertEqual(srt_one.count("-->"), 1)
        self.assertEqual(self.client.get(f"/jobs/{job['id']}/export", params={"format": "doc"}).status_code, 400)
        self.assertEqual(self.client.delete(f"/jobs/{job['id']}").status_code, 204)
        self.assertEqual(self.client.get(f"/jobs/{job['id']}").status_code, 404)

    def test_local_file_url_when_allowed(self):
        client = TestClient(create_app(Config(token=None, allow_local_files=True)))
        r = client.post("/jobs", json={"url": "file://" + ONE, "options": {"mode": "visual", "language": "en"}})
        self.assertEqual(r.status_code, 202)
        job = wait_done(client, r.json()["id"])
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["result"]["segments"][0]["text"], "set blue with e five now")
        self.assertEqual(job["source"], "file://" + ONE)

    def test_unavailable_language_gives_no_text_and_a_warning(self):
        with open(ONE, "rb") as f:
            r = self.client.post("/jobs", files={"file": ("one.mp4", f, "video/mp4")},
                                 data={"options": json.dumps({"mode": "visual", "language": "he"})})
        job = wait_done(self.client, r.json()["id"])
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["result"]["segments"], [])
        self.assertEqual(job["result"]["language"]["detection"], "unavailable")
        self.assertTrue(job["result"]["warnings"])

    def test_streamed_session(self):
        cap = cv2.VideoCapture(ONE)
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(bgr)
        cap.release()
        r = self.client.post("/sessions", json={"width": 360, "height": 288, "fps": 25, "options": {"mode": "visual", "language": "en"}})
        self.assertEqual(r.status_code, 201)
        sid = r.json()["id"]
        for start in range(0, len(frames), 25):
            batch = frames[start : start + 25]
            files = []
            for i, bgr in enumerate(batch):
                ok, jpeg = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                files.append(("frames", (f"f{start + i}.jpg", io.BytesIO(jpeg.tobytes()), "image/jpeg")))
            timestamps = json.dumps([(start + i) / 25.0 for i in range(len(batch))])
            r = self.client.post(f"/sessions/{sid}/frames", files=files, data={"timestamps": timestamps})
            self.assertEqual(r.status_code, 200, r.text)
            snapshot = r.json()
            self.assertEqual(snapshot["frames"], min(start + 25, len(frames)))
            self.assertTrue(snapshot["tracks"], "the face is tracked while streaming")
        r = self.client.post(f"/sessions/{sid}/finish")
        self.assertEqual(r.status_code, 200)
        result = r.json()["result"]
        self.assertEqual(len(result["segments"]), 1)
        self.assertGreaterEqual(sum(1 for a, b in zip("set blue with e five now".split(), result["segments"][0]["text"].split()) if a == b), 5)
        self.assertEqual(self.client.delete(f"/sessions/{sid}").status_code, 204)


if __name__ == "__main__":
    unittest.main(verbosity=2)
