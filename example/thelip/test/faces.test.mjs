// The page's face detector (faces.js with blazeface.bin) against MediaPipe's
// own, on the pictures in faces-ref.json (tools/faces_reference.py): the same
// faces, scores and keypoints; and how long one picture takes here.
//   node test/faces.test.mjs
import fs from "node:fs";
import path from "node:path";
import zlib from "node:zlib";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { FaceDetector, loadFaceModel, letterbox, unletterbox, FACE_INPUT, MOUTH } from "../faces.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const toBuffer = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const detector = new FaceDetector(loadFaceModel(toBuffer(fs.readFileSync(path.join(here, "..", "blazeface.bin")))));
const ref = JSON.parse(fs.readFileSync(path.join(here, "faces-ref.json"), "utf8"));
const pixels = zlib.gunzipSync(fs.readFileSync(path.join(here, "faces-ref.rgb.gz")));
let failures = 0;
const check = (name, ok, detail = "") => { console.log((ok ? "ok   " : "FAIL ") + name + (ok || !detail ? "" : `: ${detail}`)); if (!ok) failures++; };

const rgbaOf = (rgb) => { const out = new Uint8ClampedArray((rgb.length / 3) * 4); for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) { out[j] = rgb[i]; out[j + 1] = rgb[i + 1]; out[j + 2] = rgb[i + 2]; out[j + 3] = 255; } return out; };
// A picture into the 128x128 input as the page's canvas does it: letterboxed, bilinear, black around.
function letterboxed(rgb, w, h) {
  const { scale, dx, dy } = letterbox(w, h), S = FACE_INPUT, out = new Uint8ClampedArray(S * S * 4);
  for (let oy = 0; oy < S; oy++) for (let ox = 0; ox < S; ox++) {
    const o = (oy * S + ox) * 4; out[o + 3] = 255;
    const sx = (ox + 0.5 - dx) / scale - 0.5, sy = (oy + 0.5 - dy) / scale - 0.5;
    if (sx < -0.5 || sy < -0.5 || sx > w - 0.5 || sy > h - 0.5) continue;
    const x0 = Math.max(0, Math.min(w - 1, Math.floor(sx))), y0 = Math.max(0, Math.min(h - 1, Math.floor(sy)));
    const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1), fx = Math.max(0, Math.min(1, sx - x0)), fy = Math.max(0, Math.min(1, sy - y0));
    for (let c = 0; c < 3; c++) {
      const p = (x, y) => rgb[(y * w + x) * 3 + c];
      out[o + c] = (p(x0, y0) * (1 - fx) + p(x1, y0) * fx) * (1 - fy) + (p(x0, y1) * (1 - fx) + p(x1, y1) * fx) * fy;
    }
  }
  return out;
}
// Matches each of MediaPipe's faces to the nearest found one, by the mouth.
function compare(name, found, expected, tol) {
  if (found.length !== expected.length) return check(name, false, `found ${found.length} faces, MediaPipe ${expected.length}`);
  let worst = { score: 0, point: 0, box: 0 };
  for (const e of expected) {
    const f = found.reduce((best, g) => (Math.hypot(g.keypoints[MOUTH][0] - e.keypoints[MOUTH][0], g.keypoints[MOUTH][1] - e.keypoints[MOUTH][1]) < Math.hypot(best.keypoints[MOUTH][0] - e.keypoints[MOUTH][0], best.keypoints[MOUTH][1] - e.keypoints[MOUTH][1]) ? g : best));
    worst.score = Math.max(worst.score, Math.abs(f.score - e.score));
    e.keypoints.forEach(([x, y], k) => { worst.point = Math.max(worst.point, Math.abs(f.keypoints[k][0] - x), Math.abs(f.keypoints[k][1] - y)); });
    e.box.forEach((v, k) => { worst.box = Math.max(worst.box, Math.abs(f.box[k] - v)); });
  }
  check(`${name}: ${expected.length} face${expected.length === 1 ? "" : "s"}, score within ${worst.score.toFixed(4)}, keypoints within ${worst.point.toFixed(4)}, box within ${worst.box.toFixed(4)}`,
        worst.score <= tol.score && worst.point <= tol.point && worst.box <= tol.box, JSON.stringify(tol));
}

let offset = 0;
for (const p of ref.pictures) {
  const [w, h] = p.size;
  if (p.clip) {
    const rgb = execFileSync("ffmpeg", ["-v", "error", "-i", path.join(here, "..", "..", "..", "docs", "test", "grid", p.clip + ".webm"), "-vf", p.vf, "-vframes", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], { maxBuffer: 1 << 24 });
    const found = detector.detect(letterboxed(rgb, w, h)).map((f) => unletterbox(f, w, h));
    // MediaPipe letterboxes with its own resampling: within about a pixel of its input (2.8 of this frame).
    compare(p.name, found, p.faces, { score: 0.03, point: 0.012, box: 0.015 });
    continue;
  }
  const rgb = pixels.subarray(offset, offset + w * h * 3); offset += w * h * 3;
  // At the network's own size nothing is resampled: the same numbers to rounding
  // (MediaPipe's boxes are whole pixels, hence a box within a pixel).
  compare(p.name, detector.detect(rgbaOf(rgb)), p.faces, { score: 0.002, point: 0.0015, box: 1 / 128 + 0.001 });
}
const sample = rgbaOf(pixels.subarray(0, 128 * 128 * 3));
detector.detect(sample);
const t0 = performance.now();
for (let i = 0; i < 20; i++) detector.detect(sample);
console.log(`     ${((performance.now() - t0) / 20).toFixed(1)} ms per picture here`);
console.log(failures ? `${failures} FAILED` : "ALL PASSED");
if (failures) process.exit(1);
