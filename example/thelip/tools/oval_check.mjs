// How well LipNet reads GRID's ten sample clips (docs/test/grid) through an oval the
// page's face finder placed: each clip read whole, the oval at a share of the face's
// width, on the median of the mouth's sightings. What OVAL_OF_FACE in the page and
// its reading of a sightings' median were chosen from.
//   node tools/oval_check.mjs [first|early|spread] [shares, e.g. 0.8,0.85,0.9]
//     first: the first frame alone (a mouth still closed); early: frames 0, 5, 10;
//     spread (the page's, for a clip): a frame every 0.48 s over the first 2.9 s.
// Frames come from ffmpeg; the crop is the page's cropFrame (100x50, bilinear).
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { FaceDetector, loadFaceModel, letterbox, unletterbox, FACE_INPUT, MOUTH } from "../faces.js";
import { LipNet, loadWeights, decode, correct } from "../thelip.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const toBuffer = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const detector = new FaceDetector(loadFaceModel(toBuffer(fs.readFileSync(path.join(here, "..", "blazeface.bin")))));
const net = new LipNet(loadWeights(toBuffer(fs.readFileSync(path.join(here, "..", "thelip-weights.bin")))));
const GRID = path.join(here, "..", "..", "..", "docs", "test", "grid");
const SAID = { bbaf2n: "bin blue at f two now", brbk7n: "bin red by k seven now", lbax4n: "lay blue at x four now", lbbc2a: "lay blue by c two again",
  lrwp9a: "lay red with p nine again", lwbsza: "lay white by s zero again", pwij3p: "place white in j three please", sbia1a: "set blue in a one again",
  sbwe5n: "set blue with e five now", swiz3n: "set white in z three now" };
const [how = "spread", shares = "0.8,0.85,0.9,0.94"] = process.argv.slice(2);
const AT = { first: [0], early: [0, 5, 10], spread: [0, 12, 24, 36, 48, 60, 72] }[how];

function sampler(rgb, W, H) {
  return (sx, sy, c) => {
    const x0 = Math.max(0, Math.min(W - 1, Math.floor(sx))), y0 = Math.max(0, Math.min(H - 1, Math.floor(sy)));
    const x1 = Math.min(W - 1, x0 + 1), y1 = Math.min(H - 1, y0 + 1), fx = Math.max(0, Math.min(1, sx - x0)), fy = Math.max(0, Math.min(1, sy - y0));
    const p = (x, y) => rgb[(y * W + x) * 3 + c];
    return (p(x0, y0) * (1 - fx) + p(x1, y0) * fx) * (1 - fy) + (p(x0, y1) * (1 - fx) + p(x1, y1) * fx) * fy;
  };
}
function letterboxed(rgb, W, H) {
  const at = sampler(rgb, W, H), { scale, dx, dy } = letterbox(W, H), S = FACE_INPUT, out = new Uint8ClampedArray(S * S * 4);
  for (let oy = 0; oy < S; oy++) for (let ox = 0; ox < S; ox++) {
    const o = (oy * S + ox) * 4, sx = (ox + 0.5 - dx) / scale - 0.5, sy = (oy + 0.5 - dy) / scale - 0.5;
    out[o + 3] = 255;
    if (sx >= -0.5 && sy >= -0.5 && sx <= W - 0.5 && sy <= H - 0.5) for (let c = 0; c < 3; c++) out[o + c] = at(sx, sy, c);
  }
  return out;
}
function crop(rgb, W, H, g) {
  const at = sampler(rgb, W, H), out = new Float32Array(100 * 50 * 3);
  for (let y = 0; y < 50; y++) for (let x = 0; x < 100; x++) {
    const sx = g.x + ((x + 0.5) * g.w) / 100 - 0.5, sy = g.y + ((y + 0.5) * g.h) / 50 - 0.5, o = (x * 50 + y) * 3;
    for (let c = 0; c < 3; c++) out[o + c] = Math.round(at(sx, sy, c)) / 255;
  }
  return out;
}
const W = 360, H = 288, clips = {};
for (const clip of Object.keys(SAID)) {
  const raw = execFileSync("ffmpeg", ["-v", "error", "-i", path.join(GRID, clip + ".webm"), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], { maxBuffer: 1 << 26 });
  const frames = [];
  for (let i = 0; i < raw.length / (W * H * 3); i++) frames.push(raw.subarray(i * W * H * 3, (i + 1) * W * H * 3));
  const seen = AT.filter((i) => i < frames.length).map((i) => detector.detect(letterboxed(frames[i], W, H)).map((f) => unletterbox(f, W, H)).sort((a, b) => b.box[2] - a.box[2])[0]).filter(Boolean);
  const med = (v) => v.slice().sort((a, b) => a - b)[v.length >> 1];
  clips[clip] = { frames, mouth: [med(seen.map((f) => f.keypoints[MOUTH][0] * W)), med(seen.map((f) => f.keypoints[MOUTH][1] * H))], face: med(seen.map((f) => f.box[2] * W)) };
}
for (const share of shares.split(",").map(Number)) {
  let right = 0, total = 0;
  const each = [];
  for (const [clip, said] of Object.entries(SAID)) {
    const { frames, mouth, face } = clips[clip], w = face * share, g = { x: mouth[0] - w / 2, y: mouth[1] - w / 4, w, h: w / 2 };
    const probs = net.predict(frames.map((f) => crop(f, W, H, g)));
    const got = decode(probs, frames.length).map((x) => correct(x.text)), want = said.split(" ");
    const hits = want.filter((x, i) => got[i] === x).length;
    right += hits; total += want.length; each.push(`${clip} ${hits}`);
  }
  console.log(`oval ${share} of the face, mouth from ${how}: ${right} of ${total} words (${each.join(", ")})`);
}
