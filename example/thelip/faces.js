/**
 * Faces in a frame, in the browser: BlazeFace (short range), the face
 * detector MediaPipe ships (Apache 2.0), run from its own weights with no
 * TFLite runtime. blazeface.bin holds the network as a list of ops and its
 * weights as float16 (tools/export_blazeface.py); this file runs those ops
 * on a 128x128 picture and turns the network's 896 anchor outputs into
 * faces the way MediaPipe's face detector graph does: SSD anchors (strides
 * 8, 16, 16, 16; two per cell and layer; fixed size), boxes and six
 * keypoints decoded against them, a sigmoid score of at least 0.5, and
 * weighted non-maximum suppression at an overlap of 0.3.
 *
 * A face: { score, box: [x, y, w, h], keypoints: [[x, y] x 6] }, all as
 * fractions of the picture. Keypoints: right eye, left eye, nose tip,
 * mouth centre, right ear, left ear (the person's right and left).
 */
export const FACE_INPUT = 128;
export const MOUTH = 3, NOSE = 2, RIGHT_EYE = 0, LEFT_EYE = 1;

function halfToFloat(h) {
  const out = new Float32Array(h.length);
  for (let i = 0; i < h.length; i++) {
    const x = h[i], sign = x & 0x8000 ? -1 : 1, e = (x >> 10) & 0x1f, m = x & 0x3ff;
    out[i] = e === 0 ? sign * m * 5.960464477539063e-8 : e === 31 ? (m ? NaN : sign * Infinity) : sign * (1 + m / 1024) * 2 ** (e - 15);
  }
  return out;
}

export function loadFaceModel(buffer) {
  const bytes = new Uint8Array(buffer);
  if (String.fromCharCode(...bytes.subarray(0, 4)) !== "BLZF") throw new Error("not a blazeface.bin");
  const n = new DataView(buffer).getUint32(4, true);
  const spec = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + n)));
  const half = new Uint16Array(buffer, 8 + n);
  const consts = {};
  for (const [id, [off, count]] of Object.entries(spec.consts)) consts[id] = halfToFloat(half.subarray(off, off + count));
  return { spec, consts };
}

// ---- the ops (NHWC, batch 1) --------------------------------------------------------------------------
function samePad(size, out, k, stride) { return Math.floor(Math.max((out - 1) * stride + k - size, 0) / 2); }

// Four output channels at a time (each input value is read once for the four);
// a larger kernel goes through its input patch gathered once per output pixel.
function dot4(x, xo, w, C, co, b, y, yo, relu, CO) {
  let c = co;
  for (; c + 3 < CO; c += 4) {
    let s0 = b[c], s1 = b[c + 1], s2 = b[c + 2], s3 = b[c + 3];
    const w0 = c * C, w1 = w0 + C, w2 = w1 + C, w3 = w2 + C;
    for (let i = 0; i < C; i++) { const v = x[xo + i]; s0 += v * w[w0 + i]; s1 += v * w[w1 + i]; s2 += v * w[w2 + i]; s3 += v * w[w3 + i]; }
    y[yo + c] = relu && s0 < 0 ? 0 : s0; y[yo + c + 1] = relu && s1 < 0 ? 0 : s1; y[yo + c + 2] = relu && s2 < 0 ? 0 : s2; y[yo + c + 3] = relu && s3 < 0 ? 0 : s3;
  }
  for (; c < CO; c++) {
    let s = b[c];
    for (let i = 0, wo = c * C; i < C; i++) s += x[xo + i] * w[wo + i];
    y[yo + c] = relu && s < 0 ? 0 : s;
  }
}

function conv(x, [, H, W, C], w, [CO, KH, KW], b, [sh, sw], pad, [, OH, OW], relu, y) {
  if (KH === 1 && KW === 1 && sh === 1 && sw === 1) {
    const n = OH * OW;
    let p = 0;
    // Two pixels at a time as well: each weight read serves both.
    for (; p + 1 < n; p += 2) {
      const xa = p * C, xb = xa + C, ya = p * CO, yb = ya + CO;
      let c = 0;
      for (; c + 3 < CO; c += 4) {
        let a0 = b[c], a1 = b[c + 1], a2 = b[c + 2], a3 = b[c + 3], b0 = a0, b1 = a1, b2 = a2, b3 = a3;
        const w0 = c * C, w1 = w0 + C, w2 = w1 + C, w3 = w2 + C;
        for (let i = 0; i < C; i++) {
          const u = x[xa + i], v = x[xb + i], k0 = w[w0 + i], k1 = w[w1 + i], k2 = w[w2 + i], k3 = w[w3 + i];
          a0 += u * k0; a1 += u * k1; a2 += u * k2; a3 += u * k3; b0 += v * k0; b1 += v * k1; b2 += v * k2; b3 += v * k3;
        }
        if (relu) { a0 = a0 < 0 ? 0 : a0; a1 = a1 < 0 ? 0 : a1; a2 = a2 < 0 ? 0 : a2; a3 = a3 < 0 ? 0 : a3; b0 = b0 < 0 ? 0 : b0; b1 = b1 < 0 ? 0 : b1; b2 = b2 < 0 ? 0 : b2; b3 = b3 < 0 ? 0 : b3; }
        y[ya + c] = a0; y[ya + c + 1] = a1; y[ya + c + 2] = a2; y[ya + c + 3] = a3; y[yb + c] = b0; y[yb + c + 1] = b1; y[yb + c + 2] = b2; y[yb + c + 3] = b3;
      }
      if (c < CO) { dot4(x, xa, w, C, c, b, y, ya, relu, CO); dot4(x, xb, w, C, c, b, y, yb, relu, CO); }
    }
    for (; p < n; p++) dot4(x, p * C, w, C, 0, b, y, p * CO, relu, CO);
    return y;
  }
  const pt = pad === "same" ? samePad(H, OH, KH, sh) : 0, pl = pad === "same" ? samePad(W, OW, KW, sw) : 0;
  const K = KH * KW * C, patch = new Float32Array(K);
  for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
    for (let ky = 0, o = 0; ky < KH; ky++) {
      const iy = oy * sh - pt + ky;
      for (let kx = 0; kx < KW; kx++, o += C) {
        const ix = ox * sw - pl + kx;
        if (iy < 0 || iy >= H || ix < 0 || ix >= W) { patch.fill(0, o, o + C); continue; }
        const xo = (iy * W + ix) * C;
        for (let c = 0; c < C; c++) patch[o + c] = x[xo + c];
      }
    }
    dot4(patch, 0, w, K, 0, b, y, (oy * OW + ox) * CO, relu, CO);
  }
  return y;
}

function dwconv(x, [, H, W, C], w, [, KH, KW], b, [sh, sw], pad, [, OH, OW], relu, y) {
  const pt = pad === "same" ? samePad(H, OH, KH, sh) : 0, pl = pad === "same" ? samePad(W, OW, KW, sw) : 0;
  for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
    const yo = (oy * OW + ox) * C;
    for (let c = 0; c < C; c++) y[yo + c] = b[c];
    for (let ky = 0; ky < KH; ky++) {
      const iy = oy * sh - pt + ky;
      if (iy < 0 || iy >= H) continue;
      for (let kx = 0; kx < KW; kx++) {
        const ix = ox * sw - pl + kx;
        if (ix < 0 || ix >= W) continue;
        const xo = (iy * W + ix) * C, wo = (ky * KW + kx) * C;
        for (let c = 0; c < C; c++) y[yo + c] += x[xo + c] * w[wo + c];
      }
    }
    if (relu) for (let c = 0; c < C; c++) if (y[yo + c] < 0) y[yo + c] = 0;
  }
  return y;
}

function maxpool(x, [, H, W, C], [KH, KW], [sh, sw], pad, [, OH, OW], y) {
  y.fill(-Infinity);
  const pt = pad === "same" ? samePad(H, OH, KH, sh) : 0, pl = pad === "same" ? samePad(W, OW, KW, sw) : 0;
  for (let oy = 0; oy < OH; oy++) for (let ox = 0; ox < OW; ox++) {
    const yo = (oy * OW + ox) * C;
    for (let ky = 0; ky < KH; ky++) {
      const iy = oy * sh - pt + ky;
      if (iy < 0 || iy >= H) continue;
      for (let kx = 0; kx < KW; kx++) {
        const ix = ox * sw - pl + kx;
        if (ix < 0 || ix >= W) continue;
        const xo = (iy * W + ix) * C;
        for (let c = 0; c < C; c++) if (x[xo + c] > y[yo + c]) y[yo + c] = x[xo + c];
      }
    }
  }
  return y;
}

function padded(x, [, H, W, C], pads, [, OH, OW, OC], y) {
  y.fill(0);
  const [, [top], [left], [front]] = pads;
  for (let iy = 0; iy < H; iy++) for (let ix = 0; ix < W; ix++) {
    const xo = (iy * W + ix) * C, yo = ((iy + top) * OW + ix + left) * OC + front;
    for (let c = 0; c < C; c++) y[yo + c] = x[xo + c];
  }
  return y;
}

// ---- anchors and decoding, as MediaPipe's face_detection_short_range graph ------------------------------
function anchors() {
  const strides = [8, 16, 16, 16], out = [];
  for (let layer = 0; layer < strides.length;) {
    let last = layer;
    while (last < strides.length && strides[last] === strides[layer]) last++;
    const per = 2 * (last - layer), fm = Math.ceil(FACE_INPUT / strides[layer]);
    for (let y = 0; y < fm; y++) for (let x = 0; x < fm; x++) for (let k = 0; k < per; k++) out.push((x + 0.5) / fm, (y + 0.5) / fm);
    layer = last;
  }
  return new Float32Array(out);   // x, y per anchor; 896 anchors
}

function iou(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[0] + a[2], b[0] + b[2]), y1 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0), union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

export class FaceDetector {
  constructor(model, { minScore = 0.5, overlap = 0.3 } = {}) {
    this.spec = model.spec; this.consts = model.consts; this.minScore = minScore; this.overlap = overlap;
    this.anchors = anchors();
    this.input = new Float32Array(FACE_INPUT * FACE_INPUT * 3);
  }

  /** The network on a 128x128x3 picture in [-1, 1] -> [regressors (896x16), scores (896)]. */
  run(input) {
    const { spec, consts } = this, shape = (id) => spec.shapes[id], vals = new Map([[spec.input, input]]);
    const get = (id) => vals.get(id) || consts[id];
    // Each op's output has its own buffer, kept from one picture to the next.
    const buffers = this.buffers || (this.buffers = new Map());
    const out = (id) => { let b = buffers.get(id); if (!b) { b = new Float32Array(spec.shapes[id].reduce((n, d) => n * d, 1)); buffers.set(id, b); } return b; };
    for (const op of spec.ops) {
      const [a, b, c] = op.in, relu = op.act === "relu";
      let y;
      switch (op.op) {
        case "conv": y = conv(get(a), shape(a), get(b), shape(b), get(c), op.stride, op.pad, shape(op.out), relu, out(op.out)); break;
        case "dwconv": y = dwconv(get(a), shape(a), get(b), shape(b), get(c), op.stride, op.pad, shape(op.out), relu, out(op.out)); break;
        case "add": { const p = get(a), q = get(b); y = out(op.out); for (let i = 0; i < p.length; i++) { const s = p[i] + q[i]; y[i] = relu && s < 0 ? 0 : s; } break; }
        case "relu": { const p = get(a); y = out(op.out); for (let i = 0; i < p.length; i++) y[i] = p[i] > 0 ? p[i] : 0; break; }
        case "pad": y = padded(get(a), shape(a), op.pads, shape(op.out), out(op.out)); break;
        case "maxpool": y = maxpool(get(a), shape(a), op.size, op.stride, op.pad, shape(op.out), out(op.out)); break;
        case "reshape": y = get(a); break;
        case "concat": {
          if (![1, -2].includes(op.axis) || shape(op.out).length !== 3) throw new Error(`concat on axis ${op.axis}`);
          y = out(op.out);
          let o = 0; for (const p of op.in.map(get)) { y.set(p, o); o += p.length; }
          break;
        }
        default: throw new Error(`op ${op.op}`);
      }
      vals.set(op.out, y);
    }
    return spec.outputs.map((id) => vals.get(id));
  }

  /** Faces in a 128x128 RGBA picture (canvas ImageData.data): fractions of that picture. */
  detect(rgba) {
    const x = this.input;
    for (let i = 0, j = 0; i < x.length; i += 3, j += 4) { x[i] = rgba[j] / 127.5 - 1; x[i + 1] = rgba[j + 1] / 127.5 - 1; x[i + 2] = rgba[j + 2] / 127.5 - 1; }
    return this.decode(...this.run(x));
  }

  decode(boxes, scores) {
    const A = this.anchors, S = FACE_INPUT, found = [];
    for (let i = 0; i < scores.length; i++) {
      const score = 1 / (1 + Math.exp(-Math.max(-100, Math.min(100, scores[i]))));
      if (score < this.minScore) continue;
      const r = boxes.subarray(i * 16, i * 16 + 16), ax = A[2 * i], ay = A[2 * i + 1];
      const cx = r[0] / S + ax, cy = r[1] / S + ay, w = r[2] / S, h = r[3] / S;
      const keypoints = [];
      for (let k = 0; k < 6; k++) keypoints.push([r[4 + 2 * k] / S + ax, r[5 + 2 * k] / S + ay]);
      found.push({ score, box: [cx - w / 2, cy - h / 2, w, h], keypoints });
    }
    // Weighted non-maximum suppression: each face is the score-weighted mean of the boxes that overlap it.
    found.sort((a, b) => b.score - a.score);
    const faces = [];
    let rest = found;
    while (rest.length) {
      const top = rest[0], near = [], far = [];
      for (const d of rest) (iou(d.box, top.box) > this.overlap ? near : far).push(d);
      let total = 0, x0 = 0, y0 = 0, x1 = 0, y1 = 0;
      const kp = top.keypoints.map(() => [0, 0]);
      for (const d of near) {
        total += d.score; x0 += d.box[0] * d.score; y0 += d.box[1] * d.score;
        x1 += (d.box[0] + d.box[2]) * d.score; y1 += (d.box[1] + d.box[3]) * d.score;
        d.keypoints.forEach(([u, v], k) => { kp[k][0] += u * d.score; kp[k][1] += v * d.score; });
      }
      faces.push({ score: top.score, box: [x0 / total, y0 / total, (x1 - x0) / total, (y1 - y0) / total], keypoints: kp.map(([u, v]) => [u / total, v / total]) });
      if (far.length === rest.length) break;
      rest = far;
    }
    return faces;
  }
}

/** Where a picture of width x height goes in the square input, letterboxed (as MediaPipe's
 *  keep_aspect_ratio): {scale, dx, dy} in input pixels; and faces back in the picture's fractions. */
export function letterbox(width, height) {
  const scale = FACE_INPUT / Math.max(width, height);
  return { scale, dx: (FACE_INPUT - width * scale) / 2, dy: (FACE_INPUT - height * scale) / 2 };
}
export function unletterbox(face, width, height) {
  const { scale, dx, dy } = letterbox(width, height), S = FACE_INPUT;
  const fx = (u) => (u * S - dx) / (width * scale), fy = (v) => (v * S - dy) / (height * scale);
  const [x, y, w, h] = face.box;
  return { score: face.score, box: [fx(x), fy(y), (w * S) / (width * scale), (h * S) / (height * scale)], keypoints: face.keypoints.map(([u, v]) => [fx(u), fy(v)]) };
}
