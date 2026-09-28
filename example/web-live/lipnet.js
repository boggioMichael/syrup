/**
 * LipNet in plain JavaScript: typed arrays, no framework, runs in a Web
 * Worker on a phone. Reads the int8 export of export_lipnet_web.py.
 *
 * The convolution stack is computed incrementally: every new frame
 * finalises one more time step of each layer (a 3-deep kernel needs the
 * frame after), and a decode recomputes only the tail the way a clip that
 * ended here would be padded. The GRUs and the dense layer run over the
 * window of the last <= 75 frames, on the CPU, in a few milliseconds.
 *
 * Layout: frames are (W=100, H=50, C=3) float32 in [0, 1], width-major as
 * LipNet stores them; conv kernels are (kt, kw, kh, cin, cout).
 */

const LETTERS = "abcdefghijklmnopqrstuvwxyz";
const BLANK = 27;
const SPACE = 26;

/** Reads the LIPW file: {name: Float32Array (dequantised), shapes}. */
export function loadWeights(buffer) {
  const view = new DataView(buffer);
  if (String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)) !== "LIPW") throw new Error("not a LipNet web weight file");
  const headLength = view.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headLength)));
  const base = 8 + headLength;
  const tensors = {};
  for (const t of header.tensors) {
    const q = new Int8Array(buffer, base + t.offset, t.size);
    const f = new Float32Array(t.size);
    const s = t.scale;
    for (let i = 0; i < t.size; i++) f[i] = q[i] * s;
    tensors[t.name] = { data: f, shape: t.shape };
  }
  return tensors;
}

/** A 3-D convolution layer with the kernel laid out for a cout-innermost inner loop. */
class Conv3d {
  constructor(kernel, bias, pad, stride, inW, inH) {
    const [kt, kw, kh, cin, cout] = kernel.shape;
    Object.assign(this, { kt, kw, kh, cin, cout, pad, stride, inW, inH });
    this.w = kernel.data; // index: (((t*kw + x)*kh + y)*cin + c)*cout + o
    this.b = bias.data;
    this.outW = Math.floor((inW + 2 * pad[1] - kw) / stride[1]) + 1;
    this.outH = Math.floor((inH + 2 * pad[2] - kh) / stride[2]) + 1;
    this.poolW = Math.floor(this.outW / 2);
    this.poolH = Math.floor(this.outH / 2);
    this.raw = new Float32Array(this.outW * this.outH * cout);
  }

  /**
   * One output time step from three input frames (each (inW, inH, cin) or
   * null for zero padding), followed by ReLU and 2x2 max pooling.
   * Returns a new Float32Array (poolW, poolH, cout).
   */
  step(frames) {
    const { kt, kw, kh, cin, cout, pad, stride, inW, inH, outW, outH, w, b, raw } = this;
    const sw = stride[1], sh = stride[2], pw = pad[1], ph = pad[2];
    const acc = new Float32Array(cout);
    for (let ox = 0; ox < outW; ox++) {
      for (let oy = 0; oy < outH; oy++) {
        acc.set(b);
        for (let t = 0; t < kt; t++) {
          const frame = frames[t];
          if (!frame) continue;
          for (let x = 0; x < kw; x++) {
            const ix = ox * sw - pw + x;
            if (ix < 0 || ix >= inW) continue;
            for (let y = 0; y < kh; y++) {
              const iy = oy * sh - ph + y;
              if (iy < 0 || iy >= inH) continue;
              const inBase = (ix * inH + iy) * cin;
              const wBase = ((t * kw + x) * kh + y) * cin * cout;
              for (let c = 0; c < cin; c++) {
                const v = frame[inBase + c];
                if (v === 0) continue;
                const wb = wBase + c * cout;
                // cout is a multiple of 8 in every LipNet layer (32, 64, 96).
                for (let o = 0; o < cout; o += 8) {
                  const k = wb + o;
                  acc[o] += v * w[k]; acc[o + 1] += v * w[k + 1]; acc[o + 2] += v * w[k + 2]; acc[o + 3] += v * w[k + 3];
                  acc[o + 4] += v * w[k + 4]; acc[o + 5] += v * w[k + 5]; acc[o + 6] += v * w[k + 6]; acc[o + 7] += v * w[k + 7];
                }
              }
            }
          }
        }
        const outBase = (ox * outH + oy) * cout;
        for (let o = 0; o < cout; o++) raw[outBase + o] = acc[o] > 0 ? acc[o] : 0;
      }
    }
    // 2x2 max pool over (x, y), channels kept.
    const { poolW, poolH } = this;
    const pooled = new Float32Array(poolW * poolH * cout);
    for (let px = 0; px < poolW; px++) {
      for (let py = 0; py < poolH; py++) {
        const a = ((2 * px) * outH + 2 * py) * cout, bq = ((2 * px) * outH + 2 * py + 1) * cout;
        const c2 = ((2 * px + 1) * outH + 2 * py) * cout, d = ((2 * px + 1) * outH + 2 * py + 1) * cout;
        const ob = (px * poolH + py) * cout;
        for (let o = 0; o < cout; o++) {
          let m = raw[a + o];
          if (raw[bq + o] > m) m = raw[bq + o];
          if (raw[c2 + o] > m) m = raw[c2 + o];
          if (raw[d + o] > m) m = raw[d + o];
          pooled[ob + o] = m;
        }
      }
    }
    return pooled;
  }
}

function hardSigmoid(x) {
  const y = 0.2 * x + 0.5;
  return y < 0 ? 0 : y > 1 ? 1 : y;
}

/** Keras 2.0.2 GRU over a sequence: x (T, D) -> (T, units); gates z, r, h; reset before the recurrent product. */
function gru(x, T, D, kernel, recurrent, bias, reverse) {
  const units = recurrent.shape[0];
  const K = kernel.data, R = recurrent.data, B = bias.data;
  const out = new Float32Array(T * units);
  const h = new Float32Array(units);
  const xw = new Float32Array(3 * units);
  const hz = new Float32Array(units), hr = new Float32Array(units), rh = new Float32Array(units), hh = new Float32Array(units);
  for (let step = 0; step < T; step++) {
    const t = reverse ? T - 1 - step : step;
    // Input projection for this frame: xw = x[t] @ K + B.
    xw.set(B);
    const xb = t * D;
    for (let d = 0; d < D; d++) {
      const v = x[xb + d];
      if (v === 0) continue;
      const kb = d * 3 * units;
      for (let j = 0; j < 3 * units; j++) xw[j] += v * K[kb + j];
    }
    // Recurrent projections: h @ R[:, :units], h @ R[:, units:2u], (r*h) @ R[:, 2u:].
    hz.fill(0); hr.fill(0);
    for (let i = 0; i < units; i++) {
      const v = h[i];
      if (v === 0) continue;
      const rb = i * 3 * units;
      for (let j = 0; j < units; j++) { hz[j] += v * R[rb + j]; hr[j] += v * R[rb + units + j]; }
    }
    for (let j = 0; j < units; j++) {
      const z = hardSigmoid(xw[j] + hz[j]);
      const r = hardSigmoid(xw[units + j] + hr[j]);
      hz[j] = z;          // reuse: z
      rh[j] = r * h[j];   // reset applied to the state before the recurrent matmul
    }
    hh.fill(0);
    for (let i = 0; i < units; i++) {
      const v = rh[i];
      if (v === 0) continue;
      const rb = i * 3 * units + 2 * units;
      for (let j = 0; j < units; j++) hh[j] += v * R[rb + j];
    }
    const ob = t * units;
    for (let j = 0; j < units; j++) {
      const z = hz[j];
      const cand = Math.tanh(xw[2 * units + j] + hh[j]);
      h[j] = z * h[j] + (1 - z) * cand;
      out[ob + j] = h[j];
    }
  }
  return out;
}

export class LipNet {
  constructor(tensors) {
    this.t = tensors;
    this.conv = [
      new Conv3d(tensors.conv1_kernel, tensors.conv1_bias, [1, 2, 2], [1, 2, 2], 100, 50),
      null, null,
    ];
    this.conv[1] = new Conv3d(tensors.conv2_kernel, tensors.conv2_bias, [1, 2, 2], [1, 1, 1], this.conv[0].poolW, this.conv[0].poolH);
    this.conv[2] = new Conv3d(tensors.conv3_kernel, tensors.conv3_bias, [1, 1, 1], [1, 1, 1], this.conv[1].poolW, this.conv[1].poolH);
    this.featureSize = this.conv[2].poolW * this.conv[2].poolH * this.conv[2].cout; // 1728
    this.reset();
  }

  reset() {
    // Finalised outputs per layer, indexed by time (arrays grow; callers trim).
    this.frames = [];
    this.layers = [[], [], []];
  }

  /** Feed one frame (Float32Array 100*50*3). Finalises conv1[t-1], conv2[t-2], conv3[t-3]. */
  push(frame) {
    this.frames.push(frame);
    const t = this.frames.length - 1;
    this._finalise(0, t - 1, this.frames);
    this._finalise(1, t - 2, this.layers[0]);
    this._finalise(2, t - 3, this.layers[1]);
  }

  _finalise(layer, at, inputs) {
    if (at < 0) return;
    const out = this.layers[layer];
    if (out[at] !== undefined) return;
    out[at] = this.conv[layer].step([inputs[at - 1] ?? null, inputs[at], inputs[at + 1] ?? null]);
  }

  /** Forget everything before `from` (keeps memory bounded for a sliding window). */
  trim(from) {
    for (let i = 0; i < from; i++) {
      if (this.frames[i]) this.frames[i] = null;
      for (const l of this.layers) if (l[i]) l[i] = null;
    }
  }

  /**
   * Features (T, 1728) for frames [from, to] as if the clip started at
   * `from` and ended at `to`: cached values inside, the tail recomputed with
   * end padding, the head recomputed with start padding.
   */
  features(from, to) {
    const T = to - from + 1;
    const out = new Float32Array(T * this.featureSize);
    // Layer by layer, recompute what the window's edges change.
    const l0 = [], l1 = [], l2 = [];
    const inputAt = (i) => (i < from || i > to ? null : this.frames[i]);
    for (let i = from; i <= to; i++) {
      const edge = i === from || i === to;
      l0[i] = !edge && this.layers[0][i] ? this.layers[0][i] : this.conv[0].step([inputAt(i - 1), inputAt(i), inputAt(i + 1)]);
    }
    const l0At = (i) => (i < from || i > to ? null : l0[i]);
    for (let i = from; i <= to; i++) {
      const edge = i <= from + 1 || i >= to - 1;
      l1[i] = !edge && this.layers[1][i] ? this.layers[1][i] : this.conv[1].step([l0At(i - 1), l0At(i), l0At(i + 1)]);
    }
    const l1At = (i) => (i < from || i > to ? null : l1[i]);
    for (let i = from; i <= to; i++) {
      const edge = i <= from + 2 || i >= to - 2;
      l2[i] = !edge && this.layers[2][i] ? this.layers[2][i] : this.conv[2].step([l1At(i - 1), l1At(i), l1At(i + 1)]);
      out.set(l2[i], (i - from) * this.featureSize);
    }
    return out;
  }

  /** Per-frame probabilities (T, 28) from features. */
  probabilities(features, T) {
    const t = this.t;
    let x = features, D = this.featureSize;
    for (const i of [1, 2]) {
      const fw = gru(x, T, D, t[`gru${i}_fw_kernel`], t[`gru${i}_fw_recurrent`], t[`gru${i}_fw_bias`], false);
      const bw = gru(x, T, D, t[`gru${i}_bw_kernel`], t[`gru${i}_bw_recurrent`], t[`gru${i}_bw_bias`], true);
      const units = t[`gru${i}_fw_recurrent`].shape[0];
      const cat = new Float32Array(T * 2 * units);
      for (let s = 0; s < T; s++) {
        cat.set(fw.subarray(s * units, (s + 1) * units), s * 2 * units);
        cat.set(bw.subarray(s * units, (s + 1) * units), s * 2 * units + units);
      }
      x = cat; D = 2 * units;
    }
    const W = t.dense_kernel.data, B = t.dense_bias.data, C = t.dense_kernel.shape[1];
    const probs = new Float32Array(T * C);
    const logits = new Float32Array(C);
    for (let s = 0; s < T; s++) {
      logits.set(B);
      for (let d = 0; d < D; d++) {
        const v = x[s * D + d];
        if (v === 0) continue;
        const wb = d * C;
        for (let c = 0; c < C; c++) logits[c] += v * W[wb + c];
      }
      let max = -Infinity;
      for (let c = 0; c < C; c++) if (logits[c] > max) max = logits[c];
      let sum = 0;
      for (let c = 0; c < C; c++) { const e = Math.exp(logits[c] - max); logits[c] = e; sum += e; }
      for (let c = 0; c < C; c++) probs[s * C + c] = logits[c] / sum;
    }
    return probs;
  }

  /** Whole-clip inference for a list of frames (no incremental state used). */
  predict(frames) {
    const saved = [this.frames, this.layers];
    this.reset();
    for (const f of frames) this.push(f);
    const feats = this.features(0, frames.length - 1);
    const probs = this.probabilities(feats, frames.length);
    [this.frames, this.layers] = saved;
    return probs;
  }
}

/** Greedy CTC: words with their frame span and mean peak probability. */
export function decode(probs, T, C = 28) {
  const words = [];
  let letters = "", start = -1, end = -1, confSum = 0, n = 0, previous = -1;
  const flush = () => {
    if (letters) words.push({ text: letters, start, end, confidence: confSum / n });
    letters = ""; start = end = -1; confSum = 0; n = 0;
  };
  for (let t = 0; t < T; t++) {
    let best = 0, peak = -1;
    for (let c = 0; c < C; c++) { const p = probs[t * C + c]; if (p > peak) { peak = p; best = c; } }
    if (best !== previous && best !== BLANK) {
      if (best === SPACE) flush();
      else { if (!letters) start = t; letters += LETTERS[best]; end = t; confSum += peak; n++; }
    }
    previous = best;
  }
  flush();
  return words;
}

// GRID vocabulary (51 words) and Norvig's correction as in the original code.
export const GRID = {
  command: ["bin", "lay", "place", "set"],
  colour: ["blue", "green", "red", "white"],
  preposition: ["at", "by", "in", "with"],
  letter: "abcdefghijklmnopqrstuvxyz".split(""),
  digit: ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
  adverb: ["again", "now", "please", "soon"],
};
export const VOCABULARY = new Set([...GRID.command, ...GRID.colour, ...GRID.preposition, ...GRID.letter, ...GRID.digit, ...GRID.adverb]);

function edits1(word) {
  const out = new Set();
  for (let i = 0; i <= word.length; i++) {
    const L = word.slice(0, i), R = word.slice(i);
    if (R) out.add(L + R.slice(1));
    if (R.length > 1) out.add(L + R[1] + R[0] + R.slice(2));
    for (const c of LETTERS) { if (R) out.add(L + c + R.slice(1)); out.add(L + c + R); }
  }
  return out;
}

export function correct(word) {
  if (VOCABULARY.has(word)) return word;
  const e1 = [...edits1(word)].filter((w) => VOCABULARY.has(w));
  if (e1.length) return e1.sort()[0];
  for (const e of edits1(word)) {
    for (const f of edits1(e)) if (VOCABULARY.has(f)) return f;
  }
  return word;
}

export function randomSentence() {
  const pick = (a) => a[Math.floor(Math.random() * a.length)];
  return [pick(GRID.command), pick(GRID.colour), pick(GRID.preposition), pick(GRID.letter), pick(GRID.digit), pick(GRID.adverb)].join(" ");
}
