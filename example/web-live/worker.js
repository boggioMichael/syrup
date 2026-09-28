/**
 * The Web Worker: owns the LipNet engine, takes frames as they are
 * captured, keeps the incremental convolution state, and reads a window
 * on request. Built into the page with the engine prepended (build.py).
 *
 *   -> { type: "init", weights: ArrayBuffer }
 *   <- { type: "ready", perFrameMs }
 *   -> { type: "frame", index, data: Float32Array }      (100*50*3, width-major, [0,1])
 *   <- { type: "processed", index, queued }
 *   -> { type: "read", from, to, final }                 (frame indices, inclusive)
 *   <- { type: "reading", from, to, final, words: [{text, raw, confidence, start, end}], ms }
 *   -> { type: "reset" }
 */
let net = null;
const HISTORY = 200; // frames kept; older ones are forgotten

function readWindow(from, to) {
  const t0 = performance.now();
  const T = to - from + 1;
  const probs = net.probabilities(net.features(from, to), T);
  const words = decode(probs, T).map((w) => {
    const text = correct(w.text);
    let confidence = w.confidence;
    if (text !== w.text) confidence *= 0.8;
    if (!VOCABULARY.has(text)) confidence *= 0.5;
    return { text, raw: w.text, confidence, start: from + w.start, end: from + w.end };
  });
  return { words, ms: performance.now() - t0 };
}

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === "init") {
    net = new LipNet(loadWeights(m.weights));
    // How fast is this device? Ten frames of noise through the convolutions.
    const probe = new Float32Array(100 * 50 * 3);
    for (let i = 0; i < probe.length; i++) probe[i] = Math.random();
    const t0 = performance.now();
    for (let i = 0; i < 10; i++) net.push(probe);
    const perFrameMs = (performance.now() - t0) / 10;
    net.reset();
    self.postMessage({ type: "ready", perFrameMs });
  } else if (m.type === "frame") {
    if (!net) return;
    if (m.index !== net.frames.length) {
      // A gap (frames dropped upstream): the sequence restarts here.
      net.reset();
      for (let i = 0; i < m.index; i++) { net.frames.push(null); net.layers.forEach((l) => l.push(null)); }
    }
    net.push(m.data);
    if (net.frames.length > HISTORY) net.trim(net.frames.length - HISTORY);
    self.postMessage({ type: "processed", index: m.index });
  } else if (m.type === "read") {
    if (!net || m.to >= net.frames.length || m.from < 0 || m.to < m.from) {
      self.postMessage({ type: "reading", from: m.from, to: m.to, final: m.final, words: [], ms: 0 });
      return;
    }
    const from = Math.max(m.from, net.frames.length - HISTORY, 0);
    const { words, ms } = readWindow(from, m.to);
    self.postMessage({ type: "reading", from, to: m.to, final: m.final, words, ms });
  } else if (m.type === "reset") {
    if (net) net.reset();
  }
};
