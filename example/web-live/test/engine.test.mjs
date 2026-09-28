// The JavaScript engine against the numpy reference: same probabilities,
// same words; and how fast it is on this machine.
//   node test/engine.test.mjs [weights.bin] [crops.f32] [probs.f32]
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { LipNet, loadWeights, decode, correct } from "../lipnet.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const [weightsPath = path.join(here, "..", "lipnet-web.bin"), cropsPath = path.join(here, "..", "lipnet-web.sbwe5n.crops.f32"), probsPath = path.join(here, "..", "lipnet-web.sbwe5n.probs.f32")] = process.argv.slice(2);

const toBuffer = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
const t0 = performance.now();
const net = new LipNet(loadWeights(toBuffer(fs.readFileSync(weightsPath))));
console.log(`weights loaded in ${(performance.now() - t0).toFixed(0)} ms`);
const crops = new Float32Array(toBuffer(fs.readFileSync(cropsPath)));
const reference = new Float32Array(toBuffer(fs.readFileSync(probsPath)));
const F = 100 * 50 * 3;
const T = crops.length / F;
const frames = [];
for (let i = 0; i < T; i++) frames.push(crops.subarray(i * F, (i + 1) * F));

// Whole clip.
let t1 = performance.now();
const probs = net.predict(frames);
const whole = performance.now() - t1;
let maxDiff = 0;
for (let i = 0; i < probs.length; i++) maxDiff = Math.max(maxDiff, Math.abs(probs[i] - reference[i]));
const words = decode(probs, T).map((w) => correct(w.text)).join(" ");
console.log(`whole clip (${T} frames): ${whole.toFixed(0)} ms, max |js - numpy| = ${maxDiff.toExponential(2)}, words: "${words}"`);
if (maxDiff > 1e-3) { console.error("FAIL: probabilities differ from the reference"); process.exit(1); }
if (words !== "set blue with e five now") { console.error("FAIL: wrong words"); process.exit(1); }

// Incremental: push frame by frame, decode the window at the end and a few times on the way.
net.reset();
t1 = performance.now();
for (const f of frames) net.push(f);
const perFrame = (performance.now() - t1) / T;
t1 = performance.now();
const feats = net.features(0, T - 1);
const p2 = net.probabilities(feats, T);
const decodeMs = performance.now() - t1;
let d2 = 0;
for (let i = 0; i < p2.length; i++) d2 = Math.max(d2, Math.abs(p2[i] - reference[i]));
console.log(`incremental: ${perFrame.toFixed(1)} ms per frame pushed, decode of the 75-frame window ${decodeMs.toFixed(0)} ms, max diff ${d2.toExponential(2)}`);
if (d2 > 1e-3) { console.error("FAIL: incremental features differ"); process.exit(1); }
// A window that starts late must equal the numpy result for that sub-clip: compare decode of frames 10..74 with predict().
const sub = net.probabilities(net.features(10, T - 1), T - 10);
const ref2 = new LipNet(loadWeights(toBuffer(fs.readFileSync(weightsPath)))).predict(frames.slice(10));
let d3 = 0;
for (let i = 0; i < sub.length; i++) d3 = Math.max(d3, Math.abs(sub[i] - ref2[i]));
console.log(`window 10..74 vs fresh predict: max diff ${d3.toExponential(2)}`);
if (d3 > 1e-3) { console.error("FAIL: windowed features differ"); process.exit(1); }
console.log("ALL PASSED");
