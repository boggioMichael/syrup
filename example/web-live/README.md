# The Lip Live (in the browser)

LipNet running in plain JavaScript in a Web Worker, on a phone: record a
clip of yourself saying a GRID sentence (or use the camera on an https
page) and the words are read from your lips, in the browser, with no
audio and nothing uploaded.

- `lipnet.js` — the engine: int8 weights dequantised, batch norm folded,
  the convolution stack computed incrementally frame by frame, GRUs and
  the dense layer on the CPU, CTC decoding, the GRID corrector. Matches the
  numpy network to 2e-6 (`test/engine.test.mjs`).
- `worker.js` — the worker protocol (init, frame, read, reset).
- `src/page.html` — the page: camera or file, a guide box for the mouth,
  utterance detection from lip motion, readings that form while you speak
  on a fast device, a sentence to say, what the network sees.
- `build.py` — inlines everything into `the-lip-live.html` (6.1 MB with
  the weights as base64).
- `test/page.test.mjs` — the built page in Chromium reads a silent GRID
  clip correctly.

```sh
python3 ../ml/conversion/export_lipnet_web.py --out lipnet-web.bin --reference sbwe5n
node test/engine.test.mjs
python3 build.py
NODE_PATH=$(npm root -g) node test/page.test.mjs
```

Camera access needs a secure page (https or localhost); a page opened
from a file or over plain http, and a claude.ai artifact, get the "record
a clip" flow instead, which processes the recording at the speed the
network reads it with the subtitle forming underneath. Hosting the built
file on GitHub Pages gives the camera flow. Speed measured here: 47–53 ms
per frame in Node/Chromium on a 2.1 GHz Xeon core; a recent phone is
comparable or faster. 64/66 words on the GRID sample clips with the int8
weights; a new face and a phone camera are harder than the lab data.
