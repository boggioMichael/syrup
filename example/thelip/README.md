# thelip.syrup

The Lip in your browser, on a phone: point the camera at your mouth (or
record a clip) and say a sentence; the words are read from your lips, in
the browser, with no audio and nothing uploaded. The live site is
**https://boggiomichael.github.io/syrup/thelip/** (GitHub Pages, from
`docs/thelip/`, deployed by `.github/workflows/pages.yml` on push).

- `thelip.js` — the engine: The Lip's network (LipNet, Assael et al. 2016;
  weights rizkiarm/LipNet, MIT) in plain JavaScript: int8 weights
  dequantised, batch norm folded, the convolution stack computed
  incrementally frame by frame, GRUs and the dense layer on the CPU, CTC
  decoding, the GRID corrector. Matches the numpy network to 2e-6
  (`test/engine.test.mjs`).
- `worker.js` — the Web Worker protocol (init, frame, read, reset).
- `src/page.html` — the page: camera or file, a guide box for the mouth,
  utterance detection from lip motion, readings that form while you speak
  on a fast device, a sentence to say, what thelip sees.
- `build.py` — inlines everything into one file: `thelip.syrup.html` (page
  content, for a claude.ai artifact) or, with `--standalone`, a complete
  document for hosting (`docs/thelip/index.html`).
- `test/page.test.mjs` — the built page in Chromium reads a silent GRID
  clip correctly.

```sh
python3 ../ml/conversion/export_thelip_weights.py --out thelip-weights.bin --reference sbwe5n
node test/engine.test.mjs
python3 build.py && python3 build.py --standalone --out ../../docs/thelip/index.html
NODE_PATH=$(npm root -g) node test/page.test.mjs
```

Camera access needs a secure page (https or localhost), which the Pages
site is; a page opened from a file, over plain http, or as a claude.ai
artifact gets the "record a clip" flow instead, which plays the recording
at the speed thelip reads it with the subtitle forming underneath. Speed
measured here: 47–53 ms per frame in Node/Chromium on a 2.1 GHz Xeon
core; a recent phone is comparable or faster. 64/66 words on the GRID
sample clips with the int8 weights; a new face and a phone camera are
harder than the lab data.
