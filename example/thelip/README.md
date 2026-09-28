# thelip.syrup

The Lip in your browser, on a phone: point the camera at your mouth (or
record a clip) and say a sentence; the words are read from your lips, in
the browser, with no audio and nothing uploaded. The live site is
**https://boggiomichael.github.io/syrup/thelip/** (GitHub Pages, from
`docs/thelip/`, committed to the `gh-pages` branch by
`.github/workflows/pages.yml` on every push that changes `docs/`).

- `thelip.js` — the engine: The Lip's network (LipNet, Assael et al. 2016;
  weights rizkiarm/LipNet, MIT) in plain JavaScript: int8 weights
  dequantised, batch norm folded, the convolution stack computed
  incrementally frame by frame, GRUs and the dense layer on the CPU, CTC
  decoding, the GRID corrector. Matches the numpy network to 2e-6
  (`test/engine.test.mjs`).
- `worker.js` — the Web Worker protocol (init, frame, read, reset).
- `src/page.html` — the page, and nothing else on it: the camera opens
  when the page opens, a dashed oval marks where the lips go (drag to
  move, pinch to resize), an utterance is detected from lip motion and
  read as soon as the lips go still, the words appear as subtitles at the
  bottom (forming word by word on a fast device), a tiny strip shows what
  thelip sees, and a "?" holds the vocabulary, the credits and a way to
  open a recorded clip instead. Where the camera is refused (a claude.ai
  artifact, a plain-http page) it says so and offers the clip.
- `build.py` — inlines everything into one file: `thelip.syrup.html` (page
  content, for a claude.ai artifact) or, with `--standalone`, a complete
  document for hosting (`docs/thelip/index.html`).
- `test/page.test.mjs` — the built page in Chromium reads a silent GRID
  clip through the clip path; `test/camera.test.mjs` — Chromium's fake
  camera plays the clip on a loop and the page, opening the camera on
  load, reads the sentence live.

```sh
python3 ../ml/conversion/export_thelip_weights.py --out thelip-weights.bin --reference sbwe5n
node test/engine.test.mjs
python3 build.py && python3 build.py --standalone --out ../../docs/thelip/index.html
NODE_PATH=$(npm root -g) node test/page.test.mjs && NODE_PATH=$(npm root -g) node test/camera.test.mjs
```

Camera access needs a secure page (https or localhost), which the Pages
site is; a page opened from a file, over plain http, or as a claude.ai
artifact gets the "record a clip" flow instead, which plays the recording
at the speed thelip reads it with the subtitle forming underneath. Speed
measured here: 37–42 ms per frame in Node/Chromium on a 2.1 GHz Xeon
core (37 ms with doubles, which V8 runs faster than float32 stores); a recent phone is comparable or faster. 64/66 words on the GRID
sample clips with the int8 weights; a new face and a phone camera are
harder than the lab data.

## Your own domain

GitHub Pages serves the same site under a domain you own, at
`https://<domain>/thelip/`, in two steps:

1. At the registrar, point the domain at GitHub Pages: for the apex
   (`syrup.ai`) four `A` records to `185.199.108.153`, `185.199.109.153`,
   `185.199.110.153`, `185.199.111.153` (and `AAAA` records to
   `2606:50c0:8000::153`, `2606:50c0:8001::153`, `2606:50c0:8002::153`,
   `2606:50c0:8003::153`); for `www` or another subdomain, one `CNAME`
   record to `boggiomichael.github.io`.
2. Put the domain, alone, in `docs/CNAME`; the Pages workflow publishes it
   with the site, GitHub picks it up as the custom domain and issues the
   HTTPS certificate once the DNS records resolve (minutes to a day).

Do step 1 first: the moment `docs/CNAME` is published, GitHub redirects
`boggiomichael.github.io/syrup/` to the domain, so a domain that does not
resolve yet takes the site down with it.
