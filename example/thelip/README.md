# thelip.syrup

The Lip in your browser, on a phone: point the camera at your mouth (or
record a clip) and say a sentence; the words are read from your lips, in
the browser, with no audio and nothing uploaded. The live site is
**https://thelip.ai/** (GitHub Pages: `docs/index.html`, committed to the
`gh-pages` branch by `.github/workflows/pages.yml` on every push that
changes `docs/`; `docs/CNAME` names the domain; the GitHub address
boggiomichael.github.io/syrup/ redirects there). A plain-http visit is sent
to https by the page itself, because a camera needs a secure page.

- `thelip.js` — the engine: The Lip's network (LipNet, Assael et al. 2016;
  weights rizkiarm/LipNet, MIT) in plain JavaScript: int8 weights
  dequantised, batch norm folded, the convolution stack computed
  incrementally frame by frame, GRUs and the dense layer on the CPU, CTC
  decoding, the GRID corrector. Matches the numpy network to 2e-6
  (`test/engine.test.mjs`).
- `worker.js` — the Web Worker protocol (init, frame, read, reset).
- `faces.js`, `face-worker.js`, `blazeface.bin` — the face finder, in a
  worker of its own: BlazeFace (short range), MediaPipe's face detector
  (Apache 2.0), run from its own weights with no TFLite runtime: the
  network's 90 ops in plain JavaScript on a 128x128 letterboxed frame,
  MediaPipe's anchors, decoding and weighted non-maximum suppression.
  `tools/export_blazeface.py` turns MediaPipe's model file into
  `blazeface.bin` (212 KB of float16 weights); `test/faces.test.mjs`
  checks it against MediaPipe's own detector (`tools/faces_reference.py`):
  the same faces, scores and keypoints to four decimals at the network's
  own size, within a pixel of its input on letterboxed frames; 35 ms a
  look in Node here.
- `src/page.html` — the page, and nothing else on it: the camera opens
  when the page opens, and an oval finds the mouth by itself (red while
  it follows it; drag it to place it by hand, pinch to resize, and it
  stays); an utterance is detected from lip motion and read as soon as
  the lips go still, the words appear as subtitles at the bottom (forming
  word by word on a fast device), a tiny strip shows what thelip sees,
  and a "?" holds the vocabulary, the credits and a way to open a
  recorded clip instead. Where the camera is refused (a claude.ai
  artifact, a plain-http page) it says so and offers the clip.
- The oval: 0.85 of the face's width, on the median of the mouth
  keypoint's last two seconds of sightings (a few a second), moving only
  when the mouth really moved, so that what starts a sentence is the lips.
  A speaking mouth's centre drops as the jaw opens, and LipNet reads best
  from an oval between closed and open: on GRID's ten sample clips
  (`docs/test/grid`), each read whole through the oval the finder placed,
  60 of 60 words at 0.85 and 0.9 of the face, 58 at 0.8, 59 at 0.94, and
  54 at 0.85 from the first frame alone, mouth closed
  (`tools/oval_check.mjs`). The page itself, opening those clips with no
  oval given: 60 of 60 (`test/page.test.mjs`). Other faces' mouths get
  ovals of their own; with a server, which reads every face, their motion
  starts a sentence too.
- Subtitles stay: when the next sentence starts, a reading moves up into
  a crawl above the subtitle, with what was heard under it, tilted back
  like the opening of Star Wars, smaller and fainter the older it is, gone
  after three minutes (six at most).
- Server mode: with `?server=<address>` (or the address typed in the "?"
  sheet) the page sends each sentence's frames (320 px JPEGs, 25 fps) to
  [`../thelip-server`](../thelip-server) and shows its answer — any English
  words, a few seconds late, nothing read locally meanwhile. The address is
  kept in the browser; "Stop" forgets it. Before anything is kept the page
  asks, once, at the start (a card over the subtitles, in Hebrew for a
  Hebrew browser): each sentence's small grey video of the mouth and the
  words said, for training; the face and the voice are not kept. After a
  yes, the sound of each sentence goes too, to be written down (speech recognition)
  and kept as its label: the microphone's, or an opened clip's own —
  taken out of the file in the browser when it can (`decodeAudioData`),
  else the clip itself goes and the server takes the stretch's sound out.
  `test/server.test.mjs` covers it against the server's `--fake` mode.
- `build.py` — inlines everything into one file: `thelip.syrup.html` (page
  content, for a claude.ai artifact) or, with `--standalone`, a complete
  document for hosting (`docs/index.html`).
- `test/page.test.mjs` — the built page in Chromium reads a silent GRID
  clip through the clip path, with the oval placed by hand and then by
  the finder (all ten GRID clips; two people in one clip);
  `test/camera.test.mjs` — Chromium's fake camera plays the clip on a
  loop: the oval finds the mouth in it, dragging places it by hand, and
  the page, opening the camera on load, reads the sentence live.

```sh
python3 ../ml/conversion/export_thelip_weights.py --out thelip-weights.bin --reference sbwe5n
node test/engine.test.mjs && node test/faces.test.mjs
python3 build.py && python3 build.py --standalone --out ../../docs/index.html
NODE_PATH=$(npm root -g) node test/page.test.mjs && NODE_PATH=$(npm root -g) node test/camera.test.mjs
```

Camera access needs a secure page (https or localhost), which the Pages
site is; a page opened from a file, over plain http, or as a claude.ai
artifact gets the "record a clip" flow instead, which plays the recording
at the speed thelip reads it with the subtitle forming underneath. Speed
measured here: 37–42 ms per frame in Node/Chromium on a 2.1 GHz Xeon
core (37 ms with doubles, which V8 runs faster than float32 stores); a recent phone is comparable or faster. 64/66 words on the GRID
sample clips with the int8 weights (the Python pipeline, which finds the
mouth its own way); 60/60 on the ten in `docs/test/grid` through the
page's own oval; a new face and a phone camera are harder than the lab
data. A clip of three seconds or less (GRID's length) is read whole, the
way LipNet was trained; a longer one sentence by sentence, as the camera.

## The domain

thelip.ai is served by GitHub Pages; the same two steps put the site
under any domain:

1. At the registrar, point the domain at GitHub Pages: for the apex
   (`thelip.ai`) four `A` records to `185.199.108.153`, `185.199.109.153`,
   `185.199.110.153`, `185.199.111.153` (and `AAAA` records to
   `2606:50c0:8000::153`, `2606:50c0:8001::153`, `2606:50c0:8002::153`,
   `2606:50c0:8003::153`); for `www` or another subdomain, one `CNAME`
   record to `boggiomichael.github.io`.
2. Put the domain, alone, in `docs/CNAME`; the Pages workflow publishes it
   with the site, GitHub picks it up as the custom domain and issues the
   HTTPS certificate once the DNS records resolve (minutes to a day).

Do step 1 first: the moment `docs/CNAME` is published, GitHub redirects
`boggiomichael.github.io/syrup/` to the domain, so a domain that does not
resolve yet takes the site down with it. Both were done on 2026-09-28:
the four `A` records at the registrar, then `docs/CNAME`; the certificate
was issued within minutes.
