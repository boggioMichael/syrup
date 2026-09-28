# Lip Read for YouTube (Chrome extension)

A Manifest V3 extension that adds a **LIP** button to the YouTube player.
Press it and the people in the video get subtitles read from their lips —
per person, with a confidence, uncertain words marked — drawn over the
video and listed in a side panel where every line seeks the video and the
whole transcript exports as TXT, JSON, SRT or VTT.

<p><img src="../docs/images/extension-overlay.png" width="480" alt="Two people boxed and labelled, a subtitle line forming under one of them"> <img src="../docs/images/extension-panel.png" width="240" alt="The side panel with mode, language, speakers, status, notice and the transcript"></p>

Processing happens in the **lipreader service** on your own machine
(`example/inference-api`); the extension never sends the video anywhere
else, and the panel says where processing happened if you point it at a
server instead.

## Install

```sh
# 1. the service (once)
cd example/lipreader && python3 tests/make_fixtures.py     # optional: only for the tests
cd ../inference-api && python3 -m lipreader_api             # http://127.0.0.1:8765
pip install yt-dlp                                          # lets the service fetch YouTube videos

# 2. the extension
cd ../chrome-extension && tsc -p tsconfig.json              # writes dist/
```

Then in Chrome: `chrome://extensions` → Developer mode → *Load unpacked* →
this directory. Open a YouTube video, press **LIP** in the player
controls, and open the side panel from the extension's toolbar icon.

## What it does, and how

- **Source.** A content script cannot read YouTube's frames (the media is
  cross-origin and taints any canvas), so by default the extension sends
  the *page URL* to the service, which fetches the video with `yt-dlp`,
  analyses it whole, and the overlay syncs the result with the player's
  clock. On a page whose `<video>` has a plain media URL, that URL is sent
  instead. The alternative source, *tab capture*, streams what is on
  screen to the service's `/sessions` endpoint through an offscreen
  document (`chrome.tabCapture`); it is implemented but could not be
  exercised in the environment this was built in — see Status.
- **Modes.** Visual only (the audio track is never opened), audio +
  visual, or audio transcript with speakers attributed from the lips. The
  mode is stamped on every transcript segment.
- **Speakers.** Every visible speaker, or the current one (the largest face
  when you press the button). Click a face box to follow that person only;
  click again to release.
- **Honesty.** Words below the confidence threshold are shown as `[word?]`.
  A language with no runnable model produces no text and a warning in the
  panel. `auto` in visual mode is reported as *assumed* (no visual
  language-identification model exists publicly). The probabilistic notice
  is always visible.
- **YouTube integration** lives in one file, `src/youtube.ts`: the
  selectors, the navigation event, the letterboxing maths. When YouTube
  changes its markup, that is the file to change.

## Files

```
manifest.json         MV3 manifest (sidePanel, storage, tabs, tabCapture, offscreen; localhost hosts)
content-loader.js     loads dist/content.js as a module into the page
src/youtube.ts        the only file that knows YouTube's DOM
src/content.ts        button, overlay (subtitles, boxes, follow), state mirror
src/worker.ts         service worker: the API client, one state per tab, capture orchestration
src/panel.ts          side panel: controls, transcript, seek, export
src/options.ts        service address, token, source, boxes
src/capture.ts        offscreen document: tab stream -> JPEG frames
src/shared.ts         messages, settings, helpers; types from ../shared-types
src/types/chrome.d.ts the subset of the Chrome API used, typed by hand (no npm here)
test/e2e.mjs          Playwright: the built extension against the real service and a fake YouTube page
test/fake_youtube.py  that page: YouTube's selectors, a fixture as WebM, range requests
```

## Test

```sh
tsc -p tsconfig.json
NODE_PATH=$(npm root -g) node test/e2e.mjs        # needs playwright + chromium, ffmpeg, the fixtures
```

The end-to-end test starts the service and the fake page, loads the built
extension into Chromium, presses the button, and checks 25 things: the
button and overlay appear; the analysis runs; two people are boxed; the
subtitle at 1.6 s reads "set blue…" for one person and at 4.6 s "bin red…"
for the other; the speaking person's box is highlighted; clicking a face
follows only that person; the panel lists both lines with speaker,
timestamps, language and confidence and says where processing happened;
clicking a line seeks the video; all four exports download well-formed
files; stop, pause and resume work. It passes in this repository's build
environment (Chromium 1194 via Playwright 1.56).

## Status

- Works, end to end, against the local service: URL source, all three
  modes as far as the service supports them (the audio modes need a
  Whisper backend installed on the service's machine), overlay, follow,
  panel, seek, export, pause/resume/stop.
- Not exercised here: real youtube.com (no network in the build
  environment; the adapter targets `#movie_player`, `video.html5-main-video`,
  `.ytp-right-controls` and `yt-navigate-finish`, which are YouTube's
  current markup, and falls back to the largest `<video>`), and the tab
  capture source (no capturable tab in headless Chromium). Both are small,
  isolated code paths; the first YouTube session should start with the
  service running, the panel open, and `chrome://extensions` → service
  worker → console for any error.
- Type checking uses a hand-written `chrome.d.ts`; `npm i -D @types/chrome`
  replaces it when npm is available.
