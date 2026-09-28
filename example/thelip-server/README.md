# thelip-server

Any English words for [thelip.ai](https://thelip.ai/). The page reads 51
words on its own, in the browser; pointed at this server it sends the
frames of each sentence to a computer that runs an open-vocabulary English
lip reader and shows the answer a few seconds later.

```
thelip.ai (phone)  --frames of one sentence (JPEG, 25 fps)-->  thelip-server (your computer)
                   <--          {"text": "HELLO THERE"}      --
```

## Run it (Windows)

Double-click **`thelip-server.cmd`**. The first run takes a while: it
creates a Python environment, installs PyTorch and mediapipe, fetches the
Chaplin pipeline and the model files (1.2 GB) and the tunnel client — all
under `C:\Users\Public\thelip-server`, an ASCII path, because PyTorch's
DLLs fail to initialise from a path with non-ASCII characters (a Hebrew
user name, say). Then it prints a link and a QR code:

```
Open this on the phone (it is the site with the server's address in it):
  https://thelip.ai/?server=https://<random>.trycloudflare.com
```

Scan it, or type it; the page keeps the address (it is in the "?" sheet,
with "Stop" to forget it). Say a sentence in English; the subtitle says
"…" while the computer reads, then the words. Ctrl+C in the window stops
the server.

**Working from thelip.ai directly.** The page also looks for the server on
its own, in two places, so nobody needs the link: `https://thelip.ai/server.json`
and `https://api.thelip.ai`. `run.py` writes the tunnel's address into
`server.json` on the site's `gh-pages` branch after every start (it needs
`git` and push rights on the machine; it says so when it cannot), and a
Cloudflare *named* tunnel gives the fixed address instead of a throwaway one:
put its token in `tunnel-token.txt` and the hostname it routes in
`public-url.txt` next to `run.py` (both ignored by git); see "A fixed
address" below. Otherwise every run gets a new throwaway address
(`--no-tunnel` serves this computer only, at `http://127.0.0.1:8791`).

### A fixed address (api.thelip.ai)

1. A free Cloudflare account; add the site `thelip.ai` to it and set the two
   nameservers it gives at the registrar (Namecheap → Domain → Nameservers →
   Custom DNS). Re-create the site's records in Cloudflare's DNS: the four
   GitHub Pages `A` records for `@` (185.199.108–111.153, proxy off).
2. Zero Trust → Networks → Tunnels → Create a tunnel (cloudflared) named
   `thelip`; copy the token from the install command into `tunnel-token.txt`.
3. In the tunnel's Public hostnames: `api.thelip.ai` → `http://localhost:8791`.
   Write `https://api.thelip.ai` into `public-url.txt`. Start the launcher.

Needs Python 3.10–3.12 (mediapipe has no wheels for newer Pythons) from
[python.org](https://www.python.org/downloads/windows/) with "Add to PATH"
and "py launcher" ticked. macOS/Linux: `python3 -m venv .venv &&
.venv/bin/pip install -r requirements.txt && .venv/bin/python run.py`
(macOS needs `brew install cloudflared` for the tunnel).

## What runs

- **Model**: `LRS3_V_WER19.1` — the visual speech recogniser of Ma,
  Petridis & Pantic (Auto-AVSR, 2023; ResNet-18 3D front end, Conformer
  encoder, transformer decoder, unigram-5000 subwords) with its subword
  language model, trained on LRS3 (TED talks). 19.1% word error rate on
  LRS3's test set, in the lab; a phone in a bedroom is harder. English only.
  Weights from Hugging Face (`Amanvir/LRS3_V_WER19.1`, `Amanvir/lm_en_subword`),
  which re-host the release of
  [mpc001/Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages):
  **non-commercial** (comparative and benchmarking use). Fine for trying it;
  a product needs a model with its own licence.
- **Pipeline**: [Chaplin](https://github.com/amanvirparhar/chaplin)
  (Amanvir Parhar, MIT; pinned commit in `run.py`), which carries the
  Imperial College preprocessing (Apache-2.0): alignment of four face points
  to a mean face, a 96×96 grey mouth crop, and the ESPnet beam search.
  `server.py` calls exactly the steps of Chaplin's
  `InferencePipeline.forward`, on frames from memory instead of a file, with
  one substitution: the four points (right eye, left eye, nose tip, mouth
  centre) come from BlazeFace through mediapipe's Tasks API, because the
  legacy Solutions API Chaplin's detector uses is gone from mediapipe
  0.10.3x; same model, same keypoints, same order.
- **Speed**: the model has 250M parameters; on a desktop CPU a 3-second
  sentence takes a few seconds (beam 20; `--beam 40` is Chaplin's setting,
  slower; `--no-lm` drops the language model, faster and worse). A CUDA GPU
  is used when PyTorch sees one.
- **Tunnel**: `cloudflared tunnel --url` gives a throwaway https address
  with no account, because the page is https and a phone cannot call a plain
  http server from it. `--token secret` makes the server refuse frames
  without that token (the link carries it).

## Privacy, and learning from use

Frames leave the phone only when a server is set, only for the sentence
being read, only to that address. The server decodes them in memory and
drops them with the answer; nothing is written to disk — unless the reader
turns on "Keep my sentences for training" in the page's "?" sheet. Then
each read is sent with `improve=1` and the server keeps, under
`data/<id>/`, the 96×96 grey mouth crops its model saw (`crops.npy`; not
the frames, not the face) and `meta.json` with what it read; tapping the
subtitle on the page and fixing the text posts `/feedback`, which adds the
correction (`corrected`, and `confirmed` when the reading was right).
Those samples are the material for `example/thelip-train`: a model adapted
to real phones and real speakers. Off is the default; the page says all of
this next to the switch.

## Tests

```sh
python3 test_server.py                                    # the routes, against --fake (no model)
cd ../thelip && NODE_PATH=$(npm root -g) node test/server.test.mjs   # the page against --fake, fake camera
```

Both run in CI. The model itself could not be run where this was written
(no PyTorch there, no way to download the weights): the contract and the
page are tested with `--fake`; the real runs happen on the user's computer,
and `server.log` says what happened. The first such run found two things
the code now handles: PyTorch's DLLs refusing to initialise from the
Hebrew-named user folder (hence `C:\Users\Public\thelip-server`), and
mediapipe 0.10.35 without the Solutions API (hence the Tasks detector).

## Files

- `server.py` — the HTTP server: `GET /health`, `POST /read` (multipart:
  `fps`, `frames`…, `improve`), `POST /feedback`, CORS open, optional bearer
  token, `--fake`; samples under `THELIP_DATA` (default `<work>/data`).
- `run.py` — fetches Chaplin, the models and cloudflared, starts the
  server and the tunnel, prints the link and the QR code, writes `link.txt`.
- `thelip-server.cmd` — the Windows launcher: Python, `.venv`, packages,
  then `run.py`.
- `requirements.txt`, `test_server.py`.
