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
Chaplin pipeline and the model files (1.2 GB) and the tunnel client. Then
it prints a link and a QR code:

```
Open this on the phone (it is the site with the server's address in it):
  https://thelip.ai/?server=https://<random>.trycloudflare.com
```

Scan it, or type it; the page keeps the address (it is in the "?" sheet,
with "Stop" to forget it). Say a sentence in English; the subtitle says
"…" while the computer reads, then the words. Ctrl+C in the window stops
the server. Every run gets a new tunnel address, so open the new link each
time (`--no-tunnel` serves this computer only, at `http://127.0.0.1:8791`).

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
  Imperial College preprocessing (Apache-2.0): mediapipe face detection,
  alignment of the four stable points to a mean face, a 96×96 grey mouth
  crop, and the ESPnet beam search. `server.py` calls exactly the steps of
  Chaplin's `InferencePipeline.forward`, on frames from memory instead of a
  file.
- **Speed**: the model has 250M parameters; on a desktop CPU a 3-second
  sentence takes a few seconds (beam 20; `--beam 40` is Chaplin's setting,
  slower; `--no-lm` drops the language model, faster and worse). A CUDA GPU
  is used when PyTorch sees one.
- **Tunnel**: `cloudflared tunnel --url` gives a throwaway https address
  with no account, because the page is https and a phone cannot call a plain
  http server from it. `--token secret` makes the server refuse frames
  without that token (the link carries it).

## Privacy

Frames leave the phone only when a server is set, only for the sentence
being read, only to that address. The server decodes them in memory and
drops them with the answer; nothing is written to disk. The page says so in
its "?" sheet.

## Tests

```sh
python3 test_server.py                                    # the routes, against --fake (no model)
cd ../thelip && NODE_PATH=$(npm root -g) node test/server.test.mjs   # the page against --fake, fake camera
```

Both run in CI. The model itself could not be run where this was written
(no PyTorch there, no way to download the weights): the contract and the
page are tested with `--fake`; the first real run is the one on your
computer, and `server.log` says what happened.

## Files

- `server.py` — the HTTP server: `GET /health`, `POST /read` (multipart:
  `fps`, `frames`…), CORS open, optional bearer token, `--fake`.
- `run.py` — fetches Chaplin, the models and cloudflared, starts the
  server and the tunnel, prints the link and the QR code, writes `link.txt`.
- `thelip-server.cmd` — the Windows launcher: Python, `.venv`, packages,
  then `run.py`.
- `requirements.txt`, `test_server.py`.
