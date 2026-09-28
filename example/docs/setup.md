# Setup

## Requirements

- Rust (for syrup's library; the face and eye detector), `cargo build --release` at the repository root.
- Python 3.11+ with `numpy opencv-python-headless pillow scipy` for the pipeline,
  plus `starlette uvicorn httpx python-multipart` for the service, `jsonschema`
  for the schema test, and optionally `faster-whisper` (audio modes) and
  `yt-dlp` (YouTube fetching by the service).
- ffmpeg on PATH (frames and audio; fixtures; the demo videos).
- Node 18+ with TypeScript (`npm i -g typescript`) for the extension and the
  shared types; Playwright with Chromium (`npm i -g playwright && npx playwright install chromium`)
  for the end-to-end tests.
- Xcode 15+ on a Mac for the iOS app.

Nothing needs a GPU. Nothing needs an account.

## Build and test, in order

```sh
cargo build --release
example/ml/models/get_lipnet.sh                        # weights + sample clips (MIT / CC BY 4.0)

cd example/lipreader
python3 tests/make_fixtures.py                          # composited multi-person videos with ground truth
python3 tests/test_lipreader.py                         # 25 tests + 1 skip (~2 min)
python3 -m lipreader evaluate tests/fixtures            # the benchmark table
python3 -m lipreader analyze tests/fixtures/five_faces.mp4 --srt five.srt

cd ../inference-api
python3 tests/test_api.py                               # 9 tests
python3 -m lipreader_api                                # serve on http://127.0.0.1:8765

cd ../shared-types && tsc -p tsconfig.json
cd ../chrome-extension && tsc -p tsconfig.json
NODE_PATH=$(npm root -g) node test/e2e.mjs              # 25 checks in Chromium
cd ../web && NODE_PATH=$(npm root -g) node test/e2e.mjs # 7 checks
```

`LIPREADER_QUICK=1` runs only the tests that need no fixtures. `LIPREADER_DETECTOR=opencv`
switches the face detector (faster; see benchmarks.md).

## Service configuration (environment)

| variable | default | meaning |
|---|---|---|
| `LIPREADER_API_TOKEN` | unset | require `Authorization: Bearer` |
| `LIPREADER_RATE_LIMIT` | 30 | jobs+sessions per minute per client |
| `LIPREADER_MAX_UPLOAD_MB` | 512 | upload and download limit |
| `LIPREADER_RESULT_TTL_MINUTES` | 60 | results expire |
| `LIPREADER_ALLOW_LOCAL_FILES` | 0 | allow `file://` URLs (development) |
| `LIPREADER_PROCESSING` | local | what clients are told; `--host` other than localhost switches it to `server` |
| `LIPREADER_WORKERS` | 1 | analyses in parallel |
| `LIPREADER_CORS_ORIGINS` | extension + localhost + null | regex of allowed origins |
| `LIPNET_DIR` | python/thelip/LipNet | LipNet checkout |
| `LIPREADER_MUAVIC_DIR`, `LIPREADER_MPC001_DIR`, `LIPREADER_AUTO_AVSR_DIR` | unset | research checkpoints (see models.md) |

Deploying the service on a server: run it behind TLS (a reverse proxy),
set a token, keep `LIPREADER_RESULT_TTL_MINUTES` short; uploads are
deleted after reading regardless, and clients show "processed on the
server".

## Continuous integration

`.github/workflows/ci.yml` (job `example`) runs the Python tests, the API
tests, builds the shared types and the extension, and runs both Playwright
suites on Ubuntu; the iOS package is not built (no macOS runner is assumed).
