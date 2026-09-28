# Inference API

`example/inference-api` — a small HTTP service the Chrome extension, the web
app and scripts talk to. It runs on the user's own machine by default
(`http://127.0.0.1:8765`, processing "local"); the same service deployed on a
server is processing "server", and every response says which.

All bodies are JSON (`example/shared-types/schema`). Errors are
`{"error": "...", "code": "..."}` with a 4xx/5xx status.

## Authentication and limits

- If the service is started with `LIPREADER_API_TOKEN=...`, every request must
  carry `Authorization: Bearer <token>`; otherwise no auth (local use).
- Rate limit: `LIPREADER_RATE_LIMIT` jobs per minute per client (default 30);
  `429` with `Retry-After` when exceeded.
- Uploads: `LIPREADER_MAX_UPLOAD_MB` (default 512).
- CORS: `chrome-extension://*` and `http://localhost:*` origins are allowed
  (configurable with `LIPREADER_CORS_ORIGINS`).

## Endpoints

### `GET /health`
```json
{ "ok": true, "version": "0.1.0", "processing": "local",
  "languages": [ { "code": "en", "name": "English",
                   "visual": { "available": true, "model": "lipnet-grid", "license": "MIT (code+weights, rizkiarm/LipNet); GRID corpus CC BY 4.0",
                               "note": "51-word GRID vocabulary" },
                   "audio": { "available": false, "note": "install faster-whisper for audio modes" } }, ... ],
  "fetchers": ["upload", "local-file", "http", "youtube"] }
```
`languages` is the honest availability table: a language is listed with
`available: false` and the reason when no model can be run here.

### `GET /languages`
The `languages` array alone.

### `POST /jobs`
Create an analysis. Either multipart form data with a `file` field (the
video) and an optional `options` field (JSON), or a JSON body
`{"url": "...", "options": {...}}`. URL fetchers: `http(s)://` direct media
URLs; `youtube` when `yt-dlp` is installed on the service's machine;
`file://` only when the service runs with `LIPREADER_ALLOW_LOCAL_FILES=1`.

Response `202`: the Job (`status: "queued"`). The upload is deleted from disk
as soon as its frames have been read, and always when the job ends.

### `GET /jobs/{id}?wait=<seconds>`
The Job; with `wait`, long-polls up to that many seconds (max 30) for a status
change. `result` is present when `status == "done"`.

### `GET /jobs/{id}/export?format=srt|vtt|json|txt&track=<id>&track=<id>`
The transcript as a file (`Content-Disposition: attachment`). `track` limits
to those speakers. Uncertain words are written as `[word?]`.

### `DELETE /jobs/{id}`
Immediate deletion of the result and any remaining files. `204`. Jobs also
expire after `LIPREADER_RESULT_TTL_MINUTES` (default 60).

### `POST /sessions`, `POST /sessions/{id}/frames`, `GET /sessions/{id}`, `DELETE /sessions/{id}`
Streaming variant for a browser that captures frames itself
(`chrome.tabCapture`): `frames` takes multipart JPEGs with a `timestamps`
JSON array (seconds on the player's clock) and returns the segments produced
so far. Same deletion rules.

## Job lifecycle

```
POST /jobs  ──►  queued  ──►  running (progress 0..1)  ──►  done | failed
                                                     ──►  deleted (DELETE, or TTL)
```

## Privacy contract

- Frames are processed and discarded; nothing about a face is stored beyond
  the boxes in the result.
- No identity: tracks are numbered within the video only.
- The result's `notice` text must be shown wherever a transcript is shown.
