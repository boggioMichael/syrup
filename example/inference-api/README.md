# lipreader inference API

```sh
python3 -m lipreader_api                         # http://127.0.0.1:8765, processing "local"
LIPREADER_API_TOKEN=secret python3 -m lipreader_api --host 0.0.0.0 --port 8765   # a server: token + "server"
python3 tests/test_api.py
```

`POST /jobs` (upload or URL) → `GET /jobs/{id}?wait=20` → `GET /jobs/{id}/export?format=srt`
→ `DELETE /jobs/{id}`; `POST /sessions` + `/frames` + `/finish` for
streamed frames; `GET /health` with the honest language table. The full
contract is in [../docs/api.md](../docs/api.md). Starlette + uvicorn;
analysis in worker threads; uploads deleted as soon as read; results
memory-only with a TTL; CORS for the extension and local pages.
