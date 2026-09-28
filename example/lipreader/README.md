# lipreader (Python)

The pipeline: faces → tracks → mouths → speaking → reading → subtitles per
person, with word timing and confidence, for one or many visible people.
Built on syrup's `find_face` and `find_eyes` and on The Lip's LipNet port.

```python
import sys; sys.path.insert(0, "example/lipreader")
from lipreader import analyze
result = analyze("clip.mp4", mode="visual", language="auto")
for s in result.segments:
    print(s.track_id, f"{s.start:.2f}-{s.end:.2f}", s.text, s.confidence)
print(result.to_dict()["notice"])
```

```sh
python3 -m lipreader analyze clip.mp4 --mode visual --language en --srt clip.srt --out clip.json
python3 -m lipreader languages
python3 -m lipreader evaluate tests/fixtures
```

Modules: `video` (frames at the model's rate), `detect` (syrup / OpenCV),
`track` (IoU + Hungarian, grace, held frames), `cuts` (shot changes),
`mouth` (lip line, corners, LipNet's crop), `activity` (speaking
probability and spans), `vsr/` (model interface, LipNet backend, the honest
language registry), `decode` (CTC with timing and confidence), `audio`
(VAD, Whisper/fake recognisers), `pipeline` (the orchestration, batched,
incremental), `export` (TXT/JSON/SRT/VTT), `evaluation` (WER, CER,
attribution, track consistency, speaking overlap, cuts, latency, RTF),
`schema` (the shared contract).

Tests: `python3 tests/make_fixtures.py` composes the GRID sample clips into
seven videos with exact ground truth; `python3 tests/test_lipreader.py`
runs 25 tests over them (one speaker, two alternating, two simultaneous,
five faces, low resolution, rapid cuts, muted, unavailable languages,
current/selected speakers, the three modes, the schema). See
`../docs/benchmarks.md` for the measured numbers.
