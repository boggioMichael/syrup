# ml

- `models/get_lipnet.sh` — the one model that runs here (MIT), cloned next to The Lip.
- `models/fetch_research_models.py` — the research checkpoints for other languages, behind their licences (none redistributed, none run here; see ../docs/models.md).
- `conversion/export_lipnet_weights.py` — LipNet's weights as `.npz` + a manifest of the layer semantics; verified against the numpy network.
- `conversion/lipnet_to_coreml.py` — the Core ML package for the iOS app via coremltools' MIL builder; **not run here** (coremltools not installable), `--check` compares with numpy on first use.
- Evaluation lives in the package: `python3 -m lipreader evaluate tests/fixtures` (`lipreader/evaluation.py`: WER, CER, speaker attribution, track consistency, cut recall, speaking overlap, latency, realtime factor).
