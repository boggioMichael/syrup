# Models

What exists for visual speech recognition, what it may be used for, and
what runs in this repository today. Read together with the decisions in
[decisions.md](decisions.md) (D2, D3).

## Survey

| model | task | languages | reported accuracy (visual only) | size | licence | hosted at | runs here |
|---|---|---|---|---|---|---|---|
| **LipNet** — Assael, Shillingford, Whiteson, de Freitas, arXiv:1611.01599 (2016); weights from [rizkiarm/LipNet](https://github.com/rizkiarm/LipNet) | sentence-level VSR, closed vocabulary | en (GRID: 51 words, fixed sentence shape) | 3.38% WER overlapped speakers, 14.19% unseen (that repository's README); this pipeline: 63/66 words on its 11 sample clips | 4.6M parameters, 18 MB | MIT (code and weights); GRID corpus CC BY 4.0 | GitHub (in the repository) | **yes** — numpy, no framework |
| **Auto-AVSR** — Ma, Haliassos, Fernandez-Lopez, Chen, Petridis, Pantic (2023); [mpc001/auto_avsr](https://github.com/mpc001/auto_avsr) | open-vocabulary VSR/AVSR | en | 20.3% WER on LRS3 (vsr_trlrs2lrs3vox2avsp_base) | 250M parameters | Apache-2.0 code; weights "may have their own licenses or terms and conditions derived from the dataset used for training" (LRS2/LRS3 are research datasets) | Google Drive; the LRS3 19.1% model re-hosted on Hugging Face (`Amanvir/LRS3_V_WER19.1`) | on your own machine, yes: [`example/thelip-server`](../thelip-server) runs it through Chaplin for thelip.ai; not here (no PyTorch, no download) |
| **VSR for Multiple Languages** — Ma, Petridis, Pantic (2022); [mpc001/Visual_Speech_Recognition_for_Multiple_Languages](https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages) | open-vocabulary VSR | en, es, fr, pt, zh | LRS3 32.3% WER; CMU-MOSEAS es 44.5%, pt 51.4%, fr 58.6%; CMLR zh 8.0% CER | ~186 MB each | non-commercial: "comparative or benchmarking purposes" only | Google Drive / Baidu | no: PyTorch + ESPnet-based code + download |
| **AV-HuBERT + MuAViC** — Anwar, Shi, Ghosh, et al. (2023); [facebookresearch/muavic](https://github.com/facebookresearch/muavic) | AVSR (audio-visual) | en, ar, de, el, es, fr, it, pt, ru | video-only WER not reported by the authors; AVSR checkpoints can be run with the audio stream zeroed, at unknown cost | AV-HuBERT large | CC BY-NC 4.0 | dl.fbaipublicfiles.com | no: PyTorch + fairseq + av_hubert + download |
| Hebrew | — | he | — | — | — | — | **no public visual speech model or corpus was found** (search on 2026-09-28); reachable through the audio modes (Whisper supports Hebrew) |

Comparison criteria the spec asked for, honestly applied: on accuracy the
open-vocabulary models are far better than LipNet on real speech (LipNet
reads only GRID's vocabulary); on multilingual coverage only MuAViC covers
Arabic, German and Italian; on speed and size LipNet is the only one that
runs in numpy on a CPU in real time; on licence LipNet is the only one
usable commercially without further permission; on Core ML suitability
LipNet converts (conv3d, GRU, dense; `ml/conversion/lipnet_to_coreml.py`),
the transformer models would need a PyTorch → Core ML conversion with
their own repositories' code; on non-frontal faces and low resolution
every one of them degrades — LipNet measurably (docs/benchmarks.md: WER
0.5 at half resolution) — and none reads a profile mouth.

## What this means for the product

- English works today with a closed vocabulary. A user gets exactly what
  the model can do and a note saying what it cannot (`vocabulary` in the
  model info, shown by the clients).
- The five other requested languages need one of the research models. The
  code has a `ModelSpec` for each with its licence and the exact steps
  (`example/ml/models/fetch_research_models.py`, `LIPREADER_*_DIR`), the
  registry reports them as unavailable with the reason, and `analyze`
  produces no text for them. Integrating one is a backend class of ~150
  lines in `lipreader/vsr/` around that repository's inference code, plus
  its download — and, for anything sold, a licence conversation.
- Hebrew visual-only is not a matter of engineering here: there is no model
  to integrate. The honest path is `audio-attributed` mode (Whisper reads
  the sound, the lips decide who spoke), which the pipeline supports.
- Language identification from lips does not exist publicly; `auto` in
  visual mode is reported as `assumed`.

## A quirk worth knowing (LipNet's crop scale)

LipNet's original preprocessing pads the mouth by multiplying the corners'
absolute x coordinates by 1 ± 0.19, so the crop scale it was trained with
depends on where the mouth sat in the 360x288 GRID frame (x ≈ 180). A face
on the right of a wide frame gets, with the literal formula, a crop the
model has never seen — that is what broke the two-person fixtures before
`mouth.py` pinned the padding to GRID's position (`GRID_MOUTH_X`). Any port
must do the same.

## Getting the models

```sh
example/ml/models/get_lipnet.sh                                          # MIT, runs
python3 example/ml/models/fetch_research_models.py muavic --language ar --accept-noncommercial
python3 example/ml/models/fetch_research_models.py mpc001 --language es --accept-noncommercial
python3 example/ml/models/fetch_research_models.py auto-avsr
python3 example/ml/conversion/export_lipnet_weights.py --out lipnet-grid-weights.npz   # verified here
python3 example/ml/conversion/lipnet_to_coreml.py --weights lipnet-grid-weights.npz    # needs coremltools; not run here
```

No weights are committed to this repository.
