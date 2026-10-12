# Licence review (P0)

Checked on 12 October 2026, at the sources linked below. This is an engineering
review, not legal advice. Gate G0 needs the owner's sign-off on it, ideally
with a lawyer's.

Each candidate gets one of three decisions:

- **Analyse:** may be trained by us, read from the inside (activations, weights,
  circuits) and turned into explicit code.
- **Label only:** may produce labels or masks for our own training data. Its
  internals are not analysed.
- **Excluded:** not used in this research.

## Teachers and labellers

| Candidate | Role | Licence, as stated at the source | Terms that matter here | Decision |
| --- | --- | --- | --- | --- |
| [D-FINE](https://github.com/Peterande/D-FINE) | teacher architecture | Apache-2.0 (repository) | The README warns that checkpoints trained on Objects365 "should not be assumed to be commercially cleared" | **Analyse.** Train our own; no Objects365 checkpoints |
| [RT-DETR](https://huggingface.co/PekingU/rtdetr_r18vd) | teacher architecture | Apache-2.0 (Hugging Face model card) | The card names the original repository but not its licence | **Analyse**, once the original repository's licence is read |
| [RTMDet / MMDetection](https://www.pypi.org/project/mmdet/3.1.0/) | teacher architecture | Apache 2.0 | — | **Analyse** |
| [YuNet](https://huggingface.co/opencv/opencv_zoo/blob/d33294bfaec5f2386dab1783e1a4031330645f57/models/face_detection_yunet/README.md) | face teacher and fallback | MIT | — | **Analyse** |
| [Le-DETR](https://arxiv.org/pdf/2602.21010) | teacher candidate | none yet: "Code and weights will be open-sourced" | Its appendix notes neighbourhood attention does not export to ONNX or TensorRT | **On hold** until code and a licence exist |
| [Ultralytics YOLO](https://ultralytics.com/license) | — | AGPL-3.0 by default, including models trained with it | Closed-source use requires an Enterprise licence | **Excluded** |
| [SAM 3](https://huggingface.co/facebook/sam3/blob/main/LICENSE) | labeller | SAM License (custom) | Royalty-free limited licence; no reverse engineering (1.b.iv); commercial use and training on outputs are not addressed; publications must acknowledge use (1.b.ii) | **Label only**, after a legal check of commercial use |
| [DINOv3](https://ai.meta.com/resources/models-and-libraries/dinov3-license) | — | DINOv3 License (custom) | No reverse engineering or "discover the underlying components", and the materials include trained weights (1.b.iv); "Built with DINOv3" must be displayed (1.b.i) | **Excluded** from analysis. Feature extraction only after a legal check |

## Student families and runtimes

| Candidate | Role | Licence | Decision |
| --- | --- | --- | --- |
| Differentiable logic-gate networks ([2022](https://arxiv.org/abs/2210.08277), [2024](https://arxiv.org/abs/2411.04732)) | student family | the authors' code licence was not found | **Reimplement** from the papers |
| Transformer Programs ([NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/995f693b73050f90977ed2828202645c-Abstract.html)) | student family | not checked | **Reimplement** if used |
| ONNX Runtime 1.29.0 | runtime for learned baselines | MIT (package metadata) | **Use** |
| tract 0.23.8 | Syrup's network runtime | MIT OR Apache-2.0 (crate metadata) | **Use** |
| scikit-learn 1.9.1 | small learned baselines | BSD-3-Clause (package metadata) | **Use** |
| OpenCV 5.0.0 | baseline pipelines | Apache 2.0 (package metadata) | **Use** |

## What this means for the plan

- The teacher whose insides we read in P4 is trained by us from D-FINE,
  RT-DETR, RTMDet or YuNet code, with no Objects365 checkpoints.
- SAM 3 may label P2's training data, but only after its commercial terms are
  cleared. Until then, labels come from people and from our own models.
- Nothing from DINOv3 or SAM 3 is opened up, probed or decompiled.
- Game graphics are not a model licence question. [DATA.md](DATA.md) covers them.
