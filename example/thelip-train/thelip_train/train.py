"""Fine-tuning with Auto-AVSR's training code, unchanged except for the
lines it needs outside the authors' cluster (no wandb, no SLURM, a wall-clock
limit; and on one GPU no distributed strategy — Windows has no NCCL — with
the precision and the data workers set from the environment), run in a
per-language copy of the repository whose tokenizer files are the
language's own.

English: the whole model starts from the authors' VSR checkpoint trained on
3,448 hours (vsr_trlrs2lrs3vox2avsp_base, 20.3% WER on LRS3) and adapts to
the new clips at a low learning rate. Another language: the visual front
end and the Conformer encoder are transferred from that checkpoint, the
decoder and the CTC head are new for the language's tokens.

    exp/<lang>/auto_avsr/            the patched copy
    exp/<lang>/<run>/epoch=N.ckpt    Lightning checkpoints
    exp/<lang>/<run>/model_avg.pth   the average of the last epochs, the model to serve
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional

from .common import DIRS, say
from .tokenizer import auto_avsr_dir, train_tokenizer

BASE_CHECKPOINT_DRIVE = "https://drive.google.com/file/d/1r1kx7l9sWnDOCnaFHIGvOtzuhFyFA88_/view?usp=sharing"   # vsr_trlrs2lrs3vox2avsp_base.pth


def base_checkpoint() -> str:
    path = os.path.join(DIRS["tools"], "vsr_trlrs2lrs3vox2avsp_base.pth")
    if os.path.isfile(path) and os.path.getsize(path) > 500_000_000:
        return path
    import gdown

    say("downloading Auto-AVSR's VSR checkpoint (3,448 h, 20.3% WER on LRS3) from the authors' Google Drive")
    got = gdown.download(BASE_CHECKPOINT_DRIVE, path, quiet=False, fuzzy=True)
    if not got or os.path.getsize(path) < 500_000_000:
        raise SystemExit(f"could not download the base checkpoint; fetch it by hand from {BASE_CHECKPOINT_DRIVE} to {path}")
    return path


def _patch(path: str, replacements: List[tuple]) -> None:
    src = open(path, encoding="utf-8").read()
    for old, new in replacements:
        if old not in src:
            raise SystemExit(f"{path}: expected to find {old!r} to patch; the pinned auto_avsr changed")
        src = src.replace(old, new)
    open(path, "w", encoding="utf-8").write(src)


def prepare_copy(language: str) -> str:
    """A copy of auto_avsr for the language, patched, with its tokenizer in place."""
    dst = os.path.join(DIRS["exp"], language, "auto_avsr")
    if not os.path.isdir(dst):
        shutil.copytree(auto_avsr_dir(), dst, ignore=shutil.ignore_patterns(".git", "*.ipynb", "doc"))
        _patch(os.path.join(dst, "train.py"), [
            ("from pytorch_lightning.loggers import WandbLogger", "from pytorch_lightning.loggers import CSVLogger"),
            ('logger=WandbLogger(name=args.exp_name, project="auto_avsr_lipreader", group=args.group_name),',
             'logger=CSVLogger(args.exp_dir, name=args.exp_name),\n        max_time=os.environ.get("THELIP_MAX_TIME") or None,'),
            ('args.slurm_job_id = os.environ["SLURM_JOB_ID"]', 'args.slurm_job_id = os.environ.get("SLURM_JOB_ID", "0")'),
            ("    ensemble(args)\n", "    # checkpoints are averaged by thelip_train.train.average_last\n"),
            # One GPU (a PC): no DDP (NCCL does not exist on Windows), no synced batch norm;
            # bf16 on a card that has it, set by THELIP_PRECISION.
            ("        sync_batchnorm=True,\n", "        sync_batchnorm=args.gpus * args.num_nodes > 1,\n"),
            ("        strategy=DDPStrategy(find_unused_parameters=False),\n",
             '        strategy=DDPStrategy(find_unused_parameters=False) if args.gpus * args.num_nodes > 1 else "auto",\n'
             '        precision=os.environ.get("THELIP_PRECISION") or "32-true",\n'),
        ])
        _patch(os.path.join(dst, "datamodule", "data_module.py"), [
            ("        num_workers=10,\n", '        num_workers=int(os.environ.get("THELIP_NUM_WORKERS") or 10),\n'),
        ])
    if language != "en":
        model, units = train_tokenizer(language)
        spm_dir = os.path.join(dst, "spm", "unigram")
        shutil.copyfile(model, os.path.join(spm_dir, "unigram5000.model"))
        shutil.copyfile(units, os.path.join(spm_dir, "unigram5000_units.txt"))
    return dst


def average_last(run_dir: str, k: int = 5) -> str:
    """Average the parameters of the last k epoch checkpoints (Auto-AVSR averages 10; small runs have fewer)."""
    import torch

    ckpts = sorted(glob.glob(os.path.join(run_dir, "epoch=*.ckpt")), key=lambda p: int(re.search(r"epoch=(\d+)", p).group(1)))
    if not ckpts:
        last = os.path.join(run_dir, "last.ckpt")
        if not os.path.isfile(last):
            raise SystemExit(f"no checkpoints in {run_dir}")
        ckpts = [last]
    ckpts = ckpts[-k:]
    avg = None
    for path in ckpts:
        state = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
        state = {key[len("model."):]: v for key, v in state.items() if key.startswith("model.")}
        if avg is None:
            avg = {key: v.clone().float() for key, v in state.items()}
        else:
            for key, v in state.items():
                avg[key] += v.float()
    for key in avg:
        avg[key] /= len(ckpts)
    out = os.path.join(run_dir, "model_avg.pth")
    torch.save(avg, out)
    say(f"averaged {len(ckpts)} checkpoints -> {out}")
    return out


def train(language: str, run: str, epochs: int, lr: float, max_hours: Optional[float] = None,
          max_frames: Optional[int] = None, gpus: int = 1, resume: bool = True) -> str:
    """Runs the fine-tuning; returns the averaged model path. max_frames: frames per
    batch (THELIP_MAX_FRAMES, else 1600, the authors' per-GPU figure; a 12 GB card
    wants about half, in bf16)."""
    max_frames = max_frames or int(os.environ.get("THELIP_MAX_FRAMES") or 1600)
    copy = prepare_copy(language)
    manifests = os.path.join(DIRS["manifests"], language)
    for name in ("train", "val", "test"):
        if not os.path.isfile(os.path.join(manifests, f"{name}.csv")):
            raise SystemExit(f"missing {manifests}/{name}.csv; run the manifest stage first")
    run_dir = os.path.join(DIRS["exp"], language, run)
    base = base_checkpoint()
    cmd = [sys.executable, "train.py", "--exp-dir", os.path.join(DIRS["exp"], language), "--exp-name", run, "--modality", "video",
           "--root-dir", DIRS["clips"], "--train-file", f"{language}/train.csv", "--val-file", f"{language}/val.csv",
           "--test-file", f"{language}/test.csv", "--num-nodes", "1", "--gpus", str(gpus),
           "--pretrained-model-path", base, "--max-epochs", str(epochs), "--lr", str(lr), "--warmup-epochs", "1",
           "--max-frames", str(max_frames), "--train-num-buckets", "100", "--ctc-weight", "0.1"]
    if language != "en":
        cmd.append("--transfer-encoder")
    last = os.path.join(run_dir, "last.ckpt")
    if resume and os.path.isfile(last):
        cmd += ["--ckpt-path", last]
    env = dict(os.environ, SLURM_JOB_ID="0", PYTHONUNBUFFERED="1")
    if max_hours:
        h = int(max_hours)
        m = int((max_hours - h) * 60)
        env["THELIP_MAX_TIME"] = f"00:{h:02d}:{m:02d}:00"
    say(f"{language}/{run}: {' '.join(cmd[1:])}" + (f" (wall clock at most {env.get('THELIP_MAX_TIME')})" if max_hours else ""))
    log = open(os.path.join(DIRS["exp"], language, f"{run}.log"), "a", buffering=1, encoding="utf-8")
    proc = subprocess.run(cmd, cwd=copy, env=env, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise SystemExit(f"training exited with {proc.returncode}; see {log.name}")
    return average_last(run_dir)
