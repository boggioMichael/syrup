"""The process that runs a model example/thelip-train exported.

Such a model is Auto-AVSR's network (Ma et al. 2023), and Auto-AVSR's code
ships its own `espnet` package, as does the Chaplin pipeline the server
loads for the published models; the two differ and cannot share one
process. So server.py starts this worker once per trained model, and talks
to it over pipes:

    request   4 bytes: T, little-endian; then T x 96 x 96 bytes: grey mouth crops
    answer    one JSON line: {"text": "..."} or {"error": "..."}
    T = 0     stop

The first line the worker writes is {"ready": true, ...} once the model is
loaded. Everything else it has to say goes to stderr. Decoding is what
Auto-AVSR's ModelModule.forward does: frontend, encoder, joint CTC and
attention beam search, no language model.

    python trained_worker.py <model folder> --auto-avsr <checkout> [--device cpu] [--beam 20]
    python trained_worker.py --fake     # no torch: answers with a fixed text (tests)
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?")
    ap.add_argument("--auto-avsr", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--fake", action="store_true")
    args = ap.parse_args()

    # the protocol owns the real stdout; anything printed goes to stderr
    out = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    inp = os.fdopen(os.dup(sys.stdin.fileno()), "rb", buffering=0)
    sys.stdout = sys.stderr

    def answer(obj: dict) -> None:
        out.write((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

    def read_exact(n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = inp.read(n - len(buf))
            if not chunk:
                raise EOFError
            buf += chunk
        return bytes(buf)

    if args.fake:
        answer({"ready": True, "fake": True})
        infer = lambda crops: f"FAKE TRAINED {len(crops)} FRAMES"  # noqa: E731
    else:
        import numpy as np
        import torch

        auto = args.auto_avsr or os.environ.get("THELIP_AUTO_AVSR") or ""
        if not os.path.isdir(os.path.join(auto, "espnet")):
            answer({"error": f"auto_avsr is not at {auto!r}"})
            return
        sys.path.insert(0, auto)
        from datamodule.transforms import TextTransform, VideoTransform  # noqa: E402  (auto_avsr's)
        from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: E402
        from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E  # noqa: E402
        from espnet.nets.scorers.length_bonus import LengthBonus  # noqa: E402

        folder = args.folder
        text = TextTransform(sp_model_path=os.path.join(folder, "unigram.model"), dict_path=os.path.join(folder, "unigram_units.txt"))
        transform = VideoTransform("test")
        model = E2E(len(text.token_list), "video", ctc_weight=0.1)
        state = torch.load(os.path.join(folder, "model.pth"), map_location="cpu", weights_only=False)
        if "state_dict" in state:  # a Lightning checkpoint rather than an export
            state = {k[len("model."):]: v for k, v in state["state_dict"].items() if k.startswith("model.")}
        model.load_state_dict(state)
        device = args.device if (not args.device.startswith("cuda") or torch.cuda.is_available()) else "cpu"
        model.to(device).eval()
        scorers = model.scorers()
        scorers["lm"] = None
        scorers["length_bonus"] = LengthBonus(len(text.token_list))
        beam_search = BatchBeamSearch(
            beam_size=args.beam, vocab_size=len(text.token_list),
            weights={"decoder": 0.9, "ctc": 0.1, "lm": 0.0, "length_bonus": 0.0},
            scorers=scorers, sos=model.odim - 1, eos=model.odim - 1,
            token_list=text.token_list, pre_beam_score_key="decoder").to(device).eval()

        def infer(crops) -> str:
            with torch.no_grad():
                x = torch.from_numpy(np.ascontiguousarray(crops)).unsqueeze(1).expand(-1, 3, -1, -1)  # T x 3 x 96 x 96
                x = transform(x).to(device)
                x = model.frontend(x.unsqueeze(0))
                x = model.proj_encoder(x)
                enc, _ = model.encoder(x, None)
                hyps = beam_search(enc.squeeze(0))
                ids = torch.tensor(list(map(int, hyps[0].asdict()["yseq"][1:])))
                return text.post_process(ids).replace("<eos>", "").strip()

        answer({"ready": True, "device": device, "tokens": len(text.token_list), "parameters": sum(p.numel() for p in model.parameters())})

    while True:
        try:
            (frames,) = struct.unpack("<I", read_exact(4))
        except EOFError:
            return
        if frames == 0:
            return
        raw = read_exact(frames * 96 * 96)
        try:
            if args.fake:
                crops = [None] * frames
            else:
                crops = np.frombuffer(raw, dtype=np.uint8).reshape(frames, 96, 96)
            answer({"text": infer(crops)})
        except Exception as e:  # noqa: BLE001
            answer({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
