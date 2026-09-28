"""LipNet as a Core ML package for the iOS app, built layer by layer with
coremltools' MIL builder from the exported weights (no TensorFlow/Keras
needed: the .h5 is Keras 2.0.2 for TensorFlow 1, which no converter loads
any more).

    python3 export_lipnet_weights.py --out lipnet-grid-weights.npz
    python3 lipnet_to_coreml.py --weights lipnet-grid-weights.npz --out LipNetGRID.mlpackage
    python3 lipnet_to_coreml.py --weights lipnet-grid-weights.npz --out LipNetGRID.mlpackage --check

NOT RUN HERE: coremltools is not installable in the environment this was
written in, so this script has not been executed. It follows the
coremltools 7/8 MIL Builder API as documented (mb.program, mb.conv with
3-D kernels, mb.batch_norm, mb.relu, mb.max_pool, mb.gru, mb.linear,
mb.softmax). `--check` runs the numpy reference and the converted model on
the same random clip and reports the largest difference; do that on the
first Mac run before trusting the package.

Layout notes for whoever runs it first:
- MIL's conv expects (N, C, D, H, W) and a (Cout, Cin, kD, kH, kW) weight;
  LipNet's crops are (T, W=100, H=50, C) with a (kt, kw, kh, Cin, Cout)
  kernel, so both are transposed here; the app feeds (1, T, 100, 50, 3)
  and the model transposes on entry.
- MIL's gru expects (S, B, D) input and a weight_ih of shape (3H, D),
  weight_hh (3H, H), bias (3H), gate order z, r, o (update, reset,
  output), which matches Keras' z, r, h; `reset_after=False` (the reset
  gate multiplies the state before the recurrent product) is the Keras
  2.0.2 behaviour and must be set explicitly.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
except ImportError:  # pragma: no cover - documented above
    ct = mb = types = None


def hard_sigmoid_note():
    return ("Keras' hard_sigmoid (clip(0.2x + 0.5)) is LipNet's recurrent activation. MIL's gru offers "
            "recurrent_activation='hard_sigmoid'; if a coremltools version lacks it, express the GRU with mb ops per "
            "time step (matmul, clip, tanh) instead.")


def build(weights: dict, frames: int):
    if mb is None:
        sys.exit("coremltools is not installed (pip install coremltools); this script was not run in the build environment")

    T = frames

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, T, 100, 50, 3), dtype=types.fp32)])
    def prog(x):
        # (1, T, W, H, C) -> (1, C, T, W, H): N C D H W with D = time, H = width, W = height.
        h = mb.transpose(x=x, perm=[0, 4, 1, 2, 3])
        for i, (pad, stride) in enumerate([((1, 2, 2), (1, 2, 2)), ((1, 2, 2), (1, 1, 1)), ((1, 1, 1), (1, 1, 1))], start=1):
            k = weights[f"conv{i}_kernel"]  # (kt, kw, kh, Cin, Cout)
            w = np.ascontiguousarray(np.transpose(k, (4, 3, 0, 1, 2)))  # (Cout, Cin, kt, kw, kh)
            h = mb.conv(x=h, weight=w, bias=weights[f"conv{i}_bias"], strides=list(stride),
                        pad_type="custom", pad=[pad[0], pad[0], pad[1], pad[1], pad[2], pad[2]])
            h = mb.batch_norm(x=h, mean=weights[f"bn{i}_mean"], variance=weights[f"bn{i}_var"],
                              gamma=weights[f"bn{i}_gamma"], beta=weights[f"bn{i}_beta"], epsilon=1e-3)
            h = mb.relu(x=h)
            h = mb.max_pool(x=h, kernel_sizes=[1, 2, 2], strides=[1, 2, 2], pad_type="valid")
        # (1, C, T, W', H') -> (T, C*W'*H') = (T, 1728), the order LipNet's TimeDistributed(Flatten) uses (W, H, C).
        h = mb.transpose(x=h, perm=[2, 3, 4, 1, 0])          # (T, W', H', C, 1)
        h = mb.reshape(x=h, shape=[T, 1, 1728])               # (S, B, D) for the GRU
        for i in (1, 2):
            fw = mb.gru(x=h, initial_h=np.zeros((1, 256), np.float32),
                        weight_ih=np.ascontiguousarray(weights[f"gru{i}_fw_kernel"].T),
                        weight_hh=np.ascontiguousarray(weights[f"gru{i}_fw_recurrent"].T),
                        bias=weights[f"gru{i}_fw_bias"], direction="forward", output_sequence=True,
                        recurrent_activation="hard_sigmoid", activation="tanh", reset_after=False)
            bw = mb.gru(x=h, initial_h=np.zeros((1, 256), np.float32),
                        weight_ih=np.ascontiguousarray(weights[f"gru{i}_bw_kernel"].T),
                        weight_hh=np.ascontiguousarray(weights[f"gru{i}_bw_recurrent"].T),
                        bias=weights[f"gru{i}_bw_bias"], direction="reverse", output_sequence=True,
                        recurrent_activation="hard_sigmoid", activation="tanh", reset_after=False)
            h = mb.concat(values=[fw[0], bw[0]], axis=2)     # (T, 1, 512)
        h = mb.reshape(x=h, shape=[T, 512])
        logits = mb.linear(x=h, weight=np.ascontiguousarray(weights["dense_kernel"].T), bias=weights["dense_bias"])
        probs = mb.softmax(x=logits, axis=1)
        return mb.reshape(x=probs, shape=[1, T, 28], name="probabilities")

    return prog


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="the .npz from export_lipnet_weights.py")
    ap.add_argument("--out", default="LipNetGRID.mlpackage")
    ap.add_argument("--frames", type=int, default=75, help="clip length the model is built for (LipNet: 75)")
    ap.add_argument("--check", action="store_true", help="compare with the numpy reference on a random clip")
    args = ap.parse_args()
    weights = {k: np.asarray(v, dtype=np.float32) for k, v in np.load(args.weights).items()}
    prog = build(weights, args.frames)
    model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                       compute_precision=ct.precision.FLOAT32)
    model.short_description = "LipNet (GRID vocabulary) visual speech recognition; MIT weights from rizkiarm/LipNet"
    model.save(args.out)
    print(f"wrote {args.out} for {args.frames}-frame clips. {hard_sigmoid_note()}")
    if args.check:
        import os

        sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "python", "thelip")))
        from lipnet_np import LipNet
        from lipreader.vsr.lipnet import WEIGHTS, lipnet_dir

        net = LipNet(os.path.join(lipnet_dir(), WEIGHTS))
        clip = np.random.default_rng(0).random((args.frames, 100, 50, 3), dtype=np.float32)
        reference = net.predict(clip)
        out = model.predict({"x": clip[None]})["probabilities"][0]
        print(f"max |coreml - numpy| = {np.abs(out - reference).max():.5f} (expect < 1e-3)")


if __name__ == "__main__":
    main()
