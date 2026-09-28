"""LipNet's weights as plain arrays: a .npz plus a JSON manifest of shapes
and the layer order, from the Keras .h5 (read with The Lip's minimal HDF5
reader, no h5py). This is the input for any port — Core ML, ONNX, a Swift
or Rust forward pass — and it is verified here: the exported arrays are
loaded back and the numpy network is run on a sample clip from both.

    python3 export_lipnet_weights.py --out lipnet-grid-weights.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
THELIP = os.path.normpath(os.path.join(HERE, "..", "..", "..", "python", "thelip"))
sys.path.insert(0, THELIP)
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "lipreader")))
from minih5 import H5  # noqa: E402
from lipreader.vsr.lipnet import WEIGHTS, lipnet_dir  # noqa: E402

# Layer order and the Keras 2.0.2 semantics a port must reproduce.
MANIFEST = {
    "model": "LipNet (Assael, Shillingford, Whiteson, de Freitas 2016), weights github.com/rizkiarm/LipNet overlapped-weights368.h5 (MIT)",
    "input": "(T, 100, 50, 3) float32 in [0, 1]; width-major mouth crops as mouth.py produces them",
    "layers": [
        {"name": "conv1", "type": "conv3d", "pad": [1, 2, 2], "stride": [1, 2, 2], "then": ["batchnorm(eps=1e-3)", "relu", "maxpool(1,2,2)"]},
        {"name": "conv2", "type": "conv3d", "pad": [1, 2, 2], "stride": [1, 1, 1], "then": ["batchnorm(eps=1e-3)", "relu", "maxpool(1,2,2)"]},
        {"name": "conv3", "type": "conv3d", "pad": [1, 1, 1], "stride": [1, 1, 1], "then": ["batchnorm(eps=1e-3)", "relu", "maxpool(1,2,2)"]},
        {"name": "flatten", "type": "time-distributed flatten", "out": 1728},
        {"name": "gru1", "type": "bidirectional gru", "units": 256, "gates": "z,r,h", "recurrent_activation": "hard_sigmoid",
         "reset_before_recurrent_matmul": True},
        {"name": "gru2", "type": "bidirectional gru", "units": 256, "gates": "z,r,h", "recurrent_activation": "hard_sigmoid",
         "reset_before_recurrent_matmul": True},
        {"name": "dense1", "type": "dense", "out": 28, "then": ["softmax"]},
    ],
    "output": "(T, 28): 26 letters, space (26), CTC blank (27)",
}

KEYS = {
    "conv1_kernel": "/conv1/conv1/kernel:0", "conv1_bias": "/conv1/conv1/bias:0",
    "conv2_kernel": "/conv2/conv2/kernel:0", "conv2_bias": "/conv2/conv2/bias:0",
    "conv3_kernel": "/conv3/conv3/kernel:0", "conv3_bias": "/conv3/conv3/bias:0",
    "bn1_gamma": "/batc1/batc1/gamma:0", "bn1_beta": "/batc1/batc1/beta:0", "bn1_mean": "/batc1/batc1/moving_mean:0", "bn1_var": "/batc1/batc1/moving_variance:0",
    "bn2_gamma": "/batc2/batc2/gamma:0", "bn2_beta": "/batc2/batc2/beta:0", "bn2_mean": "/batc2/batc2/moving_mean:0", "bn2_var": "/batc2/batc2/moving_variance:0",
    "bn3_gamma": "/batc3/batc3/gamma:0", "bn3_beta": "/batc3/batc3/beta:0", "bn3_mean": "/batc3/batc3/moving_mean:0", "bn3_var": "/batc3/batc3/moving_variance:0",
    "gru1_fw_kernel": "/bidirectional_1/bidirectional_1/kernel:0", "gru1_fw_recurrent": "/bidirectional_1/bidirectional_1/recurrent_kernel:0", "gru1_fw_bias": "/bidirectional_1/bidirectional_1/bias:0",
    "gru1_bw_kernel": "/bidirectional_1/bidirectional_1/kernel_1:0", "gru1_bw_recurrent": "/bidirectional_1/bidirectional_1/recurrent_kernel_1:0", "gru1_bw_bias": "/bidirectional_1/bidirectional_1/bias_1:0",
    "gru2_fw_kernel": "/bidirectional_2/bidirectional_2/kernel:0", "gru2_fw_recurrent": "/bidirectional_2/bidirectional_2/recurrent_kernel:0", "gru2_fw_bias": "/bidirectional_2/bidirectional_2/bias:0",
    "gru2_bw_kernel": "/bidirectional_2/bidirectional_2/kernel_1:0", "gru2_bw_recurrent": "/bidirectional_2/bidirectional_2/recurrent_kernel_1:0", "gru2_bw_bias": "/bidirectional_2/bidirectional_2/bias_1:0",
    "dense_kernel": "/dense1/dense1/kernel:0", "dense_bias": "/dense1/dense1/bias:0",
}


def export(h5_path: str, out: str) -> dict:
    arrays = H5(h5_path).arrays()
    tensors = {name: np.ascontiguousarray(arrays[key]).astype(np.float32) for name, key in KEYS.items()}
    np.savez(out, **tensors)
    manifest = dict(MANIFEST, shapes={k: list(v.shape) for k, v in tensors.items()}, source=os.path.basename(h5_path))
    with open(os.path.splitext(out)[0] + ".json", "w") as f:
        json.dump(manifest, f, indent=1)
    return tensors


def verify(npz: str, h5_path: str) -> None:
    """The exported arrays drive the numpy network to the same output as the .h5."""
    from lipnet_np import LipNet, greedy_decode

    net = LipNet(h5_path)
    loaded = np.load(npz)
    for i in (1, 2, 3):
        assert np.array_equal(net.conv[i - 1]["kernel"], loaded[f"conv{i}_kernel"])
    assert np.array_equal(net.dense[0], loaded["dense_kernel"])
    rng = np.random.default_rng(0)
    clip = rng.random((30, 100, 50, 3), dtype=np.float32)
    a = net.predict(clip)
    # Swap the network's arrays for the exported ones and run again.
    net.conv[0]["kernel"] = loaded["conv1_kernel"]
    net.gru[1]["bw"] = (loaded["gru2_bw_kernel"], loaded["gru2_bw_recurrent"], loaded["gru2_bw_bias"])
    b = net.predict(clip)
    assert np.allclose(a, b), "exported arrays differ"
    print(f"verified: {len(loaded.files)} arrays, sample decode {greedy_decode(a)!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lipnet", default=lipnet_dir())
    ap.add_argument("--out", default="lipnet-grid-weights.npz")
    args = ap.parse_args()
    if not args.lipnet:
        sys.exit("no LipNet checkout (example/ml/models/get_lipnet.sh)")
    h5_path = os.path.join(args.lipnet, WEIGHTS)
    tensors = export(h5_path, args.out)
    total = sum(v.size for v in tensors.values())
    print(f"wrote {args.out}: {len(tensors)} arrays, {total:,} parameters, manifest {os.path.splitext(args.out)[0]}.json")
    verify(args.out, h5_path)


if __name__ == "__main__":
    main()
