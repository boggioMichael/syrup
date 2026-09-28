"""BlazeFace (short range) from MediaPipe's model file into blazeface.bin,
the page's own face detector: the graph as a list of ops and the weights as
float16, read by faces.js with no TFLite runtime in the browser.

    python3 tools/export_blazeface.py blaze_face_short_range.tflite [--out blazeface.bin]

The model is https://storage.googleapis.com/mediapipe-models/face_detector/
blaze_face_short_range/float16/1/blaze_face_short_range.tflite (Apache 2.0,
the model card of MediaPipe's face detector); thelip-server's run.py fetches
the same file. Reading it needs only mediapipe's copy of the TFLite schema
(mediapipe.tasks.metadata.schema_py_generated) and flatbuffers.

blazeface.bin: b"BLZF", uint32 length of the JSON that follows, the JSON
(padded to 4 bytes with spaces), then the float16 weights. The JSON:
  {"input": id, "outputs": [regressors id, classificators id],
   "shapes": {id: [dims]}, "consts": {id: [offset, count]} (in float16s),
   "ops": [{"op": "conv"|"dwconv"|"add"|"relu"|"pad"|"maxpool"|"reshape"|"concat", "in": [...], "out": id, ...}],
   "source": {"file": ..., "sha256": ...}}
"""
import argparse
import hashlib
import json
import os
import struct

import numpy as np
from mediapipe.tasks.metadata import schema_py_generated as S

OPS = {v: k for k, v in S.BuiltinOperator.__dict__.items() if not k.startswith("_")}
PADDING = {S.Padding.SAME: "same", S.Padding.VALID: "valid"}
ACT = {S.ActivationFunctionType.NONE: None, S.ActivationFunctionType.RELU: "relu"}


def options(op, cls):
    table = op.BuiltinOptions()
    o = cls()
    o.Init(table.Bytes, table.Pos)
    return o


def export(path: str) -> tuple:
    buf = open(path, "rb").read()
    model = S.Model.GetRootAsModel(buf, 0)
    g = model.Subgraphs(0)
    codes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        codes.append(OPS[max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())])

    def tensor(t):
        return g.Tensors(int(t))

    def data(t):
        T = tensor(t)
        b = model.Buffers(T.Buffer())
        return b.DataAsNumpy() if b.DataLength() else None

    shapes, consts, weights, ops = {}, {}, [], []
    offset = 0
    alias = {}   # a DEQUANTIZE's output -> its float16 input

    def const_of(t):
        """The float16 constant behind tensor t (through a DEQUANTIZE), registered once."""
        nonlocal offset
        t = alias.get(int(t), int(t))
        if str(t) in consts:
            return t
        T = tensor(t)
        raw = data(t)
        assert raw is not None and T.Type() == S.TensorType.FLOAT16, (t, T.Name())
        arr = np.frombuffer(raw.tobytes(), dtype="<f2")
        consts[str(t)] = [offset, int(arr.size)]
        shapes[str(t)] = [int(d) for d in T.ShapeAsNumpy()]
        weights.append(arr)
        offset += arr.size
        return t

    def shape_of(t):
        shapes[str(int(t))] = [int(d) for d in tensor(t).ShapeAsNumpy()]
        return int(t)

    for i in range(g.OperatorsLength()):
        op = g.Operators(i)
        kind = codes[op.OpcodeIndex()]
        ins, outs = [int(t) for t in op.InputsAsNumpy()], [int(t) for t in op.OutputsAsNumpy()]
        if kind == "DEQUANTIZE":
            alias[outs[0]] = ins[0]
            continue
        out = shape_of(outs[0])
        if kind in ("CONV_2D", "DEPTHWISE_CONV_2D"):
            o = options(op, S.Conv2DOptions if kind == "CONV_2D" else S.DepthwiseConv2DOptions)
            assert o.DilationWFactor() == 1 and o.DilationHFactor() == 1
            if kind == "DEPTHWISE_CONV_2D":
                assert o.DepthMultiplier() in (0, 1), o.DepthMultiplier()
            ops.append({"op": "conv" if kind == "CONV_2D" else "dwconv", "in": [shape_of(ins[0]), const_of(ins[1]), const_of(ins[2])], "out": out,
                        "stride": [o.StrideH(), o.StrideW()], "pad": PADDING[o.Padding()], "act": ACT[o.FusedActivationFunction()]})
        elif kind == "ADD":
            o = options(op, S.AddOptions)
            ops.append({"op": "add", "in": [shape_of(t) for t in ins], "out": out, "act": ACT[o.FusedActivationFunction()]})
        elif kind == "RELU":
            ops.append({"op": "relu", "in": [shape_of(ins[0])], "out": out})
        elif kind == "PAD":
            pads = np.frombuffer(data(ins[1]).tobytes(), dtype="<i4").reshape(-1, 2).tolist()
            ops.append({"op": "pad", "in": [shape_of(ins[0])], "out": out, "pads": pads})
        elif kind == "MAX_POOL_2D":
            o = options(op, S.Pool2DOptions)
            ops.append({"op": "maxpool", "in": [shape_of(ins[0])], "out": out, "size": [o.FilterHeight(), o.FilterWidth()],
                        "stride": [o.StrideH(), o.StrideW()], "pad": PADDING[o.Padding()], "act": ACT[o.FusedActivationFunction()]})
        elif kind == "RESHAPE":
            ops.append({"op": "reshape", "in": [shape_of(ins[0])], "out": out})
        elif kind == "CONCATENATION":
            o = options(op, S.ConcatenationOptions)
            ops.append({"op": "concat", "in": [shape_of(t) for t in ins], "out": out, "axis": o.Axis(), "act": ACT[o.FusedActivationFunction()]})
        else:
            raise SystemExit(f"op {kind} is not handled by faces.js")
    names = {int(t): tensor(t).Name().decode() for t in g.OutputsAsNumpy()}
    outputs = sorted(names, key=lambda t: {"regressors": 0, "classificators": 1}[names[t]])
    spec = {"input": shape_of(g.InputsAsNumpy()[0]), "outputs": outputs, "shapes": shapes, "consts": consts, "ops": ops,
            "source": {"file": os.path.basename(path), "sha256": hashlib.sha256(buf).hexdigest()}}
    return spec, np.concatenate(weights).astype("<f2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blazeface.bin"))
    args = ap.parse_args()
    spec, weights = export(args.model)
    head = json.dumps(spec, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 4)
    with open(args.out, "wb") as f:
        f.write(b"BLZF" + struct.pack("<I", len(head)) + head + weights.tobytes())
    print(f"wrote {args.out}: {len(spec['ops'])} ops, {weights.size} weights, {os.path.getsize(args.out)} bytes")


if __name__ == "__main__":
    main()
