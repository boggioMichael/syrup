#!/usr/bin/env python3
"""Convert an OpenCV Haar cascade (new-style XML) into syrup's compact
binary form, embedded by `src/cascade.rs`.

    python3 tools/cascade2bin.py haarcascade_frontalface_alt2.xml \
        assets/cascades/frontalface_alt2.bin

Layout (little-endian):
    magic b"SYRC", u8 version (1)
    u16 window_w, u16 window_h
    u32 n_features; per feature: u8 n_rects; per rect: u8 x, y, w, h; f32 weight
    u32 n_stages; per stage: f32 threshold; u32 n_weak; per weak classifier:
        u8 n_nodes; per node: i32 left, i32 right, u32 feature, f32 threshold
        (n_nodes + 1) x f32 leaf values
Node links follow OpenCV: a positive value is the index of the next node,
zero or negative is -(leaf index).
"""
import struct
import sys
import xml.etree.ElementTree as ET


def main(src, dst):
    cascade = ET.parse(src).getroot().find("cascade")
    if cascade.findtext("featureType").strip() != "HAAR":
        raise SystemExit("only HAAR cascades are supported")
    if cascade.findtext("stageType").strip() != "BOOST":
        raise SystemExit("only BOOST cascades are supported")
    out = bytearray(b"SYRC")
    out += struct.pack("<BHH", 1, int(cascade.findtext("width")), int(cascade.findtext("height")))

    features = cascade.find("features")
    out += struct.pack("<I", len(features))
    for feature in features:
        if (feature.findtext("tilted") or "0").strip() == "1":
            raise SystemExit("tilted features are not supported")
        rects = [r.text.split() for r in feature.find("rects")]
        out += struct.pack("<B", len(rects))
        for x, y, w, h, weight in rects:
            out += struct.pack("<BBBBf", int(x), int(y), int(w), int(h), float(weight))

    stages = cascade.find("stages")
    out += struct.pack("<I", len(stages))
    for stage in stages:
        weak = stage.find("weakClassifiers")
        out += struct.pack("<fI", float(stage.findtext("stageThreshold")), len(weak))
        for classifier in weak:
            nodes = classifier.findtext("internalNodes").split()
            leaves = [float(v) for v in classifier.findtext("leafValues").split()]
            n_nodes = len(nodes) // 4
            assert len(leaves) == n_nodes + 1, "a tree with n nodes has n + 1 leaves"
            out += struct.pack("<B", n_nodes)
            for i in range(n_nodes):
                left, right, feature, threshold = nodes[i * 4 : i * 4 + 4]
                out += struct.pack("<iiIf", int(left), int(right), int(feature), float(threshold))
            for value in leaves:
                out += struct.pack("<f", value)
    with open(dst, "wb") as f:
        f.write(out)
    print(f"{dst}: {len(out)} bytes, {len(features)} features, {len(stages)} stages")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
