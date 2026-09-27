# The Lip

Lip reading from muted video, subtitled as the person speaks. Named after
Tony Lip.

<p align="center"><img src="../../docs/the-lip.gif" alt="A muted clip being subtitled from the lips" width="480"></p>

The clips are real recordings of people speaking. Only their frames are
read — the audio track never enters — and the subtitle forms under the
video while the mouth moves. Over the eleven sample clips that ship with
the LipNet repository, 64 of 66 words are read correctly and 9 of 11
sentences are exact; `evaluate.py` reproduces the table below.

## How it works

```
frames ──► syrup.find_eyes ──► lip line + corners ──► 100x50 mouth crops
                                                            │
                     subtitle ◄── CTC decode ◄── LipNet (numpy) ◄─┘
```

- **Mouth localisation** (`mouth.py`) — `syrup.find_eyes` gives the
  interocular distance and the horizontal centre; the line between the
  lips is the row below the eyes along which the most pixels are clearly
  darker than the skin around them (a nostril shadow is dark but short, the
  lip line is dark and long); the dark run along that row gives the
  corners. Estimates are smoothed over time and frames without eyes borrow
  the nearest frame's. The crop replicates LipNet's preprocessing: the
  frame is resized so the padded mouth spans 100 px and a 100x50 window is
  cut around its centre.
- **The network** (`lipnet_np.py`) — LipNet (Assael, Shillingford, Whiteson,
  de Freitas, 2016): three 3-D convolutions with batch norm and pooling,
  two bidirectional GRUs, a softmax over 26 letters, space and the CTC
  blank. It is re-implemented in numpy from the Keras 2.0.2 weights
  published with [rizkiarm/LipNet](https://github.com/rizkiarm/LipNet),
  reproducing that version's semantics (gate order z, r, h; hard-sigmoid
  recurrent activation; reset gate before the recurrent product; batch-norm
  epsilon 1e-3). The weight file is read with `minih5.py`, a minimal HDF5
  parser, so no TensorFlow or h5py is needed. Decoding is best-path CTC
  followed by Norvig's spelling corrector over the GRID dictionary, as in
  the original code.
- **Streaming subtitles** (`the_lip.py`) — after every frame the prefix
  seen so far is decoded again. Characters the network emits in its last 8
  frames are its guess at how the sentence continues, so they are held
  back; what appears has settled. Convolution features are temporally
  local, so the full-clip features are reused and only the last frames
  are recomputed with the padding a shorter sequence would have.

## Run it

```sh
cargo build --release                                     # the syrup library
pip install numpy pillow opencv-python-headless
cd python/thelip
git clone --depth 1 https://github.com/rizkiarm/LipNet    # weights, dictionary, sample clips
python3 evaluate.py                                       # the accuracy table
./make_demo.sh                                            # the-lip-demo.mp4
python3 the_lip.py my_clip.mp4 --out subtitled.mp4        # any 25 fps clip of a face, 75 frames per utterance
python3 test_thelip.py                                    # no weights needed
```

Set `LIPNET_DIR` (or pass `--lipnet`) if the checkout lives elsewhere.
Clip arguments may be glob patterns; `the_lip.py` expands them itself,
so the same command works from `cmd.exe`.

## What it reads, honestly

| clip | spoken | read from the lips |
|---|---|---|
| bbaf2n | bin blue at f two now | bin blue at f two now |
| brbk7n | bin red by k seven now | bin red by k seven now |
| lbax4n | lay blue at x four now | lay blue at x four now |
| lbbc2a | lay blue by **c** two again | lay blue by **d** two again |
| lrwp9a | lay red with p nine again | lay red with p nine again |
| lwbsza | lay white by s zero again | lay white by s zero again |
| pwij3p | place white in j three please | place white in j three please |
| sbia1a | set blue in a one again | set blue in a one again |
| sbwe5n | set blue with e five now | set blue with e five now |
| swiz3n | set white in z **three** now | set white in z **one** now |
| id2_vcd_swwp2s | set white with p two soon | set white with p two soon |

64/66 words (97.0%). The figure the weights' authors report for them is
3.38% word error rate (96.6% of words) on the overlapped-speakers test
set; eleven clips are a small slice of it, and the numbers agree.

LipNet is trained on the GRID corpus: 51 words in a fixed sentence shape
(*command colour preposition letter digit adverb*), 34 speakers, frontal
faces at 25 fps. It reads that vocabulary well and will not read open
English — that needs a model trained on open-vocabulary data (LRS2/LRS3
class), which is orders of magnitude larger. The mouth localisation, the
streaming decode and the overlay are the parts that carry over unchanged
to such a model.

Timing on one core: the network takes about 4 ms per frame; locating the
mouth through the Python binding takes about 80 ms per frame, which is
the cost of running the eye cascade over the whole frame. The Rust
examples (`examples/lip_reading.rs`) track the face and search the eyes
inside it, which is how a live version would do it.

## Credits and licences

- LipNet: Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, Nando
  de Freitas, *LipNet: End-to-End Sentence-level Lipreading*,
  arXiv:1611.01599 (2016).
- Weights, dictionary, sample clips and reference preprocessing:
  [rizkiarm/LipNet](https://github.com/rizkiarm/LipNet), MIT licence.
  Nothing from that repository is copied here; `make_demo.sh` clones it.
- Sample clips: the GRID audiovisual sentence corpus (Cooke, Barker,
  Cunningham, Shao, 2006), CC BY 4.0.
- The code in this directory is part of syrup and under its licence.
