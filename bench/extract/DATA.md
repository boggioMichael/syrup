# Data protocol (P0)

Written on 12 October 2026, before any data for this research was collected
or measured. Changes after that go to [AMENDMENTS.md](AMENDMENTS.md) with the
reason. This is an engineering protocol, not legal advice.

## The three kinds of data

| Kind | What | Where it lives | Used for |
| --- | --- | --- | --- |
| Dev | Files already public in the `ms` repository (below), and the synthetic glyph lines of `bench/vs-opencv` | the repositories, as they are | building, tuning and the P1 shakedown |
| Test | New MapleStory recordings made **after** this protocol is committed | a private store, never this repository | the gates' decisions, once each |
| Shadow | Disagreements between explicit code and the fallback model, from players who opted in (P7) | a private store | finding gaps; never the test set |

### Dev files

Hashes are in [manifests/dev-v1.sha256](manifests/dev-v1.sha256), from `ms` at
`c1564d79` (10 October 2026).

| File | Size | Labels |
| --- | --- | --- |
| `resources/maplestory.png` | 1366 × 767 | none |
| `resources/hud-classic-4k-strip.png` | bottom 216 rows of a 3840 × 2160 Classic World frame | `tests/classic_hud.rs`: HP 178/178, MP 101/101, EXP 619 [49.84%] |
| `resources/hud-4k-strip.png` | bottom 216 rows of a 3840 × 2160 frame, modern HUD | `tests/classic_hud.rs` |
| `resources/owner-classic/*.png` | four Classic World HUD crops, about 307 × 90 | in the file name (`659_659` is 659/659) |
| `chaos-zakum-solo-lvl230.mp4` | 1152 × 720, 30 fps, 5,813 frames | none |

The synthetic glyph lines (120 dev and 400 test lines, `seed 20261012`) are
hashed in `bench/vs-opencv/results/inputs.json`. Their test split was already
used by the OpenCV benchmark, so here it is dev data: a shakedown, never a
gate's evidence.

### Test recordings to collect

Recorded after this file is committed, by the owner playing, before any other
player's data is used:

- Both HUD styles (Classic World and modern), at 1366 × 768, 1920 × 1080,
  2560 × 1440 and 3840 × 2160.
- At least eight sessions per sealed set, on different maps and classes, each
  tagged with the world, the client patch date, the resolution and the HUD style.
- Enough for the four sealed sets of the pre-registration: 7,000 HUD text
  lines (task A), 7,000 icons (task B) and 3,200 frames with boxes (task C)
  in all.

## Consent and privacy

- The owner's own recordings are used with the owner's consent. Frames stay on
  the owner's PC; the recorder keeps only the bottom strip of the game area by
  default, and the extraction, run on that PC, copies only the HP, MP and EXP
  crops, so chat and names never leave it (amendment 3). For task C, the text
  inside name tags is blanked before labelling; the tag's box stays, because it
  locates the player.
- Other players take part only by opting in inside MapleSyrup, and can withdraw.
  Processing happens on their machine. Only the crop around a disagreement
  leaves it, never chat, never a whole frame.
- Raw recordings are deleted once their labelled crops are extracted. A player's
  crops are deleted on request.
- Before any other player's data is collected, the consent text and the flow
  are checked by a lawyer (Israeli privacy law, and GDPR for EU players).

## Game graphics

Templates, glyph tables and icon sets derived from the game's graphics are
learned on the player's own machine and are not shipped inside MapleSyrup. The
research keeps its copies in the private store, only to measure. Nothing new
from the game enters this public repository.

## Splits and sealing

- Splits are by session, never by frame, because frames of one session are
  near copies of each other. Within a session, a crop identical pixel for
  pixel to one already kept is dropped, so sizes count distinct items
  (amendment 4).
- Test sessions are sealed before anything runs on them: `seal.py write` lists
  the SHA-256 of every file in `sealed/<name>.sha256`, and that list is
  committed. The files stay private.
- `seal.py check` runs before every measurement on test data and refuses a file
  that is missing from the list, or whose hash changed.
- Each sealed set decides the gate the pre-registration assigns to it, and is
  never used for tuning. Any other use needs an amendment that says why.

## Labelling

- **Task A, HUD text:** every line is read by a person. The current reader's
  output is shown as a suggestion, and the person confirms or corrects it at full
  resolution. A label is the visible characters, left to right, without spaces
  (amendment 5).
- **Task B, icons:** the label set is written first, from the game's own skill
  and buff names. A person labels every icon.
- **Task C, objects:** class definitions are written before labelling. Boxes may
  be suggested by our own detector, or by SAM 3 once its commercial terms are
  cleared ([LICENSES.md](LICENSES.md)). A person checks every box.
- A random 5% of each set is labelled again, blind, and the agreement is
  reported beside the results.

## Game updates

A client patch that changes the HUD, fonts or sprites starts a new test tag.
Old results stay valid for their tag; nothing is relabelled silently.
