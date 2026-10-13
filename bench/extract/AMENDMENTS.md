# Amendments

Every change to [PREREGISTRATION.md](PREREGISTRATION.md) or [DATA.md](DATA.md)
after gate G0 is recorded here: the date, what changed, why, and what data had
been seen when it changed.

## Applied at G0 (13 October 2026)

The owner approved both when signing off G0. They came from the P1 shakedown
([SHAKEDOWN.md](SHAKEDOWN.md)), which used only synthetic dev data; no test
data existed.

1. **A line reader in the tiny-network family for task A.** Implementation 3
   now also includes a small CNN over the whole text line with CTC decoding, of
   at most 100,000 parameters. *Why:* in the shakedown, every wrong read of both
   learned readers came from the shared per-glyph segmentation (26 of 26, all
   on the small size). A learned classifier added nothing over one nearest
   neighbour, so a baseline that cuts glyphs first is not the best learned
   reader.
2. **G1 on a dedicated machine.** G1 is measured on a machine that runs nothing
   else, with CPU frequency scaling fixed, not on a shared cloud container. The
   rule itself (under 5% between runs) stays. *Why:* on a two-vCPU shared
   container, the unchanged OpenCV reader's p50 moved by 27% between two runs.

## After G0

3. **Chat and names are kept off every copy, not masked at capture**
   (13 October 2026, DATA.md "Consent and privacy"). The recorder
   ([capture/record.ps1](capture/record.ps1)) keeps frames on the owner's PC,
   saves only the bottom strip of the game area unless asked for whole frames,
   and the extraction, which runs on that PC, copies only the HP, MP and EXP
   crops. *Why:* a recorder cannot find the chat window reliably on every
   layout, while copying only field crops keeps chat and names out of every
   copy without having to find them. *Data seen:* no test data exists; the
   owner's learned crops (dev data) showed where the fields sit.
4. **Identical crops count once per session** (13 October 2026, DATA.md
   "Splits and sealing"). Within a session, a crop identical pixel for pixel
   to one already kept is dropped, so a set's size counts distinct lines,
   icons or tiles. *Why:* the HUD shows the same value for many frames in a
   row; repeated identical crops add no evidence and would overstate certainty
   (the rule of three counts them as independent). *Data seen:* none of the
   test data.
5. **A task A label is the visible characters, left to right, without
   spaces** (13 October 2026, DATA.md "Labelling"). For example
   `7,360[53.66%]`, not the teacher's `7,36053.66%`. *Why:* MapleSyrup's
   teacher spells the same line in several ways, and an exact-string measure
   needs one spelling. *Data seen:* the owner's learned font samples (dev
   data), where the same EXP line was spelled with and without brackets.
