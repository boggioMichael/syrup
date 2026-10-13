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

None yet.
