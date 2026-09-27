"""Title and end cards for the demo video, sized to the clip frames.

    python3 cards.py --summary clips.json --out-dir .

The summary is what `the_lip.py --truth` writes next to its output; the
numbers on the end card come from it. Text is measured and wrapped, so
nothing runs off the frame.
"""
import argparse
import json
import subprocess

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG = (12, 12, 16)


def wrap(draw, text, font, width):
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = (line + " " + w).strip()
        if draw.textlength(trial, font=font) <= width:
            line = trial
        else:
            lines.append(line)
            line = w
    if line:
        lines.append(line)
    return lines


def card(size, blocks, margin=48):
    """blocks: list of (text, font, colour, gap_after). Centred vertically."""
    W, H = size
    img = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(img)
    laid = []
    for text, font, colour, gap in blocks:
        for line in wrap(draw, text, font, W - 2 * margin):
            laid.append((line, font, colour, 0))
        laid[-1] = laid[-1][:3] + (gap,)
    heights = [font.getbbox("Ag")[3] + 6 for _, font, _, _ in laid]
    total = sum(h + gap for h, (_, _, _, gap) in zip(heights, laid))
    y = (H - total) / 2
    for (line, font, colour, gap), h in zip(laid, heights):
        tw = draw.textlength(line, font=font)
        draw.text(((W - tw) / 2, y), line, font=font, fill=colour)
        y += h + gap
    return img


def write_video(img, path, seconds, fps):
    W, H = img.size
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    raw = img.tobytes()
    for _ in range(int(seconds * fps)):
        proc.stdin.write(raw)
    proc.stdin.close()
    proc.wait()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="the .json the_lip.py --truth wrote")
    ap.add_argument("--seconds", type=float, default=6)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    with open(args.summary) as f:
        summary = json.load(f)
    size = tuple(int(v) for v in summary["size"].split("x"))
    fps = summary["fps"]
    words = f"{summary['words_right']}/{summary['words']}"
    sentences = f"{summary['sentences_right']}/{summary['clips']}"
    clips = summary["clips"]

    title = ImageFont.truetype(FONT_BOLD, 64)
    h2 = ImageFont.truetype(FONT, 26)
    body = ImageFont.truetype(FONT, 20)
    fine = ImageFont.truetype(FONT, 15)
    white, grey, dim, yellow, green = (255, 255, 255), (200, 200, 200), (140, 140, 140), (255, 230, 80), (120, 255, 150)

    intro = card(size, [
        ("THE LIP", title, yellow, 6),
        ("lip reading from muted video, subtitled live", h2, white, 34),
        ("The clips are real recordings of people speaking. Their audio is removed before anything runs.", body, grey, 14),
        ("syrup finds the face and the eyes and locates the mouth on every frame.", body, grey, 14),
        ("LipNet (Assael et al. 2016) reads the lips: its published weights, run in numpy.", body, grey, 14),
        ("The subtitle is re-decoded on every frame from what has been seen so far.", body, grey, 34),
        ("Video: GRID corpus (Cooke et al., CC BY 4.0), via github.com/rizkiarm/LipNet (MIT).", fine, dim, 0),
    ])
    outro = card(size, [
        ("THE LIP", title, yellow, 6),
        (f"{words} words read correctly over {clips} muted clips", h2, green, 10),
        (f"{sentences} sentences fully correct, no audio used", h2, white, 34),
        ("Honest limits: LipNet is trained on the GRID vocabulary (51 words, fixed sentence shape).", body, grey, 14),
        ("It reads that vocabulary well and will not read open English; that needs a larger model.", body, grey, 14),
        ("The mouth localisation, the streaming decode and the overlay are the reusable parts.", body, grey, 34),
        ("github.com/boggioMichael/syrup", h2, yellow, 0),
    ])
    intro.save(f"{args.out_dir}/card_in.png")
    outro.save(f"{args.out_dir}/card_out.png")
    write_video(intro, f"{args.out_dir}/card_in.mp4", args.seconds, fps)
    write_video(outro, f"{args.out_dir}/card_out.mp4", args.seconds, fps)


if __name__ == "__main__":
    main()
