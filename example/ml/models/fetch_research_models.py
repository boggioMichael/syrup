"""Research visual-speech checkpoints for other languages, behind their
licences. None of these is redistributed here and none runs in lipreader's
current build (they need PyTorch and, for MuAViC, fairseq + av_hubert; see
docs/models.md). This script fetches what can be fetched directly and tells
you exactly what to do for the rest.

    python3 fetch_research_models.py muavic --language ar --accept-noncommercial --out models/
    python3 fetch_research_models.py mpc001 --language es
    python3 fetch_research_models.py auto-avsr
"""
import argparse
import os
import sys
import urllib.request

MUAVIC = "https://dl.fbaipublicfiles.com/muavic/models/{lang}_avsr/"
MUAVIC_FILES = ["checkpoint_best.pt", "dict.{lang}.txt", "tokenizer.model"]
MUAVIC_LANGS = ["en", "ar", "de", "el", "es", "fr", "it", "pt", "ru"]

LICENCES = {
    "muavic": ("CC BY-NC 4.0", "https://github.com/facebookresearch/muavic/blob/main/LICENSE"),
    "mpc001": ("non-commercial (\"comparative or benchmarking purposes\")",
               "https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages/blob/master/LICENSE"),
    "auto-avsr": ("Apache-2.0 code; weights carry the training data's terms (LRS2/LRS3: research)",
                  "https://github.com/mpc001/auto_avsr"),
}


def download(url: str, target: str) -> None:
    print(f"  {url} -> {target}")
    with urllib.request.urlopen(url, timeout=60) as r, open(target, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=list(LICENCES))
    ap.add_argument("--language", default="en")
    ap.add_argument("--accept-noncommercial", action="store_true", help="you have read the licence and your use is non-commercial")
    ap.add_argument("--out", default="models")
    args = ap.parse_args()
    licence, url = LICENCES[args.model]
    print(f"{args.model}: licence {licence}\n  {url}")
    if args.model in ("muavic", "mpc001") and not args.accept_noncommercial:
        sys.exit("non-commercial licence: pass --accept-noncommercial after reading it")
    if args.model == "muavic":
        if args.language not in MUAVIC_LANGS:
            sys.exit(f"MuAViC languages: {MUAVIC_LANGS}")
        target = os.path.join(args.out, f"muavic-{args.language}")
        os.makedirs(target, exist_ok=True)
        for name in MUAVIC_FILES:
            name = name.format(lang=args.language)
            download(MUAVIC.format(lang=args.language) + name, os.path.join(target, name))
        print(f"done; set LIPREADER_MUAVIC_DIR={target}. Running it needs PyTorch, fairseq and github.com/facebookresearch/av_hubert.")
    elif args.model == "mpc001":
        print("checkpoints are on Google Drive, linked from the repository README (one per language: en, es, fr, pt, zh);\n"
              "download the one you need by hand, then set LIPREADER_MPC001_DIR to its directory.\n"
              "Running it needs PyTorch and that repository's code (ESPnet-based).")
    else:
        print("checkpoints are on Google Drive, linked from https://github.com/mpc001/auto_avsr (vsr_trlrs2lrs3vox2avsp_base.pth);\n"
              "download by hand, then set LIPREADER_AUTO_AVSR_DIR. Running it needs PyTorch and that repository's code.")


if __name__ == "__main__":
    main()
