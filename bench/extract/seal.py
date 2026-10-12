"""Seal a test set before anything runs on it, and check it before every run.

    python3 bench/extract/seal.py write DATA_DIR bench/extract/sealed/NAME.sha256
    python3 bench/extract/seal.py check DATA_DIR bench/extract/sealed/NAME.sha256

`write` lists the SHA-256 of every file under DATA_DIR (relative paths, sorted)
and refuses to overwrite an existing list: a sealed set is sealed once. The list
is committed; the files themselves stay outside the repository (DATA.md).

`check` exits with an error when a listed file is missing, a file's hash
changed, or DATA_DIR holds a file the list does not name. It prints nothing
else on success but the count, so it can guard any measurement script.
"""
from __future__ import annotations

import hashlib
import os
import sys


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def files(root: str) -> list[str]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            out.append(os.path.relpath(os.path.join(dirpath, name), root).replace(os.sep, "/"))
    return sorted(out)


def read_list(path: str) -> dict[str, str]:
    listed = {}
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            sha, rel = line.split("  ", 1)
            listed[rel] = sha
    return listed


def write(root: str, out: str) -> int:
    if os.path.exists(out):
        print(f"{out} exists: a sealed set is sealed once (amend instead)", file=sys.stderr)
        return 1
    names = files(root)
    if not names:
        print(f"{root} holds no files", file=sys.stderr)
        return 1
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        for rel in names:
            f.write(f"{digest(os.path.join(root, rel))}  {rel}\n")
    print(f"sealed {len(names)} files in {out}")
    return 0


def check(root: str, sealed: str) -> int:
    listed = read_list(sealed)
    present = set(files(root))
    problems = []
    for rel, sha in sorted(listed.items()):
        path = os.path.join(root, rel)
        if rel not in present:
            problems.append(f"missing: {rel}")
        elif digest(path) != sha:
            problems.append(f"changed: {rel}")
    for rel in sorted(present - set(listed)):
        problems.append(f"not sealed: {rel}")
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        print(f"{len(problems)} problem(s): refusing to measure on {root}", file=sys.stderr)
        return 1
    print(f"{len(listed)} files match {sealed}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 4 or argv[1] not in ("write", "check"):
        print(__doc__, file=sys.stderr)
        return 2
    return (write if argv[1] == "write" else check)(argv[2], argv[3])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
