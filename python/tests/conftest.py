import os
import tempfile
from pathlib import Path

# A fresh artifact cache per test session, and no settings from the shell.
os.environ["SYRUP_CACHE_DIR"] = tempfile.mkdtemp(prefix="syrup-pytest-")
for var in ("SYRUP_MODE", "SYRUP_RUSTC", "SYRUP_FACE_MODEL"):
    os.environ.pop(var, None)

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "crates" / "syrup-runtime" / "tests" / "fixtures"
MODEL = ROOT / "crates" / "syrup-runtime" / "models" / "face_detection_yunet_2023mar.onnx"
