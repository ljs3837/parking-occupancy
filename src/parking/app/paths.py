from pathlib import Path

# src/parking/app/paths.py → project root
ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
ANNOT_DIR = DATA_DIR / "annotations"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
