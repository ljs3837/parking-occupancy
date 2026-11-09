"""호환용. 통합 앱으로 이동합니다 → python src/main.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking.app import run

if __name__ == "__main__":
    print("통합 실행: python src/main.py")
    raise SystemExit(run())
