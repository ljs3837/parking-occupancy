"""통합 실행 진입점.

사용:
  python src/main.py
  # 또는
  cd src && python -m parking
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking.app import run


def main() -> int:
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
