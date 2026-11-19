"""통합 실행 진입점.

사용:
  python src/main.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> int:
    if find_spec("PyQt6") is None:
        venv = SRC.parent / ".venv"
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if python.is_file() and Path(sys.prefix).resolve() != venv.resolve():
            return subprocess.call([str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
        print(
            f"PyQt6를 찾을 수 없습니다. 현재 Python: {sys.executable}\n"
            "이 Python 환경에 설치하려면: python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 1

    from parking.app.application import run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
