from __future__ import annotations

import sys

from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication, QMessageBox

from parking.app.paths import DATA_DIR
from parking.app.shell import AppShell


def _ui_font() -> QFont:
    """Windows는 Segoe/맑은 고딕, Mac은 Apple 고딕. 없는 글꼴은 다음 후보를 쓴다."""
    families = set(QFontDatabase.families())
    for name in ("Segoe UI", "Malgun Gothic", "Apple SD Gothic Neo", "Helvetica Neue"):
        if name in families:
            return QFont(name, 10)
    font = QFont()
    font.setPointSize(10)
    return font


def run() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("Parking Occupancy")
    app.setFont(_ui_font())

    win = AppShell()
    win.show()

    if not DATA_DIR.exists():
        QMessageBox.warning(win, "알림", f"data 폴더가 없습니다.\n{DATA_DIR}")

    return app.exec()
