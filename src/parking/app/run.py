from __future__ import annotations

import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QMessageBox

from parking.app.paths import DATA_DIR
from parking.app.shell import AppShell


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Parking Occupancy")
    app.setFont(QFont("Segoe UI", 10))

    win = AppShell()
    win.show()

    if not DATA_DIR.exists():
        QMessageBox.warning(win, "알림", f"data 폴더가 없습니다.\n{DATA_DIR}")

    return app.exec()
