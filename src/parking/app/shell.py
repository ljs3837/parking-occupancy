from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from parking.app.annotator import AnnotatorPage
from parking.app.pages.home import HomePage
from parking.app.pages.placeholder import PlaceholderPage
from parking.app.style import APP_STYLE


class AppShell(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Parking Occupancy")
        self.resize(1480, 900)
        self.setStyleSheet(APP_STYLE)

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        nav = QFrame()
        nav.setObjectName("navRail")
        nav.setFixedWidth(220)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(14, 18, 14, 18)
        nav_layout.setSpacing(8)

        brand = QLabel("Parking One")
        brand.setObjectName("brandTitle")
        brand_sub = QLabel("통합 실행")
        brand_sub.setObjectName("brandSub")
        nav_layout.addWidget(brand)
        nav_layout.addWidget(brand_sub)
        nav_layout.addSpacing(12)

        self.stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {}
        self.nav_buttons: dict[str, QPushButton] = {}

        self.home = HomePage()
        self.annotator = AnnotatorPage()
        self.annotator.status_message.connect(self._set_status)

        self.monitor = PlaceholderPage(
            "관제",
            "실시간 점유·빈자리 현황 화면은 이후 이 메뉴에 붙입니다.",
        )
        self.train = PlaceholderPage(
            "학습",
            "YOLO/분류 학습 파이프라인은 이후 이 메뉴에 붙입니다.",
        )

        for key, title, page in [
            ("home", "홈", self.home),
            ("annotate", "색인", self.annotator),
            ("monitor", "관제", self.monitor),
            ("train", "학습", self.train),
        ]:
            self.pages[key] = page
            self.stack.addWidget(page)
            btn = QPushButton(title)
            btn.setObjectName("navBtn")
            btn.setProperty("active", "false")
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, k=key: self.goto(k))
            self.nav_buttons[key] = btn
            nav_layout.addWidget(btn)

        nav_layout.addStretch(1)
        hint = QLabel("실행: python src/main.py")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#5f6f84;font-size:11px;")
        nav_layout.addWidget(hint)

        layout.addWidget(nav)
        layout.addWidget(self.stack, 1)

        self.status = self.statusBar()
        self.status.showMessage("준비")
        self.goto("home")

    def _set_status(self, msg: str):
        self.status.showMessage(msg)

    def goto(self, key: str):
        if key != "annotate" and self.stack.currentWidget() is self.annotator:
            if not self.annotator.confirm_leave():
                return

        page = self.pages[key]
        self.stack.setCurrentWidget(page)
        for k, btn in self.nav_buttons.items():
            btn.setProperty("active", "true" if k == key else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

        titles = {
            "home": "홈",
            "annotate": "색인",
            "monitor": "관제",
            "train": "학습",
        }
        self.status.showMessage(titles.get(key, key))

    def closeEvent(self, event):
        if self.stack.currentWidget() is self.annotator and not self.annotator.confirm_leave():
            event.ignore()
            return
        event.accept()
