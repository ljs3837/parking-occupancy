from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from parking.app.annotator import AnnotatorPage
from parking.app.pages.home import HomePage
from parking.app.pages.monitor import DETECTOR, MonitorPage
from parking.app.pages.train import TrainPage
from parking.app.style import APP_STYLE
from parking.perception.live import list_models


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
        nav.setFixedWidth(230)
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

        self.monitor = MonitorPage()
        self.monitor.status_message.connect(self._set_status)
        self.train = TrainPage()
        self.train.status_message.connect(self._set_status)

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
        model_title = QLabel("모델")
        model_title.setObjectName("brandSub")
        nav_layout.addWidget(model_title)
        refresh_models = QPushButton("새로고침")
        refresh_models.clicked.connect(self.refresh_models)
        nav_layout.addWidget(refresh_models)
        self.model_list = QListWidget()
        self.model_list.setMinimumHeight(140)
        self.model_list.itemClicked.connect(self.select_model)
        nav_layout.addWidget(self.model_list)

        layout.addWidget(nav)
        layout.addWidget(self.stack, 1)

        self.status = self.statusBar()
        self.status.showMessage("준비")
        self.refresh_models()
        self.goto("home")

    def refresh_models(self):
        current = self.model_list.currentItem()
        current_label = current.text() if current else ""
        self.model_list.clear()
        models = list_models()
        chosen = 0
        for index, model in enumerate(models):
            item = QListWidgetItem(model["label"])
            item.setData(Qt.ItemDataRole.UserRole, model)
            item.setToolTip(model["path"])
            self.model_list.addItem(item)
            if model["label"] == current_label or (not current_label and model["kind"] == "occupancy"):
                chosen = index
        if self.model_list.count():
            self.model_list.setCurrentRow(chosen)
            self.select_model(self.model_list.item(chosen))

    def select_model(self, item: QListWidgetItem):
        model = item.data(Qt.ItemDataRole.UserRole)
        DETECTOR.use(model["kind"], model["path"], model["label"])
        if model["kind"] == "cnn":
            self.status.showMessage("CNN은 잘라 둔 칸의 점유만 분류합니다. 칸 위치는 YOLO 분할 모델을 학습해야 합니다.")
        else:
            self.status.showMessage(f"모델 선택: {model['label']}")

    def _set_status(self, msg: str):
        self.status.showMessage(msg)

    def goto(self, key: str):
        if key != "annotate" and self.stack.currentWidget() is self.annotator:
            if not self.annotator.confirm_leave():
                return
        if key != "monitor" and self.stack.currentWidget() is self.monitor:
            self.monitor.pause_all()

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
        self.monitor.stop_all()
        event.accept()
