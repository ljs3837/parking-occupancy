from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from parking.app.paths import DATA_DIR, ROOT


class HomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        layout.setSpacing(16)

        title = QLabel("Parking Occupancy")
        title.setObjectName("brandTitle")
        title.setStyleSheet("color:#e7eef7;font-size:28px;font-weight:700;")
        layout.addWidget(title)

        sub = QLabel("CCTV 주차면 점유 · 색인 · 관제를 한 앱에서")
        sub.setStyleSheet("color:#8b9bb0;font-size:14px;")
        layout.addWidget(sub)

        card = QFrame()
        card.setObjectName("heroCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 18, 18, 18)
        card_layout.setSpacing(8)

        for line in [
            "왼쪽 메뉴에서 기능을 선택하세요.",
            "· 색인 — 주차면 다각형 / 점유 태그",
            "· 관제 · 학습 — 이후 연결 예정",
            f"프로젝트: {ROOT}",
            f"데이터: {DATA_DIR}",
        ]:
            lab = QLabel(line)
            lab.setWordWrap(True)
            card_layout.addWidget(lab)

        layout.addWidget(card)
        layout.addStretch(1)
