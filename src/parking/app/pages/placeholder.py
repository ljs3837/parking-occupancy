from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlaceholderPage(QWidget):
    def __init__(self, title: str, description: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 28)
        h = QLabel(title)
        h.setStyleSheet("color:#e7eef7;font-size:22px;font-weight:700;")
        layout.addWidget(h)
        d = QLabel(description)
        d.setWordWrap(True)
        d.setStyleSheet("color:#8b9bb0;")
        layout.addWidget(d)
        layout.addStretch(1)
