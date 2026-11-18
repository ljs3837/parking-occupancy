from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class TrainWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(str)
    fail = pyqtSignal(str)

    def __init__(self, kind: str, epochs: int, batch: int):
        super().__init__()
        self.kind = kind
        self.epochs = epochs
        self.batch = batch
        self._stop = False

    def request_stop(self):
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop

    def run(self):
        try:
            def log(msg: str):
                self.log.emit(str(msg))

            from parking.training.yolo_ft import train_yolo

            path = train_yolo(
                epochs=self.epochs,
                batch=self.batch,
                log=log,
                should_stop=self._should_stop,
            )
            self.done.emit(str(path))
        except Exception as e:
            self.fail.emit(str(e))


class ModelCard(QFrame):
    start_clicked = pyqtSignal()

    def __init__(self, title: str, subtitle: str, detail: str, parent=None):
        super().__init__(parent)
        self.setObjectName("heroCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        h = QLabel(title)
        h.setStyleSheet("color:#e7eef7;font-size:18px;font-weight:700;")
        layout.addWidget(h)

        s = QLabel(subtitle)
        s.setStyleSheet("color:#2ec4b6;font-size:12px;font-weight:600;")
        layout.addWidget(s)

        d = QLabel(detail)
        d.setWordWrap(True)
        d.setStyleSheet("color:#8b9bb0;")
        layout.addWidget(d)

        form = QFormLayout()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 200)
        self.epochs.setValue(5)
        self.batch = QSpinBox()
        self.batch.setRange(1, 128)
        self.batch.setValue(16)
        form.addRow("epochs", self.epochs)
        form.addRow("batch", self.batch)
        layout.addLayout(form)

        self.btn = QPushButton("학습 시작")
        self.btn.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.btn)
        layout.addStretch(1)

    def params(self) -> tuple[int, int]:
        return self.epochs.value(), self.batch.value()

    def set_busy(self, busy: bool):
        self.btn.setEnabled(not busy)


class TrainPage(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: TrainWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(14)

        title = QLabel("학습")
        title.setStyleSheet("color:#e7eef7;font-size:22px;font-weight:700;")
        root.addWidget(title)

        desc = QLabel(
            "주차칸 YOLO를 학습합니다. 우리 색인과 허깅페이스 PKLot 사진을 전부 사용합니다."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#8b9bb0;")
        root.addWidget(desc)

        self.yolo_card = ModelCard(
            "YOLO 파인튜닝",
            "YOLO11s-seg",
            "주차칸 좌표만으로 면을 찾습니다. 차 있음/없음은 학습에 쓰지 않고, 나중에 차량 YOLO로 판정합니다.",
        )
        self.yolo_card.batch.setValue(4)
        self.yolo_card.start_clicked.connect(self._start)
        root.addWidget(self.yolo_card)

        row = QHBoxLayout()
        self.stop_btn = QPushButton("중지")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        row.addWidget(self.stop_btn)
        row.addStretch(1)
        root.addLayout(row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("학습 로그가 여기에 표시됩니다…")
        self.log.setMinimumHeight(220)
        root.addWidget(self.log, 1)

    def _append(self, msg: str):
        self.log.append(msg)
        self.status_message.emit(msg[:80])

    def _set_busy(self, busy: bool):
        self.yolo_card.set_busy(busy)
        self.stop_btn.setEnabled(busy)

    def _start(self):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "학습", "이미 학습이 진행 중입니다.")
            return
        epochs, batch = self.yolo_card.params()
        self.log.clear()
        self._append(f"=== YOLO 학습 시작 (epochs={epochs}, batch={batch}, 로컬+허깅페이스 전부) ===")
        self.worker = TrainWorker("yolo", epochs, batch)
        self.worker.log.connect(self._append)
        self.worker.done.connect(self._on_done)
        self.worker.fail.connect(self._on_fail)
        self._set_busy(True)
        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.request_stop()
            self._append("[UI] 중지 요청… (현재 epoch/배치 끝난 뒤 반영)")

    def _on_done(self, path: str):
        self._set_busy(False)
        self._append(f"=== 완료: {path} ===")
        QMessageBox.information(self, "학습 완료", f"가중치 저장:\n{path}")

    def _on_fail(self, err: str):
        self._set_busy(False)
        self._append(f"=== 실패: {err} ===")
        QMessageBox.warning(self, "학습 실패", err)
