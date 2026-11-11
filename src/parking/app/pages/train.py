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

    def __init__(self, kind: str, epochs: int, batch: int, max_images: int):
        super().__init__()
        self.kind = kind
        self.epochs = epochs
        self.batch = batch
        self.max_images = max_images
        self._stop = False

    def request_stop(self):
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop

    def run(self):
        try:
            def log(msg: str):
                self.log.emit(str(msg))

            if self.kind == "cnn":
                from parking.training.cnn import train_cnn

                path = train_cnn(
                    epochs=self.epochs,
                    batch=self.batch,
                    max_images=self.max_images,
                    log=log,
                    should_stop=self._should_stop,
                )
            elif self.kind == "yolo":
                from parking.training.yolo_ft import train_yolo

                path = train_yolo(
                    epochs=self.epochs,
                    batch=self.batch,
                    max_images=self.max_images,
                    log=log,
                    should_stop=self._should_stop,
                )
            else:
                raise ValueError(self.kind)
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
        self.max_images = QSpinBox()
        self.max_images.setRange(10, 5000)
        self.max_images.setValue(80)
        form.addRow("epochs", self.epochs)
        form.addRow("batch", self.batch)
        form.addRow("max images", self.max_images)
        layout.addLayout(form)

        self.btn = QPushButton("학습 시작")
        self.btn.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.btn)
        layout.addStretch(1)

    def params(self) -> tuple[int, int, int]:
        return self.epochs.value(), self.batch.value(), self.max_images.value()

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
            "주차칸 라벨(칸 좌표 + 점유여부)로 두 가지 모델을 학습합니다. "
            "우리 색인(`data/annotations`)과 PKLot `samples.json`을 함께 사용합니다."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#8b9bb0;")
        root.addWidget(desc)

        cards = QHBoxLayout()
        cards.setSpacing(14)

        self.cnn_card = ModelCard(
            "CNN 분류",
            "OccupancyCNN",
            "이미 잘라 둔 칸이 빈칸인지 점유인지만 분류합니다. 칸 위치는 찾지 않습니다.",
        )
        self.cnn_card.batch.setValue(32)
        self.cnn_card.start_clicked.connect(lambda: self._start("cnn"))

        self.yolo_card = ModelCard(
            "YOLO 파인튜닝",
            "YOLOv8n-seg",
            "주차칸 폴리곤 그대로 empty / occupied 영역을 찾습니다. 대각선 칸도 사각형으로 누르지 않습니다.",
        )
        self.yolo_card.batch.setValue(8)
        self.yolo_card.start_clicked.connect(lambda: self._start("yolo"))

        cards.addWidget(self.cnn_card)
        cards.addWidget(self.yolo_card)
        root.addLayout(cards)

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
        self.cnn_card.set_busy(busy)
        self.yolo_card.set_busy(busy)
        self.stop_btn.setEnabled(busy)

    def _start(self, kind: str):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "학습", "이미 학습이 진행 중입니다.")
            return
        card = self.cnn_card if kind == "cnn" else self.yolo_card
        epochs, batch, max_images = card.params()
        self.log.clear()
        self._append(f"=== {kind.upper()} 학습 시작 (epochs={epochs}, batch={batch}, max_images={max_images}) ===")
        self.worker = TrainWorker(kind, epochs, batch, max_images)
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
