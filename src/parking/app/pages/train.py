from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from parking.training.prepare import MODELS_DIR, list_data_folders
from parking.training.yolo_ft import TEST_ROUND_2_AUGMENTATION

# 주차칸 탐지/분할/꼭짓점 파인튜닝에 쓰는 Ultralytics 가중치.
YOLO_MODELS = (
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8n-seg", "yolov8n-seg.pt"),
    ("YOLOv8s-seg", "yolov8s-seg.pt"),
    ("YOLOv8m-seg", "yolov8m-seg.pt"),
    ("YOLOv8l-seg", "yolov8l-seg.pt"),
    ("YOLO11n", "yolo11n.pt"),
    ("YOLO11n-seg", "yolo11n-seg.pt"),
    ("YOLO11s-seg", "yolo11s-seg.pt"),
    ("YOLO11m-seg", "yolo11m-seg.pt"),
    ("YOLO11l-seg", "yolo11l-seg.pt"),
    ("YOLO11s-pose", "yolo11s-pose.pt"),
)


class TrainWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(str)
    fail = pyqtSignal(str)

    def __init__(self, kind: str, epochs: int, batch: int, model_name: str, folders: list[str], *, test_round_2: bool = False):
        super().__init__()
        self.kind = kind
        self.epochs = epochs
        self.batch = batch
        self.model_name = model_name
        self.folders = folders
        self.test_round_2 = test_round_2
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
                model_name=self.model_name,
                folders=set(self.folders),
                test_round_2=self.test_round_2,
                log=log,
                should_stop=self._should_stop,
            )
            self.done.emit(str(path))
        except Exception as e:
            self.fail.emit(str(e))


class ModelCard(QFrame):
    start_clicked = pyqtSignal()

    def __init__(self, title: str, detail: str, parent=None):
        super().__init__(parent)
        self.setObjectName("heroCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        h = QLabel(title)
        h.setStyleSheet("color:#e7eef7;font-size:18px;font-weight:700;")
        layout.addWidget(h)

        self.subtitle = QLabel(YOLO_MODELS[0][0])
        self.subtitle.setStyleSheet("color:#2ec4b6;font-size:12px;font-weight:600;")
        layout.addWidget(self.subtitle)

        d = QLabel(detail)
        d.setWordWrap(True)
        d.setStyleSheet("color:#8b9bb0;")
        layout.addWidget(d)

        form = QFormLayout()
        self.model = QComboBox()
        for label, weights in YOLO_MODELS:
            self.model.addItem(label, weights)
        for weights in sorted((MODELS_DIR / "yolo_corners").glob("*.pt")):
            self.model.addItem(f"Pose · {weights.stem}", str(weights))
        self.model.setCurrentIndex(self.model.findData("yolo11s-seg.pt"))
        self.subtitle.setText(self.model.currentText())
        self.model.currentIndexChanged.connect(self._sync_subtitle)
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 200)
        self.epochs.setValue(5)
        self.batch = QSpinBox()
        self.batch.setRange(1, 128)
        self.batch.setValue(4)
        form.addRow("YOLO", self.model)
        form.addRow("epochs", self.epochs)
        form.addRow("batch", self.batch)
        self.test_preset = QComboBox()
        self.test_preset.addItem("기존 설정", False)
        self.test_preset.addItem("2차 테스트", True)
        self.test_preset.setCurrentIndex(1)
        form.addRow("테스트", self.test_preset)
        self.test_values = QLabel(", ".join(
            f"{name}={value:g}" for name, value in TEST_ROUND_2_AUGMENTATION.items()
        ))
        self.test_values.setWordWrap(True)
        form.addRow(self.test_values)
        self.test_preset.currentIndexChanged.connect(
            lambda: self.test_values.setVisible(bool(self.test_preset.currentData()))
        )
        layout.addLayout(form)

        folder_label = QLabel("data 폴더")
        folder_label.setStyleSheet("color:#8b9bb0;")
        layout.addWidget(folder_label)
        self.folders = QListWidget()
        self.folders.setMaximumHeight(140)
        layout.addWidget(self.folders)
        self.reload_folders()

        self.btn = QPushButton("학습 시작")
        self.btn.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.btn)
        layout.addStretch(1)

    def _sync_subtitle(self):
        self.subtitle.setText(self.model.currentText())

    def reload_folders(self) -> None:
        previous = {
            self.folders.item(index).text(): self.folders.item(index).checkState()
            for index in range(self.folders.count())
        }
        self.folders.clear()
        for name in list_data_folders():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(previous.get(name, Qt.CheckState.Checked))
            self.folders.addItem(item)

    def selected_folders(self) -> list[str]:
        names = []
        for index in range(self.folders.count()):
            item = self.folders.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                names.append(item.text())
        return names

    def params(self) -> tuple[int, int, str]:
        return self.epochs.value(), self.batch.value(), str(self.model.currentData())

    def set_busy(self, busy: bool):
        self.btn.setEnabled(not busy)
        self.model.setEnabled(not busy)
        self.epochs.setEnabled(not busy)
        self.batch.setEnabled(not busy)
        self.test_preset.setEnabled(not busy)
        self.folders.setEnabled(not busy)


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
            "주차칸 YOLO를 학습합니다. data 폴더는 여러 개 고를 수 있고, 기본은 전체입니다."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#8b9bb0;")
        root.addWidget(desc)

        self.yolo_card = ModelCard(
            "YOLO 파인튜닝",
            "주차칸 좌표만으로 면을 찾습니다. 차 있음/없음은 학습에 쓰지 않고, 나중에 차량 YOLO로 판정합니다.",
        )
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
        epochs, batch, model_name = self.yolo_card.params()
        folders = self.yolo_card.selected_folders()
        if not folders:
            QMessageBox.information(self, "학습", "사용할 data 폴더를 하나 이상 선택하세요.")
            return
        self.log.clear()
        self._append(
            f"=== YOLO 학습 시작 ({model_name}, epochs={epochs}, batch={batch}, folders={', '.join(folders)}) ==="
        )
        self.worker = TrainWorker(
            "yolo", epochs, batch, model_name, folders,
            test_round_2=bool(self.yolo_card.test_preset.currentData()),
        )
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
