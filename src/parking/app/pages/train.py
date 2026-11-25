from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
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
from parking.training.yolo_ft import TEST_ROUND_2_AUGMENTATION, TEST_ROUND_2_HELP

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

    def __init__(
        self,
        kind: str,
        epochs: int,
        batch: int,
        model_name: str,
        folders: list[str],
        *,
        augmentation: dict[str, float] | None = None,
    ):
        super().__init__()
        self.kind = kind
        self.epochs = epochs
        self.batch = batch
        self.model_name = model_name
        self.folders = folders
        self.augmentation = augmentation
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
                augmentation_overrides=self.augmentation,
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
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(8)

        h = QLabel(title)
        h.setStyleSheet("color:#e7eef7;font-size:18px;font-weight:700;")
        layout.addWidget(h)

        self.subtitle = QLabel("YOLO11n-seg")
        self.subtitle.setStyleSheet("color:#2ec4b6;font-size:12px;font-weight:600;")
        layout.addWidget(self.subtitle)

        d = QLabel(detail)
        d.setWordWrap(True)
        d.setStyleSheet("color:#8b9bb0;")
        layout.addWidget(d)

        form = QFormLayout()
        form.setSpacing(6)
        self.model = QComboBox()
        self.reload_models()
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 200)
        self.epochs.setValue(5)
        self.epoch_label = QLabel()
        self.batch = QSpinBox()
        self.batch.setRange(1, 128)
        self.batch.setValue(4)
        form.addRow("시작 모델", self.model)
        form.addRow(self.epoch_label, self.epochs)
        form.addRow("batch", self.batch)
        layout.addLayout(form)

        aug_title = QLabel("증강 (마우스 올리면 설명)")
        aug_title.setStyleSheet("color:#8b9bb0;")
        layout.addWidget(aug_title)
        aug_grid = QGridLayout()
        aug_grid.setHorizontalSpacing(10)
        aug_grid.setVerticalSpacing(4)
        self.aug_spins: dict[str, QDoubleSpinBox] = {}
        spin_ranges = {
            "degrees": (0.0, 45.0, 1),
            "perspective": (0.0, 0.001, 4),
            "scale": (0.0, 1.0, 2),
            "translate": (0.0, 0.5, 2),
            "mosaic": (0.0, 1.0, 2),
        }
        for index, (name, default) in enumerate(TEST_ROUND_2_AUGMENTATION.items()):
            low, high, decimals = spin_ranges[name]
            spin = QDoubleSpinBox()
            spin.setRange(low, high)
            spin.setDecimals(decimals)
            spin.setSingleStep(10 ** (-decimals) if decimals else 1.0)
            spin.setValue(float(default))
            spin.setToolTip(TEST_ROUND_2_HELP[name])
            label = QLabel(name)
            label.setToolTip(TEST_ROUND_2_HELP[name])
            row, col = divmod(index, 3)
            cell = QHBoxLayout()
            cell.addWidget(label)
            cell.addWidget(spin)
            aug_grid.addLayout(cell, row, col)
            self.aug_spins[name] = spin
        layout.addLayout(aug_grid)

        folder_label = QLabel("data 폴더")
        folder_label.setStyleSheet("color:#8b9bb0;")
        layout.addWidget(folder_label)
        self.folders = QListWidget()
        self.folders.setMaximumHeight(110)
        layout.addWidget(self.folders)
        self.reload_folders()

        self.btn = QPushButton("학습 시작")
        self.model.currentIndexChanged.connect(self._sync_model)
        self._sync_model()
        self.btn.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.btn)

    def reload_models(self) -> None:
        selected = self.model.currentData() or "yolo11n-seg.pt"
        self.model.clear()
        for label, weights in YOLO_MODELS:
            self.model.addItem(label, weights)
        for directory in ("yolo_spaces", "yolo_corners"):
            for weights in sorted((MODELS_DIR / directory).glob("*.pt")):
                if weights.is_file():
                    self.model.addItem(f"추가 학습 · {weights.name}", str(weights))
        index = self.model.findData(selected)
        self.model.setCurrentIndex(index if index >= 0 else self.model.findData("yolo11n-seg.pt"))

    def _sync_model(self):
        saved_model = self.model.currentIndex() >= len(YOLO_MODELS)
        self.subtitle.setText(self.model.currentText())
        self.epoch_label.setText("추가 에폭" if saved_model else "에폭")
        self.btn.setText("추가 학습 시작" if saved_model else "학습 시작")

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

    def augmentation_values(self) -> dict[str, float]:
        return {name: float(spin.value()) for name, spin in self.aug_spins.items()}

    def set_busy(self, busy: bool):
        self.btn.setEnabled(not busy)
        self.model.setEnabled(not busy)
        self.epochs.setEnabled(not busy)
        self.batch.setEnabled(not busy)
        for spin in self.aug_spins.values():
            spin.setEnabled(not busy)
        self.folders.setEnabled(not busy)


class TrainPage(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.worker: TrainWorker | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(12)

        title = QLabel("학습")
        title.setStyleSheet("color:#e7eef7;font-size:22px;font-weight:700;")
        root.addWidget(title)

        desc = QLabel("주차칸 YOLO 학습. data 폴더는 여러 개 선택 가능(기본 전체).")
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#8b9bb0;")
        root.addWidget(desc)

        self.yolo_card = ModelCard(
            "YOLO 파인튜닝",
            "주차칸 좌표만 학습합니다. 점유는 나중에 차량 YOLO로 판정합니다.",
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

    def showEvent(self, event):
        if self.worker is None or not self.worker.isRunning():
            self.yolo_card.reload_models()
        super().showEvent(event)

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
        augmentation = self.yolo_card.augmentation_values()
        self.log.clear()
        self._append(
            f"=== YOLO 학습 시작 ({model_name}, epochs={epochs}, batch={batch}, folders={', '.join(folders)}) ==="
        )
        self._append(
            "[UI] 증강: " + ", ".join(f"{name}={value:g}" for name, value in augmentation.items())
        )
        self.worker = TrainWorker(
            "yolo",
            epochs,
            batch,
            model_name,
            folders,
            augmentation=augmentation,
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
        self.yolo_card.reload_models()
        self._set_busy(False)
        self._append(f"=== 완료: {path} ===")
        QMessageBox.information(self, "학습 완료", f"가중치 저장:\n{path}")

    def _on_fail(self, err: str):
        self._set_busy(False)
        self._append(f"=== 실패: {err} ===")
        QMessageBox.warning(self, "학습 실패", err)
