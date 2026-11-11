"""여러 CCTV 영상을 재생하고 인식 결과를 겹쳐 보여 준다."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from parking.app.paths import OBSERVE_DIR, VIDEO_EXTS
from parking.perception.live import LiveDetector, draw_detections, summarize

DETECTOR = LiveDetector()


def grid_columns(count: int) -> int:
    if count <= 1:
        return 1
    if count <= 4:
        return 2
    return 3


def default_videos() -> list[Path]:
    numbered = [OBSERVE_DIR / f"{index}.mp4" for index in range(1, 5)]
    if all(path.is_file() for path in numbered):
        return numbered
    merged = [OBSERVE_DIR / "merged" / f"{index}.mp4" for index in range(1, 5)]
    if all(path.is_file() for path in merged):
        return merged
    return list_videos()[:4]


def list_videos() -> list[Path]:
    OBSERVE_DIR.mkdir(parents=True, exist_ok=True)
    videos = [
        path
        for path in OBSERVE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]
    videos.sort()
    return videos


def format_clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remain = divmod(seconds, 3600)
    minutes, secs = divmod(remain, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def to_qimage(frame_bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
    return image.copy()


class VideoWorker(QThread):
    frame_ready = pyqtSignal(int, QImage, str)
    failed = pyqtSignal(int, str)
    model_ready = pyqtSignal(str)

    def __init__(self, index: int, path: str, detect: bool):
        super().__init__()
        self.index = index
        self.path = path
        self.detect = detect
        self._running = True
        self._paused = False

    def pause(self, paused: bool):
        self._paused = paused

    def stop(self):
        self._running = False

    def run(self):
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            self.failed.emit(self.index, "영상을 열 수 없습니다")
            return
        boxes: list[dict] = []
        frame_index = 0
        model_sent = False
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        if fps < 1:
            fps = 30
        frame_interval = 1.0 / fps
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        total_seconds = frame_count / fps if fps else 0
        while self._running:
            if self._paused:
                self.msleep(40)
                continue
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            frame_index += 1
            view = draw_detections(frame, boxes) if boxes else frame
            position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            clock = f"{format_clock(position)} / {format_clock(total_seconds)}"
            detail = summarize(boxes) if self.detect and boxes else ""
            self.frame_ready.emit(self.index, to_qimage(view), f"{clock}|{detail}")
            if self.detect and frame_index % 6 == 1:
                try:
                    boxes = DETECTOR.predict(frame)
                    if not model_sent:
                        self.model_ready.emit(DETECTOR.label)
                        model_sent = True
                except Exception as exc:
                    boxes = []
                    self.failed.emit(self.index, f"인식 실패: {exc}")
                    self.detect = False
            remain = frame_interval - (time.perf_counter() - started)
            if remain > 0:
                self.msleep(int(remain * 1000))
        capture.release()


class CameraTile(QFrame):
    model_ready = pyqtSignal(str)
    picked = pyqtSignal(int)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.path: str | None = None
        self.worker: VideoWorker | None = None
        self.setObjectName("heroCard")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        self.title = QLabel(f"CAM {index + 1:02d}")
        self.title.setStyleSheet("color:#9aabbd;font-size:11px;font-weight:600;")
        self.clock = QLabel("--:-- / --:--")
        self.clock.setStyleSheet("color:#e7eef7;font-size:12px;")
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.clock)
        layout.addLayout(header)

        self.screen = QLabel("오른쪽 목록에서 영상을 선택하세요")
        self.screen.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.screen.setMinimumSize(420, 260)
        self.screen.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.screen.setStyleSheet("background:#070b10;border:1px solid #2a3545;border-radius:8px;color:#8b9bb0;")
        layout.addWidget(self.screen, 1)

    def mousePressEvent(self, event):
        self.picked.emit(self.index)
        super().mousePressEvent(event)

    def set_active(self, active: bool):
        color = "#2ec4b6" if active else "#2a3545"
        self.setStyleSheet(f"QFrame#heroCard {{ border: 1px solid {color}; }}")

    def set_video(self, path: str):
        self.stop()
        self.path = path
        self.clock.setText("--:-- / --:--")
        self.screen.setText(Path(path).name)

    def play(self, detect: bool = True):
        if not self.path:
            self.clock.setText("영상을 선택")
            return
        if self.worker and self.worker.isRunning():
            self.worker.detect = detect
            self.worker.pause(False)
            return
        self.worker = VideoWorker(self.index, self.path, detect)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.failed.connect(self._show_error)
        self.worker.model_ready.connect(self.model_ready.emit)
        self.worker.start()

    def pause(self):
        if self.worker:
            self.worker.pause(True)

    def stop(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait(1500)
            self.worker = None

    def _show_frame(self, index: int, image: QImage, summary: str):
        if index != self.index:
            return
        pixmap = QPixmap.fromImage(image)
        self.screen.setPixmap(pixmap.scaled(
            self.screen.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        clock, _, detail = summary.partition("|")
        self.clock.setText(clock)
        self.clock.setToolTip(detail)

    def _show_error(self, index: int, message: str):
        if index == self.index:
            self.clock.setText(message)


class MonitorPage(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tiles: list[CameraTile] = []
        self.selected_index = 0
        self._playing = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("CCTV 수"))
        self.count = QSpinBox()
        self.count.setRange(1, 9)
        self.count.setValue(4)
        bar.addWidget(self.count)
        apply_btn = QPushButton("화면 적용")
        apply_btn.clicked.connect(self.apply_count)
        self.play_btn = QPushButton("재생")
        self.play_btn.clicked.connect(self.toggle_all)
        self.detect_box = QCheckBox("인식 표시")
        self.detect_box.setChecked(True)
        bar.addWidget(apply_btn)
        bar.addWidget(self.play_btn)
        bar.addWidget(self.detect_box)
        bar.addStretch(1)
        root.addLayout(bar)

        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setSpacing(10)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.grid_host)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(8)
        side_layout.addWidget(QLabel("관제 영상"))
        refresh_btn = QPushButton("목록 새로고침")
        refresh_btn.clicked.connect(self.refresh_videos)
        side_layout.addWidget(refresh_btn)
        self.video_list = QListWidget()
        self.video_list.itemClicked.connect(self.assign_selected_video)
        side_layout.addWidget(self.video_list, 1)
        self.video_hint = QLabel("목록을 누르면 선택한 화면에 들어갑니다.")
        self.video_hint.setWordWrap(True)
        side_layout.addWidget(self.video_hint)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 2)
        side.setMaximumWidth(280)
        root.addWidget(splitter, 1)
        self.apply_count()
        self.refresh_videos()

    def apply_count(self):
        paths = [tile.path for tile in self.tiles]
        if not any(paths):
            paths = [str(path) for path in default_videos()]
        self.stop_all()
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        count = self.count.value()
        columns = grid_columns(count)
        self.tiles = []
        for index in range(count):
            tile = CameraTile(index)
            if index < len(paths) and paths[index]:
                tile.set_video(paths[index])
            tile.picked.connect(self.select_tile)
            row, col = divmod(index, columns)
            self.grid.addWidget(tile, row, col)
            self.grid.setRowStretch(row, 1)
            self.grid.setColumnStretch(col, 1)
            self.tiles.append(tile)
        self.select_tile(min(self.selected_index, count - 1))
        self.status_message.emit(f"CCTV {count}화면")

    def refresh_videos(self):
        self.video_list.clear()
        videos = list_videos()
        for path in videos:
            item = QListWidgetItem(path.relative_to(OBSERVE_DIR).as_posix())
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.video_list.addItem(item)
        if videos:
            self.video_hint.setText(f"{len(videos)}개 · 누르면 선택한 화면으로 들어갑니다.")
        else:
            self.video_hint.setText(f"영상이 없습니다. {OBSERVE_DIR} 에 넣어 주세요.")

    def select_tile(self, index: int):
        self.selected_index = index
        for tile in self.tiles:
            tile.set_active(tile.index == index)

    def assign_selected_video(self, item: QListWidgetItem):
        if not self.tiles:
            return
        tile = self.tiles[min(self.selected_index, len(self.tiles) - 1)]
        tile.set_video(item.data(Qt.ItemDataRole.UserRole))
        self.status_message.emit(f"CAM {tile.index + 1:02d} ← {item.text()}")

    def toggle_all(self):
        if self._playing:
            self.pause_all()
        else:
            self.play_all()

    def play_all(self):
        detect = self.detect_box.isChecked()
        ready = [tile for tile in self.tiles if tile.path]
        if not ready:
            self.status_message.emit("오른쪽 목록에서 영상을 먼저 넣으세요")
            return
        if detect:
            try:
                DETECTOR.ensure()
            except Exception as exc:
                self.status_message.emit(f"모델 로드 실패: {exc}")
                detect = False
        for tile in ready:
            tile.play(detect)
        self._playing = True
        self.play_btn.setText("일시정지")
        self.status_message.emit(f"{len(ready)}개 화면 재생")

    def pause_all(self):
        for tile in self.tiles:
            tile.pause()
        self._playing = False
        self.play_btn.setText("재생")
        self.status_message.emit("일시정지")

    def stop_all(self):
        for tile in self.tiles:
            tile.stop()
        self._playing = False
        if hasattr(self, "play_btn"):
            self.play_btn.setText("재생")

