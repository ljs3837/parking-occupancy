"""여러 CCTV 영상을 재생하고, 고정 슬롯 + 차량 YOLO로 점유를 보여 준다."""

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
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from parking.app.paths import OBSERVE_DIR, VIDEO_EXTS
from parking.app.pages.layout_editor import LayoutEditor
from parking.fusion.resolver import SpaceFusion, merge_space_states
from parking.perception.layouts import (
    assign_class_names,
    delete_layout,
    is_layout_confirmed,
    load_layout_for_video,
    save_layout,
    spaces_from_polygons,
    stream_id_from_video,
)
from parking.perception.live import (
    LiveDetector,
    draw_detections,
    draw_occupancy,
    space_polygons,
    summarize,
)
from parking.perception.occupancy import OccupancyFilter, assign_occupancy, summarize_occupancy

DETECTOR = LiveDetector()
# 관제 추론 주기 (프레임). 4화면이 YOLO 1대를 공유하므로 6보다 크게.
DETECT_EVERY_N = 15


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


def capture_layout_from_video(video_path: str) -> tuple[str, Path, int]:
    """레이아웃 확정: 주차칸 1회 탐지 → confirmed 저장 → 슬롯 모델 언로드."""
    if DETECTOR.kind != "occupancy":
        raise RuntimeError("왼쪽에서 주차칸(.pt) 모델을 선택하세요")
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("영상을 열 수 없습니다")
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError("프레임을 읽지 못했습니다")
    DETECTOR.ensure_space_model()
    try:
        boxes = DETECTOR.predict_spaces(frame)
        polygons = space_polygons(frame, boxes)
        stream_id = stream_id_from_video(video_path)
        spaces = assign_class_names(spaces_from_polygons(polygons), stream_id)
    finally:
        DETECTOR.unload_space_model()
    if not spaces:
        raise RuntimeError("주차칸을 찾지 못했습니다. 모델·임계값을 확인하세요")
    try:
        rel = Path(video_path).resolve().relative_to(OBSERVE_DIR.resolve()).as_posix()
    except ValueError:
        rel = Path(video_path).name
    path = save_layout(stream_id, spaces, source_video=rel, confirmed=True)
    return stream_id, path, len(spaces)


class VideoWorker(QThread):
    frame_ready = pyqtSignal(int, QImage, str)
    failed = pyqtSignal(int, str)
    model_ready = pyqtSignal(str)
    occupancy_ready = pyqtSignal(int, dict)

    def __init__(self, index: int, path: str, detect: bool):
        super().__init__()
        self.index = index
        self.path = path
        self.detect = detect
        self._running = True
        self._paused = False
        self._reload = False
        self.force_space = False
        self._detect_now = False
        self.model_sent = False

    def pause(self, paused: bool):
        self._paused = paused

    def set_space_preview(self, on: bool):
        """모델을 바꾸면 확정 레이아웃 대신 그 주차칸 모델로 다시 탐지한다."""
        self.force_space = on
        self.model_sent = False
        self._detect_now = True

    def request_reload(self):
        """확정 레이아웃(칸 이름·모양)이 바뀌면 다음 프레임에 다시 읽는다."""
        self._reload = True

    def stop(self):
        self._running = False

    def run(self):
        capture = cv2.VideoCapture(self.path)
        if not capture.isOpened():
            self.failed.emit(self.index, "영상을 열 수 없습니다")
            return

        # 레이아웃 확정된 영상만 고정 슬롯 + 차량 YOLO. 그 전에는 학습 주차칸 모델을 계속 돌림.
        layout_locked = is_layout_confirmed(self.path)
        _stream_id, spaces = load_layout_for_video(self.path, confirmed_only=True)
        camera_id = f"CAM_{self.index + 1:02d}"
        occ_filter = OccupancyFilter()
        fusion = SpaceFusion()
        vehicle_boxes: list[dict] = []
        space_boxes: list[dict] = []
        space_states: list[dict] = [
            {
                "space_id": s.get("space_id"),
                "state": "unknown",
                "confidence": None,
                "points": s.get("points") or [],
            }
            for s in spaces
        ]
        frame_index = 0
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
            height, width = frame.shape[:2]
            if self._reload:
                self._reload = False
                layout_locked = is_layout_confirmed(self.path)
                _stream_id, spaces = load_layout_for_video(self.path, confirmed_only=True)
                space_states = [
                    {
                        "space_id": s.get("space_id"),
                        "state": "unknown",
                        "confidence": None,
                        "points": s.get("points") or [],
                    }
                    for s in spaces
                ]
                occ_filter = OccupancyFilter()
                fusion = SpaceFusion()

            preview = self.force_space and DETECTOR.kind == "occupancy"
            if self.detect and preview and space_boxes:
                view = draw_detections(frame, space_boxes)
            elif self.detect and not preview and layout_locked and spaces:
                view = draw_occupancy(frame, space_states, vehicle_boxes)
            elif self.detect and space_boxes:
                view = draw_detections(frame, space_boxes)
            elif self.detect and vehicle_boxes:
                view = draw_detections(frame, vehicle_boxes)
            else:
                view = frame

            position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
            clock = f"{format_clock(position)} / {format_clock(total_seconds)}"
            if self.detect and preview:
                detail = f"모델 재탐지 · {DETECTOR.label}"
                if space_boxes:
                    detail += f" · {summarize(space_boxes)}"
            elif self.detect and layout_locked and spaces:
                occupied, vacant, _unknown = summarize_occupancy(space_states)
                detail = f"확정 · 칸 {len(spaces)} · 점유 {occupied} · 여유 {vacant}"
                if vehicle_boxes:
                    detail += f" · {summarize(vehicle_boxes)}"
            elif self.detect and space_boxes:
                detail = f"주차칸 탐색 중 · {summarize(space_boxes)}"
            elif self.detect and not layout_locked:
                detail = "주차칸 모델로 탐색 중 · 레이아웃 확정 전"
            else:
                detail = ""
            self.frame_ready.emit(self.index, to_qimage(view), f"{clock}|{detail}")

            due = frame_index % DETECT_EVERY_N == 1 or self._detect_now
            if self._detect_now:
                self._detect_now = False
            if self.detect and due:
                try:
                    preview = self.force_space and DETECTOR.kind == "occupancy"
                    if preview or not layout_locked:
                        if DETECTOR.kind != "occupancy":
                            space_boxes = []
                            if not self.model_sent:
                                self.failed.emit(self.index, "주차칸(.pt) 모델을 왼쪽에서 선택하세요")
                                self.model_sent = True
                        else:
                            space_boxes = DETECTOR.predict_spaces(frame)
                            if not self.model_sent:
                                self.model_ready.emit(f"{DETECTOR.label} · 주차칸 탐색")
                                self.model_sent = True
                    elif layout_locked:
                        # 확정 후: 기본 차량 YOLO + 고정 슬롯 점유
                        vehicle_boxes = DETECTOR.predict(frame)
                        space_boxes = []
                        if not self.model_sent:
                            self.model_ready.emit("차량 YOLO · 레이아웃 확정")
                            self.model_sent = True
                        if spaces:
                            raw = assign_occupancy(spaces, vehicle_boxes, width=width, height=height)
                            filtered = occ_filter.update(raw)
                            space_states = fusion.update(camera_id, filtered)
                            occupied, vacant, _unknown = summarize_occupancy(space_states)
                            self.occupancy_ready.emit(
                                self.index,
                                {
                                    "states": {
                                        str(item.get("space_id")): item.get("state")
                                        for item in space_states
                                        if item.get("space_id")
                                    },
                                    "free": vacant,
                                    "total": len(space_states),
                                    "occupied": occupied,
                                },
                            )
                except Exception as exc:
                    vehicle_boxes = []
                    space_boxes = []
                    self.failed.emit(self.index, f"인식 실패: {exc}")
                    self.detect = False
            remain = frame_interval - (time.perf_counter() - started)
            if remain > 0:
                self.msleep(int(remain * 1000))
        capture.release()


class CameraTile(QFrame):
    model_ready = pyqtSignal(str)
    occupancy_ready = pyqtSignal(int, dict)
    picked = pyqtSignal(int)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.path: str | None = None
        self.worker: VideoWorker | None = None
        self.last_image: QImage | None = None
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

        self.free_label = QLabel("여유 —")
        self.free_label.setStyleSheet("color:#8b9bb0;font-size:11px;")
        layout.addWidget(self.free_label)

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
        stream_id, spaces = load_layout_for_video(path, confirmed_only=True)
        if spaces:
            self.free_label.setText(f"확정됨 {stream_id} · 칸 {len(spaces)}")
        else:
            self.free_label.setText("미확정 · 학습 모델로 주차칸 탐색")

    def play(self, detect: bool = True, preview_spaces: bool = False):
        if not self.path:
            self.clock.setText("영상을 선택")
            return
        if self.worker and self.worker.isRunning():
            self.worker.detect = detect
            self.worker.set_space_preview(preview_spaces)
            self.worker.pause(False)
            return
        self.worker = VideoWorker(self.index, self.path, detect)
        self.worker.set_space_preview(preview_spaces)
        self.worker.frame_ready.connect(self._show_frame)
        self.worker.failed.connect(self._show_error)
        self.worker.model_ready.connect(self.model_ready.emit)
        self.worker.occupancy_ready.connect(self._on_occupancy)
        self.worker.occupancy_ready.connect(self.occupancy_ready.emit)
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
        self.last_image = image
        pixmap = QPixmap.fromImage(image)
        self.screen.setPixmap(pixmap.scaled(
            self.screen.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
        clock, _, detail = summary.partition("|")
        self.clock.setText(clock)
        self.clock.setToolTip(detail)

    def _on_occupancy(self, index: int, counts: dict):
        if index != self.index:
            return
        free = counts.get("free", counts.get("vacant", 0))
        total = counts.get("total", 0)
        occupied = counts.get("occupied", 0)
        self.free_label.setText(f"여유 {free}/{total} · 점유 {occupied}")

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
        self.preview_spaces = False
        self._preview_locked: set[str] = set()

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
        self.layout_btn = QPushButton("레이아웃 적용")
        self.layout_btn.setToolTip(
            "지금 화면의 주차칸을 이때만 저장합니다. 적용 전에는 레이아웃이 없습니다."
        )
        self.layout_btn.clicked.connect(self.confirm_layout)
        bar.addWidget(apply_btn)
        bar.addWidget(self.play_btn)
        bar.addWidget(self.detect_box)
        bar.addWidget(self.layout_btn)
        self.total_free = QLabel("전체 여유 —")
        self.total_free.setStyleSheet("color:#2ec4b6;font-weight:600;")
        bar.addWidget(self.total_free)
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
        self.video_hint = QLabel(
            f"확정 전: 학습 주차칸 모델 / 확정 후: 고정 슬롯+차량 YOLO · {DETECT_EVERY_N}프레임"
        )
        self.video_hint.setWordWrap(True)
        side_layout.addWidget(self.video_hint)
        side_layout.addWidget(QLabel("클래스명"))
        self.space_list = QListWidget()
        self.space_list.currentItemChanged.connect(self._fill_space_name)
        side_layout.addWidget(self.space_list, 1)
        self.space_name = QLineEdit()
        self.space_name.setPlaceholderText("클래스명 · 다른 캠과 같으면 같은 칸")
        side_layout.addWidget(self.space_name)
        name_btn = QPushButton("클래스명 적용")
        name_btn.clicked.connect(self.rename_selected_space)
        delete_btn = QPushButton("폴리곤 삭제")
        delete_btn.clicked.connect(self.delete_selected_space)
        edit_btn = QPushButton("폴리곤 수정")
        edit_btn.clicked.connect(self.edit_layout_polygons)
        side_layout.addWidget(name_btn)
        side_layout.addWidget(delete_btn)
        side_layout.addWidget(edit_btn)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 8)
        splitter.setStretchFactor(1, 2)
        side.setMaximumWidth(280)
        root.addWidget(splitter, 1)
        self._tile_counts: dict[int, dict] = {}
        self._tile_states: dict[int, dict[str, str]] = {}
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
        self._tile_counts = {}
        self._tile_states = {}
        for index in range(count):
            tile = CameraTile(index)
            if index < len(paths) and paths[index]:
                tile.set_video(paths[index])
            tile.picked.connect(self.select_tile)
            tile.occupancy_ready.connect(self._on_tile_occupancy)
            row, col = divmod(index, columns)
            self.grid.addWidget(tile, row, col)
            self.grid.setRowStretch(row, 1)
            self.grid.setColumnStretch(col, 1)
            self.tiles.append(tile)
        self.select_tile(min(self.selected_index, count - 1))
        self.status_message.emit(f"CCTV {count}화면")

    def on_model_selected(self, model: dict):
        """왼쪽 모델 목록을 바꾸면, 확정된 화면이라도 그 주차칸 모델로 다시 탐지한다."""
        self.preview_spaces = model.get("kind") == "occupancy"
        self._preview_locked.clear()
        if self.preview_spaces:
            try:
                DETECTOR.ensure_space_model()
            except Exception as exc:
                self.status_message.emit(f"주차칸 모델 로드 실패: {exc}")
                return
        for tile in self.tiles:
            if tile.worker and tile.worker.isRunning():
                tile.worker.set_space_preview(self.preview_spaces)
        if self.preview_spaces:
            self.status_message.emit(f"{model.get('label')} 로 주차칸 다시 탐지")

    def refresh_videos(self):
        self.video_list.clear()
        videos = list_videos()
        for path in videos:
            item = QListWidgetItem(path.relative_to(OBSERVE_DIR).as_posix())
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.video_list.addItem(item)
        if videos:
            self.video_hint.setText(
                f"{len(videos)}개 · 확정 전 주차칸 모델 계속 / 확정 후 점유 · {DETECT_EVERY_N}프레임"
            )
        else:
            self.video_hint.setText(f"영상이 없습니다. {OBSERVE_DIR} 에 넣어 주세요.")

    def select_tile(self, index: int):
        self.selected_index = index
        for tile in self.tiles:
            tile.set_active(tile.index == index)
        self._reload_space_list()

    def _selected_tile(self) -> CameraTile | None:
        if not self.tiles:
            return None
        return self.tiles[min(self.selected_index, len(self.tiles) - 1)]

    def _reload_space_list(self):
        self.space_list.clear()
        tile = self._selected_tile()
        if tile is None or not tile.path or not is_layout_confirmed(tile.path):
            self.space_name.clear()
            return
        _stream_id, spaces = load_layout_for_video(tile.path, confirmed_only=True)
        for index, space in enumerate(spaces):
            item = QListWidgetItem(str(space.get("space_id") or f"S{index + 1:03d}"))
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.space_list.addItem(item)
        if self.space_list.count():
            self.space_list.setCurrentRow(0)

    def _fill_space_name(self, current, _previous):
        self.space_name.setText(current.text() if current is not None else "")

    def rename_selected_space(self):
        tile = self._selected_tile()
        item = self.space_list.currentItem()
        if tile is None or not tile.path or item is None:
            self.status_message.emit("확정된 화면의 칸을 선택하세요")
            return
        name = self.space_name.text().strip()
        if not name:
            return
        _stream_id, spaces = load_layout_for_video(tile.path, confirmed_only=True)
        index = int(item.data(Qt.ItemDataRole.UserRole))
        if index < 0 or index >= len(spaces):
            return
        spaces[index]["space_id"] = name
        was_playing = bool(tile.worker and tile.worker.isRunning() and not tile.worker._paused)
        detect = tile.worker.detect if tile.worker else self.detect_box.isChecked()
        stream_id = stream_id_from_video(tile.path)
        try:
            rel = Path(tile.path).resolve().relative_to(OBSERVE_DIR.resolve()).as_posix()
        except ValueError:
            rel = Path(tile.path).name
        save_layout(stream_id, spaces, source_video=rel, confirmed=True)
        if tile.worker and tile.worker.isRunning():
            tile.worker.request_reload()
        else:
            tile.set_video(tile.path)
            if was_playing:
                tile.play(detect)
        self._reload_space_list()
        self.status_message.emit(f"클래스명 {name} · 같은 클래스명은 전체 여유에서 한 칸")

    def delete_selected_space(self):
        tile = self._selected_tile()
        item = self.space_list.currentItem()
        if tile is None or not tile.path or item is None:
            self.status_message.emit("삭제할 폴리곤을 선택하세요")
            return
        _stream_id, spaces = load_layout_for_video(tile.path, confirmed_only=True)
        index = int(item.data(Qt.ItemDataRole.UserRole))
        if index < 0 or index >= len(spaces):
            return
        removed = str(spaces[index].get("space_id") or index + 1)
        del spaces[index]
        stream_id = stream_id_from_video(tile.path)
        if spaces:
            try:
                rel = Path(tile.path).resolve().relative_to(OBSERVE_DIR.resolve()).as_posix()
            except ValueError:
                rel = Path(tile.path).name
            save_layout(stream_id, spaces, source_video=rel, confirmed=True)
            if tile.worker and tile.worker.isRunning():
                tile.worker.request_reload()
        else:
            delete_layout(stream_id)
            self._preview_locked.discard(tile.path)
            self._tile_states.pop(tile.index, None)
            self._tile_counts.pop(tile.index, None)
            if tile.worker and tile.worker.isRunning():
                tile.worker.set_space_preview(self.preview_spaces)
                tile.worker.request_reload()
            else:
                tile.set_video(tile.path)
            merged = merge_space_states(self._tile_states)
            self.total_free.setText(
                f"전체 여유 {merged['free']}/{merged['total']} · 점유 {merged['occupied']}"
            )
        self._reload_space_list()
        self.status_message.emit(f"폴리곤 {removed} 삭제")

    def edit_layout_polygons(self):
        tile = self._selected_tile()
        if tile is None or not tile.path:
            self.status_message.emit("먼저 화면을 선택하세요")
            return
        if not is_layout_confirmed(tile.path):
            QMessageBox.information(self, "폴리곤 수정", "레이아웃 확정 후에 칸을 고칠 수 있습니다.")
            return
        image = tile.last_image
        if image is None or image.isNull():
            capture = cv2.VideoCapture(tile.path)
            ok, frame = capture.read()
            capture.release()
            if not ok or frame is None:
                self.status_message.emit("프레임을 읽지 못했습니다")
                return
            image = to_qimage(frame)
        _stream_id, spaces = load_layout_for_video(tile.path, confirmed_only=True)
        editor = LayoutEditor(image, spaces, self)
        if editor.exec() != editor.DialogCode.Accepted:
            return
        updated = editor.result_spaces()
        if not updated:
            self.status_message.emit("칸이 없습니다")
            return
        stream_id = stream_id_from_video(tile.path)
        try:
            rel = Path(tile.path).resolve().relative_to(OBSERVE_DIR.resolve()).as_posix()
        except ValueError:
            rel = Path(tile.path).name
        save_layout(stream_id, updated, source_video=rel, confirmed=True)
        if tile.worker and tile.worker.isRunning():
            tile.worker.request_reload()
        self._reload_space_list()
        self.status_message.emit(f"칸 {len(updated)}개 저장 · {stream_id}")

    def assign_selected_video(self, item: QListWidgetItem):
        if not self.tiles:
            return
        tile = self.tiles[min(self.selected_index, len(self.tiles) - 1)]
        tile.set_video(item.data(Qt.ItemDataRole.UserRole))
        self.status_message.emit(f"CAM {tile.index + 1:02d} ← {item.text()}")

    def confirm_layout(self):
        if not self.tiles:
            return
        tile = self.tiles[min(self.selected_index, len(self.tiles) - 1)]
        if not tile.path:
            self.status_message.emit("먼저 화면에 영상을 넣으세요")
            return
        if DETECTOR.kind != "occupancy":
            QMessageBox.information(
                self,
                "레이아웃 확정",
                "왼쪽에서 주차칸(.pt) 모델을 선택한 뒤 다시 눌러 주세요.",
            )
            return
        try:
            was_playing = self._playing
            if was_playing:
                self.pause_all()
            _stream_id, path, count = capture_layout_from_video(tile.path)
            tile.set_video(tile.path)
            self._preview_locked.add(tile.path)
            if tile.worker and tile.worker.isRunning():
                tile.worker.set_space_preview(False)
                tile.worker.request_reload()
            self.status_message.emit(f"레이아웃 확정 {path.name} · 칸 {count} · 이후 차량 YOLO 점유")
            self._reload_space_list()
            if was_playing:
                self.play_all()
        except Exception as exc:
            QMessageBox.warning(self, "레이아웃 확정 실패", str(exc))
            self.status_message.emit(f"레이아웃 실패: {exc}")

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
        any_locked = any(is_layout_confirmed(tile.path) for tile in ready if tile.path)
        any_unlocked = any(not is_layout_confirmed(tile.path) for tile in ready if tile.path)
        if detect and any_unlocked and DETECTOR.kind != "occupancy":
            self.status_message.emit("미확정 화면은 왼쪽 주차칸(.pt) 모델이 필요합니다")
        if detect:
            try:
                if any_unlocked:
                    if DETECTOR.kind == "occupancy":
                        DETECTOR.ensure_space_model()
                if any_locked:
                    DETECTOR.ensure()
                    if not any_unlocked:
                        DETECTOR.unload_space_model()
            except Exception as exc:
                self.status_message.emit(f"모델 로드 실패: {exc}")
                detect = False
        for tile in ready:
            preview = self.preview_spaces and tile.path not in self._preview_locked
            tile.play(detect, preview_spaces=preview)
        self._playing = True
        self.play_btn.setText("일시정지")
        if any_unlocked and not any_locked:
            self.status_message.emit(
                f"{len(ready)}개 재생 · 학습 주차칸 모델 탐색 / {DETECT_EVERY_N}프레임 · 확정 버튼으로 고정"
            )
        elif any_locked and not any_unlocked:
            self.status_message.emit(
                f"{len(ready)}개 재생 · 확정 레이아웃 + 차량 YOLO / {DETECT_EVERY_N}프레임"
            )
        else:
            self.status_message.emit(
                f"{len(ready)}개 재생 · 미확정=주차칸모델 / 확정=차량YOLO · {DETECT_EVERY_N}프레임"
            )

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

    def _on_tile_occupancy(self, index: int, counts: dict):
        self._tile_counts[index] = counts
        if "states" in counts:
            self._tile_states[index] = dict(counts.get("states") or {})
        merged = merge_space_states(self._tile_states)
        self.total_free.setText(
            f"전체 여유 {merged['free']}/{merged['total']} · 점유 {merged['occupied']}"
        )
