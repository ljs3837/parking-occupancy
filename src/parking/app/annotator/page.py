from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPushButton, QSplitter, QToolBar,
    QVBoxLayout, QWidget,
)

from parking.app.annotator.canvas import Canvas
from parking.app.annotator.io import annot_path, list_images, load_annotation, occupancy_of, save_annotation
from parking.app.paths import ANNOT_DIR, DATA_DIR


class AnnotatorPage(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        ANNOT_DIR.mkdir(parents=True, exist_ok=True)
        self.all_files = list_images()
        self.current_path: str | None = None
        self.canvas = Canvas()
        self.canvas.changed.connect(self._on_canvas_changed)
        self.canvas.selection_changed.connect(self._refresh_inspector)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.mode_box = QComboBox()
        self.mode_box.addItems(["그리기 (D)", "선택/이동 (V)"])
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.toolbar.addWidget(self.mode_box)
        self.target_box = QComboBox()
        self.target_box.addItems(["주차면", "가림 영역"])
        self.target_box.currentIndexChanged.connect(self._target_changed)
        self.toolbar.addWidget(self.target_box)
        for text, slot in [
            ("폴리곤 닫기", self.canvas.close_draft), ("복사", self._copy),
            ("붙여넣기", self._paste), ("삭제", self.canvas.delete_selected),
            ("맞춤", self.canvas.fit), ("저장", self.save_current),
            ("실행 취소", self._undo), ("다시 실행", self._redo),
        ]:
            self.toolbar.addAction(text, slot)
        root.addWidget(self.toolbar)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(8)
        side_layout.addWidget(QLabel("파일"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("경로 검색")
        self.search.textChanged.connect(self._filter_files)
        side_layout.addWidget(self.search)
        self.file_stats = QLabel()
        side_layout.addWidget(self.file_stats)
        self.dashboard_label = QLabel()
        self.dashboard_label.setWordWrap(True)
        self.dashboard_label.setStyleSheet("color:#dbe7f5;font-weight:600;padding:6px 0;")
        side_layout.addWidget(self.dashboard_label)
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self._open_item)
        side_layout.addWidget(self.file_list, 1)
        side_layout.addWidget(QLabel("폴리곤"))
        self.poly_list = QListWidget()
        self.poly_list.itemClicked.connect(self._select_poly_item)
        side_layout.addWidget(self.poly_list)
        side_layout.addWidget(QLabel("상태 단축키: 1 빈자리 / 2 점유 / 3 미정 / 4 가림"))
        tag_row = QHBoxLayout()
        for name, key in [("빈자리", "empty"), ("점유", "occupied"), ("미정", "unknown"), ("가림", "occlusion")]:
            button = QPushButton(name)
            button.clicked.connect(lambda _=False, k=key: self.canvas.set_occupancy(k))
            tag_row.addWidget(button)
        side_layout.addLayout(tag_row)
        self.tag_label = QLabel("태그: -")
        self.tag_label.setWordWrap(True)
        side_layout.addWidget(self.tag_label)
        add_row = QHBoxLayout()
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("사용자 태그")
        add_button = QPushButton("추가")
        add_button.clicked.connect(self._add_custom_tag)
        add_row.addWidget(self.tag_input)
        add_row.addWidget(add_button)
        side_layout.addLayout(add_row)

        splitter = QSplitter()
        splitter.addWidget(self.canvas)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)
        self._bind_shortcuts()
        self._filter_files("")
        self._refresh_inspector()

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("D"), self, activated=lambda: self.mode_box.setCurrentIndex(0))
        QShortcut(QKeySequence("V"), self, activated=lambda: self.mode_box.setCurrentIndex(1))
        QShortcut(QKeySequence("Return"), self, activated=self.canvas.close_draft)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self._copy)
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self._paste)
        QShortcut(QKeySequence("Delete"), self, activated=self.canvas.delete_selected)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_current)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo)
        QShortcut(QKeySequence("1"), self, activated=lambda: self.canvas.set_occupancy("empty"))
        QShortcut(QKeySequence("2"), self, activated=lambda: self.canvas.set_occupancy("occupied"))
        QShortcut(QKeySequence("3"), self, activated=lambda: self.canvas.set_occupancy("unknown"))
        QShortcut(QKeySequence("4"), self, activated=lambda: self.canvas.set_occupancy("occlusion"))
        QShortcut(QKeySequence("Esc"), self, activated=self._cancel_draft)

    def _mode_changed(self, index: int):
        self.canvas.mode = "draw" if index == 0 else "select"

    def _target_changed(self, index: int):
        self.canvas.draw_label = "parking_space" if index == 0 else "occlusion"

    def _on_canvas_changed(self):
        self.status_message.emit("저장되지 않은 변경")
        self._refresh_inspector()

    def _undo(self):
        if self.canvas.undo():
            self.status_message.emit("실행 취소")

    def _redo(self):
        if self.canvas.redo():
            self.status_message.emit("다시 실행")

    def _cancel_draft(self):
        self.canvas.draft.clear()
        self.canvas.update()

    def _filter_files(self, query: str):
        query = (query or "").strip().lower()
        self.file_list.clear()
        matched = [p for p in self.all_files if not query or query in p.lower()]
        for rel in matched[:800]:
            item = QListWidgetItem(Path(rel).name)
            item.setData(Qt.ItemDataRole.UserRole, rel)
            item.setToolTip(rel)
            if annot_path(Path(rel)).exists():
                item.setForeground(QColor("#2ec4b6"))
            self.file_list.addItem(item)
        self.file_stats.setText(f"표시 {min(len(matched), 800)} / 전체 {len(matched)}")

    def confirm_leave(self) -> bool:
        if not self.canvas.dirty:
            return True
        result = QMessageBox.question(self, "저장 확인", "저장하지 않은 변경이 있습니다. 이동할까요?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return result == QMessageBox.StandardButton.Yes

    def _open_item(self, item: QListWidgetItem):
        self.open_image(item.data(Qt.ItemDataRole.UserRole))

    def open_image(self, rel: str):
        if not self.confirm_leave():
            return
        self.canvas.load_image(DATA_DIR / rel, load_annotation(rel))
        self.current_path = rel
        self.status_message.emit(f"불러옴: {rel}")
        self._refresh_inspector()

    def save_current(self):
        if not self.current_path:
            return
        save_annotation(self.current_path, self.canvas.export_polygons())
        self.canvas._dirty = False
        self.status_message.emit("data/annotations에 저장됨")
        self._filter_files(self.search.text())

    def _refresh_inspector(self):
        self.poly_list.clear()
        counts = {"empty": 0, "occupied": 0, "unknown": 0, "occlusion": 0}
        for index, polygon in enumerate(self.canvas.polygons):
            state = "occlusion" if polygon.get("label") == "occlusion" or "occlusion" in polygon.get("tags", []) else occupancy_of(polygon["tags"])
            counts[state] += 1
            item = QListWidgetItem(f"#{index + 1}  {state}  ({len(polygon['points'])}개 꼭짓점)")
            item.setData(Qt.ItemDataRole.UserRole, polygon["id"])
            self.poly_list.addItem(item)
            if polygon["id"] == self.canvas.selected_id:
                item.setSelected(True)
        polygon = self.canvas.selected()
        self.tag_label.setText("태그: " + (", ".join(polygon["tags"]) if polygon and polygon["tags"] else "-"))
        self.dashboard_label.setText(
            f"주차 현황 | 전체 {sum(counts.values())} | 빈자리 {counts['empty']} | "
            f"점유 {counts['occupied']} | 미정 {counts['unknown']} | 가림 {counts['occlusion']}"
        )

    def _select_poly_item(self, item: QListWidgetItem):
        self.canvas.selected_id = item.data(Qt.ItemDataRole.UserRole)
        self.canvas.update()
        self._refresh_inspector()

    def _copy(self):
        self.status_message.emit("복사됨" if self.canvas.copy_selected() else "선택된 폴리곤 없음")

    def _paste(self):
        self.status_message.emit("붙여넣음" if self.canvas.paste_clipboard() else "클립보드가 비어 있음")

    def _add_custom_tag(self):
        tag = self.tag_input.text().strip()
        if not tag:
            tag, ok = QInputDialog.getText(self, "태그", "태그 이름")
            if not ok:
                return
            tag = tag.strip()
        self.canvas.add_tag(tag)
        self.tag_input.clear()
        self._refresh_inspector()
