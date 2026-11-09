from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from parking.app.annotator.canvas import Canvas
from parking.app.annotator.io import (
    annot_path,
    list_images,
    load_annotation,
    occupancy_of,
    save_annotation,
)
from parking.app.paths import ANNOT_DIR, DATA_DIR


class AnnotatorPage(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        ANNOT_DIR.mkdir(parents=True, exist_ok=True)
        self.all_files = list_images()
        self.current_path: str | None = None

        self.canvas = Canvas()
        self.canvas.changed.connect(lambda: self.status_message.emit("저장 안 됨"))
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

        for text, slot in [
            ("폴리곤 닫기", self.canvas.close_draft),
            ("복사", self._copy),
            ("붙여넣기", self._paste),
            ("삭제", self.canvas.delete_selected),
            ("맞춤", self.canvas.fit),
            ("저장", self.save_current),
        ]:
            self.toolbar.addAction(text, slot)
        root.addWidget(self.toolbar)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(8)

        side_layout.addWidget(QLabel("파일"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("경로 검색…")
        self.search.textChanged.connect(self._filter_files)
        side_layout.addWidget(self.search)

        self.file_stats = QLabel()
        side_layout.addWidget(self.file_stats)

        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self._open_item)
        side_layout.addWidget(self.file_list, 1)

        side_layout.addWidget(QLabel("폴리곤"))
        self.poly_list = QListWidget()
        self.poly_list.itemClicked.connect(self._select_poly_item)
        side_layout.addWidget(self.poly_list)

        side_layout.addWidget(QLabel("점유 태그"))
        tag_row = QHBoxLayout()
        for name, key in [("occupied", "occupied"), ("empty", "empty"), ("unknown", "unknown")]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _=False, k=key: self.canvas.set_occupancy(k))
            tag_row.addWidget(btn)
        side_layout.addLayout(tag_row)

        self.tag_label = QLabel("태그: -")
        self.tag_label.setWordWrap(True)
        side_layout.addWidget(self.tag_label)

        add_row = QHBoxLayout()
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("커스텀 태그")
        add_btn = QPushButton("추가")
        add_btn.clicked.connect(self._add_custom_tag)
        add_row.addWidget(self.tag_input)
        add_row.addWidget(add_btn)
        side_layout.addLayout(add_row)

        splitter = QSplitter()
        splitter.addWidget(self.canvas)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        self._bind_shortcuts()
        self._filter_files("")

    def _bind_shortcuts(self):
        QShortcut(QKeySequence("D"), self, activated=lambda: self.mode_box.setCurrentIndex(0))
        QShortcut(QKeySequence("V"), self, activated=lambda: self.mode_box.setCurrentIndex(1))
        QShortcut(QKeySequence("Return"), self, activated=self.canvas.close_draft)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self._copy)
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self._paste)
        QShortcut(QKeySequence("Delete"), self, activated=self.canvas.delete_selected)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_current)
        QShortcut(QKeySequence("Esc"), self, activated=self._cancel_draft)

    def _mode_changed(self, idx: int):
        self.canvas.mode = "draw" if idx == 0 else "select"

    def _cancel_draft(self):
        self.canvas.draft.clear()
        self.canvas.update()

    def _filter_files(self, q: str):
        q = (q or "").strip().lower()
        self.file_list.clear()
        matched = [p for p in self.all_files if not q or q in p.lower()]
        shown = matched[:800]
        for rel in shown:
            item = QListWidgetItem(Path(rel).name)
            item.setData(Qt.ItemDataRole.UserRole, rel)
            item.setToolTip(rel)
            if annot_path(Path(rel)).exists():
                item.setForeground(QColor("#2ec4b6"))
            self.file_list.addItem(item)
        self.file_stats.setText(f"표시 {len(shown)} / 전체 {len(matched)}")

    def confirm_leave(self) -> bool:
        if not self.canvas.dirty:
            return True
        r = QMessageBox.question(
            self,
            "저장 안 됨",
            "저장하지 않은 색인이 있습니다. 이동할까요?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        return r == QMessageBox.StandardButton.Yes

    def _open_item(self, item: QListWidgetItem):
        self.open_image(item.data(Qt.ItemDataRole.UserRole))

    def open_image(self, rel: str):
        if not self.confirm_leave():
            return
        self.canvas.load_image(DATA_DIR / rel, load_annotation(rel))
        self.current_path = rel
        self.status_message.emit(f"불러옴 · {rel}")
        self._refresh_inspector()

    def save_current(self):
        if not self.current_path:
            return
        save_annotation(self.current_path, self.canvas.export_polygons())
        self.canvas._dirty = False
        self.status_message.emit("저장됨 → data/annotations/")
        self._filter_files(self.search.text())

    def _refresh_inspector(self):
        self.poly_list.clear()
        for i, p in enumerate(self.canvas.polygons):
            item = QListWidgetItem(f"#{i + 1}  {occupancy_of(p['tags'])}  ({len(p['points'])}pts)")
            item.setData(Qt.ItemDataRole.UserRole, p["id"])
            self.poly_list.addItem(item)
            if p["id"] == self.canvas.selected_id:
                item.setSelected(True)
        p = self.canvas.selected()
        self.tag_label.setText(
            "태그: " + (", ".join(p["tags"]) if p and p["tags"] else "-")
        )

    def _select_poly_item(self, item: QListWidgetItem):
        self.canvas.selected_id = item.data(Qt.ItemDataRole.UserRole)
        self.canvas.update()
        self._refresh_inspector()

    def _copy(self):
        ok = self.canvas.copy_selected()
        self.status_message.emit("복사됨" if ok else "복사할 폴리곤 없음")

    def _paste(self):
        ok = self.canvas.paste_clipboard()
        self.status_message.emit("붙여넣기 완료" if ok else "클립보드 비어 있음")

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
