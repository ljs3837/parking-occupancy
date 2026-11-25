"""확정된 관제 레이아웃의 칸 모양·이름을 색인 캔버스로 고친다."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from parking.app.annotator.canvas import Canvas


class LayoutEditor(QDialog):
    def __init__(self, image: QImage, spaces: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("주차칸 수정")
        self.resize(1100, 720)
        self._names: dict[str, str] = {}

        root = QVBoxLayout(self)
        hint = QLabel(
            "선택 모드에서 꼭짓점을 끌어 칸을 고칩니다. "
            "오른쪽 클래스명이 다른 캠과 같으면 같은 주차칸입니다. D=그리기 V=선택 Del=삭제"
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        self.canvas = Canvas()
        polygons = []
        for index, space in enumerate(spaces, start=1):
            poly_id = f"p{index}"
            name = str(space.get("space_id") or f"new-{index}")
            self._names[poly_id] = name
            polygons.append(
                {
                    "id": poly_id,
                    "points": space.get("points") or [],
                    "tags": [],
                    "label": "parking_space",
                }
            )
        self.canvas.load_qimage(image, polygons)
        self.canvas.mode = "select"

        side = QVBoxLayout()
        side.addWidget(QLabel("클래스명"))
        self.space_list = QListWidget()
        self.space_list.currentItemChanged.connect(self._select_from_list)
        side.addWidget(self.space_list, 1)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("클래스명")
        self.name_edit.editingFinished.connect(self._apply_name)
        side.addWidget(self.name_edit)
        mode_row = QHBoxLayout()
        draw_btn = QPushButton("그리기")
        draw_btn.clicked.connect(lambda: self._set_mode("draw"))
        select_btn = QPushButton("선택")
        select_btn.clicked.connect(lambda: self._set_mode("select"))
        mode_row.addWidget(draw_btn)
        mode_row.addWidget(select_btn)
        side.addLayout(mode_row)

        side_widget = QWidget()
        side_widget.setLayout(side)
        splitter = QSplitter()
        splitter.addWidget(self.canvas)
        splitter.addWidget(side_widget)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("취소")
        cancel.clicked.connect(self.reject)
        save = QPushButton("저장")
        save.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        root.addLayout(buttons)

        self.canvas.selection_changed.connect(self._sync_list)
        self.canvas.changed.connect(self._sync_list)
        QShortcut(QKeySequence(Qt.Key.Key_Delete), self, activated=self._delete)
        QShortcut(QKeySequence("D"), self, activated=lambda: self._set_mode("draw"))
        QShortcut(QKeySequence("V"), self, activated=lambda: self._set_mode("select"))
        self._sync_list()

    def _set_mode(self, mode: str):
        self.canvas.mode = mode
        self.canvas.update()

    def _delete(self):
        self.canvas.delete_selected()
        self._sync_list()

    def _sync_list(self):
        current = self.canvas.selected_id
        self.space_list.blockSignals(True)
        self.space_list.clear()
        chosen = None
        for index, polygon in enumerate(self.canvas.polygons, start=1):
            poly_id = polygon["id"]
            name = self._names.get(poly_id) or f"new-{index}"
            self._names[poly_id] = name
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, poly_id)
            self.space_list.addItem(item)
            if poly_id == current:
                chosen = item
        if chosen is not None:
            self.space_list.setCurrentItem(chosen)
            self.name_edit.setText(chosen.text())
        elif self.space_list.count():
            self.space_list.setCurrentRow(0)
            self.name_edit.setText(self.space_list.item(0).text())
        else:
            self.name_edit.clear()
        self.space_list.blockSignals(False)

    def _select_from_list(self, current, _previous):
        if current is None:
            return
        poly_id = current.data(Qt.ItemDataRole.UserRole)
        self.canvas.selected_id = poly_id
        self.canvas.selected_ids = {poly_id}
        self.name_edit.setText(current.text())
        self.canvas.update()

    def _apply_name(self):
        item = self.space_list.currentItem()
        if item is None:
            return
        name = self.name_edit.text().strip()
        if not name:
            return
        poly_id = item.data(Qt.ItemDataRole.UserRole)
        self._names[poly_id] = name
        item.setText(name)

    def result_spaces(self) -> list[dict]:
        self._apply_name()
        spaces = []
        used: dict[str, int] = {}
        for index, polygon in enumerate(self.canvas.export_polygons(), start=1):
            raw = self._names.get(polygon["id"]) or f"new-{index}"
            name = raw.strip() or f"new-{index}"
            used[name] = used.get(name, 0) + 1
            if used[name] > 1:
                name = f"{name}_{used[name]}"
            points = polygon.get("points") or []
            if len(points) < 3:
                continue
            spaces.append({"space_id": name, "points": points, "label": "parking_space"})
        return spaces
