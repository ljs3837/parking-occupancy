from __future__ import annotations

import uuid

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from parking.app.annotator.io import occupancy_of

OCC_COLORS = {
    "occupied": QColor("#ef476f"),
    "empty": QColor("#2ec4b6"),
    "unknown": QColor("#f4c95d"),
}
OCCLUSION_COLOR = QColor("#a78bfa")


class Canvas(QWidget):
    changed = pyqtSignal()
    selection_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(640, 480)

        self.image: QImage | None = None
        self.polygons: list[dict] = []
        self.selected_id: str | None = None
        self.draft: list[QPointF] = []
        self.mode = "draw"
        self.draw_label = "parking_space"
        self.draw_status = "not occupied"
        self.clipboard: dict | None = None
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self._panning = False
        self._pan_last = QPointF()
        self._dragging = False
        self._drag_vertex_index: int | None = None
        self._drag_snapshot: list[dict] | None = None
        self._drag_last_norm = QPointF()
        self._dirty = False
        self._undo_stack: list[list[dict]] = []
        self._redo_stack: list[list[dict]] = []

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self, v: bool = True):
        self._dirty = v
        if v:
            self.changed.emit()

    def load_image(self, path, polygons: list[dict]):
        img = QImage(str(path))
        if img.isNull():
            raise RuntimeError(f"이미지 로드 실패: {path}")
        self.image = img
        self.polygons = []
        for p in polygons:
            self.polygons.append(
                {
                    "id": p.get("id") or f"p_{uuid.uuid4().hex[:8]}",
                    "points": [QPointF(float(x), float(y)) for x, y in p.get("points") or []],
                    "tags": list(p.get("tags") or []),
                    "label": p.get("label") or "parking_space",
                }
            )
        self.selected_id = self.polygons[0]["id"] if self.polygons else None
        self.draft.clear()
        self._dirty = False
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.fit()
        self.selection_changed.emit()
        self.update()

    def export_polygons(self) -> list[dict]:
        return [
            {
                "id": p["id"],
                "points": [[pt.x(), pt.y()] for pt in p["points"]],
                "tags": list(p["tags"]),
                "label": p.get("label") or "parking_space",
            }
            for p in self.polygons
        ]

    def fit(self):
        if not self.image:
            return
        iw, ih = self.image.width(), self.image.height()
        if iw <= 0 or ih <= 0:
            return
        self.scale = min(self.width() / iw, self.height() / ih) * 0.96
        self.offset = QPointF(
            (self.width() - iw * self.scale) / 2,
            (self.height() - ih * self.scale) / 2,
        )
        self.update()

    def to_norm(self, pos: QPointF) -> QPointF:
        if not self.image:
            return QPointF()
        x = (pos.x() - self.offset.x()) / self.scale / self.image.width()
        y = (pos.y() - self.offset.y()) / self.scale / self.image.height()
        return QPointF(x, y)

    def to_screen(self, n: QPointF) -> QPointF:
        if not self.image:
            return QPointF()
        return QPointF(
            self.offset.x() + n.x() * self.image.width() * self.scale,
            self.offset.y() + n.y() * self.image.height() * self.scale,
        )

    def clamp01(self, p: QPointF) -> QPointF:
        return QPointF(min(1.0, max(0.0, p.x())), min(1.0, max(0.0, p.y())))

    def selected(self) -> dict | None:
        for p in self.polygons:
            if p["id"] == self.selected_id:
                return p
        return None

    def _snapshot(self) -> list[dict]:
        return [
            {
                "id": p["id"],
                "points": [[pt.x(), pt.y()] for pt in p["points"]],
                "tags": list(p["tags"]),
                "label": p.get("label") or "parking_space",
            }
            for p in self.polygons
        ]

    def _restore(self, snapshot: list[dict]):
        self.polygons = [
            {
                "id": p["id"],
                "points": [QPointF(float(x), float(y)) for x, y in p["points"]],
                "tags": list(p.get("tags") or []),
                "label": p.get("label") or "parking_space",
            }
            for p in snapshot
        ]
        self.selected_id = self.polygons[0]["id"] if self.polygons else None

    def _record_undo(self):
        self._undo_stack.append(self._snapshot())
        self._redo_stack.clear()

    def _finish_drag_history(self):
        if self._drag_snapshot is not None and self._snapshot() != self._drag_snapshot:
            self._undo_stack.append(self._drag_snapshot)
            self._redo_stack.clear()
        self._drag_snapshot = None

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        self._restore(self._undo_stack.pop())
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        self._restore(self._redo_stack.pop())
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()
        return True

    def close_draft(self):
        if len(self.draft) < 3:
            return False
        self._record_undo()
        is_occlusion = self.draw_label == "occlusion" or self.draw_status == "occlusion"
        if is_occlusion:
            tags = ["occlusion"]
            label = "occlusion"
        elif self.draw_status == "occupied":
            tags = ["occupied"]
            label = "parking_space"
        else:
            tags = ["empty"]
            label = "parking_space"
        poly = {
            "id": f"p_{uuid.uuid4().hex[:8]}",
            "points": [self.clamp01(p) for p in self.draft],
            "tags": tags,
            "label": label,
        }
        self.polygons.append(poly)
        self.selected_id = poly["id"]
        self.draft.clear()
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()
        return True

    def copy_selected(self):
        p = self.selected()
        if not p:
            return False
        self.clipboard = {
            "points": [QPointF(pt) for pt in p["points"]],
            "tags": list(p["tags"]),
            "label": p.get("label") or "parking_space",
        }
        return True

    def paste_clipboard(self):
        if not self.clipboard:
            return False
        self._record_undo()
        points = [
            self.clamp01(QPointF(pt.x() + 0.02, pt.y() + 0.02))
            for pt in self.clipboard["points"]
        ]
        poly = {
            "id": f"p_{uuid.uuid4().hex[:8]}",
            "points": points,
            "tags": list(self.clipboard["tags"]),
            "label": self.clipboard.get("label") or "parking_space",
        }
        self.polygons.append(poly)
        self.selected_id = poly["id"]
        self.clipboard = {
            "points": [QPointF(pt) for pt in points],
            "tags": list(poly["tags"]),
            "label": poly["label"],
        }
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()
        return True

    def delete_selected(self):
        if not self.selected_id:
            return
        self._record_undo()
        self.polygons = [p for p in self.polygons if p["id"] != self.selected_id]
        self.selected_id = self.polygons[-1]["id"] if self.polygons else None
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()

    def set_occupancy(self, tag: str):
        p = self.selected()
        if not p:
            return
        self._record_undo()
        occ = {"occupied", "empty", "not occupied", "unknown", "occlusion"}
        p["tags"] = [t for t in p["tags"] if t not in occ]
        p["tags"].append(tag)
        p["label"] = "occlusion" if tag == "occlusion" else "parking_space"
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()

    def add_tag(self, tag: str):
        p = self.selected()
        if not p or not tag:
            return
        if tag not in p["tags"]:
            self._record_undo()
            p["tags"].append(tag)
            self.mark_dirty()
            self.selection_changed.emit()

    def _hit(self, n: QPointF) -> dict | None:
        for p in reversed(self.polygons):
            poly = QPolygonF([self.to_screen(pt) for pt in p["points"]])
            if poly.containsPoint(self.to_screen(n), Qt.FillRule.OddEvenFill):
                return p
        return None

    def _hit_vertex(self, n: QPointF) -> tuple[dict, int] | None:
        if not self.selected_id:
            return None
        p = self.selected()
        if not p:
            return None
        cursor = self.to_screen(n)
        for i, point in enumerate(p["points"]):
            screen = self.to_screen(point)
            if (screen.x() - cursor.x()) ** 2 + (screen.y() - cursor.y()) ** 2 <= 12 ** 2:
                return p, i
        return None

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0c1016"))

        if not self.image:
            painter.setPen(QColor("#8b9bb0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "오른쪽에서 이미지를 선택하세요")
            return

        target = self.image.scaled(
            int(self.image.width() * self.scale),
            int(self.image.height() * self.scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawImage(self.offset.toPoint(), target)

        for p in self.polygons:
            if len(p["points"]) < 2:
                continue
            is_occlusion = p.get("label") == "occlusion" or "occlusion" in p.get("tags", [])
            occ = "occlusion" if is_occlusion else occupancy_of(p["tags"])
            color = OCCLUSION_COLOR if is_occlusion else OCC_COLORS.get(occ, OCC_COLORS["unknown"])
            selected = p["id"] == self.selected_id
            fill = QColor(color)
            fill.setAlpha(90 if selected else 55)
            screen_pts = [self.to_screen(pt) for pt in p["points"]]
            painter.setBrush(fill)
            painter.setPen(QPen(color, 2.6 if selected else 1.6))
            painter.drawPolygon(QPolygonF(screen_pts))

            if selected:
                painter.setBrush(QColor("#ffffff"))
                painter.setPen(Qt.PenStyle.NoPen)
                for s in screen_pts:
                    painter.drawEllipse(s, 4, 4)

            c = QPointF(
                sum(pt.x() for pt in screen_pts) / len(screen_pts),
                sum(pt.y() for pt in screen_pts) / len(screen_pts),
            )
            painter.setPen(QColor("#e7eef7"))
            painter.drawText(c, "가림" if is_occlusion else occ[:8])

        if self.draft:
            painter.setPen(QPen(QColor("#f4a261"), 2, Qt.PenStyle.DashLine))
            for i in range(1, len(self.draft)):
                painter.drawLine(self.to_screen(self.draft[i - 1]), self.to_screen(self.draft[i]))
            painter.setBrush(QColor("#f4a261"))
            painter.setPen(Qt.PenStyle.NoPen)
            for pt in self.draft:
                painter.drawEllipse(self.to_screen(pt), 4, 4)

    def wheelEvent(self, event):
        if not self.image:
            return
        pos = event.position()
        before = self.to_norm(pos)
        factor = 1.12 if event.angleDelta().y() > 0 else 0.89
        self.scale = max(0.05, min(12.0, self.scale * factor))
        after = self.to_screen(before)
        self.offset += pos - after
        self.update()

    def mousePressEvent(self, event):
        if not self.image:
            return
        pos = event.position()
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self._panning = True
            self._pan_last = pos
            return

        n = self.to_norm(pos)
        if not (0 <= n.x() <= 1 and 0 <= n.y() <= 1):
            return

        if self.mode == "select" or event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            vertex_hit = self._hit_vertex(n)
            if vertex_hit:
                self.selected_id = vertex_hit[0]["id"]
                self._dragging = True
                self._drag_vertex_index = vertex_hit[1]
                self._drag_snapshot = self._snapshot()
                self._drag_last_norm = n
                self.selection_changed.emit()
                self.update()
                return
            hit = self._hit(n)
            self.selected_id = hit["id"] if hit else None
            if hit:
                self._dragging = True
                self._drag_vertex_index = None
                self._drag_snapshot = self._snapshot()
                self._drag_last_norm = n
            self.selection_changed.emit()
            self.update()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.draft.append(self.clamp01(n))
            self.update()

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self._panning:
            self.offset += pos - self._pan_last
            self._pan_last = pos
            self.update()
            return
        if self._dragging and self.selected_id:
            n = self.to_norm(pos)
            dx = n.x() - self._drag_last_norm.x()
            dy = n.y() - self._drag_last_norm.y()
            p = self.selected()
            if p and self._drag_vertex_index is not None:
                p["points"][self._drag_vertex_index] = self.clamp01(n)
                self._drag_last_norm = n
                self.mark_dirty()
                self.update()
                return
            if p:
                p["points"] = [
                    self.clamp01(QPointF(pt.x() + dx, pt.y() + dy)) for pt in p["points"]
                ]
                self._drag_last_norm = n
                self.mark_dirty()
                self.update()

    def mouseReleaseEvent(self, _event):
        self._panning = False
        self._finish_drag_history()
        self._dragging = False
        self._drag_vertex_index = None

    def mouseDoubleClickEvent(self, _event):
        if self.mode == "draw":
            self.close_draft()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update()
