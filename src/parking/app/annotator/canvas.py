from __future__ import annotations

import math
import uuid

from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QImage, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QApplication, QWidget

from parking.app.annotator.io import occupancy_of

OCC_COLORS = {
    "occupied": QColor("#ef476f"),
    "empty": QColor("#2ec4b6"),
    "unknown": QColor("#f4c95d"),
}
OCCLUSION_COLOR = QColor("#a78bfa")
# 화면 픽셀. 확대와 상관없이 같은 거리 안에서 기존 꼭짓점에 붙인다.
SNAP_RADIUS_PX = 16


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
        self.selected_ids: set[str] = set()
        self.draft: list[QPointF] = []
        self.mode = "draw"
        self.draw_label = "parking_space"
        self.draw_status = "not occupied"
        self.assign_tag: str | None = None
        self.clipboard: list[dict] | None = None
        self.scale = 1.0
        self.offset = QPointF(0, 0)
        self._panning = False
        self._pan_last = QPointF()
        self._dragging = False
        self._drag_vertex_index: int | None = None
        self._drag_snapshot: list[dict] | None = None
        self._drag_last_norm = QPointF()
        self._dirty = False
        self._wheel_scale_accum = 0
        self._undo_stack: list[list[dict]] = []
        self._redo_stack: list[list[dict]] = []
        self._snap_norm: QPointF | None = None
        self._hover_norm: QPointF | None = None
        self._warping = False

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
        self.load_qimage(img, polygons)

    def load_qimage(self, image: QImage, polygons: list[dict]):
        if image.isNull():
            raise RuntimeError("이미지 로드 실패")
        self.image = image.copy()
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
        self.selected_ids = {self.selected_id} if self.selected_id else set()
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
        alive = {p["id"] for p in self.polygons}
        self.selected_ids = {pid for pid in self.selected_ids if pid in alive}
        if self.selected_id not in alive:
            self.selected_id = next(iter(self.selected_ids), None)

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
        self.selected_ids = {poly["id"]}
        self.draft.clear()
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()
        return True

    def replace_with_polygons(self, polygons: list[dict]) -> int:
        """기존 칸을 지우고 새 칸만 남긴다. 실행 취소로 이전 칸을 되돌린다."""
        added = []
        for polygon in polygons:
            points = [QPointF(float(x), float(y)) for x, y in polygon.get("points") or []]
            if len(points) < 3:
                continue
            added.append(
                {
                    "id": polygon.get("id") or f"p_{uuid.uuid4().hex[:8]}",
                    "points": points,
                    "tags": list(polygon.get("tags") or ["unknown"]),
                    "label": polygon.get("label") or "parking_space",
                    "source_name": polygon.get("source_name"),
                }
            )
        self._record_undo()
        self.polygons = added
        self.selected_id = added[-1]["id"] if added else None
        self.selected_ids = {item["id"] for item in added}
        self.draft.clear()
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()
        return len(added)

    def undo_last_draft_point(self) -> bool:
        if not self.draft:
            return False
        self.draft.pop()
        self.update()
        return True

    def replace_polygons(self, polygons: list[dict]):
        self._record_undo()
        self.polygons = [
            {
                "id": polygon.get("id") or f"p_{uuid.uuid4().hex[:8]}",
                "points": [QPointF(float(x), float(y)) for x, y in polygon.get("points") or []],
                "tags": list(polygon.get("tags") or ["empty"]),
                "label": polygon.get("label") or "parking_space",
            }
            for polygon in polygons
            if len(polygon.get("points") or []) >= 3
        ]
        self.selected_id = self.polygons[-1]["id"] if self.polygons else None
        self.selected_ids = {self.selected_id} if self.selected_id else set()
        self.draft.clear()
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()

    def _chosen(self) -> list[dict]:
        chosen = [p for p in self.polygons if p["id"] in self.selected_ids or p["id"] == self.selected_id]
        return chosen

    def copy_selected(self) -> int:
        chosen = self._chosen()
        if not chosen:
            return 0
        self.clipboard = [
            {
                "points": [QPointF(pt) for pt in p["points"]],
                "tags": list(p["tags"]),
                "label": p.get("label") or "parking_space",
            }
            for p in chosen
        ]
        return len(self.clipboard)

    def paste_clipboard(self) -> int:
        if not self.clipboard:
            return 0
        self._record_undo()
        pasted: list[dict] = []
        new_ids: list[str] = []
        for item in self.clipboard:
            points = [
                self.clamp01(QPointF(pt.x() + 0.02, pt.y() + 0.02))
                for pt in item["points"]
            ]
            poly = {
                "id": f"p_{uuid.uuid4().hex[:8]}",
                "points": points,
                "tags": list(item["tags"]),
                "label": item.get("label") or "parking_space",
            }
            self.polygons.append(poly)
            new_ids.append(poly["id"])
            pasted.append(
                {
                    "points": [QPointF(pt) for pt in points],
                    "tags": list(poly["tags"]),
                    "label": poly["label"],
                }
            )
        self.clipboard = pasted
        self.selected_ids = set(new_ids)
        self.selected_id = new_ids[-1]
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()
        return len(new_ids)

    def delete_selected(self):
        ids = set(self.selected_ids)
        if self.selected_id:
            ids.add(self.selected_id)
        if not ids:
            return
        self._record_undo()
        self.polygons = [p for p in self.polygons if p["id"] not in ids]
        self.selected_id = self.polygons[-1]["id"] if self.polygons else None
        self.selected_ids = {self.selected_id} if self.selected_id else set()
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()

    def set_occupancy(self, tag: str):
        targets = self._chosen()
        if not targets:
            return
        self._record_undo()
        occ = {"occupied", "empty", "not occupied", "unknown", "occlusion"}
        for p in targets:
            p["tags"] = [t for t in p["tags"] if t not in occ]
            p["tags"].append(tag)
            p["label"] = "occlusion" if tag == "occlusion" else "parking_space"
        self.mark_dirty()
        self.selection_changed.emit()
        self.update()

    def toggle_occupancy(self, poly: dict) -> str:
        if poly.get("label") == "occlusion" or "occlusion" in poly.get("tags", []):
            current = "occlusion"
        else:
            current = occupancy_of(poly.get("tags") or [])
        nxt = "empty" if current == "occupied" else "occupied"
        self.selected_id = poly["id"]
        self.selected_ids = {poly["id"]}
        self.set_occupancy(nxt)
        return nxt

    def _status_is(self, poly: dict, tag: str) -> bool:
        if poly.get("label") == "occlusion" or "occlusion" in poly.get("tags", []):
            current = "occlusion"
        else:
            current = occupancy_of(poly.get("tags") or [])
        if tag == "empty":
            return current == "empty"
        return current == tag

    def apply_assign(self) -> bool:
        tag = self.assign_tag
        if not tag:
            return False
        targets = self._chosen()
        if not targets or all(self._status_is(poly, tag) for poly in targets):
            return False
        self.set_occupancy(tag)
        return True

    def scale_selected(self, area_factor: float) -> int:
        targets = self._chosen()
        if not targets:
            return 0
        linear = math.sqrt(area_factor)
        self._record_undo()
        for poly in targets:
            points = poly["points"]
            if len(points) < 2:
                continue
            cx = sum(pt.x() for pt in points) / len(points)
            cy = sum(pt.y() for pt in points) / len(points)
            poly["points"] = [
                self.clamp01(QPointF(cx + (pt.x() - cx) * linear, cy + (pt.y() - cy) * linear))
                for pt in points
            ]
        self.mark_dirty()
        self.update()
        return len(targets)

    def _scale_by_wheel(self, dy: int):
        if not dy:
            return
        self._wheel_scale_accum += dy
        notches = int(self._wheel_scale_accum / 120)
        if notches == 0:
            return
        self._wheel_scale_accum -= notches * 120
        factor = 1.01 ** notches if notches > 0 else 0.99 ** (-notches)
        self.scale_selected(factor)

    def _hit(self, n: QPointF) -> dict | None:
        for p in reversed(self.polygons):
            poly = QPolygonF([self.to_screen(pt) for pt in p["points"]])
            if poly.containsPoint(self.to_screen(n), Qt.FillRule.OddEvenFill):
                return p
        return None

    def _control_held(self) -> bool:
        return bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ControlModifier)

    def _snap_norm_at(self, pos: QPointF) -> QPointF | None:
        """이미 그린 폴리곤 꼭짓점 중 화면 반경 안의 가장 가까운 점."""
        if not self.image:
            return None
        best: QPointF | None = None
        best_d = SNAP_RADIUS_PX ** 2
        for poly in self.polygons:
            for point in poly["points"]:
                screen = self.to_screen(point)
                dist = (screen.x() - pos.x()) ** 2 + (screen.y() - pos.y()) ** 2
                if dist <= best_d:
                    best_d = dist
                    best = point
        return None if best is None else QPointF(best)

    def _move_cursor_to_norm(self, norm: QPointF):
        if self._warping:
            return
        target = self.mapToGlobal(self.to_screen(norm).toPoint())
        if QCursor.pos() == target:
            return
        self._warping = True
        try:
            QCursor.setPos(target)
        finally:
            self._warping = False

    def _refresh_draw_snap(self, pos: QPointF, ctrl: bool):
        snap = self._snap_norm_at(pos) if self.mode == "draw" and ctrl else None
        self._snap_norm = snap
        if snap is not None:
            self._hover_norm = QPointF(snap)
            self._move_cursor_to_norm(snap)
        elif self.mode == "draw" and self.image:
            self._hover_norm = self.to_norm(pos)
        else:
            self._hover_norm = None
        self.update()

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
            selected = p["id"] in self.selected_ids or p["id"] == self.selected_id
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
                    painter.drawEllipse(s, 5, 5)

            c = QPointF(
                sum(pt.x() for pt in screen_pts) / len(screen_pts),
                sum(pt.y() for pt in screen_pts) / len(screen_pts),
            )
            painter.setPen(QColor("#e7eef7"))
            source = (p.get("source_name") or "").strip()
            if is_occlusion:
                text = "가림"
            elif source:
                text = source[:14]
            else:
                text = occ[:8]
            painter.drawText(c, text)

        if self.draft:
            painter.setPen(QPen(QColor("#f4a261"), 2, Qt.PenStyle.DashLine))
            for i in range(1, len(self.draft)):
                painter.drawLine(self.to_screen(self.draft[i - 1]), self.to_screen(self.draft[i]))
            painter.setBrush(QColor("#f4a261"))
            painter.setPen(Qt.PenStyle.NoPen)
            for pt in self.draft:
                painter.drawEllipse(self.to_screen(pt), 4, 4)
            if self._hover_norm is not None:
                painter.setPen(QPen(QColor("#f4a261"), 1.4, Qt.PenStyle.DotLine))
                painter.drawLine(self.to_screen(self.draft[-1]), self.to_screen(self._hover_norm))
        if self._snap_norm is not None:
            center = self.to_screen(self._snap_norm)
            painter.setBrush(QColor("#ffffff"))
            painter.setPen(QPen(QColor("#f4a261"), 2))
            painter.drawEllipse(center, 7, 7)

    def wheelEvent(self, event):
        if not self.image:
            return
        if self.mode != "draw" and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self._scale_by_wheel(event.angleDelta().y())
            event.accept()
            return
        pos = event.position()
        before = self.to_norm(pos)
        factor = 1.12 if event.angleDelta().y() > 0 else 0.89
        self.scale = max(0.05, min(12.0, self.scale * factor))
        after = self.to_screen(before)
        self.offset += pos - after
        self.update()

    def _start_move(self, n: QPointF):
        self._dragging = True
        self._drag_vertex_index = None
        self._drag_snapshot = self._snapshot()
        self._drag_last_norm = n

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Space and self.mode == "draw":
            self.close_draft()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Control and self.mode == "draw" and not event.isAutoRepeat():
            local = self.mapFromGlobal(QCursor.pos())
            self._refresh_draw_snap(QPointF(local), True)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key.Key_Control and not event.isAutoRepeat():
            self._snap_norm = None
            self._hover_norm = None
            self.update()
        super().keyReleaseEvent(event)

    def leaveEvent(self, event):
        self._snap_norm = None
        self._hover_norm = None
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if not self.image:
            return
        self.setFocus()
        pos = event.position()
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.AltModifier
        ):
            self._panning = True
            self._pan_last = pos
            return

        n = self.to_norm(pos)
        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        snap = self._snap_norm_at(pos) if self.mode == "draw" and ctrl else None
        if snap is not None:
            n = snap
        elif not (0 <= n.x() <= 1 and 0 <= n.y() <= 1):
            return

        if self.mode == "toggle" and event.button() == Qt.MouseButton.LeftButton:
            hit = self._hit(n)
            if hit:
                self.toggle_occupancy(hit)
            return

        # 그리기 중 Ctrl은 선택 대신 기존 꼭짓점에 붙인다.
        selecting = self.mode == "select" or bool(
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ) or (ctrl and self.mode != "draw")
        if selecting:
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            hit = self._hit(n)
            if shift and hit:
                self.selected_ids.add(hit["id"])
                self.selected_id = hit["id"]
                self.apply_assign()
                self._start_move(n)
                self.selection_changed.emit()
                self.update()
                return
            vertex_hit = None if shift else self._hit_vertex(n)
            if vertex_hit and len(self.selected_ids) <= 1:
                self.selected_id = vertex_hit[0]["id"]
                self.selected_ids = {self.selected_id}
                self._dragging = True
                self._drag_vertex_index = vertex_hit[1]
                self._drag_snapshot = self._snapshot()
                self._drag_last_norm = n
                self.selection_changed.emit()
                self.update()
                return
            if hit and hit["id"] in self.selected_ids and len(self.selected_ids) > 1:
                self.selected_id = hit["id"]
                self._start_move(n)
            elif hit:
                self.selected_id = hit["id"]
                self.selected_ids = {hit["id"]}
                self._start_move(n)
            else:
                self.selected_id = None
                self.selected_ids = set()
            self.selection_changed.emit()
            self.update()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            self.draft.append(self.clamp01(n))
            self._snap_norm = snap
            if (
                self.mode == "draw"
                and len(self.draft) == 4
                and self.draw_label != "occlusion"
                and self.draw_status != "occlusion"
            ):
                self.close_draft()
            else:
                self.update()

    def mouseMoveEvent(self, event):
        if self._warping:
            return
        pos = event.position()
        if self.mode == "draw" and not self._panning and not self._dragging:
            self._refresh_draw_snap(
                pos,
                bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier),
            )
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
            targets = [poly for poly in self.polygons if poly["id"] in self.selected_ids]
            if not targets and p:
                targets = [p]
            if targets:
                for poly in targets:
                    poly["points"] = [
                        self.clamp01(QPointF(pt.x() + dx, pt.y() + dy)) for pt in poly["points"]
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
