from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QDir, QSortFilterProxyModel, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFileSystemModel, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QPushButton, QSizePolicy, QSplitter,
    QToolBar, QTreeView, QVBoxLayout, QWidget,
)

from parking.app.annotator.canvas import Canvas
from parking.app.annotator.io import (
    annot_path, apply_folder_annotations, load_annotation, occupancy_of, polygons_from_folder_layout, save_annotation,
)
from parking.app.paths import ANNOT_DIR, DATA_DIR, IMAGE_EXTS


class _AnnotatedFsModel(QFileSystemModel):
    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.ForegroundRole and index.column() == 0 and not self.isDir(index):
            path = Path(self.filePath(index))
            if path.suffix.lower() in IMAGE_EXTS:
                try:
                    rel = path.resolve().relative_to(DATA_DIR.resolve())
                except (OSError, ValueError):
                    rel = None
                if rel is not None and annot_path(rel).exists():
                    return QColor("#2ec4b6")
        return super().data(index, role)


class _NameFilterProxy(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.query = ""
        self.setRecursiveFilteringEnabled(True)

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        if not self.query:
            return True
        model = self.sourceModel()
        index = model.index(source_row, 0, source_parent)
        if not index.isValid() or model.isDir(index):
            return True
        return self.query in model.fileName(index).lower()


class _DataTreeView(QTreeView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._on_resize = None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._on_resize is not None:
            self._on_resize()


class AnnotatorPage(QWidget):
    status_message = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        ANNOT_DIR.mkdir(parents=True, exist_ok=True)
        self.current_path: str | None = None
        self.canvas = Canvas()
        self.canvas.changed.connect(self._on_canvas_changed)
        self.canvas.selection_changed.connect(self._refresh_inspector)
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.timeout.connect(self._autosave)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.mode_box = QComboBox()
        self.mode_box.addItems(["그리기 (D)", "선택/이동 (V)", "상태변경 (E)"])
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        self.toolbar.addWidget(self.mode_box)
        self.target_box = QComboBox()
        self.target_box.addItems(["차 없는 칸", "차 있는 칸", "가림"])
        self.target_box.currentIndexChanged.connect(self._target_changed)
        self.toolbar.addWidget(self.target_box)
        for text, slot in [
            ("닫기 (Space)", self.canvas.close_draft), ("복사", self._copy),
            ("붙여넣기", self._paste), ("색인 삭제", self._delete_selected_polygon),
            ("맞춤", self.canvas.fit), ("폴더 칸 적용", self._apply_folder_polygons),
            ("저장", self.save_current),
            ("실행 취소", self._undo), ("다시 실행", self._redo),
        ]:
            self.toolbar.addAction(text, slot)
        root.addWidget(self.toolbar)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(8)
        side_layout.addWidget(QLabel("data"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("이름 검색")
        self.search.textChanged.connect(self._filter_files)
        side_layout.addWidget(self.search)
        self.file_stats = QLabel("data 폴더")
        self.file_stats.setWordWrap(True)
        side_layout.addWidget(self.file_stats)
        self.dashboard_label = QLabel()
        self.dashboard_label.setWordWrap(True)
        self.dashboard_label.setStyleSheet("color:#dbe7f5;font-weight:600;padding:6px 0;")
        side_layout.addWidget(self.dashboard_label)
        self.fs_model = _AnnotatedFsModel(self)
        self.fs_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.NoDotAndDotDot)
        root_index = self.fs_model.setRootPath(str(DATA_DIR))
        self.file_proxy = _NameFilterProxy(self)
        self.file_proxy.setSourceModel(self.fs_model)
        self.file_tree = _DataTreeView()
        self.file_tree.setModel(self.file_proxy)
        self.file_tree.setUniformRowHeights(True)
        self.file_tree.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.file_tree.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.file_tree.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.file_tree.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.file_tree.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.file_tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.file_tree.setMinimumWidth(0)
        # 헤더를 숨기면 가로 스크롤 범위가 생기지 않는다. 높이를 0으로 둔다.
        self.file_tree.setHeaderHidden(False)
        header = self.file_tree.header()
        header.setFixedHeight(0)
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        for col in (1, 2, 3):
            self.file_tree.hideColumn(col)
        self.file_tree.setRootIndex(self.file_proxy.mapFromSource(root_index))
        self.file_tree.clicked.connect(self._open_tree_index)
        self.file_tree.selectionModel().currentChanged.connect(self._on_tree_current_changed)
        self._scroll_passes = 0
        self._fitting_column = False
        self.file_tree.expanded.connect(self._on_tree_expanded)
        self.file_tree.collapsed.connect(self._schedule_fit_tree_column)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit_tree_column)
        self.file_tree._on_resize = self._schedule_fit_tree_column
        self.fs_model.directoryLoaded.connect(self._on_directory_loaded)
        self.fs_model.directoryLoaded.connect(self._schedule_fit_tree_column)
        side_layout.addWidget(self.file_tree, 1)
        side_layout.addWidget(QLabel("폴리곤"))
        self.poly_list = QListWidget()
        self.poly_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._updating_poly_list = False
        self.poly_list.itemSelectionChanged.connect(self._on_poly_selection)
        side_layout.addWidget(self.poly_list)
        hint = QLabel("폴더 칸 적용으로 같은 폴더 칸을 가져옴 · 기본은 차 없음 · E 로 상태 전환")
        hint.setWordWrap(True)
        side_layout.addWidget(hint)
        tag_row = QHBoxLayout()
        for name, key in [("차 없음", "empty"), ("차 있음", "occupied"), ("미정", "unknown"), ("가림", "occlusion")]:
            button = QPushButton(name)
            button.clicked.connect(lambda _=False, k=key: self._use_status(k))
            tag_row.addWidget(button)
        side_layout.addLayout(tag_row)

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
        QShortcut(QKeySequence("E"), self, activated=lambda: self.mode_box.setCurrentIndex(2))
        space = QShortcut(QKeySequence(Qt.Key.Key_Space), self.canvas)
        space.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        space.activated.connect(self.canvas.close_draft)
        QShortcut(QKeySequence("Ctrl+C"), self, activated=self._copy)
        QShortcut(QKeySequence("Ctrl+V"), self, activated=self._paste)
        for widget in (self.canvas, self.poly_list):
            shortcut = QShortcut(QKeySequence(Qt.Key.Key_Delete), widget)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(self._delete_selected_polygon)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self._delete_selected_file)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_current)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self._undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self._redo)
        QShortcut(QKeySequence("1"), self, activated=lambda: self._use_status("empty"))
        QShortcut(QKeySequence("2"), self, activated=lambda: self._use_status("occupied"))
        QShortcut(QKeySequence("3"), self, activated=lambda: self._use_status("unknown"))
        QShortcut(QKeySequence("4"), self, activated=lambda: self._use_status("occlusion"))
        QShortcut(QKeySequence("Esc"), self, activated=self._cancel_draft)

    def _mode_changed(self, index: int):
        self.canvas.mode = {0: "draw", 1: "select"}.get(index, "toggle")
        self.canvas.setCursor(
            Qt.CursorShape.PointingHandCursor if self.canvas.mode == "toggle" else Qt.CursorShape.ArrowCursor
        )

    def _apply_target(self, index: int):
        if index == 2:
            self.canvas.draw_label = "occlusion"
            self.canvas.draw_status = "occlusion"
            self.canvas.assign_tag = "occlusion"
            return
        self.canvas.draw_label = "parking_space"
        occupied = index == 1
        self.canvas.draw_status = "occupied" if occupied else "not occupied"
        self.canvas.assign_tag = "occupied" if occupied else "empty"

    def _target_changed(self, index: int):
        self._apply_target(index)
        self.canvas.apply_assign()

    def _use_status(self, tag: str):
        self.canvas.assign_tag = tag
        index = {"empty": 0, "occupied": 1, "occlusion": 2}.get(tag)
        if index is not None:
            self.target_box.blockSignals(True)
            self.target_box.setCurrentIndex(index)
            self.target_box.blockSignals(False)
            self._apply_target(index)
        self.canvas.set_occupancy(tag)

    def _on_canvas_changed(self):
        self._refresh_inspector()
        self._autosave_timer.start(150)

    def _autosave(self):
        if self.current_path and self.canvas.dirty:
            self.save_current()

    def _flush_autosave(self):
        self._autosave_timer.stop()
        self._autosave()

    def _undo(self):
        if self.canvas.undo():
            self.status_message.emit("실행 취소")

    def _redo(self):
        if self.canvas.redo():
            self.status_message.emit("다시 실행")

    def _cancel_draft(self):
        self.canvas.draft.clear()
        self.canvas.update()

    def _on_tree_expanded(self, _index):
        self._scroll_passes = 1
        self._schedule_fit_tree_column()

    def _schedule_fit_tree_column(self, *_args):
        self._fit_timer.start(40)

    def _fit_tree_column(self):
        view = self.file_tree
        model = view.model()
        if model is None or self._fitting_column:
            return
        self._fitting_column = True
        try:
            metrics = view.fontMetrics()
            indent = max(view.indentation(), 1)
            max_width = 0
            index = model.index(0, 0, view.rootIndex())
            seen = 0
            while index.isValid() and seen < 2500:
                text = model.data(index, Qt.ItemDataRole.DisplayRole)
                depth = 1
                parent = index.parent()
                root = view.rootIndex()
                while parent.isValid() and parent != root:
                    depth += 1
                    parent = parent.parent()
                    if depth > 64:
                        break
                text_w = metrics.horizontalAdvance(text if isinstance(text, str) else "")
                width = depth * indent + text_w + 56
                if width > max_width:
                    max_width = width
                index = view.indexBelow(index)
                seen += 1
            target = max(max_width, view.viewport().width())
            if view.columnWidth(0) != target:
                view.setColumnWidth(0, target)
            if self._scroll_passes and view.horizontalScrollBar().maximum() > 0:
                self._scroll_passes = 0
                view.horizontalScrollBar().setValue(view.horizontalScrollBar().maximum())
        finally:
            self._fitting_column = False

    def _on_directory_loaded(self, path: str):
        if self.file_tree.rootIndex().isValid():
            return
        if Path(path).resolve() != DATA_DIR.resolve():
            return
        root = self.fs_model.index(str(DATA_DIR))
        if root.isValid():
            self.file_tree.setRootIndex(self.file_proxy.mapFromSource(root))

    def _filter_files(self, query: str):
        query = (query or "").strip().lower()
        self.file_proxy.query = query
        self.file_proxy.invalidateFilter()
        self.file_stats.setText("data 폴더" if not query else f"이름 검색: {query}")

    def _on_tree_current_changed(self, current, _previous):
        self._reveal_tree_index(current)
        self._open_tree_index(current)

    def _reveal_tree_index(self, proxy_index):
        view = self.file_tree
        if not proxy_index.isValid():
            return
        view.scrollTo(proxy_index, QAbstractItemView.ScrollHint.EnsureVisible)
        rect = view.visualRect(proxy_index)
        if not rect.isValid():
            return
        bar = view.horizontalScrollBar()
        viewport_w = view.viewport().width()
        if rect.right() > viewport_w:
            bar.setValue(min(bar.maximum(), bar.value() + rect.right() - viewport_w + 12))
        elif rect.left() < 0:
            bar.setValue(max(0, bar.value() + rect.left() - 12))

    def _open_tree_index(self, proxy_index):
        if not proxy_index.isValid():
            return
        index = self.file_proxy.mapToSource(proxy_index)
        if not index.isValid() or self.fs_model.isDir(index):
            return
        path = Path(self.fs_model.filePath(index))
        if path.suffix.lower() not in IMAGE_EXTS:
            return
        try:
            rel = path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
        except (OSError, ValueError):
            return
        self.open_image(rel)

    def _image_rel(self, proxy_index) -> str | None:
        source = self.file_proxy.mapToSource(proxy_index)
        if not source.isValid() or self.fs_model.isDir(source):
            return None
        path = Path(self.fs_model.filePath(source))
        if path.suffix.lower() not in IMAGE_EXTS:
            return None
        try:
            return path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
        except (OSError, ValueError):
            return None

    def _neighbor_image_rel(self, index, deleted: Path) -> str | None:
        deleted_res = deleted.resolve()
        for step in (self.file_tree.indexBelow, self.file_tree.indexAbove):
            cursor = index
            for _ in range(40):
                cursor = step(cursor)
                if not cursor.isValid():
                    break
                rel = self._image_rel(cursor)
                if rel and (DATA_DIR / rel).resolve() != deleted_res:
                    return rel
        return None

    def _delete_selected_polygon(self):
        if not self.canvas.selected_ids and not self.canvas.selected_id:
            self.status_message.emit("선택된 색인이 없습니다")
            return
        self.canvas.delete_selected()
        self.status_message.emit("색인 삭제")

    def _delete_selected_file(self):
        index = self.file_tree.currentIndex()
        if not index.isValid():
            self.status_message.emit("삭제할 이미지를 탐색기에서 선택하세요")
            return
        source = self.file_proxy.mapToSource(index)
        if not source.isValid() or self.fs_model.isDir(source):
            self.status_message.emit("폴더는 삭제할 수 없습니다")
            return
        path = Path(self.fs_model.filePath(source))
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            self.status_message.emit("이미지 파일만 삭제할 수 있습니다")
            return
        try:
            rel = path.resolve().relative_to(DATA_DIR.resolve()).as_posix()
        except (OSError, ValueError):
            return
        next_rel = self._neighbor_image_rel(index, path)
        self._autosave_timer.stop()
        self.canvas._dirty = False
        self.current_path = None
        try:
            path.unlink()
        except OSError as exc:
            self.status_message.emit(f"삭제 실패: {exc}")
            return
        ann = annot_path(Path(rel))
        if ann.is_file():
            try:
                ann.unlink()
            except OSError:
                pass
        self.status_message.emit(f"삭제됨: {rel}")
        if next_rel:
            try:
                self.open_image(next_rel)
            except Exception as exc:
                self.status_message.emit(f"삭제됨: {rel} / 다음 이미지 열기 실패: {exc}")
            return
        self.canvas.image = None
        self.canvas.polygons.clear()
        self.canvas.selected_id = None
        self.canvas.selected_ids = set()
        self.canvas.update()
        self._refresh_inspector()

    def confirm_leave(self) -> bool:
        self._flush_autosave()
        return True

    def open_image(self, rel: str):
        if rel == self.current_path:
            return
        if not self.confirm_leave():
            return
        self.canvas.load_image(DATA_DIR / rel, load_annotation(rel))
        self.current_path = rel
        self.status_message.emit(f"불러옴: {rel}")
        self._refresh_inspector()

    def _apply_folder_polygons(self):
        if not self.current_path:
            self.status_message.emit("사진을 먼저 여세요")
            return
        if not Path(self.current_path).as_posix().startswith("cctv_image/"):
            self.status_message.emit("cctv_image 폴더에서만 적용됩니다")
            return
        polygons = self.canvas.export_polygons()
        if not polygons:
            polygons = polygons_from_folder_layout(self.current_path)
        if not polygons:
            self.status_message.emit("이 폴더에 적용할 칸이 없습니다. 한 장에서 칸을 먼저 그리세요")
            return
        self._autosave_timer.stop()
        try:
            applied = apply_folder_annotations(self.current_path, polygons)
        except (OSError, ValueError) as exc:
            self.status_message.emit(f"폴더 칸 적용 실패: {exc}")
            return
        self.canvas.replace_polygons(polygons)
        self.canvas._dirty = False
        for rel in applied:
            index = self.fs_model.index(str((DATA_DIR / rel).resolve()))
            if index.isValid():
                self.fs_model.dataChanged.emit(index, index, [Qt.ItemDataRole.ForegroundRole])
        self.status_message.emit(f"사진 {len(applied)}장에 폴더 칸 {len(polygons)}개 적용 · 새 칸은 차 없음")

    def save_current(self):
        if not self.current_path:
            return
        save_annotation(self.current_path, self.canvas.export_polygons())
        self.canvas._dirty = False
        self.status_message.emit("data/annotations에 저장됨")
        index = self.fs_model.index(str((DATA_DIR / self.current_path).resolve()))
        if index.isValid():
            self.fs_model.dataChanged.emit(index, index, [Qt.ItemDataRole.ForegroundRole])

    def _refresh_inspector(self):
        self._updating_poly_list = True
        self.poly_list.clear()
        counts = {"empty": 0, "occupied": 0, "unknown": 0, "occlusion": 0}
        for index, polygon in enumerate(self.canvas.polygons):
            state = "occlusion" if polygon.get("label") == "occlusion" or "occlusion" in polygon.get("tags", []) else occupancy_of(polygon["tags"])
            state_name = {"empty": "차 없음", "occupied": "차 있음", "unknown": "미정", "occlusion": "가림"}.get(state, state)
            counts[state] += 1
            item = QListWidgetItem(f"#{index + 1}  {state_name}  ({len(polygon['points'])}개 꼭짓점)")
            item.setData(Qt.ItemDataRole.UserRole, polygon["id"])
            self.poly_list.addItem(item)
            if polygon["id"] in self.canvas.selected_ids or polygon["id"] == self.canvas.selected_id:
                item.setSelected(True)
        self._updating_poly_list = False
        self.dashboard_label.setText(
            f"주차 현황 | 전체 {sum(counts.values())} | 차 없음 {counts['empty']} | "
            f"차 있음 {counts['occupied']} | 미정 {counts['unknown']} | 가림 {counts['occlusion']}"
        )

    def _on_poly_selection(self):
        if self._updating_poly_list:
            return
        ids = [item.data(Qt.ItemDataRole.UserRole) for item in self.poly_list.selectedItems()]
        self.canvas.selected_ids = set(ids)
        self.canvas.selected_id = ids[-1] if ids else None
        if len(ids) > 1:
            self.canvas.apply_assign()
        self.canvas.update()

    def _copy(self):
        count = self.canvas.copy_selected()
        self.status_message.emit(f"{count}개 복사됨" if count else "선택된 폴리곤 없음")

    def _paste(self):
        count = self.canvas.paste_clipboard()
        self.status_message.emit(f"{count}개 붙여넣음" if count else "클립보드가 비어 있음")
