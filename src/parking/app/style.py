APP_STYLE = """
QMainWindow, QWidget {
  background: #0c1016;
  color: #e7eef7;
  font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
  font-size: 13px;
}
QToolBar {
  background: #141a22;
  border-bottom: 1px solid #2a3545;
  spacing: 8px;
  padding: 6px;
}
QToolButton, QPushButton {
  background: #1b2330;
  border: 1px solid #2a3545;
  border-radius: 8px;
  padding: 6px 10px;
}
QPushButton:hover, QToolButton:hover {
  border-color: #2ec4b6;
}
QPushButton#navBtn {
  text-align: left;
  padding: 12px 14px;
  border: 1px solid transparent;
  border-radius: 10px;
  background: transparent;
  color: #8b9bb0;
  font-weight: 600;
}
QPushButton#navBtn:hover {
  background: #1b2330;
  color: #e7eef7;
}
QPushButton#navBtn[active="true"] {
  background: rgba(46, 196, 182, 0.16);
  border-color: rgba(46, 196, 182, 0.35);
  color: #b8fff7;
}
QLineEdit, QComboBox, QListWidget, QTreeView, QSpinBox, QTextEdit {
  background: #141a22;
  border: 1px solid #2a3545;
  border-radius: 8px;
  padding: 6px;
  color: #e7eef7;
}
QListWidget::item:selected, QTreeView::item:selected {
  background: rgba(46, 196, 182, 0.22);
  color: #e7eef7;
}
QStatusBar {
  background: #141a22;
  color: #8b9bb0;
}
QLabel {
  color: #8b9bb0;
}
QLabel#brandTitle {
  color: #e7eef7;
  font-size: 15px;
  font-weight: 700;
}
QLabel#brandSub {
  color: #8b9bb0;
  font-size: 11px;
}
QFrame#navRail {
  background: #10161e;
  border-right: 1px solid #2a3545;
}
QSplitter::handle {
  background: #2a3545;
  width: 2px;
}
QScrollBar:horizontal {
  height: 12px;
  background: #10161e;
  margin: 0;
}
QScrollBar:vertical {
  width: 12px;
  background: #10161e;
  margin: 0;
}
QScrollBar::handle:horizontal {
  background: #3d4d63;
  border-radius: 4px;
  min-width: 24px;
}
QScrollBar::handle:vertical {
  background: #3d4d63;
  border-radius: 4px;
  min-height: 24px;
}
QScrollBar::handle:horizontal:hover, QScrollBar::handle:vertical:hover {
  background: #2ec4b6;
}
QScrollBar::add-line, QScrollBar::sub-line {
  width: 0;
  height: 0;
}
QFrame#heroCard {
  background: #141a22;
  border: 1px solid #2a3545;
  border-radius: 14px;
}
"""
