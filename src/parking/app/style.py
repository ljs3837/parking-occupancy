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
QLineEdit, QComboBox, QListWidget, QSpinBox, QTextEdit {
  background: #141a22;
  border: 1px solid #2a3545;
  border-radius: 8px;
  padding: 6px;
  color: #e7eef7;
}
QListWidget::item:selected {
  background: rgba(46, 196, 182, 0.22);
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
QFrame#heroCard {
  background: #141a22;
  border: 1px solid #2a3545;
  border-radius: 14px;
}
"""
