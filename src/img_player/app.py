"""Qt application bootstrap — V0.1 smoke test (empty window, auto-close)."""

from __future__ import annotations

import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


def run_gui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("img_player (smoke test)")
    window.resize(600, 400)
    window.setCentralWidget(QLabel("img_player — setup OK", window))
    window.show()

    QTimer.singleShot(2000, app.quit)
    return int(app.exec())
