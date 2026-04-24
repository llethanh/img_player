"""Tiny Qt app to visually check the OCIO pipeline + GL viewport.

Loads ONE image file, displays it via :class:`GLViewport`, and lets the user
change the source colorspace, the display/view, and tweak exposure & gamma
with sliders.

Usage:
    python scripts/show_frame.py <path-to-image>
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QWidget,
)

from img_player.color.gpu_processor import build_shader_bundle
from img_player.color.ocio_manager import OCIOManager
from img_player.io.reader import read_frame
from img_player.render.gl_viewport import GLViewport


class ShowFrameWindow(QMainWindow):
    def __init__(self, image_path: Path) -> None:
        super().__init__()
        self.setWindowTitle(f"img_player — {image_path.name}")
        self.resize(1280, 720)

        self._manager = OCIOManager()
        self._pixels = read_frame(image_path)
        # Drop extra channels beyond RGBA for display.
        if self._pixels.shape[2] > 4:
            self._pixels = self._pixels[:, :, :4]

        self._viewport = GLViewport()
        self._viewport.set_frame(self._pixels)

        # Controls
        self._src_combo = QComboBox()
        self._src_combo.addItems(self._manager.list_colorspaces())
        # Try to pick a sensible default for the source colorspace.
        scene_linear = self._manager.role("scene_linear") or self._manager.list_colorspaces()[0]
        self._src_combo.setCurrentText(scene_linear)

        self._display_combo = QComboBox()
        self._display_combo.addItems(self._manager.list_displays())
        default_display = self._manager.default_display()
        self._display_combo.setCurrentText(default_display)

        self._view_combo = QComboBox()
        self._refresh_views(default_display)

        self._exposure_spin = QDoubleSpinBox()
        self._exposure_spin.setRange(-6.0, 6.0)
        self._exposure_spin.setSingleStep(0.25)
        self._exposure_spin.setValue(0.0)
        self._exposure_spin.setSuffix(" stops")

        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setRange(0.1, 4.0)
        self._gamma_spin.setSingleStep(0.05)
        self._gamma_spin.setValue(1.0)

        self._info_label = QLabel(
            f"Source : {image_path.name}\n"
            f"Shape  : {self._pixels.shape} ({self._pixels.dtype})\n"
            f"OCIO   : {self._manager.source.description}"
        )
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Wire signals
        self._src_combo.currentTextChanged.connect(lambda _: self._rebuild_shader())
        self._display_combo.currentTextChanged.connect(self._on_display_changed)
        self._view_combo.currentTextChanged.connect(lambda _: self._rebuild_shader())
        self._exposure_spin.valueChanged.connect(
            lambda v: self._viewport.set_color_params(exposure=v)
        )
        self._gamma_spin.valueChanged.connect(lambda v: self._viewport.set_color_params(gamma=v))

        # Layout
        form = QFormLayout()
        form.addRow("Source colorspace:", self._src_combo)
        form.addRow("Display:", self._display_combo)
        form.addRow("View:", self._view_combo)
        form.addRow("Exposure:", self._exposure_spin)
        form.addRow("Gamma:", self._gamma_spin)
        form.addRow(self._info_label)

        side_panel = QWidget()
        side_panel.setLayout(form)
        side_panel.setMaximumWidth(360)

        root = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(self._viewport, stretch=1)
        layout.addWidget(side_panel)
        root.setLayout(layout)
        self.setCentralWidget(root)

        self._rebuild_shader()

    def _refresh_views(self, display: str) -> None:
        self._view_combo.blockSignals(True)
        self._view_combo.clear()
        self._view_combo.addItems(self._manager.list_views(display))
        default_view = self._manager.default_view(display)
        if default_view:
            self._view_combo.setCurrentText(default_view)
        self._view_combo.blockSignals(False)

    def _on_display_changed(self, display: str) -> None:
        self._refresh_views(display)
        self._rebuild_shader()

    def _rebuild_shader(self) -> None:
        src = self._src_combo.currentText()
        display = self._display_combo.currentText()
        view = self._view_combo.currentText()
        if not src or not display or not view:
            return
        try:
            bundle = build_shader_bundle(
                self._manager, source_colorspace=src, display=display, view=view
            )
        except Exception as err:
            logging.exception("failed to build shader (%s -> %s/%s): %s", src, display, view, err)
            return
        self._viewport.set_color_params(bundle=bundle)


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("usage: python scripts/show_frame.py <path-to-image>", file=sys.stderr)
        return 2
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"file not found: {image_path}", file=sys.stderr)
        return 2

    app = QApplication(sys.argv)
    window = ShowFrameWindow(image_path)
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    sys.exit(main())
