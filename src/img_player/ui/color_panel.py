"""Color management panel: source, display, view, exposure, gamma."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from img_player.color.ocio_manager import OCIOManager


class ColorPanel(QWidget):  # type: ignore[misc]
    """Emits ``color_params_changed(src, display, view, exposure, gamma)`` on any change."""

    color_params_changed = Signal(str, str, str, float, float)

    def __init__(self, manager: OCIOManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._emit_enabled = True

        self._src_combo = QComboBox()
        self._src_combo.addItems(manager.list_colorspaces())
        default_src = manager.role("scene_linear") or manager.list_colorspaces()[0]
        self._src_combo.setCurrentText(default_src)

        self._display_combo = QComboBox()
        self._display_combo.addItems(manager.list_displays())
        default_display = manager.default_display()
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

        self._reset_btn = QPushButton("Reset exposure / gamma")
        self._reset_btn.clicked.connect(self._reset_adjustments)

        self._src_combo.currentTextChanged.connect(self._notify)
        self._display_combo.currentTextChanged.connect(self._on_display_changed)
        self._view_combo.currentTextChanged.connect(self._notify)
        self._exposure_spin.valueChanged.connect(self._notify)
        self._gamma_spin.valueChanged.connect(self._notify)

        form = QFormLayout()
        form.addRow("Source colorspace:", self._src_combo)
        form.addRow("Display:", self._display_combo)
        form.addRow("View:", self._view_combo)
        form.addRow("Exposure:", self._exposure_spin)
        form.addRow("Gamma:", self._gamma_spin)

        group = QGroupBox("Color management")
        group.setLayout(form)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(group)
        layout.addWidget(self._reset_btn)
        layout.addStretch(1)

    # -------------------------------------------------------------- Public helpers

    def current_params(self) -> tuple[str, str, str, float, float]:
        return (
            self._src_combo.currentText(),
            self._display_combo.currentText(),
            self._view_combo.currentText(),
            float(self._exposure_spin.value()),
            float(self._gamma_spin.value()),
        )

    def set_source_colorspace(self, name: str) -> None:
        """Set the source colorspace without emitting — used when opening a new
        sequence where we can guess the input colorspace from metadata."""
        self._emit_enabled = False
        self._src_combo.setCurrentText(name)
        self._emit_enabled = True

    def bump_exposure(self, delta: float) -> None:
        """Nudge the exposure spinbox (used by keyboard shortcuts)."""
        self._exposure_spin.setValue(self._exposure_spin.value() + delta)

    def emit_current(self) -> None:
        """Force an emission of ``color_params_changed`` with current values."""
        self._notify()

    # -------------------------------------------------------------- Internals

    def _reset_adjustments(self) -> None:
        self._emit_enabled = False
        self._exposure_spin.setValue(0.0)
        self._gamma_spin.setValue(1.0)
        self._emit_enabled = True
        self._notify()

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
        self._notify()

    def _notify(self) -> None:
        if not self._emit_enabled:
            return
        src, display, view, exposure, gamma = self.current_params()
        if not (src and display and view):
            return
        self.color_params_changed.emit(src, display, view, exposure, gamma)
