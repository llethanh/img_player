"""Timeline widget: a scrub slider + frame number display."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QWidget


class Timeline(QWidget):  # type: ignore[misc]
    """Slider bound to the current frame, with `first / current / last` labels."""

    frame_requested = Signal(int)  # user scrubbed the slider

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._first = 0
        self._last = 0

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setTracking(True)
        self._slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._slider.valueChanged.connect(self._on_slider_value_changed)

        self._frame_label = QLabel("—")
        self._frame_label.setMinimumWidth(120)
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        layout.addWidget(self._slider, stretch=1)
        layout.addWidget(self._frame_label)

    def set_range(self, first: int, last: int) -> None:
        self._first = first
        self._last = max(first, last)
        self._slider.blockSignals(True)
        self._slider.setMinimum(first)
        self._slider.setMaximum(self._last)
        self._slider.blockSignals(False)
        self._refresh_label(self._slider.value())

    def set_current_frame(self, frame: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._refresh_label(frame)

    def _on_slider_value_changed(self, value: int) -> None:
        self._refresh_label(value)
        self.frame_requested.emit(value)

    def _refresh_label(self, current: int) -> None:
        if self._last <= self._first:
            self._frame_label.setText("—")
        else:
            self._frame_label.setText(f"{current} / {self._first}-{self._last}")
