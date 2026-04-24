"""Timeline widget: scrub slider + cache-fill indicator + frame number display."""

from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPaintEvent
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSlider, QVBoxLayout, QWidget


class CacheBar(QWidget):  # type: ignore[misc]
    """Thin strip that paints which frames are currently in the RAM cache."""

    BG_COLOR = QColor(32, 32, 32)
    CACHED_COLOR = QColor(56, 180, 100)
    PLAYHEAD_COLOR = QColor(240, 240, 240)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(6)
        self._first = 0
        self._last = 0
        self._cached: frozenset[int] = frozenset()
        self._current = 0

    def set_range(self, first: int, last: int) -> None:
        self._first = first
        self._last = max(first, last)
        self.update()

    def set_cached_frames(self, frames: frozenset[int]) -> None:
        if frames != self._cached:
            self._cached = frames
            self.update()

    def set_current_frame(self, frame: int) -> None:
        if frame != self._current:
            self._current = frame
            self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.BG_COLOR)

        total = self._last - self._first + 1
        if total <= 0:
            return

        w = self.width()
        h = self.height()
        pixel_per_frame = w / total
        in_range = [f for f in self._cached if self._first <= f <= self._last]
        if in_range:
            painter.setBrush(self.CACHED_COLOR)
            painter.setPen(Qt.PenStyle.NoPen)
            # Group consecutive frames into runs so we draw O(runs) rectangles
            # instead of one per frame — much faster for long sequences.
            in_range.sort()
            run_start = in_range[0]
            prev = run_start
            for f in in_range[1:]:
                if f == prev + 1:
                    prev = f
                    continue
                self._draw_run(painter, run_start, prev, pixel_per_frame, h)
                run_start = f
                prev = f
            self._draw_run(painter, run_start, prev, pixel_per_frame, h)

        # Playhead marker
        if self._first <= self._current <= self._last:
            x = (self._current - self._first) * pixel_per_frame
            painter.setPen(self.PLAYHEAD_COLOR)
            painter.drawLine(int(x), 0, int(x), h)

    def _draw_run(self, painter: QPainter, start: int, end: int, ppf: float, h: int) -> None:
        x1 = (start - self._first) * ppf
        x2 = (end + 1 - self._first) * ppf
        painter.drawRect(QRectF(x1, 0, max(1.0, x2 - x1), h))


class Timeline(QWidget):  # type: ignore[misc]
    """Slider bound to the current frame, with a cache bar and frame-range labels."""

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

        self._cache_bar = CacheBar()

        self._frame_label = QLabel("—")
        self._frame_label.setMinimumWidth(120)
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Slider + cache bar stacked vertically (inside a small column),
        # frame label to the right of the column.
        stack = QVBoxLayout()
        stack.setContentsMargins(0, 0, 0, 0)
        stack.setSpacing(0)
        stack.addWidget(self._slider)
        stack.addWidget(self._cache_bar)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)
        layout.addLayout(stack, stretch=1)
        layout.addWidget(self._frame_label)

    # ------------------------------------------------------------------ Public API

    def set_range(self, first: int, last: int) -> None:
        self._first = first
        self._last = max(first, last)
        self._slider.blockSignals(True)
        self._slider.setMinimum(first)
        self._slider.setMaximum(self._last)
        self._slider.blockSignals(False)
        self._cache_bar.set_range(first, self._last)
        self._refresh_label(self._slider.value())

    def set_current_frame(self, frame: int) -> None:
        self._slider.blockSignals(True)
        self._slider.setValue(frame)
        self._slider.blockSignals(False)
        self._cache_bar.set_current_frame(frame)
        self._refresh_label(frame)

    def set_cached_frames(self, frames: frozenset[int]) -> None:
        self._cache_bar.set_cached_frames(frames)

    # ------------------------------------------------------------------ Internals

    def _on_slider_value_changed(self, value: int) -> None:
        self._refresh_label(value)
        self._cache_bar.set_current_frame(value)
        self.frame_requested.emit(value)

    def _refresh_label(self, current: int) -> None:
        if self._last <= self._first:
            self._frame_label.setText("—")
        else:
            self._frame_label.setText(f"{current} / {self._first}-{self._last}")
