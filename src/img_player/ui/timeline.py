"""Nuke-inspired custom-painted timeline with tick marks, in/out markers, cache bar."""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import QWidget

from img_player.ui.theme import C, F, G

DisplayMode = Literal["frames", "tc"]


def frame_to_timecode(frame: int, fps: float) -> str:
    """Non-drop-frame timecode ``HH:MM:SS:FF`` for a frame index."""
    fps_int = max(1, round(fps))
    frame = max(0, frame)
    total_seconds = frame // fps_int
    ff = frame % fps_int
    hours = total_seconds // 3600
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{ff:02d}"


class Timeline(QWidget):  # type: ignore[misc]
    """Custom timeline: ticks + labels + range + in/out + playhead triangle + cache bar.

    Scrubbing emits ``frame_requested`` continuously during the drag.
    """

    frame_requested = Signal(int)

    # ---- Geometry (from charter) -------------------------------------
    MARGIN_X      = 8
    LABEL_H       = 14
    TICK_TOP      = 14
    TICK_MINOR_H  = 5
    TICK_MAJOR_H  = 9
    RANGE_Y       = 28
    RANGE_H       = 3
    CACHE_TOP     = 41
    CACHE_H       = 6
    TOTAL_H       = G.TIMELINE_H  # 52

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(self.TOTAL_H)
        self.setFixedHeight(self.TOTAL_H)
        self.setMouseTracking(True)

        self._first = 0
        self._last = 0
        self._current = 0
        self._in_frame: int | None = None
        self._out_frame: int | None = None
        self._fps = 24.0
        self._display_mode: DisplayMode = "frames"
        self._cached_frames: frozenset[int] = frozenset()
        self._scrubbing = False

        self._label_font: QFont = F.mono(F.SIZE_XS)

    # ------------------------------------------------------------------ Public API

    def set_range(self, first: int, last: int) -> None:
        self._first = first
        self._last = max(first, last)
        self.update()

    def set_current_frame(self, frame: int) -> None:
        if frame == self._current:
            return
        self._current = frame
        self.update()

    def set_in_out(self, in_frame: int | None, out_frame: int | None) -> None:
        self._in_frame = in_frame
        self._out_frame = out_frame
        self.update()

    def set_fps(self, fps: float) -> None:
        if abs(fps - self._fps) < 1e-6:
            return
        self._fps = max(0.1, fps)
        if self._display_mode == "tc":
            self.update()

    def set_display_mode(self, mode: DisplayMode) -> None:
        if mode == self._display_mode:
            return
        self._display_mode = mode
        self.update()

    def set_cached_frames(self, frames: frozenset[int]) -> None:
        if frames == self._cached_frames:
            return
        self._cached_frames = frames
        self.update()

    # ------------------------------------------------------------------ Painting

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), C.BG_DEEP)

        if self._last <= self._first:
            return

        painter.setFont(self._label_font)
        self._draw_ticks_and_labels(painter)
        self._draw_range_bar(painter)
        self._draw_in_out_markers(painter)
        self._draw_playhead(painter)
        self._draw_cache_bar(painter)

    def _usable_width(self) -> int:
        return max(1, int(self.width() - 2 * self.MARGIN_X))

    def _total_frames(self) -> int:
        return max(1, self._last - self._first + 1)

    def _frame_to_x(self, frame: int) -> float:
        ppf = self._usable_width() / self._total_frames()
        return self.MARGIN_X + (frame - self._first + 0.5) * ppf

    def _x_to_frame(self, x: float) -> int:
        ppf = self._usable_width() / self._total_frames()
        raw = round((x - self.MARGIN_X) / ppf - 0.5 + self._first)
        return max(self._first, min(self._last, raw))

    def _tick_spacings(self) -> tuple[int, int]:
        ppf = self._usable_width() / self._total_frames()
        if ppf >= 8:
            return 1, 5
        if ppf >= 2:
            return 5, 25
        if ppf >= 0.5:
            return 25, 100
        if ppf >= 0.1:
            return 100, 500
        return 500, 2500

    def _format_label(self, frame: int) -> str:
        if self._display_mode == "tc":
            return frame_to_timecode(frame, self._fps)
        return str(frame)

    def _draw_ticks_and_labels(self, painter: QPainter) -> None:
        minor, major = self._tick_spacings()
        tick_baseline = self.TICK_TOP
        metrics = QFontMetrics(self._label_font)
        label_y = self.LABEL_H - 2

        minor_pen = QPen(C.TICK_MINOR, 1)
        major_pen = QPen(C.TICK_MAJOR, 1)

        last_label_x = -9999
        for frame in range(self._first, self._last + 1):
            is_major = (frame - self._first) % major == 0 or frame == self._last
            is_minor = (frame - self._first) % minor == 0
            if not (is_minor or is_major):
                continue
            x = round(self._frame_to_x(frame))
            if is_major:
                painter.setPen(major_pen)
                painter.drawLine(x, tick_baseline, x, tick_baseline + self.TICK_MAJOR_H)
                label = self._format_label(frame)
                label_w = metrics.horizontalAdvance(label)
                lx = x - label_w // 2
                if lx - last_label_x > label_w + 8:
                    painter.setPen(C.TICK_LABEL)
                    painter.drawText(lx, label_y, label)
                    last_label_x = lx
            else:
                painter.setPen(minor_pen)
                painter.drawLine(x, tick_baseline, x, tick_baseline + self.TICK_MINOR_H)

    def _draw_range_bar(self, painter: QPainter) -> None:
        in_x  = self._frame_to_x(self._in_frame  if self._in_frame  is not None else self._first)
        out_x = self._frame_to_x(self._out_frame if self._out_frame is not None else self._last)
        if out_x <= in_x:
            return
        rect = QRectF(in_x, self.RANGE_Y, out_x - in_x, self.RANGE_H)
        painter.fillRect(rect, C.RANGE_BAR)

    def _draw_in_out_markers(self, painter: QPainter) -> None:
        painter.setPen(QPen(C.MARKER_IO, 1.5))
        painter.setBrush(C.MARKER_IO)
        y_top = self.TICK_TOP
        y_bot = self.RANGE_Y + self.RANGE_H + 4

        if self._in_frame is not None:
            x = self._frame_to_x(self._in_frame)
            painter.drawLine(QPointF(x, y_top), QPointF(x, y_bot))
            flag = QPolygonF(
                [
                    QPointF(x, y_top),
                    QPointF(x + 6, y_top + 3),
                    QPointF(x, y_top + 6),
                ]
            )
            painter.drawPolygon(flag)
        if self._out_frame is not None:
            x = self._frame_to_x(self._out_frame)
            painter.drawLine(QPointF(x, y_top), QPointF(x, y_bot))
            flag = QPolygonF(
                [
                    QPointF(x, y_top),
                    QPointF(x - 6, y_top + 3),
                    QPointF(x, y_top + 6),
                ]
            )
            painter.drawPolygon(flag)

    def _draw_playhead(self, painter: QPainter) -> None:
        x = self._frame_to_x(self._current)
        painter.setPen(QPen(C.PLAYHEAD, 1))
        painter.drawLine(QPointF(x, self.TICK_TOP), QPointF(x, self.RANGE_Y + self.RANGE_H))
        triangle = QPolygonF(
            [
                QPointF(x - 5, self.LABEL_H - 1),
                QPointF(x + 5, self.LABEL_H - 1),
                QPointF(x, self.TICK_TOP + 6),
            ]
        )
        painter.setPen(QPen(C.PLAYHEAD_OUTLINE, 1))
        painter.setBrush(C.PLAYHEAD)
        painter.drawPolygon(triangle)

    def _draw_cache_bar(self, painter: QPainter) -> None:
        # Slot background: solid deep black so empty / not-yet-cached
        # frames are clearly readable as "nothing here yet". No outer
        # border anymore — the border belongs to each cached run, not
        # to the slot itself.
        bar_rect = QRectF(self.MARGIN_X, self.CACHE_TOP, self._usable_width(), self.CACHE_H)
        painter.fillRect(bar_rect, C.CACHE_BAR_BG)

        if not self._cached_frames:
            return

        # Each cached run = translucent orange fill + opaque orange
        # border. Looks like a window over the black slot.
        painter.setBrush(C.CACHE_BAR)
        painter.setPen(QPen(C.CACHE_BAR_BORDER, 1))

        in_range = sorted(f for f in self._cached_frames if self._first <= f <= self._last)
        if not in_range:
            return
        run_start = in_range[0]
        prev = run_start
        for f in in_range[1:]:
            if f == prev + 1:
                prev = f
                continue
            self._draw_cache_run(painter, run_start, prev)
            run_start = f
            prev = f
        self._draw_cache_run(painter, run_start, prev)

    def _draw_cache_run(self, painter: QPainter, start: int, end: int) -> None:
        x1 = self._frame_to_x(start) - self._half_frame_width()
        x2 = self._frame_to_x(end)   + self._half_frame_width()
        painter.drawRect(QRectF(x1, self.CACHE_TOP, max(1.0, x2 - x1), self.CACHE_H))

    def _half_frame_width(self) -> float:
        return 0.5 * self._usable_width() / self._total_frames()

    # ------------------------------------------------------------------ Mouse

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._scrubbing = True
        self._emit_for_x(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._scrubbing:
            self._emit_for_x(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        del event
        self._scrubbing = False

    def _emit_for_x(self, x: float) -> None:
        if self._last <= self._first:
            return
        frame = self._x_to_frame(x)
        if frame != self._current:
            self._current = frame
            self.update()
        self.frame_requested.emit(frame)
