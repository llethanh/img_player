"""The :class:`AnnotationOverlay` — captures pen strokes and paints them.

The overlay is a transparent ``QWidget`` parented to the
:class:`~img_player.render.gl_viewport.GLViewport`. It sits *above*
the GL surface in the Z-order, so its :meth:`paintEvent` draws on top
of the rendered image.

Two modes:

* **Pass-through** (default, no tool active) — the overlay sets
  ``WA_TransparentForMouseEvents`` on itself, so all mouse events
  fall through to the viewport's existing handlers (drag-scrub, pan,
  zoom). The user perceives no overlay.
* **Capture** (pen or eraser active) — the overlay captures the mouse
  and either appends to a stroke-in-progress (pen) or hit-tests the
  closest existing stroke (eraser).

Three pure helpers live at module level so the math is unit-testable
without spinning up a GL context — same pattern as
:func:`~img_player.render.gl_viewport._anchored_pan_for_zoom` from
PR #34.
"""

from __future__ import annotations

import math
from enum import Enum

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from img_player.annotate.store import AnnotationStore
from img_player.annotate.stroke import Stroke
from img_player.render.gl_viewport import GLViewport


class ToolKind(Enum):
    """Active drawing tool. ``NONE`` = pass-through (no capture)."""

    NONE = "none"
    PEN = "pen"
    ERASER = "eraser"


# ============================================================================
# Pure helpers (module-level for testability)
# ============================================================================


def widget_to_image(
    *,
    widget_xy: tuple[float, float],
    widget_size: tuple[int, int],
    img_size: tuple[int, int],
    factor: float,
    pan: tuple[float, float],
) -> tuple[float, float]:
    """Inverse of the viewport's image-to-widget transform.

    Forward::

        widget_xy = win/2 + pan + (image_xy - img/2) * factor

    Inverse::

        image_xy = (widget_xy - win/2 - pan) / factor + img/2
    """
    cx, cy = widget_xy
    win_w, win_h = widget_size
    img_w, img_h = img_size
    px, py = pan
    return (
        (cx - win_w / 2.0 - px) / factor + img_w / 2.0,
        (cy - win_h / 2.0 - py) / factor + img_h / 2.0,
    )


def image_to_widget(
    *,
    image_xy: tuple[float, float],
    widget_size: tuple[int, int],
    img_size: tuple[int, int],
    factor: float,
    pan: tuple[float, float],
) -> tuple[float, float]:
    """Forward image-space-to-widget transform. See :func:`widget_to_image`."""
    ix, iy = image_xy
    win_w, win_h = widget_size
    img_w, img_h = img_size
    px, py = pan
    return (
        win_w / 2.0 + px + (ix - img_w / 2.0) * factor,
        win_h / 2.0 + py + (iy - img_h / 2.0) * factor,
    )


def _point_segment_distance(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    """Shortest distance from point ``p`` to the segment ``a-b``.

    Standard "project onto segment, clamp to endpoints" geometry. Used
    by :func:`nearest_stroke_index` for the eraser's hit-testing.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        # Degenerate segment — both endpoints coincide.
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def nearest_stroke_index(
    cursor_image_xy: tuple[float, float],
    strokes: tuple[Stroke, ...] | list[Stroke],
    *,
    hit_radius: float,
) -> int | None:
    """Return the index of the stroke closest to the cursor, or ``None``.

    Distance is the minimum point-to-segment distance across all
    polyline segments of each stroke (a one-point stroke is treated
    as a point-to-point distance).

    A stroke qualifies only if its closest distance to the cursor is
    ≤ ``hit_radius``. Among qualifying strokes, the one with the
    smallest distance wins. ``hit_radius`` and the stroke coordinates
    must be in the same units (the overlay passes both in image
    pixels — see the spec §4 for why image-space).

    Returns ``None`` when no stroke is within reach. The eraser uses
    that as its no-op signal.
    """
    best_idx: int | None = None
    best_dist = math.inf
    for idx, stroke in enumerate(strokes):
        pts = stroke.points
        if len(pts) == 1:
            d = math.hypot(
                cursor_image_xy[0] - pts[0][0],
                cursor_image_xy[1] - pts[0][1],
            )
        else:
            d = min(
                _point_segment_distance(cursor_image_xy, pts[i], pts[i + 1])
                for i in range(len(pts) - 1)
            )
        if d < best_dist and d <= hit_radius:
            best_dist = d
            best_idx = idx
    return best_idx


# ============================================================================
# The overlay widget
# ============================================================================


# Decimation: skip a new mouseMove sample if it's within this many
# image-pixels of the previous one. Avoids inflating the polyline with
# sub-pixel jitter on a still cursor.
_MIN_MOVE_IMAGE_PX = 1.0

# Eraser hit radius in image-space pixels. Generous enough that a
# slightly-off click still grabs the intended stroke.
_ERASER_HIT_RADIUS_IMAGE_PX = 8.0


class AnnotationOverlay(QWidget):
    """Transparent layer above the GL viewport — captures + paints."""

    def __init__(
        self,
        gl_viewport: GLViewport,
        store: AnnotationStore,
        parent: QObject | None = None,
    ) -> None:
        # Parent is the GL viewport so the overlay sits in its
        # coordinate space and Z-stack above it.
        super().__init__(gl_viewport)
        self._gl_viewport = gl_viewport
        self._store = store

        # Tool state (set by app.py / toolbar in slice 3).
        self._tool: ToolKind = ToolKind.NONE
        self._color: str = "#E84A4A"
        self._size: float = 5.0

        # Frame the overlay should render. Set by app.py whenever the
        # controller's current frame changes.
        self._current_frame: int = 0

        # When playing, hide annotations unless the store's flag is
        # toggled on. App.py wires controller.state_changed.is_playing
        # to set_is_playing.
        self._is_playing: bool = False

        # In-progress stroke during a pen drag. ``None`` when not
        # actively drawing. Points are in image-space.
        self._current_stroke_points: list[tuple[float, float]] | None = None

        # Mirror the viewport's geometry. We follow resizes via an
        # event filter installed on the parent.
        self.setGeometry(gl_viewport.rect())
        gl_viewport.installEventFilter(self)

        # Pass-through by default — no tool, no capture.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Transparent background — we paint on top of the GL surface.
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Repaint when the store mutates the current frame.
        self._store.frame_annotated.connect(self._on_frame_annotated)
        # Repaint when the viewport's transform changes (zoom, pan,
        # resize, frame size change). The strokes are anchored in
        # image-space so the projection has to follow.
        self._gl_viewport.transform_changed.connect(self.update)

        # Lift above the GL surface in the Z-stack.
        self.raise_()

    # ------------------------------------------------------------------ Public API

    def set_tool(self, tool: ToolKind) -> None:
        """Switch the active tool. Toggles mouse capture accordingly."""
        if tool == self._tool:
            return
        self._tool = tool
        # Capture only when there's a tool to use.
        passthrough = tool == ToolKind.NONE
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, passthrough
        )
        # Cursor shape hint.
        if tool == ToolKind.PEN:
            self.setCursor(Qt.CursorShape.CrossCursor)
        elif tool == ToolKind.ERASER:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.unsetCursor()

    def set_color(self, color: str) -> None:
        self._color = color

    def set_size(self, size: float) -> None:
        self._size = max(1.0, float(size))

    def set_current_frame(self, frame: int) -> None:
        """Update the frame whose strokes should render. Triggers repaint."""
        if frame == self._current_frame:
            return
        self._current_frame = frame
        self.update()

    def set_is_playing(self, is_playing: bool) -> None:
        """Tell the overlay whether the controller is in playback.

        Combined with ``store.show_during_playback`` to decide whether
        to render anything in :meth:`paintEvent`.
        """
        if is_playing == self._is_playing:
            return
        self._is_playing = is_playing
        self.update()

    @property
    def tool(self) -> ToolKind:
        return self._tool

    # ------------------------------------------------------------------ Event filter (resize)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._gl_viewport and event.type() == QEvent.Type.Resize:
            self.setGeometry(self._gl_viewport.rect())
        return super().eventFilter(watched, event)

    # ------------------------------------------------------------------ Store signals

    def _on_frame_annotated(self, frame: int) -> None:
        """Repaint only if the changed frame is the one we display."""
        if frame == self._current_frame:
            self.update()

    # ------------------------------------------------------------------ Mouse capture

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        cursor_image = self._cursor_in_image_space(event)
        if cursor_image is None:
            return

        if self._tool == ToolKind.PEN:
            self._current_stroke_points = [cursor_image]
            self.update()
            event.accept()
        elif self._tool == ToolKind.ERASER:
            idx = nearest_stroke_index(
                cursor_image,
                self._store.strokes_at(self._current_frame),
                hit_radius=_ERASER_HIT_RADIUS_IMAGE_PX,
            )
            if idx is not None:
                self._store.remove_stroke(self._current_frame, idx)
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._current_stroke_points is None:
            return  # not actively drawing
        cursor_image = self._cursor_in_image_space(event)
        if cursor_image is None:
            return
        last = self._current_stroke_points[-1]
        if math.hypot(
            cursor_image[0] - last[0], cursor_image[1] - last[1]
        ) >= _MIN_MOVE_IMAGE_PX:
            self._current_stroke_points.append(cursor_image)
            self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self._current_stroke_points is None
        ):
            super().mouseReleaseEvent(event)
            return
        points = tuple(self._current_stroke_points)
        self._current_stroke_points = None
        if len(points) >= 1:
            stroke = Stroke(points=points, color=self._color, size=self._size)
            self._store.add_stroke(self._current_frame, stroke)
        event.accept()

    def _cursor_in_image_space(
        self, event: QMouseEvent
    ) -> tuple[float, float] | None:
        """Convert a Qt mouse event's position to image-space pixels.

        Returns ``None`` when the viewport has no image yet (size 0)
        or the zoom factor is zero — both edge cases where the
        transform is undefined.
        """
        img_size = self._gl_viewport.image_size()
        if img_size[0] == 0 or img_size[1] == 0:
            return None
        factor, pan_x, pan_y = self._gl_viewport.current_transform()
        if factor == 0.0:
            return None
        return widget_to_image(
            widget_xy=(event.position().x(), event.position().y()),
            widget_size=(self.width(), self.height()),
            img_size=img_size,
            factor=factor,
            pan=(pan_x, pan_y),
        )

    # ------------------------------------------------------------------ Painting

    def paintEvent(self, event: QEvent) -> None:  # noqa: ARG002 — Qt API
        # Visibility gate: don't paint when playing unless the user
        # explicitly toggled show-during-playback.
        if self._is_playing and not self._store.show_during_playback:
            return

        img_size = self._gl_viewport.image_size()
        if img_size[0] == 0 or img_size[1] == 0:
            return  # no image yet
        factor, pan_x, pan_y = self._gl_viewport.current_transform()
        if factor == 0.0:
            return

        widget_size = (self.width(), self.height())

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            for stroke in self._store.strokes_at(self._current_frame):
                self._draw_stroke(
                    painter,
                    stroke.points,
                    stroke.color,
                    stroke.size,
                    widget_size,
                    img_size,
                    factor,
                    (pan_x, pan_y),
                )
            # In-progress stroke (during a pen drag) — paint it with
            # the current tool color/size.
            if self._current_stroke_points is not None:
                self._draw_stroke(
                    painter,
                    tuple(self._current_stroke_points),
                    self._color,
                    self._size,
                    widget_size,
                    img_size,
                    factor,
                    (pan_x, pan_y),
                )
        finally:
            painter.end()

    def _draw_stroke(
        self,
        painter: QPainter,
        points: tuple[tuple[float, float], ...],
        color: str,
        size: float,
        widget_size: tuple[int, int],
        img_size: tuple[int, int],
        factor: float,
        pan: tuple[float, float],
    ) -> None:
        """Draw a single stroke onto ``painter``."""
        pen = QPen(QColor(color))
        # ``size`` is in image-pixels; multiply by factor so the
        # rendered stroke stays visually proportional to the image
        # at any zoom level.
        pen.setWidthF(max(1.0, size * factor))
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        # Project the polyline into widget space once.
        widget_pts = [
            image_to_widget(
                image_xy=p,
                widget_size=widget_size,
                img_size=img_size,
                factor=factor,
                pan=pan,
            )
            for p in points
        ]

        path = QPainterPath()
        path.moveTo(*widget_pts[0])

        if len(widget_pts) == 1:
            # A click without movement — render as a dot. moveTo + a
            # zero-length lineTo leaves a round-cap dot of pen width.
            path.lineTo(*widget_pts[0])
        elif len(widget_pts) == 2:
            # Only two samples: a quadratic curve has nothing to smooth
            # over, fall back to a straight segment.
            path.lineTo(*widget_pts[1])
        else:
            # Smoothing: trace quadratic Bézier segments using each
            # captured point as a control point and the midpoint
            # between consecutive points as anchors. The mouse-event
            # sampling rate (~60-125 Hz on Windows) leaves visible
            # polygonal joints between consecutive points when the
            # user drags fast — this midpoint-quadratic technique
            # rounds those joints into smooth curves at zero data
            # cost (the stored polyline is unchanged; only the
            # rendered path commands differ).
            #
            # Reference: the same trick used in inkscape, paper.js,
            # rough sketch tools, and the canonical "smooth signature
            # capture" pattern. Tangent continuity is automatic at
            # the midpoint anchors.
            for i in range(1, len(widget_pts) - 1):
                mid_x = (widget_pts[i][0] + widget_pts[i + 1][0]) / 2.0
                mid_y = (widget_pts[i][1] + widget_pts[i + 1][1]) / 2.0
                path.quadTo(*widget_pts[i], mid_x, mid_y)
            # The final segment from the last midpoint to the actual
            # last sample — keep it sharp so the stroke ends exactly
            # where the user released the mouse.
            path.lineTo(*widget_pts[-1])

        painter.drawPath(path)
