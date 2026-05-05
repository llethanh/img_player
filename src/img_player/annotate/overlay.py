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

from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

from img_player.annotate.ephemeral import EphemeralStrokeManager
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

        # Ephemeral mode (v0.4.1) — when True, finished pen strokes
        # are routed to the EphemeralStrokeManager instead of the
        # AnnotationStore. The manager owns the live list + the
        # fade timer; we just hand strokes off and read back at
        # paintEvent. Set by app.py via set_ephemeral_mode().
        self._ephemeral_mode: bool = False
        self._ephemeral_manager: EphemeralStrokeManager | None = None
        # Snapshot of _ephemeral_mode taken at mousePress, read at
        # mouseRelease. This binds the routing decision to press-time
        # so a mid-drag mode toggle (keyboard G or click on the
        # toolbar) doesn't reroute a stroke half-way through. Cleared
        # back to None when the drag ends.
        self._current_stroke_is_ephemeral: bool | None = None

        # When a non-left mouse button starts a drag (e.g. middle for
        # pan), we forward the entire press → move* → release sequence
        # to the GL viewport so the user can still pan / right-click /
        # etc. while a drawing tool is active. This field tracks
        # which button is being pass-thru'd; ``None`` means the next
        # press is up for grabs.
        self._passthrough_button: Qt.MouseButton | None = None

        # Pen stabilizer (Lazy Mouse) — when ``_stabilizer_factor`` is
        # non-zero, mouse_move events update a *target* position; a
        # 60 Hz catch-up timer pulls the *smoothed* position toward
        # the target by ``(1 - factor)`` each tick. Stroke samples
        # are appended from the smoothed position, not the raw cursor.
        # Result: hand tremor is filtered, the line trails behind the
        # cursor with a strength proportional to ``factor``. When
        # ``factor == 0`` the smoothed position snaps to the target
        # every tick (= no smoothing, behaviour identical to the
        # legacy direct-capture path).
        self._stabilizer_factor: float = 0.0
        self._stab_target: tuple[float, float] | None = None
        self._stab_smoothed: tuple[float, float] | None = None
        self._stab_timer = QTimer(self)
        self._stab_timer.setInterval(16)  # ~60 Hz catch-up cadence
        self._stab_timer.timeout.connect(self._on_stabilizer_tick)

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

    def set_pen_stabilizer_factor(self, factor: float) -> None:
        """Set the Lazy Mouse strength (0.0 to ~0.95).

        ``factor=0`` disables smoothing (raw cursor capture). Higher
        values make the line trail behind the cursor — the catch-up
        timer pulls the smoothed point ``(1 - factor)`` of the way to
        the cursor each frame, so a value of 0.85 means each frame
        traverses 15 % of the remaining distance (~500 ms perceived
        trail). Values above 0.95 stop progressing meaningfully and
        produce a stuck cursor — the toolbar caps at 0.85 by design.
        """
        self._stabilizer_factor = max(0.0, min(0.95, float(factor)))

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

    # ------------------------------------------------------------------ Ephemeral wiring (v0.4.1)

    def set_ephemeral_manager(self, manager: EphemeralStrokeManager) -> None:
        """Inject the live-strokes manager. Call once at app startup.

        The manager is owned by ``app.py`` (so its lifetime is tied
        to the application, not to a single overlay instance — which
        could be re-created on a sequence change). We also subscribe
        to its ``repaint_needed`` signal so the alpha animation
        renders smoothly without us polling.
        """
        self._ephemeral_manager = manager
        manager.repaint_needed.connect(self.update)

    def set_ephemeral_mode(self, on: bool) -> None:
        """Toggle ephemeral mode.

        While on, strokes finished by a left-button release route to
        ``EphemeralStrokeManager.add()`` instead of
        ``AnnotationStore.add_stroke()``. The actual stroke geometry,
        smoothing, and tool capture logic is unchanged — only the
        sink differs.
        """
        self._ephemeral_mode = bool(on)

    @property
    def ephemeral_mode(self) -> bool:
        return self._ephemeral_mode

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
            # Non-left click while a tool is active: hand the entire
            # press → move* → release sequence to the GL viewport so
            # the user can still pan with the middle button (or use
            # any future right-click handler) without having to
            # toggle the pen off first. Qt's automatic propagation
            # on event.ignore() is unreliable when the overlay is a
            # child of a QOpenGLWidget — the GL widget creates an
            # internal native subwindow and event routing differs
            # from regular widget hierarchies. Explicit forwarding
            # via sendEvent is rock-solid.
            self._passthrough_button = event.button()
            QCoreApplication.sendEvent(self._gl_viewport, event)
            return

        cursor_image = self._cursor_in_image_space(event)
        if cursor_image is None:
            return

        if self._tool == ToolKind.PEN:
            self._current_stroke_points = [cursor_image]
            # Snapshot the current mode so a mid-drag toggle doesn't
            # reroute the finished stroke. The user's intent is
            # decided at press-time: "I started this stroke in
            # ephemeral/persistent mode, that's where it goes".
            self._current_stroke_is_ephemeral = self._ephemeral_mode
            # Lazy Mouse: when active, kick off the catch-up timer.
            # Smoothed and target both start at the press point, so
            # there's no initial "jump" visible.
            if self._stabilizer_factor > 0.0:
                self._stab_target = cursor_image
                self._stab_smoothed = cursor_image
                self._stab_timer.start()
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
        # Mid-drag for the non-left passthrough — keep forwarding so
        # the viewport's pan logic gets each move sample.
        if self._passthrough_button is not None:
            QCoreApplication.sendEvent(self._gl_viewport, event)
            return
        if self._current_stroke_points is None:
            return  # not actively drawing
        cursor_image = self._cursor_in_image_space(event)
        if cursor_image is None:
            return
        # Lazy Mouse path: just update the target. The 60 Hz catch-up
        # timer pulls the smoothed point toward it and appends to the
        # stroke. Mouse-event sampling rate (60-125 Hz on Windows) is
        # plenty for tracking; the visible smoothness comes from the
        # EMA in ``_on_stabilizer_tick``.
        if self._stab_timer.isActive():
            self._stab_target = cursor_image
            event.accept()
            return
        # Legacy direct-capture path (factor == 0 or stabilizer never
        # started for this stroke).
        last = self._current_stroke_points[-1]
        if math.hypot(
            cursor_image[0] - last[0], cursor_image[1] - last[1]
        ) >= _MIN_MOVE_IMAGE_PX:
            self._current_stroke_points.append(cursor_image)
            self.update()
        event.accept()

    def _on_stabilizer_tick(self) -> None:
        """60 Hz catch-up step for Lazy Mouse.

        Pulls the smoothed point a fraction ``(1 - factor)`` of the
        way to the target each tick. When the smoothed point has
        moved enough since the last sample, the new position is
        appended to the in-progress stroke. This produces the
        "trailing line" effect that filters hand tremor.

        When factor is 0 (level=Off) the timer isn't running, so
        this code path is dormant.
        """
        if (
            self._current_stroke_points is None
            or self._stab_target is None
            or self._stab_smoothed is None
        ):
            return
        alpha = 1.0 - self._stabilizer_factor
        sx, sy = self._stab_smoothed
        tx, ty = self._stab_target
        new_x = sx + (tx - sx) * alpha
        new_y = sy + (ty - sy) * alpha
        self._stab_smoothed = (new_x, new_y)
        # Filter on movement vs the LAST APPENDED point, not the
        # previous smoothed pos. Otherwise high stabilizer values
        # would never advance the stroke (per-tick movement may stay
        # below the threshold for several ticks at once).
        last_x, last_y = self._current_stroke_points[-1]
        if math.hypot(new_x - last_x, new_y - last_y) >= _MIN_MOVE_IMAGE_PX:
            self._current_stroke_points.append((new_x, new_y))
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        # Closing the non-left passthrough — forward the release and
        # clear the flag so the next press is up for grabs again.
        if (
            self._passthrough_button is not None
            and event.button() == self._passthrough_button
        ):
            QCoreApplication.sendEvent(self._gl_viewport, event)
            self._passthrough_button = None
            return
        if (
            event.button() != Qt.MouseButton.LeftButton
            or self._current_stroke_points is None
        ):
            super().mouseReleaseEvent(event)
            return
        # Lazy Mouse: stop the catch-up timer and force the stroke to
        # END at the release position, regardless of how far behind
        # the smoothed point still was. Without this, releasing while
        # the line is still trailing would leave a visible gap
        # between the line end and where the user lifted — feels
        # broken. Appending the cursor position itself snaps the
        # final visual to the user's actual release point.
        if self._stab_timer.isActive():
            self._stab_timer.stop()
            release_pos = self._cursor_in_image_space(event)
            if release_pos is not None and self._current_stroke_points:
                last_x, last_y = self._current_stroke_points[-1]
                if math.hypot(
                    release_pos[0] - last_x, release_pos[1] - last_y,
                ) >= _MIN_MOVE_IMAGE_PX:
                    self._current_stroke_points.append(release_pos)
        self._stab_target = None
        self._stab_smoothed = None
        points = tuple(self._current_stroke_points)
        is_ephemeral = self._current_stroke_is_ephemeral
        self._current_stroke_points = None
        self._current_stroke_is_ephemeral = None
        if len(points) >= 1:
            stroke = Stroke(points=points, color=self._color, size=self._size)
            # Route based on the press-time snapshot. If the manager
            # wasn't injected (test harness, partial wiring) we fall
            # back to persistent — failing safe rather than dropping
            # the stroke on the floor.
            if is_ephemeral and self._ephemeral_manager is not None:
                self._ephemeral_manager.add(stroke)
            else:
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
            # Pass 1 — persistent strokes attached to the current frame,
            # at full alpha.
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
            # Pass 2 — live ephemeral strokes (v0.4.1). Painted on
            # top of persistent so a fading gesture stays visible
            # over the underlying review notes. Each stroke gets its
            # own alpha based on how long it has been alive.
            # Rendered regardless of self._is_playing — the spec is
            # explicit that ephemeral is presentation-grade and must
            # always show during playback.
            if self._ephemeral_manager is not None:
                for stroke, alpha in self._ephemeral_manager.live_strokes_with_alpha():
                    self._draw_stroke(
                        painter,
                        stroke.points,
                        stroke.color,
                        stroke.size,
                        widget_size,
                        img_size,
                        factor,
                        (pan_x, pan_y),
                        alpha=alpha,
                    )
            # Pass 3 — in-progress stroke (during a pen drag). Paint
            # it with the current tool color/size at full alpha. Even
            # an ephemeral-mode in-progress stroke renders solid
            # until release — the timer doesn't see it yet.
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
        *,
        alpha: float = 1.0,
    ) -> None:
        """Draw a single stroke onto ``painter``.

        ``alpha`` defaults to ``1.0`` (full opacity) so persistent
        strokes keep behaving exactly as before. Ephemeral strokes
        pass an alpha in ``[0, 1]`` computed from their age via
        :func:`~img_player.annotate.ephemeral.alpha_at`.
        """
        pen_color = QColor(color)
        # Multiply (not overwrite) the alpha — if the user picked a
        # color with built-in transparency (e.g. an #RRGGBBAA from
        # the palette), the fade still respects the original alpha.
        if alpha != 1.0:
            pen_color.setAlphaF(max(0.0, min(1.0, alpha)) * pen_color.alphaF())
        pen = QPen(pen_color)
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
