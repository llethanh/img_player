"""LayerBar — custom-painted row showing one layer's range on the master timeline.

Visual model:

    │padding│··················· bar area ··················│padding│
    │       │ │┃────────── shotA.####.exr ─────────┃│       │
    │       │ ↑                                    ↑        │
    │     in handle              body          out handle   │
    │                                                       │
    │  master_first ───── playhead snap ─────── master_last │

Mouse interactions:

* Drag on the **body** → shifts ``layer.offset``.
* Drag on the **left handle** → adjusts ``layer.layer_in``.
* Drag on the **right handle** → adjusts ``layer.layer_out``.
* During the drag we show a live preview but commit only on
  release, so the LayerStack fires a single ``layer_modified``
  signal per gesture (one cache invalidation, not hundreds).

Snap targets (within ``SNAP_PX`` screen pixels):

* The master playhead.
* Master in / out points (when set).
* Every other layer's master_start / master_end edges.

The widget draws everything itself — there's no QSS — so the
visual stays consistent regardless of theme. Coordinate maths
live in pure helpers (testable without Qt event simulation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from img_player.layers import Layer
from img_player.ui.theme import F, H


# ---------------------------------------------------------------- Constants

# Pixel-space styling tokens. Tuned visually against the Studio
# Dark palette; adjust here if the row height ever changes.
HANDLE_W = 6           # vertical handle thickness in px
HANDLE_GRAB = 8        # extra hit-test margin around handles
SNAP_PX = 6            # snap distance in screen pixels
PADDING_X = 4          # left/right inner padding
BAR_RADIUS = 2         # corner rounding for the bar fill

# Fixed-height row budget.
BAR_HEIGHT = 22


# ---------------------------------------------------------------- Geometry helpers


@dataclass(frozen=True)
class BarGeometry:
    """Pure-data helper for converting between master frames and pixel x.

    All x coordinates are inside the LayerBar widget (origin at its
    top-left). The bar's drawable region is ``[PADDING_X, width - PADDING_X]``.
    """

    width: int                 # widget width in pixels
    master_first: int          # leftmost master frame represented
    master_last: int           # rightmost master frame represented (inclusive)

    @property
    def usable_w(self) -> int:
        return max(1, self.width - 2 * PADDING_X)

    @property
    def master_length(self) -> int:
        return max(1, self.master_last - self.master_first + 1)

    def frame_to_x(self, master_frame: int) -> float:
        """Map a master-frame index to its widget pixel x-coordinate.

        Out-of-range frames return clamped x; callers that care about
        the distinction should check ``master_first <= f <= master_last``.
        """
        normalized = (master_frame - self.master_first) / self.master_length
        return PADDING_X + normalized * self.usable_w

    def x_to_frame(self, x: float) -> int:
        """Inverse of :meth:`frame_to_x` — rounded to the nearest master-frame."""
        normalized = (x - PADDING_X) / self.usable_w
        return self.master_first + round(normalized * self.master_length)


def snap_master_frame(
    candidate: int,
    geometry: BarGeometry,
    targets: list[int],
    snap_distance_frames: int,
) -> int:
    """Return the closest snap target to ``candidate`` if within
    ``snap_distance_frames``; otherwise return ``candidate`` unchanged.

    Pure function — keeps the snap math out of the widget so it can
    be unit-tested without a Qt event loop. ``snap_distance_frames``
    is computed from :data:`SNAP_PX` translated to frame units by
    the caller (depends on the live widget width).
    """
    if not targets:
        return candidate
    best_target = candidate
    best_distance = snap_distance_frames + 1  # strict-less-than below
    for t in targets:
        d = abs(t - candidate)
        if d < best_distance:
            best_distance = d
            best_target = t
    return best_target if best_distance <= snap_distance_frames else candidate


# ---------------------------------------------------------------- Widget


DragKind = Literal["body", "in", "out"]


class LayerBar(QWidget):  # type: ignore[misc]
    """Single-row visualisation + drag interaction for one Layer."""

    # Emitted on mouse release with the FINAL committed value(s).
    # Carrying the layer id lets the LayerPanel route to LayerStack
    # without scanning rows.
    offset_changed = Signal(str, int)         # (layer_id, new_offset)
    # IN handle drag: standard NLE convention — the LEFT edge of the
    # visible bar moves while the RIGHT edge stays put. That requires
    # changing both ``layer_in`` and ``offset`` by the same delta in
    # one shot, so we carry them together and route to a single
    # ``LayerStack.update`` (= one ``layer_modified`` signal, one
    # cache invalidation).
    trim_in_changed = Signal(str, int, int)   # (layer_id, new_layer_in, new_offset)
    layer_out_changed = Signal(str, int)      # (layer_id, new_layer_out)
    # Anywhere on the bar, single-click without drag → focus the layer
    # (= same as clicking the row). Lets the row delegate without
    # having to forward QMouseEvent itself.
    focus_requested = Signal(str)

    def __init__(self, layer: Layer, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer = layer
        # Master timeline state is fed by the panel — defaults match
        # an empty stack so the widget paints a sensible "nothing yet".
        self._master_first: int = layer.master_start
        self._master_last: int = layer.master_end
        # Playhead position on the master timeline, used for snap +
        # the small vertical line drawn on top.
        self._playhead: int | None = None
        # Master in / out points (may be ``None`` when not set).
        self._master_in: int | None = None
        self._master_out: int | None = None
        # Other layers' edges to snap against. Stored as a flat list
        # of master-frame integers — duplicates are fine since
        # ``snap_master_frame`` picks the closest.
        self._snap_edges: list[int] = []

        # Live drag state. ``_drag_kind`` is None when the user
        # isn't pressing; the *_preview fields hold the in-progress
        # value while the user moves the mouse. On release we emit
        # the corresponding signal and reset.
        #
        # IN-handle drag updates BOTH ``_drag_preview_layer_in`` and
        # ``_drag_preview_offset`` so the bar's right edge stays put
        # while the left edge moves (standard NLE convention).
        self._drag_kind: DragKind | None = None
        self._drag_preview_offset: int | None = None
        self._drag_preview_layer_in: int | None = None
        self._drag_preview_layer_out: int | None = None
        # Capture the press position so we can detect "click without
        # drag" (= focus, no offset change) by checking total motion.
        self._drag_start_x: float = 0.0
        self._drag_start_layer_offset: int = layer.offset
        self._drag_start_layer_in: int = layer.layer_in
        self._drag_start_layer_out: int = layer.layer_out
        self._has_moved: bool = False

        self.setFixedHeight(BAR_HEIGHT)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed,
        )
        self.setMouseTracking(True)
        # Makes the cursor change snappy on hover even before the
        # user clicks.
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    # ------------------------------------------------------------------ Public API

    def set_layer(self, layer: Layer) -> None:
        """Update the underlying layer reference. Call when the row's
        layer attributes have mutated externally."""
        self._layer = layer
        # Reset the drag start positions so a future press captures
        # the fresh state.
        self._drag_start_layer_offset = layer.offset
        self._drag_start_layer_in = layer.layer_in
        self._drag_start_layer_out = layer.layer_out
        self.update()

    def set_master_range(self, first: int, last: int) -> None:
        if (first, last) == (self._master_first, self._master_last):
            return
        self._master_first = first
        self._master_last = max(first, last)
        self.update()

    def set_playhead(self, master_frame: int | None) -> None:
        if master_frame == self._playhead:
            return
        self._playhead = master_frame
        self.update()

    def set_master_in_out(self, in_frame: int | None, out_frame: int | None) -> None:
        self._master_in = in_frame
        self._master_out = out_frame
        self.update()

    def set_snap_edges(self, edges: list[int]) -> None:
        """Other layers' master_start / master_end edges to snap against."""
        self._snap_edges = list(edges)

    # ------------------------------------------------------------------ Painting

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        geom = self._geometry()
        layer = self._layer
        # Apply preview values when dragging so the visual moves
        # with the mouse before the commit.
        offset = self._drag_preview_offset
        if offset is None:
            offset = layer.offset
        layer_in = self._drag_preview_layer_in
        if layer_in is None:
            layer_in = layer.layer_in
        layer_out = self._drag_preview_layer_out
        if layer_out is None:
            layer_out = layer.layer_out
        master_start = offset
        master_end = offset + (layer_out - layer_in)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Background track (subtle, just so the bar pops). Drawn
        # behind everything so the layer fill sits on top.
        track_rect = QRectF(
            PADDING_X, 4,
            geom.usable_w, BAR_HEIGHT - 8,
        )
        painter.fillRect(track_rect, QColor("#15171B"))

        # Layer fill — accent-tinted so it matches the focused-row
        # highlight when the user clicks. Slightly translucent to
        # let the snap-edge line peek through if it overlaps.
        x1 = geom.frame_to_x(master_start)
        x2 = geom.frame_to_x(master_end)
        bar_rect = QRectF(x1, 4, max(2.0, x2 - x1), BAR_HEIGHT - 8)
        bar_color = QColor(H.ACCENT)
        bar_color.setAlpha(200)
        painter.setBrush(bar_color)
        painter.setPen(QPen(QColor(H.ACCENT_BRIGHT), 1))
        painter.drawRoundedRect(bar_rect, BAR_RADIUS, BAR_RADIUS)

        # Filename label, ellipsized to fit inside the bar minus the
        # handles. Drawn in dark-on-orange for legibility against
        # the accent fill.
        font = F.mono(F.SIZE_SM)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        text_area = bar_rect.adjusted(HANDLE_W + 2, 0, -(HANDLE_W + 2), 0)
        elided = metrics.elidedText(
            layer.name, Qt.TextElideMode.ElideMiddle, int(text_area.width()),
        )
        painter.setPen(QColor("#0A0A0A"))
        painter.drawText(text_area, int(Qt.AlignmentFlag.AlignVCenter), elided)

        # Trim handles — drawn on top of the bar so they're always
        # grabable. Brighter than the body so they read as
        # interactive.
        handle_color = QColor("#FFFFFF")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(handle_color)
        in_handle = QRectF(x1, 4, HANDLE_W, BAR_HEIGHT - 8)
        out_handle = QRectF(x2 - HANDLE_W, 4, HANDLE_W, BAR_HEIGHT - 8)
        painter.drawRect(in_handle)
        painter.drawRect(out_handle)

        # Playhead line on top of everything.
        if self._playhead is not None and \
                geom.master_first <= self._playhead <= geom.master_last:
            ph_x = geom.frame_to_x(self._playhead)
            painter.setPen(QPen(QColor("#F2F2F2"), 1))
            painter.drawLine(int(ph_x), 0, int(ph_x), BAR_HEIGHT)

    # ------------------------------------------------------------------ Mouse

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x = event.position().x()
        kind = self._hit_test(x)
        self._drag_kind = kind
        self._drag_start_x = x
        self._drag_start_layer_offset = self._layer.offset
        self._drag_start_layer_in = self._layer.layer_in
        self._drag_start_layer_out = self._layer.layer_out
        self._has_moved = False
        if kind == "in" or kind == "out":
            self.setCursor(Qt.CursorShape.SizeHorCursor)
        elif kind == "body":
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = event.position().x()
        if self._drag_kind is None:
            # Pure hover — adjust cursor based on hit test.
            kind = self._hit_test(x)
            if kind == "in" or kind == "out":
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif kind == "body":
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            return
        # Active drag.
        if abs(x - self._drag_start_x) > 1.0:
            self._has_moved = True
        geom = self._geometry()
        delta_frames = geom.x_to_frame(x) - geom.x_to_frame(self._drag_start_x)
        snap_dist = self._snap_distance_in_frames(geom)
        layer = self._layer
        if self._drag_kind == "body":
            new_offset = self._drag_start_layer_offset + delta_frames
            new_offset = snap_master_frame(
                new_offset, geom, self._snap_targets(),
                snap_dist,
            )
            self._drag_preview_offset = new_offset
        elif self._drag_kind == "in":
            # Standard NLE in-trim: the LEFT edge of the bar moves,
            # the right edge stays where it is. That means BOTH
            # ``layer_in`` and ``offset`` shift by the same delta
            # (``master_start = offset`` moves; ``master_end =
            # offset + layer_out - layer_in`` stays put).
            new_in = self._drag_start_layer_in + delta_frames
            # Clamp to source range and keep at least one frame
            # between in/out.
            new_in = max(layer.sequence.first_frame, new_in)
            new_in = min(self._drag_start_layer_out - 1, new_in)
            actual_delta = new_in - self._drag_start_layer_in
            new_offset = self._drag_start_layer_offset + actual_delta
            # Snap the *visible* left edge (= master_start = new_offset).
            snapped_master = snap_master_frame(
                new_offset, geom, self._snap_targets(exclude_self=True),
                snap_dist,
            )
            snap_delta = snapped_master - new_offset
            new_in += snap_delta
            new_offset += snap_delta
            # Re-clamp after snap.
            new_in = max(layer.sequence.first_frame, new_in)
            new_in = min(self._drag_start_layer_out - 1, new_in)
            new_offset = self._drag_start_layer_offset + (
                new_in - self._drag_start_layer_in
            )
            self._drag_preview_layer_in = new_in
            self._drag_preview_offset = new_offset
        elif self._drag_kind == "out":
            new_out = self._drag_start_layer_out + delta_frames
            new_out = min(layer.sequence.last_frame, new_out)
            new_out = max(self._drag_start_layer_in + 1, new_out)
            master_out = layer.offset + (new_out - self._drag_start_layer_in)
            snapped_master = snap_master_frame(
                master_out, geom, self._snap_targets(exclude_self=True),
                snap_dist,
            )
            new_out = self._drag_start_layer_out + (snapped_master - master_out)
            new_out = min(layer.sequence.last_frame, new_out)
            new_out = max(self._drag_start_layer_in + 1, new_out)
            self._drag_preview_layer_out = new_out
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._drag_kind is None:
            return
        kind = self._drag_kind
        if not self._has_moved:
            # Pure click → emit focus request. No mutation.
            self.focus_requested.emit(self._layer.id)
        else:
            if kind == "body" and self._drag_preview_offset is not None:
                self.offset_changed.emit(
                    self._layer.id, int(self._drag_preview_offset),
                )
            elif (
                kind == "in"
                and self._drag_preview_layer_in is not None
                and self._drag_preview_offset is not None
            ):
                # Atomic IN trim: layer_in and offset are committed
                # together so a single LayerStack.update fires one
                # ``layer_modified`` signal (one cache invalidation).
                self.trim_in_changed.emit(
                    self._layer.id,
                    int(self._drag_preview_layer_in),
                    int(self._drag_preview_offset),
                )
            elif kind == "out" and self._drag_preview_layer_out is not None:
                self.layer_out_changed.emit(
                    self._layer.id, int(self._drag_preview_layer_out),
                )
        # Reset.
        self._drag_kind = None
        self._drag_preview_offset = None
        self._drag_preview_layer_in = None
        self._drag_preview_layer_out = None
        self._has_moved = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()
        event.accept()

    # ------------------------------------------------------------------ Internals

    def _geometry(self) -> BarGeometry:
        return BarGeometry(
            width=self.width(),
            master_first=self._master_first,
            master_last=self._master_last,
        )

    def _hit_test(self, x: float) -> DragKind | None:
        """Classify a pointer x within the widget into in/out/body."""
        layer = self._layer
        geom = self._geometry()
        x1 = geom.frame_to_x(layer.master_start)
        x2 = geom.frame_to_x(layer.master_end)
        # Out-of-bar clicks return None (panel may use them for focus).
        if x < x1 - HANDLE_GRAB or x > x2 + HANDLE_GRAB:
            return None
        if abs(x - x1) <= HANDLE_GRAB:
            return "in"
        if abs(x - x2) <= HANDLE_GRAB:
            return "out"
        return "body"

    def _snap_distance_in_frames(self, geom: BarGeometry) -> int:
        """Translate :data:`SNAP_PX` into frame units for the current widget width."""
        if geom.usable_w <= 0:
            return 0
        return max(0, round(SNAP_PX * geom.master_length / geom.usable_w))

    def _snap_targets(self, exclude_self: bool = False) -> list[int]:
        """Master-frame values to snap against. ``exclude_self`` is
        useful for trim drags where the layer's own edges are already
        the thing being moved — snapping to them is a no-op."""
        targets: list[int] = []
        if self._playhead is not None:
            targets.append(self._playhead)
        if self._master_in is not None:
            targets.append(self._master_in)
        if self._master_out is not None:
            targets.append(self._master_out)
        targets.extend(self._snap_edges)
        if not exclude_self:
            targets.append(self._layer.master_start)
            targets.append(self._layer.master_end)
        return targets
