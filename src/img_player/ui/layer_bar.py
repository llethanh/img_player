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

from PySide6.QtCore import QPoint, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFontMetrics,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPen,
)
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget

from img_player.layers import Layer
from img_player.ui.theme import F, H

# ---------------------------------------------------------------- Constants

# Pixel-space styling tokens. Tuned visually against the Studio
# Dark palette; adjust here if the row height ever changes.
HANDLE_W = 8           # vertical handle thickness in px
HANDLE_GRAB = 12       # extra hit-test margin around handles —
                       # generous so the OUT handle pinned at the
                       # bar's right edge isn't fiddly to click.
SNAP_PX = 6            # snap distance in screen pixels
PADDING_X = 0          # left/right inner padding. Zero so a layer
                       # that starts at ``master_first`` has its bar
                       # fill flush with the row's left edge — any
                       # positive value leaves a thin strip of panel
                       # background showing, which reads as a black
                       # gap before / after the layer.
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
        # "Frame as point" convention: length = number of *steps*
        # between master_first and master_last, not the count of
        # frames. This places frame N at fraction
        # ``(N - master_first) / master_length`` of the usable width,
        # so frame_to_x(master_first) == 0 and frame_to_x(master_last)
        # == usable_w — i.e. the bar's IN/OUT handles sit pixel-flush
        # with the timeline's first and last ticks. No more
        # half-a-slot drift between widgets.
        return max(1, self.master_last - self.master_first)

    def frame_to_x(self, master_frame: int) -> float:
        """Map a master-frame index to its widget pixel x-coordinate.

        ``master_first`` lands at ``PADDING_X`` (the bar's drawable
        left edge); ``master_last`` lands at ``PADDING_X +
        usable_w``. Layers covering the full master range therefore
        have their visible bar fill spanning the entire drawable
        area, with their handles aligned exactly on the
        corresponding timeline ticks.

        Out-of-range frames return clamped x; callers that care about
        the distinction should check ``master_first <= f <= master_last``.
        """
        normalized = (master_frame - self.master_first) / self.master_length
        return PADDING_X + normalized * self.usable_w

    def x_to_frame(self, x: float) -> int:
        """Inverse of :meth:`frame_to_x` — rounded to the nearest
        master-frame."""
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
    # Live preview during a body drag — fires on every mouseMove that
    # advances the offset. The panel listens to propagate the delta
    # to peer bars in a multi-select group, so the user sees the
    # whole group slide in real time instead of only the dragged bar.
    # Cleared on release (``offset_changed``) or drag cancel.
    offset_preview_changed = Signal(str, int)  # (layer_id, preview_offset)
    # Cleared after offset_changed / drag cancel — panel resets peer
    # previews. Emitted with the source layer id so the panel knows
    # which group to clean up.
    offset_preview_cleared = Signal(str)
    # Mirror of :attr:`LayerRow.row_clicked` — the panel uses it to
    # drive the multi-select state machine even when the press lands
    # on the bar's body / handles (which ``accept()`` the event and
    # so don't bubble back to the row's own ``mousePressEvent``).
    # ``kind`` is one of ``"single"`` / ``"ctrl"`` / ``"shift"``.
    row_clicked = Signal(str, str)
    # IN handle drag: standard NLE convention — the LEFT edge of the
    # visible bar moves while the RIGHT edge stays put. That requires
    # changing both ``layer_in`` and ``offset`` by the same delta in
    # one shot, so we carry them together and route to a single
    # ``LayerStack.update`` (= one ``layer_modified`` signal, one
    # cache invalidation).
    trim_in_changed = Signal(str, int, int)   # (layer_id, new_layer_in, new_offset)
    layer_out_changed = Signal(str, int)      # (layer_id, new_layer_out)
    # Still-image right-edge drag: extends/contracts ``still_hold_frames``
    # rather than ``layer_out`` (stills have a 1-frame source range, so
    # the standard out-trim math collapses). Emitted instead of
    # ``layer_out_changed`` when the dragged layer is a still.
    still_hold_changed = Signal(str, int)     # (layer_id, new_still_hold_frames)
    # Anywhere on the bar, single-click without drag → focus the layer
    # (= same as clicking the row). Lets the row delegate without
    # having to forward QMouseEvent itself.
    focus_requested = Signal(str)
    # Vertical-dominant drag started inside the bar — the user wants
    # to reorder the row, not adjust ``offset``. Carries the global
    # cursor position so the row can compute the drag pixmap hot
    # spot (= where the user grabbed the row, regardless of which
    # column they pressed on).
    reorder_drag_requested = Signal(QPoint)

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
        # Still-only: live preview of ``still_hold_frames`` while the
        # right handle is being dragged on a still layer. ``None`` for
        # sequence/video layers (their out-handle drag uses
        # ``_drag_preview_layer_out`` instead).
        self._drag_preview_still_hold: int | None = None
        # External preview offset pushed by the panel during a peer's
        # drag (= "this bar is part of a multi-select group, the user
        # is dragging another bar in the group, here's what your
        # offset should look like during that drag"). Takes priority
        # over the local committed offset in paint, but yields to the
        # local ``_drag_preview_offset`` when this bar is itself
        # being dragged. Cleared by the panel on the source bar's
        # commit / cancel.
        self._external_preview_offset: int | None = None
        # Multi-select membership flag — set by the panel via
        # ``set_in_selection``. Drives the deferred-single-click
        # logic: pressing on a bar that's already part of the group
        # without a modifier could be the start of a drag (move whole
        # group) or a plain click (demote to just this layer). We
        # defer the ``row_clicked("single")`` emit to mouseRelease so
        # the disambiguation happens only after we know whether a
        # drag took place.
        self._in_selection: bool = False
        self._pending_single_click: bool = False
        # Capture the press position so we can detect "click without
        # drag" (= focus, no offset change) by checking total motion.
        # ``_drag_start_y`` is also used to detect vertical-dominant
        # motion, which we hand off as a row-reorder drag instead of
        # an offset edit.
        self._drag_start_x: float = 0.0
        self._drag_start_y: float = 0.0
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

    def set_external_preview_offset(self, offset: int | None) -> None:
        """Push a preview offset coming from a peer's drag (multi-
        select group). ``None`` clears the override and the bar
        paints at its committed offset again. Cheap — just updates
        a member and triggers a repaint."""
        if offset == self._external_preview_offset:
            return
        self._external_preview_offset = offset
        self.update()

    def set_in_selection(self, on: bool) -> None:
        """Set the multi-select membership flag for this bar.

        Drives the deferred-single-click logic in
        :meth:`mousePressEvent` / :meth:`mouseReleaseEvent`. The
        panel calls this in lockstep with ``LayerRow.set_selected``
        so press-without-drag on a grouped layer correctly demotes
        the selection to just that layer at release time.
        """
        self._in_selection = bool(on)

    # ------------------------------------------------------------------ Painting

    def paintEvent(self, event: QPaintEvent) -> None:
        del event
        geom = self._geometry()
        layer = self._layer
        # Apply preview values when dragging so the visual moves
        # with the mouse before the commit. Local drag preview wins
        # over external (= "I'm being actively dragged"), and either
        # wins over the committed layer.offset.
        if self._drag_preview_offset is not None:
            offset = self._drag_preview_offset
        elif self._external_preview_offset is not None:
            offset = self._external_preview_offset
        else:
            offset = layer.offset
        layer_in = self._drag_preview_layer_in
        if layer_in is None:
            layer_in = layer.layer_in
        layer_out = self._drag_preview_layer_out
        if layer_out is None:
            layer_out = layer.layer_out
        master_start = offset
        # Stills carry their duration in ``still_hold_frames`` rather
        # than ``layer_out - layer_in`` (the trim window collapses to
        # the single source frame), so derive ``master_end`` from a
        # preview-aware hold value when this is a still.
        if layer.is_still:
            preview_hold = self._drag_preview_still_hold
            hold = (
                preview_hold
                if preview_hold is not None
                else layer.still_hold_frames
            )
            master_end = master_start + max(1, hold) - 1
        else:
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

        # Untrimmed-source ghost — when ``layer_in`` is past the
        # sequence's first frame OR ``layer_out`` is before its last
        # frame, the layer's source covers more master frames than
        # the trimmed bar shows. Drawing a translucent fill across the
        # *full* source extent lets the user see at a glance how much
        # head/tail material is hidden behind the trim, and gives a
        # visual handle for "drag IN/OUT back outwards to reveal it".
        # Mirrors the convention NLEs use (Premiere greyed-out
        # source extension, Resolve clip extension, etc.).
        source = layer.sequence
        # Master-frame the source.first_frame would land on if no
        # head trim — i.e. ``layer_in`` were source.first_frame.
        ghost_first = offset - (layer_in - source.first_frame)
        ghost_last = ghost_first + (source.last_frame - source.first_frame)
        if ghost_first < master_start or ghost_last > master_end:
            gx1 = geom.frame_to_x(ghost_first)
            gx2 = geom.frame_to_x(ghost_last)
            ghost_rect = QRectF(gx1, 4, max(2.0, gx2 - gx1), BAR_HEIGHT - 8)
            ghost_color = QColor(H.ACCENT)
            ghost_color.setAlpha(60)  # subtle — must read as "behind"
            painter.setBrush(ghost_color)
            painter.setPen(QPen(QColor(H.ACCENT_BRIGHT), 1, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(ghost_rect, BAR_RADIUS, BAR_RADIUS)

        # Layer fill — accent-tinted so it matches the focused-row
        # highlight when the user clicks. Slightly translucent to
        # let the snap-edge line peek through if it overlaps. With
        # the "frame as point" convention in BarGeometry, x1 and x2
        # are already on the bar's intended edges — no half-slot
        # extension required.
        x1 = geom.frame_to_x(master_start)
        x2 = geom.frame_to_x(master_end)
        bar_rect = QRectF(x1, 4, max(2.0, x2 - x1), BAR_HEIGHT - 8)
        bar_color = QColor(H.ACCENT)
        bar_color.setAlpha(200)
        painter.setBrush(bar_color)
        painter.setPen(QPen(QColor(H.ACCENT_BRIGHT), 1))
        painter.drawRoundedRect(bar_rect, BAR_RADIUS, BAR_RADIUS)
        # Still-image layers: overlay 45° diagonal hatching on the
        # solid fill so they're visually distinct from sequence /
        # video bands at a glance. The hatch rides on top of the
        # accent fill (same rounded-rect clip via a save/restore
        # clipPath dance) and uses a darker tint so the stripes
        # read as "shadow" against the accent rather than as a
        # second color. ``Qt.BDiagPattern`` = back-diagonal stripes
        # (top-left → bottom-right), which conventional NLEs use
        # for "freeze frame" / "still" markers.
        if getattr(layer, "is_still", False):
            painter.save()
            painter.setClipRect(bar_rect)
            hatch_color = QColor(H.ACCENT)
            # Darker than the fill so stripes read as recessed.
            hatch_color = hatch_color.darker(170)
            hatch_color.setAlpha(180)
            painter.setBrush(QBrush(hatch_color, Qt.BrushStyle.BDiagPattern))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(bar_rect)
            painter.restore()
            # Re-stroke the rounded outline on top so the hatch
            # doesn't poke past the rounded corners.
            painter.setBrush(Qt.BrushStyle.NoBrush)
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
        # interactive. Active drag highlights the handle being
        # dragged in accent so the user has feedback that the
        # gesture was registered.
        handle_default = QColor("#FFFFFF")
        handle_active = QColor(H.ACCENT_BRIGHT)
        painter.setPen(Qt.PenStyle.NoPen)
        in_handle = QRectF(x1, 4, HANDLE_W, BAR_HEIGHT - 8)
        out_handle = QRectF(x2 - HANDLE_W, 4, HANDLE_W, BAR_HEIGHT - 8)
        painter.setBrush(
            handle_active if self._drag_kind == "in" else handle_default,
        )
        painter.drawRect(in_handle)
        painter.setBrush(
            handle_active if self._drag_kind == "out" else handle_default,
        )
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
        # Capture modifiers FIRST so the panel's selection state
        # machine sees the click before the drag setup decides what
        # to do with it. Without this, dragging a non-selected
        # layer's body would skip "replace selection with this
        # layer" (the bar accepts the event, so the row's own
        # mousePressEvent never sees it).
        mods = event.modifiers()
        if mods & Qt.KeyboardModifier.ShiftModifier:
            click_kind = "shift"
        elif mods & Qt.KeyboardModifier.ControlModifier:
            click_kind = "ctrl"
        else:
            click_kind = "single"
        # Remember the modifier at press time so the release handler
        # can skip the unconditional ``focus_requested`` for Ctrl /
        # Shift clicks. Without this, a Ctrl-click that toggles a
        # layer OFF would be followed at release by a focus_requested
        # on the same layer — the panel's selection-consistency
        # invariant ("focus must be in the selection") then re-adds
        # it. End result: nothing happens visually. The panel's
        # ``_on_row_clicked`` already sets focus correctly for the
        # ctrl / shift cases, so the release-time focus push is
        # redundant there anyway.
        self._press_kind = click_kind
        self._pending_single_click = False
        if click_kind == "single" and self._in_selection:
            # Defer: a press on an already-grouped bar might still
            # become a drag (move whole group) or a plain click
            # (demote selection). Wait for mouseRelease /
            # mouseMove to disambiguate. Without this, the panel
            # would shrink the selection to {id} on press and the
            # subsequent drag would only move one bar.
            self._pending_single_click = True
        else:
            self.row_clicked.emit(self._layer.id, click_kind)
        x = event.position().x()
        kind = self._hit_test(x)
        # ``None`` means the click landed outside the layer's actual
        # range (= empty track area before / after the coloured
        # fill). Pre-v1.0.x we bailed out of the drag here, so the
        # user could only grab the layer where the orange fill was —
        # awkward when the layer was short or sat near the timeline
        # edge. Treat the entire bar width as a "body" hit so the
        # user can drag from anywhere along the row's length. The
        # offset math is identical: it computes the cursor delta in
        # master frames and applies it to ``_drag_start_layer_offset``,
        # which is independent of where on the bar the press landed.
        if kind is None:
            kind = "body"
        self._drag_kind = kind
        self._drag_start_x = x
        self._drag_start_y = event.position().y()
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
        # Active drag — but if the user is moving more vertically
        # than horizontally past the system drag threshold, they're
        # trying to reorder the row, not edit this layer's offset /
        # trim. Hand off to the row's reorder-drag path and bail out
        # of the in-progress bar drag so we don't ALSO commit an
        # offset change on release.
        y = event.position().y()
        dx = abs(x - self._drag_start_x)
        dy = abs(y - self._drag_start_y)
        if (
            self._drag_kind in ("body", "in", "out")
            and dy > QApplication.startDragDistance()
            and dy > dx
        ):
            # Reset preview so the bar paints back at its committed
            # position while the QDrag pixmap takes over.
            was_body_drag = (self._drag_kind == "body")
            self._drag_kind = None
            self._drag_preview_offset = None
            self._drag_preview_layer_in = None
            self._drag_preview_layer_out = None
            self._has_moved = False
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self.update()
            # Cancel any peer previews that were set during the
            # body-drag we just abandoned in favour of a reorder drag.
            if was_body_drag:
                self.offset_preview_cleared.emit(self._layer.id)
            self.reorder_drag_requested.emit(event.globalPosition().toPoint())
            event.accept()
            return
        if dx > 1.0:
            self._has_moved = True
            # Drag underway → cancel any deferred single-click
            # demote. Without this, releasing after a drag would
            # also shrink the selection at release time.
            self._pending_single_click = False
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
            # Live preview hook — the panel uses this to slide every
            # peer bar in a multi-select group in lockstep with the
            # dragged one. Without this signal, peers would only
            # snap to their new positions on release.
            self.offset_preview_changed.emit(self._layer.id, int(new_offset))
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
            # Stills: the right handle controls ``still_hold_frames``,
            # not ``layer_out`` (the trim window collapses to one
            # source frame for a single-file layer). Compute the new
            # hold from the master-frame delta and emit a separate
            # signal on release.
            if layer.is_still:
                new_hold = max(1, layer.still_hold_frames + delta_frames)
                # Snap on the visible right edge against neighboring
                # layer edges + the playhead, same as the sequence
                # path — gives consistent muscle memory.
                master_out = layer.master_start + new_hold - 1
                snapped_master = snap_master_frame(
                    master_out, geom, self._snap_targets(exclude_self=True),
                    snap_dist,
                )
                new_hold += snapped_master - master_out
                new_hold = max(1, new_hold)
                self._drag_preview_still_hold = new_hold
                self.update()
                event.accept()
                return
            # Standard NLE out-trim: the RIGHT edge of the bar moves,
            # the left edge stays put. Only ``layer_out`` changes;
            # ``offset`` and ``layer_in`` are untouched.
            new_out = self._drag_start_layer_out + delta_frames
            # Clamp to source range and keep at least one frame
            # between in/out.
            new_out = min(layer.sequence.last_frame, new_out)
            new_out = max(self._drag_start_layer_in + 1, new_out)
            # Snap on the *visible* right edge (= master_end =
            # layer.offset + (layer_out - layer_in)). The previous
            # version overwrote ``new_out`` with
            # ``drag_start + (snapped - master_out)`` which reset it
            # to the original whenever no snap target was within
            # range — symptom: the OUT handle wouldn't budge at all
            # outside snap zones.
            master_out = layer.offset + (new_out - layer.layer_in)
            snapped_master = snap_master_frame(
                master_out, geom, self._snap_targets(exclude_self=True),
                snap_dist,
            )
            new_out += snapped_master - master_out
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
            # If a deferred single-click was pending (= the user
            # pressed on a grouped bar without a modifier and
            # didn't drag), fire it NOW: that's the "demote to
            # this one layer" intent the deferred logic exists for.
            if self._pending_single_click:
                self.row_clicked.emit(self._layer.id, "single")
            # Ctrl / Shift presses already routed their focus through
            # the panel's ``_on_row_clicked`` selection state machine
            # at press time — emitting ``focus_requested`` again here
            # would re-set focus on the just-clicked layer and
            # ``_validate_selection_consistency`` would re-add it to
            # the selection. That's the bug the user reported as
            # "Ctrl-click un-selects then re-selects". Plain clicks
            # still need the focus push (toolbar / sidebar listeners
            # live off ``focus_requested``).
            if getattr(self, "_press_kind", "single") == "single":
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
            elif (
                kind == "out"
                and self._layer.is_still
                and self._drag_preview_still_hold is not None
            ):
                self.still_hold_changed.emit(
                    self._layer.id, int(self._drag_preview_still_hold),
                )
        # Reset.
        was_body_drag = (kind == "body")
        self._drag_kind = None
        self._drag_preview_offset = None
        self._drag_preview_layer_in = None
        self._drag_preview_layer_out = None
        self._drag_preview_still_hold = None
        self._has_moved = False
        self._pending_single_click = False
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.update()
        # Tell the panel to drop any peer previews it set during the
        # drag. Done AFTER the offset_changed emit so the cascade
        # handler reads the committed delta from a clean state.
        if was_body_drag:
            self.offset_preview_cleared.emit(self._layer.id)
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
