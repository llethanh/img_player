"""LayerPanel — collapsible list of :class:`Layer` rows below the timeline.

Drawn as a vertical stack of :class:`LayerRow` widgets, with a tiny
header (chevron + count) that lets the user fold the panel away to
reclaim viewport vertical space.

The panel is **always present** in the main window (per Q10/A) — it
just collapses to its header when no layer is loaded or when the user
hides it manually. Single-sequence playback shows one row, mirroring
the behaviour of multi-layer setups so there's no special-case UI
when going from 1 to 2 layers.

Phase 3 scope (this commit): rows + visibility toggle + reorder via
buttons. The bar visualisation on the master timeline (offset / trim
drag handles) lands in phase 4.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMimeData, QPoint, Qt, Signal
from PySide6.QtGui import QDrag, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from img_player.layers import Layer, LayerStack
from img_player.ui.layer_bar import LayerBar
from img_player.ui.theme import C, F, G, H, S


# ---------------------------------------------------------------- LayerRow

# Widget heights — kept tight so the panel stays unobtrusive when
# multiple layers stack up. Tuned to match the existing transport-bar
# button heights for visual continuity.
_ROW_HEIGHT = 22
_NUMBER_W = 26       # leftmost "#" column
_EYE_W = 26          # visibility toggle
_T_W = 22            # transparency (alpha_composite) toggle
_ALPHA_S_W = 28      # straight-alpha toggle (wider for the "αS" glyph)

# Hue tokens for the per-row alpha toggles — kept consistent with the
# previous transport-bar buttons so the user's mental "T = teal,
# αS = purple" mapping carries over.
_T_BTN_COLOR = "#5DC9D2"
_ALPHA_S_BTN_COLOR = "#B783D9"


# Public layout constants exposed so :class:`MasterTimelinePanel` can
# build an axis row whose timeline starts at exactly the same x as the
# layer bars. Exposing them avoids hard-coding the same arithmetic in
# two places — change the row's prefix here and the timeline gutter
# tracks automatically.
def layer_row_prefix_width() -> int:
    """Total pixel width before the bar in a :class:`LayerRow`.

    Mirrors the row's HBoxLayout: ``[content-margin-left, number,
    spacing, eye, spacing, T, spacing, αS, spacing-before-bar]``.
    Used by the master-timeline composite to size the gutter that
    sits left of the timeline so the timeline drawable ends up at
    the exact same x as each bar drawable.
    """
    return (
        S.SM + _NUMBER_W
        + S.SM + _EYE_W
        + S.SM + _T_W
        + S.SM + _ALPHA_S_W
        + S.SM
    )


def layer_row_right_margin() -> int:
    """Right content margin of a :class:`LayerRow`. The composite's
    axis row uses the same value so the timeline ends at the same x
    as each layer bar."""
    return S.SM


def _row_alpha_button(
    label: str,
    tooltip: str,
    *,
    color: str,
    width: int,
    height: int,
    font_size_pt: int | None = None,
) -> QToolButton:
    """Build a checkable per-row alpha toggle styled like the
    transport-bar's old T / αS buttons: coloured when active, dim
    grey + raised background when inactive. Lives at row scope now
    that ``Layer.alpha_composite`` / ``Layer.alpha_is_straight`` are
    per-layer state."""
    btn = QToolButton()
    btn.setText(label)
    btn.setFixedSize(width, height)
    btn.setCheckable(True)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.setToolTip(tooltip)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    fs = f"font-size: {font_size_pt}pt;" if font_size_pt is not None else ""
    btn.setStyleSheet(
        "QToolButton {"
        f"  color: {color};"
        f"  font-weight: 600;"
        f"  padding: 0;"
        f"  {fs}"
        "}"
        "QToolButton:!checked {"
        f"  color: {H.TEXT_DISABLED};"
        f"  background: {H.BG_RAISED};"
        "}"
    )
    return btn

# Custom mime type for intra-panel drag-and-drop reordering. Carries
# the layer id as plain UTF-8 bytes; the panel reads it on drop and
# routes to ``LayerStack.reorder``. The application/x-... prefix
# keeps drops from foreign sources from being silently accepted.
_LAYER_ID_MIME = "application/x-img-player-layer-id"


class LayerRow(QFrame):  # type: ignore[misc]
    """One row in the panel: number + eye + name + reorder buttons.

    Highlights itself when the layer it represents is the focused
    layer (= the one the user is currently editing). Clicking
    anywhere on the row sets focus.
    """

    # Mouse-press anywhere on the row asks the panel to focus this
    # layer. The panel forwards to LayerStack.set_focus.
    focus_requested = Signal(str)
    # Eye toggle — the panel just routes to LayerStack.toggle_visible.
    visibility_toggle_requested = Signal(str)
    # Suppr key on a focused row — the panel forwards to
    # LayerStack.remove. No confirmation: stack is in-memory and the
    # source files on disk stay put.
    delete_requested = Signal(str)
    # LayerBar drag commits — forwarded as-is to the panel which
    # routes to LayerStack.update.
    offset_changed = Signal(str, int)
    trim_in_changed = Signal(str, int, int)  # (id, new_layer_in, new_offset)
    layer_out_changed = Signal(str, int)
    # Per-layer alpha-mode toggles. Both carry the layer id + new
    # boolean so the panel can route to ``LayerStack.update`` without
    # scanning rows.
    transparency_toggled = Signal(str, bool)
    alpha_straight_toggled = Signal(str, bool)

    def __init__(self, layer: Layer, index: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layer_id = layer.id
        self.setFixedHeight(_ROW_HEIGHT)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAutoFillBackground(True)
        # Click anywhere = focus.
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Make the row keyboard-focusable so ``Delete`` can be handled
        # locally via ``keyPressEvent`` — no global shortcut, no
        # ambiguity about which layer the key targets.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Track press position so ``mouseMoveEvent`` can tell a click
        # (= focus) from a drag (= reorder) using ``startDragDistance``.
        self._press_pos: QPoint | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(S.SM, 0, S.SM, 0)
        layout.setSpacing(S.SM)

        # --- Layer number ----------------------------------------------
        self._number_label = QLabel(str(index + 1))
        self._number_label.setFixedWidth(_NUMBER_W)
        self._number_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._number_label.setFont(F.mono(F.SIZE_SM))
        layout.addWidget(self._number_label)

        # --- Eye / visibility ------------------------------------------
        # Plain text "eye" emoji works for now; can swap for an SVG
        # icon later. Checkable button for visual feedback.
        self._eye_btn = QToolButton()
        self._eye_btn.setFixedSize(_EYE_W, _ROW_HEIGHT - 4)
        self._eye_btn.setCheckable(True)
        self._eye_btn.setChecked(layer.visible)
        self._eye_btn.setText("👁" if layer.visible else "·")
        self._eye_btn.setToolTip("Show / hide this layer")
        self._eye_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._eye_btn.clicked.connect(self._on_eye_clicked)
        layout.addWidget(self._eye_btn)

        # --- Per-layer alpha toggles (T + αS) --------------------------
        # Used to live in the transport bar as global controls; moved
        # here once the underlying state became per-layer so the user
        # can flick each layer's alpha mode without re-focusing the
        # row first. Same teal / purple hues as the transport
        # buttons used to have, so the mental colour mapping carries.
        self._transparency_btn = _row_alpha_button(
            "T",
            "Transparency — alpha-composite this layer over what's "
            "below in the stack",
            color=_T_BTN_COLOR,
            width=_T_W,
            height=_ROW_HEIGHT - 4,
        )
        self._transparency_btn.setChecked(layer.alpha_composite)
        self._transparency_btn.toggled.connect(self._on_transparency_clicked)
        layout.addWidget(self._transparency_btn)

        self._alpha_straight_btn = _row_alpha_button(
            "αS",
            "Straight alpha (PNG / TGA convention). Off = premultiplied "
            "(EXR / VFX rendering default).",
            color=_ALPHA_S_BTN_COLOR,
            width=_ALPHA_S_W,
            height=_ROW_HEIGHT - 4,
            font_size_pt=9,
        )
        self._alpha_straight_btn.setChecked(layer.alpha_is_straight)
        self._alpha_straight_btn.toggled.connect(self._on_alpha_straight_clicked)
        layout.addWidget(self._alpha_straight_btn)

        # --- Layer bar (range + drag handles) ---------------------------
        # Replaces the plain filename label with an interactive
        # visualisation of the layer's master range. The bar handles
        # offset / trim drags itself; signals bubble back through
        # the row to the panel and finally LayerStack.update.
        self._bar = LayerBar(layer, parent=self)
        self._bar.offset_changed.connect(self.offset_changed.emit)
        self._bar.trim_in_changed.connect(self.trim_in_changed.emit)
        self._bar.layer_out_changed.connect(self.layer_out_changed.emit)
        self._bar.focus_requested.connect(self.focus_requested.emit)
        # Vertical drag inside the bar = "I want to reorder this row".
        # The bar emits with the global cursor position so we can map
        # back to row-local coords for the drag pixmap hotspot.
        self._bar.reorder_drag_requested.connect(self._on_bar_reorder_drag)
        layout.addWidget(self._bar, 1)

        # Reorder is now drag-and-drop on the row body itself; deletion
        # is keyboard-driven (Suppr on the focused row). The bar
        # therefore stretches all the way to the panel's right edge,
        # matching the PDPlayer reference.

        # Default unfocused look. ``set_focused(True)`` paints the
        # accent background to match PDPlayer / Nuke conventions.
        self._focused = False
        self._refresh_palette()

    # ------------------------------------------------------------------ Public API

    @property
    def layer_id(self) -> str:
        return self._layer_id

    def set_index(self, index: int) -> None:
        """Update the leftmost layer number after a reorder."""
        self._number_label.setText(str(index + 1))

    def set_visible_state(self, visible: bool) -> None:
        """Sync the eye button without retriggering the signal."""
        self._eye_btn.blockSignals(True)
        self._eye_btn.setChecked(bool(visible))
        self._eye_btn.setText("👁" if visible else "·")
        self._eye_btn.blockSignals(False)

    def set_name(self, name: str) -> None:
        # Layer.name is consulted by the bar's paintEvent; we just
        # poke it to repaint via set_layer.
        self._bar.set_layer(self._bar._layer)  # type: ignore[attr-defined]

    def update_layer(self, layer: Layer) -> None:
        """Push a fresh Layer reference into the bar — call after the
        underlying layer's offset/trim/visibility/alpha-flags have
        mutated. Keeps the per-row toggles (T / αS) in sync without
        firing their own signals."""
        self._bar.set_layer(layer)
        self.set_transparency_state(layer.alpha_composite)
        self.set_alpha_straight_state(layer.alpha_is_straight)

    def set_master_range(self, first: int, last: int) -> None:
        """Coordinate-space update from the panel."""
        self._bar.set_master_range(first, last)

    def set_playhead(self, master_frame: int | None) -> None:
        self._bar.set_playhead(master_frame)

    def set_master_in_out(
        self, in_frame: int | None, out_frame: int | None,
    ) -> None:
        self._bar.set_master_in_out(in_frame, out_frame)

    def set_snap_edges(self, edges: list[int]) -> None:
        self._bar.set_snap_edges(edges)

    def set_focused(self, on: bool) -> None:
        if on == self._focused:
            return
        self._focused = on
        self._refresh_palette()
        # When a row becomes focused, also pull keyboard focus so the
        # ``Delete`` shortcut on the row fires without needing an extra
        # tab into the panel. Skip on un-focus so we don't yank focus
        # away from whatever the user clicked next.
        if on:
            self.setFocus(Qt.FocusReason.OtherFocusReason)

    # ------------------------------------------------------------------ Internals

    def _refresh_palette(self) -> None:
        """Apply the focused/unfocused background tint."""
        if self._focused:
            self.setStyleSheet(
                f"QFrame {{ background: {H.ACCENT_DIM}; }}"
                f"QLabel {{ color: #FFF; }}"
            )
        else:
            self.setStyleSheet(
                "QFrame { background: transparent; }"
                "QLabel { color: #C8C8C8; }"
            )

    def _on_transparency_clicked(self, checked: bool) -> None:
        self.transparency_toggled.emit(self._layer_id, bool(checked))

    def _on_alpha_straight_clicked(self, checked: bool) -> None:
        self.alpha_straight_toggled.emit(self._layer_id, bool(checked))

    def set_transparency_state(self, on: bool) -> None:
        """Sync the T button without retriggering its toggled signal —
        used when the underlying ``Layer.alpha_composite`` mutates
        externally (session load, programmatic stack edit)."""
        self._transparency_btn.blockSignals(True)
        try:
            self._transparency_btn.setChecked(bool(on))
        finally:
            self._transparency_btn.blockSignals(False)

    def set_alpha_straight_state(self, on: bool) -> None:
        self._alpha_straight_btn.blockSignals(True)
        try:
            self._alpha_straight_btn.setChecked(bool(on))
        finally:
            self._alpha_straight_btn.blockSignals(False)

    def _on_eye_clicked(self) -> None:
        # Toggle the glyph immediately for snappy feedback; the
        # actual mutation flows back via the LayerStack signal.
        new_visible = self._eye_btn.isChecked()
        self._eye_btn.setText("👁" if new_visible else "·")
        self.visibility_toggle_requested.emit(self._layer_id)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # Click anywhere on the row (outside its sub-widgets) =
        # focus this layer + remember the press position so the next
        # ``mouseMoveEvent`` can decide whether the user is starting a
        # reorder drag.
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self.focus_requested.emit(self._layer_id)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        # Initiate a QDrag once the user has moved beyond the system's
        # drag-start distance — Qt's standard threshold avoids
        # accidental drags from imperceptible mouse jitter on click.
        if (
            self._press_pos is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            distance = (event.position().toPoint() - self._press_pos).manhattanLength()
            if distance >= QApplication.startDragDistance():
                self._start_reorder_drag()
                self._press_pos = None
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._press_pos = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_requested.emit(self._layer_id)
            event.accept()
            return
        super().keyPressEvent(event)

    def _on_bar_reorder_drag(self, global_pos: QPoint) -> None:
        """Hand off from a vertical drag started inside the LayerBar.
        Compute a row-local press position so the drag pixmap stays
        anchored where the user grabbed the row, then kick off the
        same QDrag the row's own mouseMoveEvent uses."""
        local = self.mapFromGlobal(global_pos)
        # Clamp inside the row rect so the hotspot can't end up
        # outside the pixmap (would offset the cursor weirdly).
        local.setX(max(0, min(self.width() - 1, local.x())))
        local.setY(max(0, min(self.height() - 1, local.y())))
        self._press_pos = local
        self._start_reorder_drag()
        self._press_pos = None

    def _start_reorder_drag(self) -> None:
        """Kick off a QDrag carrying this row's layer id. The
        :class:`LayerPanel` accepts the drop and calls
        :meth:`LayerStack.reorder` based on the cursor's Y position."""
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData(_LAYER_ID_MIME, self._layer_id.encode("utf-8"))
        drag.setMimeData(mime)
        # A snapshot of the row gives the user clear feedback during
        # the drag — they see what they're moving, not a generic
        # cursor.
        pixmap = self.grab()
        drag.setPixmap(pixmap)
        # Anchor the pixmap so the cursor stays exactly where the user
        # grabbed the row — without this, Qt centres the pixmap on the
        # cursor and the bar appears to jump left at drag-start. Falls
        # back to a sensible centre if for some reason no press
        # position was recorded.
        if self._press_pos is not None:
            drag.setHotSpot(self._press_pos)
        else:
            drag.setHotSpot(QPoint(pixmap.width() // 4, pixmap.height() // 2))
        drag.exec(Qt.DropAction.MoveAction)


# ---------------------------------------------------------------- LayerPanel


_HEADER_H = 22
_PANEL_BG = "#0E0F12"
_PANEL_HEADER_BG = "#16181D"


class _RowsHost(QWidget):  # type: ignore[misc]
    """Container for :class:`LayerRow` widgets that accepts drops to
    reorder them. Lives inside :class:`LayerPanel` and forwards every
    drop to the panel via ``parent()`` — kept tiny on purpose so the
    panel's API stays the canonical entry point for stack mutations.
    """

    def __init__(self, panel: "LayerPanel") -> None:
        super().__init__(panel)
        self._panel = panel
        self.setAcceptDrops(True)
        # Thin horizontal accent line shown during a row-reorder drag
        # to preview where the row will land. A child QFrame so it
        # paints on top of the row widgets (a paintEvent on the host
        # itself would be hidden by them — children paint last).
        self._drop_indicator = QFrame(self)
        self._drop_indicator.setFixedHeight(2)
        self._drop_indicator.setStyleSheet(
            f"background: {H.ACCENT_BRIGHT};"
        )
        self._drop_indicator.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
        )
        self._drop_indicator.hide()

    def dragEnterEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasFormat(_LAYER_ID_MIME):
            event.acceptProposedAction()
            self._update_drop_indicator(event)
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.mimeData().hasFormat(_LAYER_ID_MIME):
            event.acceptProposedAction()
            self._update_drop_indicator(event)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._drop_indicator.hide()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        self._drop_indicator.hide()
        mime = event.mimeData()
        if not mime.hasFormat(_LAYER_ID_MIME):
            event.ignore()
            return
        layer_id = bytes(mime.data(_LAYER_ID_MIME)).decode("utf-8")
        # Resolve the drop position to a stack index. We compare the
        # cursor's Y to each row's vertical centre: if the drop is
        # above the centre we land *above* that row, else below.
        y = event.position().y() if hasattr(event, "position") else event.pos().y()
        target_index = self._index_for_y(y)
        self._panel._on_row_dropped(layer_id, target_index)
        event.acceptProposedAction()

    def _update_drop_indicator(self, event) -> None:  # type: ignore[no-untyped-def]
        """Position the insertion line at the gap matching the current
        cursor Y. Idx 0 = above the first row, idx N = below the last."""
        y = event.position().y() if hasattr(event, "position") else event.pos().y()
        idx = self._index_for_y(y)
        rows = list(self._panel._rows.values())
        if not rows:
            self._drop_indicator.hide()
            return
        if idx >= len(rows):
            line_y = rows[-1].geometry().bottom() - 1
        else:
            line_y = rows[idx].geometry().top()
        self._drop_indicator.setGeometry(
            0, max(0, line_y - 1), self.width(), 2,
        )
        self._drop_indicator.raise_()
        self._drop_indicator.show()

    def _index_for_y(self, y: float) -> int:
        layers = list(self._panel._rows.values())
        for i, row in enumerate(layers):
            geom = row.geometry()
            if y < geom.top() + geom.height() / 2.0:
                return i
        return len(layers)


class LayerPanel(QFrame):  # type: ignore[misc]
    """Collapsible list of LayerRows + a header with a chevron toggle.

    Reads from a :class:`LayerStack` and rebuilds itself on every
    composition change. The widget is owned by :class:`MainWindow`,
    parented under the timeline.
    """

    def __init__(
        self,
        stack: LayerStack,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._stack = stack
        self._collapsed = False
        self._rows: dict[str, LayerRow] = {}

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"LayerPanel {{ background: {_PANEL_BG}; }}")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Header (collapse chevron + count) --------------------------
        self._header = self._build_header()
        outer.addWidget(self._header)

        # --- Rows container --------------------------------------------
        # Plain QWidget that hosts a QVBoxLayout populated dynamically.
        # When collapsed, this widget is hidden — the header alone
        # remains visible at ``_HEADER_H`` px.
        self._rows_host = _RowsHost(self)
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(0)
        outer.addWidget(self._rows_host, 1)

        self._stack.layers_changed.connect(self._rebuild)
        self._stack.visibility_changed.connect(self._on_visibility_changed)
        self._stack.layer_modified.connect(self._on_layer_modified)
        self._stack.focus_changed.connect(self._on_focus_changed)

        self._rebuild()

    # ------------------------------------------------------------------ Public API

    def set_collapsed(self, on: bool) -> None:
        """Show / hide the rows; header stays visible."""
        if on == self._collapsed:
            return
        self._collapsed = bool(on)
        self._rows_host.setVisible(not self._collapsed)
        self._chevron_btn.setText("▸" if self._collapsed else "▾")

    def is_collapsed(self) -> bool:
        return self._collapsed

    # ------------------------------------------------------------------ Internals

    def _build_header(self) -> QWidget:
        header = QFrame(self)
        header.setFixedHeight(_HEADER_H)
        header.setStyleSheet(f"QFrame {{ background: {_PANEL_HEADER_BG}; }}")
        h = QHBoxLayout(header)
        h.setContentsMargins(S.SM, 0, S.SM, 0)
        h.setSpacing(S.SM)

        self._chevron_btn = QToolButton(header)
        self._chevron_btn.setFixedSize(18, 18)
        self._chevron_btn.setText("▾")
        self._chevron_btn.setToolTip("Show / hide layers")
        self._chevron_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._chevron_btn.clicked.connect(
            lambda: self.set_collapsed(not self._collapsed),
        )
        h.addWidget(self._chevron_btn)

        title = QLabel("Layers")
        title.setStyleSheet("color: #B0B0B0;")
        title.setFont(F.ui(F.SIZE_SM))
        h.addWidget(title)

        h.addStretch(1)

        self._count_label = QLabel("0")
        self._count_label.setStyleSheet("color: #707070;")
        self._count_label.setFont(F.mono(F.SIZE_XS))
        self._count_label.setToolTip("Number of layers")
        h.addWidget(self._count_label)

        return header

    def _rebuild(self) -> None:
        """Throw away every row and rebuild from the stack snapshot.

        Cheap because LayerRow construction is just a few QLabels;
        if profiling ever shows this hot we can switch to in-place
        update of existing rows.
        """
        # Clear existing rows.
        while self._rows_layout.count():
            item = self._rows_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._rows.clear()

        layers = self._stack.layers()
        self._count_label.setText(str(len(layers)))
        focused_id = self._stack.focused_id

        for i, layer in enumerate(layers):
            row = LayerRow(layer, index=i, parent=self._rows_host)
            row.set_focused(layer.id == focused_id)
            row.focus_requested.connect(self._on_row_focus_requested)
            row.visibility_toggle_requested.connect(self._on_row_visibility_toggle)
            row.delete_requested.connect(self._on_row_delete_requested)
            row.transparency_toggled.connect(self._on_row_transparency_toggled)
            row.alpha_straight_toggled.connect(
                self._on_row_alpha_straight_toggled,
            )
            row.offset_changed.connect(self._on_row_offset_changed)
            row.trim_in_changed.connect(self._on_row_trim_in_changed)
            row.layer_out_changed.connect(self._on_row_layer_out_changed)
            self._rows_layout.addWidget(row)
            self._rows[layer.id] = row
        # Synchronise master-range + snap edges across every row
        # after construction so the bars draw at the right scale on
        # first paint.
        self._sync_bar_geometry()

        # Empty stack → still draw an empty hint so the panel
        # doesn't look broken.
        if not layers:
            empty = QLabel("No layer loaded — drop a sequence onto the viewer.")
            empty.setStyleSheet("color: #606060; padding: 6px 12px;")
            empty.setFont(F.ui(F.SIZE_XS))
            self._rows_layout.addWidget(empty)
            return

    def _on_visibility_changed(self, layer_id: str) -> None:
        layer = self._stack.find(layer_id)
        row = self._rows.get(layer_id)
        if layer is not None and row is not None:
            row.set_visible_state(layer.visible)

    def _on_layer_modified(self, layer_id: str) -> None:
        layer = self._stack.find(layer_id)
        row = self._rows.get(layer_id)
        if layer is not None and row is not None:
            row.update_layer(layer)
        # Master range may have shifted (offset / trim changed) →
        # every row needs to re-scale its bar.
        self._sync_bar_geometry()

    def _on_focus_changed(self, layer_id: str) -> None:
        for lid, row in self._rows.items():
            row.set_focused(lid == layer_id)

    def _on_row_focus_requested(self, layer_id: str) -> None:
        self._stack.set_focus(layer_id)

    def _on_row_visibility_toggle(self, layer_id: str) -> None:
        self._stack.toggle_visible(layer_id)

    def _on_row_dropped(self, layer_id: str, target_index: int) -> None:
        """Drag-drop reorder commit. ``target_index`` is the destination
        index *as if the source had not been removed*; we adjust for
        the upcoming pop so passing index N from "above the row at N"
        actually lands on N. ``LayerStack.reorder`` already clamps
        out-of-range values, so the math here only has to compensate
        for the pre-removal index shift.
        """
        layers = self._stack.layers()
        src_index = next(
            (i for i, l in enumerate(layers) if l.id == layer_id), None,
        )
        if src_index is None:
            return
        # The user dragged the row over its own position — no-op.
        if target_index == src_index or target_index == src_index + 1:
            return
        # When dropping below the current position, the implicit pop
        # shifts every later index down by one — so subtract one to
        # land where the user pointed.
        adjusted = target_index - 1 if target_index > src_index else target_index
        self._stack.reorder(layer_id, adjusted)

    def _on_row_delete_requested(self, layer_id: str) -> None:
        """Remove this layer. The cache invalidates the layer's
        master range via ``layers_changed`` and the app's
        ``_refresh_after_stack_change`` re-displays whatever's
        underneath (or black if nothing covers the playhead)."""
        self._stack.remove(layer_id)

    # --- Drag commits from LayerBar ----------------------------------

    def _on_row_transparency_toggled(self, layer_id: str, on: bool) -> None:
        self._stack.update(layer_id, alpha_composite=bool(on))

    def _on_row_alpha_straight_toggled(self, layer_id: str, on: bool) -> None:
        self._stack.update(layer_id, alpha_is_straight=bool(on))

    def _on_row_offset_changed(self, layer_id: str, new_offset: int) -> None:
        self._stack.update(layer_id, offset=new_offset)

    def _on_row_trim_in_changed(
        self, layer_id: str, new_in: int, new_offset: int,
    ) -> None:
        # Atomic update so a single ``layer_modified`` fires (one
        # cache invalidation) rather than two.
        self._stack.update(layer_id, layer_in=new_in, offset=new_offset)

    def _on_row_layer_out_changed(self, layer_id: str, new_out: int) -> None:
        self._stack.update(layer_id, layer_out=new_out)

    # --- Geometry coordination across rows ---------------------------

    def set_playhead(self, master_frame: int | None) -> None:
        """Push the current master playhead to every row's bar so it
        draws the snap-target line at the right place."""
        for row in self._rows.values():
            row.set_playhead(master_frame)

    def set_master_in_out(
        self, in_frame: int | None, out_frame: int | None,
    ) -> None:
        for row in self._rows.values():
            row.set_master_in_out(in_frame, out_frame)

    def sync_bar_geometry(self) -> None:
        """Public passthrough to :meth:`_sync_bar_geometry` for callers
        that mutated a layer's underlying ``sequence`` reference
        without firing a stack signal (e.g. ``cache.reload``). Lets
        them refresh the bars without awkward access through the
        underscore-prefixed method."""
        self._sync_bar_geometry()

    def _sync_bar_geometry(self) -> None:
        """Refresh master_range + per-row snap edges across all bars.

        Master-range used for the bars is the **broadest** range
        across all layers' *source potential* (= the master coords
        each layer's untrimmed source would cover) — not the live
        ``stack.master_range`` which collapses around the trimmed
        content. With the live range, single-layer trim was
        invisible: the bar would shrink during drag preview but
        bounce back to full-width on release because the master
        range followed the trim.

        Snap edges include every other layer's ``master_start /
        master_end`` so drags can stick to neighbours.
        """
        layers = self._stack.layers()
        if not layers:
            return
        first, last = self._broad_master_range()
        for layer in layers:
            row = self._rows.get(layer.id)
            if row is None:
                continue
            row.set_master_range(first, last)
            edges = []
            for other in layers:
                if other.id == layer.id:
                    continue
                edges.append(other.master_start)
                edges.append(other.master_end)
            row.set_snap_edges(edges)

    def broad_master_range(self) -> tuple[int, int]:
        """Public accessor for the broad master range used by the bars.

        Exposed so the main timeline can mirror the same scale — without
        this, the timeline is keyed on the loaded sequence's frame range
        while the layer bars are keyed on the union of source-potentials,
        and the two scrubbers move at different speeds. Returns ``(0, 0)``
        when the stack is empty.
        """
        if not self._stack.layers():
            return (0, 0)
        return self._broad_master_range()

    def _broad_master_range(self) -> tuple[int, int]:
        """Master range that covers each layer's full source extent.

        For a layer with offset ``o``, ``layer_in == sequence.first_frame
        + a``, and ``layer_out == sequence.last_frame - b``, the
        untrimmed source maps to master ``[o - a, o + (last - first - a)]``.
        Taking the union of every layer's untrimmed extent gives a
        range broad enough that trims and shifts produce visible
        bar changes — without it, dragging in single-layer mode
        looks like a no-op.
        """
        firsts: list[int] = []
        lasts: list[int] = []
        for layer in self._stack.layers():
            source_first = layer.sequence.first_frame
            source_last = layer.sequence.last_frame
            # Master-frame where source.first_frame would land if
            # ``layer_in`` were source_first (no trim from the head).
            source_first_master = (
                layer.offset - (layer.layer_in - source_first)
            )
            source_last_master = (
                source_first_master + (source_last - source_first)
            )
            firsts.append(source_first_master)
            lasts.append(source_last_master)
        if not firsts:
            return (0, 0)
        return (min(firsts), max(lasts))


# ---------------------------------------------------------------- MasterTimelinePanel


class MasterTimelinePanel(QFrame):  # type: ignore[misc]

    # File / folder dropped on the layers area — append as a new
    # top layer to the stack. The semantically opposite gesture of
    # ``ViewerWidget.replace_requested``; route accordingly in the
    # main window.
    add_layer_requested = Signal(Path)
    """Composite that owns the master timeline + the layer panel.

    Why this exists: the timeline and the layer rows used to be two
    siblings under :class:`MainWindow`'s central layout, and we had a
    fragile signal (``bar_inset_changed`` → ``set_content_insets``)
    that measured one widget post-layout to teach the other where to
    draw. Every time a paint convention drifted between the two
    (slot-centre vs slot-edge, padding values, range-bar extension)
    the user spotted a half-slot misalignment.

    Folding both into a single composite with ONE column model fixes
    the class of bug at its root: the timeline lives in an "axis row"
    whose left gutter is *exactly* the width of a :class:`LayerRow`'s
    prefix (number + eye + their spacings) and whose right margin
    matches the row's right margin. Both widgets are then forced to
    share the same horizontal axis by the layout system itself — no
    runtime measurement, no signal coordination.

    The wrapped :class:`Timeline` is still exposed via
    :attr:`timeline` so existing wiring (controller, scrub signals,
    cache-bar refresh) keeps working unchanged.
    """

    def __init__(
        self,
        timeline: QWidget,
        layer_panel: "LayerPanel",
        frame_display: QWidget | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._timeline = timeline
        self._layer_panel = layer_panel
        self._frame_display = frame_display

        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"MasterTimelinePanel {{ background: {_PANEL_BG}; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # --- Axis row: [gutter (= LayerRow prefix width)] + timeline ---
        # Switched from a contentsMargins-based gutter to a real
        # widget so the frame readout can live INSIDE that left
        # column instead of leaving it empty. Width is still
        # ``layer_row_prefix_width()`` so the timeline starts at the
        # exact same x as each LayerBar — alignment is unchanged.
        prefix_w = layer_row_prefix_width()
        right_margin = layer_row_right_margin()
        axis_row = QWidget(self)
        axis_layout = QHBoxLayout(axis_row)
        axis_layout.setContentsMargins(0, 0, right_margin, 0)
        axis_layout.setSpacing(0)

        gutter = QWidget(axis_row)
        gutter.setFixedWidth(prefix_w)
        gutter_layout = QHBoxLayout(gutter)
        gutter_layout.setContentsMargins(0, 0, 0, 0)
        gutter_layout.setSpacing(0)
        if frame_display is not None:
            gutter_layout.addStretch(1)
            gutter_layout.addWidget(frame_display)
            gutter_layout.addStretch(1)
        self._axis_gutter = gutter
        self._axis_gutter_layout = gutter_layout
        axis_layout.addWidget(gutter)
        axis_layout.addWidget(timeline, 1)
        self._axis_row = axis_row
        outer.addWidget(axis_row)

        # --- Layer panel below ---------------------------------------
        outer.addWidget(layer_panel, 1)

        # File-drop zone with an "ADD TO LAYERS" overlay. Folder /
        # file drops anywhere on the composite (axis row OR layer
        # rows) trigger ``add_layer_requested``. The :class:`_RowsHost`
        # already accepts a different mime type for intra-panel
        # reorder — those drops get ``hasUrls() == False`` and bubble
        # past our handler back into the row's existing logic.
        from img_player.ui.drop_zone import (
            ADD_LAYER_ACCENT, DropOverlay, install_file_drop_zone,
        )
        self._drop_overlay = DropOverlay(
            "ADD TO LAYERS", ADD_LAYER_ACCENT, self,
        )
        install_file_drop_zone(
            self, self._drop_overlay,
            lambda path: self.add_layer_requested.emit(path),
        )

    @property
    def timeline(self) -> QWidget:
        return self._timeline

    @property
    def layer_panel(self) -> "LayerPanel":
        return self._layer_panel

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        if self._drop_overlay.isVisible():
            self._drop_overlay.setGeometry(self.rect())
