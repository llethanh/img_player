"""Top-of-viewer band for the two-layer compare overlay.

UI surface for picking layers A / B, the compare mode (swap / vert /
horiz / opacity) and the seam position. Sits as a child of the
:class:`MainWindow` so it floats above the GL viewport in absolute
coords; visibility is toggled by ``app.py`` based on
``CompareState.enabled``.

Pure UI — emits signals, doesn't own the state. The owning app is
the single source of truth and re-feeds the band on every state
change so a programmatic update (session load, keyboard shortcut)
keeps the widget in sync without bespoke setters.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QRectF, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from img_player.compare.state import (
    COMPARE_MODES,
    MODE_HORIZONTAL,
    MODE_OPACITY,
    MODE_VERTICAL,
)
from img_player.ui.icons import make_icon
from img_player.ui.theme import C, G, H, S


# Tooltips for each mode button — labels live as icons now.
_MODE_TOOLTIPS: dict[str, str] = {
    MODE_VERTICAL: "Vertical split",
    MODE_HORIZONTAL: "Horizontal split",
    MODE_OPACITY: "Opacity blend",
}

# Maps each mode token to the icon template name.
_MODE_ICONS: dict[str, str] = {
    MODE_VERTICAL: "compare_vert",
    MODE_HORIZONTAL: "compare_horiz",
    MODE_OPACITY: "opacity",
}


@dataclass(frozen=True)
class _LayerOption:
    """One entry in either A or B dropdown."""

    layer_id: str
    name: str


# ============================================================================
# SeamBar — replacement for QSlider, looks like a progress bar with %
# ============================================================================


class SeamBar(QWidget):  # type: ignore[misc]
    """Click/drag horizontal bar showing the seam position 0..1.

    Visually a "loading bar": dark track, orange fill from the left
    to the current value, centred percentage label rendered with the
    inverse colour over the fill so the digits stay legible across
    the boundary. Cleaner read than a thumb-on-track slider for the
    A/B-wipe use case and plays better with the rest of the design
    system (no native QSlider chrome to fight).
    """

    seam_changed = Signal(float)

    BAR_W = 140
    BAR_H = G.INPUT_H

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(self.BAR_W, self.BAR_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed,
        )
        self._value = 0.5
        self._dragging = False

    # ---- Public API ---------------------------------------------------

    def value(self) -> float:
        return self._value

    def set_value(self, value: float) -> None:
        clamped = max(0.0, min(1.0, float(value)))
        if clamped == self._value:
            return
        self._value = clamped
        self.update()

    # ---- Mouse --------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._update_from_x(event.position().x())
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._dragging:
            self._update_from_x(event.position().x())
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()

    def _update_from_x(self, x: float) -> None:
        if self.width() <= 0:
            return
        new = max(0.0, min(1.0, float(x) / float(self.width())))
        if new != self._value:
            self._value = new
            self.update()
            self.seam_changed.emit(new)

    # ---- Paint --------------------------------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        del event
        from PySide6.QtGui import QPainterPath
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0, 0, self.width(), self.height())
        radius = float(G.RADIUS_MD)
        track_rect = rect.adjusted(0.5, 0.5, -0.5, -0.5)

        # Track (full width, dark fill).
        painter.setPen(QPen(C.BORDER_DEFAULT, 1))
        painter.setBrush(QBrush(C.BG_SURFACE))
        painter.drawRoundedRect(track_rect, radius, radius)

        # Filled portion (orange + bright outline matching the layer
        # cache-bar idiom). Clipped to the rounded track so the fill
        # keeps the corner radius on the left edge.
        fill_w = max(0.0, rect.width() * self._value)
        if fill_w > 0:
            painter.save()
            track_path = QPainterPath()
            track_path.addRoundedRect(track_rect, radius, radius)
            painter.setClipPath(track_path)
            fill_rect = QRectF(rect.left(), rect.top(), fill_w, rect.height())
            # Solid orange fill.
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(C.ACCENT))
            painter.drawRect(fill_rect)
            # Bright outline tracing the entire fill (= same idiom
            # the layer cache-bar uses to lift the warm fill off the
            # dark track). The clip-path crops the strokes that
            # would land outside the track's rounded shape, so the
            # outline naturally follows the left rounded corners
            # and stops dead at the seam on the right.
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(C.ACCENT_BRIGHT, 1.0))
            painter.drawRect(fill_rect.adjusted(0.5, 0.5, -0.5, -0.5))
            painter.restore()

        # "Split" label centred. Warm cream so it stays legible
        # across the boundary between the orange fill and the dark
        # unfilled remainder.
        painter.setPen(QPen(QColor("#FFE5C0")))
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(11)
        painter.setFont(font)
        painter.drawText(
            rect, Qt.AlignmentFlag.AlignCenter, "Split",
        )


# ============================================================================
# CompareBand — toolbar widget
# ============================================================================


class CompareBand(QFrame):  # type: ignore[misc]
    """``[A ▼]  ⇄  [B ▼]  [Vert] [Horiz] [Opacity]  ▰▰▰░░ 50%  A↔B  ✕``."""

    # User picked a different layer in dropdown A or B. The app
    # writes the new id into CompareState and triggers a redraw.
    layer_a_picked = Signal(str)
    layer_b_picked = Signal(str)
    # User clicked one of the four mode buttons. Carries the mode
    # token (one of :data:`COMPARE_MODES`).
    mode_picked = Signal(str)
    # Seam slider moved (0..100 → 0.0..1.0 on the receiver side).
    # Continuous: emits while dragging. The viewer redraws live.
    seam_changed = Signal(float)
    # User clicked the always-visible "Solo B" toggle. The receiver
    # flips ``CompareState.swap_showing_b`` and re-renders.
    swap_toggled = Signal()
    # User clicked ✕ — exit compare mode entirely.
    close_requested = Signal()
    # User clicked ⇄ — permute A and B.
    swap_layers_requested = Signal()

    BAND_HEIGHT = G.INPUT_H + 4

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("compareBand")
        self.setFrameShape(QFrame.Shape.NoFrame)
        # Transparent — the band lives inside the menu bar's corner
        # widget now, so it should inherit the menu-bar background
        # rather than draw its own raised panel. Applies the global
        # QPushButton / QComboBox styles uniformly with the rest of
        # the corner controls.
        self.setStyleSheet(
            "QFrame#compareBand { background: transparent; }"
            # Mode buttons (object-named below) — pull the accent
            # fill on :checked so they read as "active button" rather
            # than the dim default-checked state which looks like a
            # tab. Same pattern the TC pill in the timeline gutter
            # uses.
            "QPushButton#cmpMode:checked {"
            f"  background-color: {H.ACCENT};"
            f"  color: {H.BG_DEEP};"
            f"  border: 1px solid {H.ACCENT_BRIGHT};"
            "}"
            "QPushButton#cmpMode:checked:hover {"
            f"  background-color: {H.ACCENT_BRIGHT};"
            "}"
        )
        self.setFixedHeight(self.BAND_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(S.SM)

        # ---- Layer A dropdown ----
        layout.addWidget(QLabel("A"))
        self._combo_a = QComboBox()
        self._combo_a.setFixedHeight(G.INPUT_H)
        self._combo_a.setMinimumWidth(140)
        self._combo_a.activated.connect(self._on_a_activated)
        layout.addWidget(self._combo_a)

        # ⇄ swap layers button (permute A/B in the dropdowns).
        self._swap_layers_btn = QPushButton("⇄")
        self._swap_layers_btn.setFixedSize(G.INPUT_H, G.INPUT_H)
        self._swap_layers_btn.setToolTip("Swap layers (Ctrl+W)")
        self._swap_layers_btn.clicked.connect(self.swap_layers_requested.emit)
        layout.addWidget(self._swap_layers_btn)

        # ---- Layer B dropdown ----
        layout.addWidget(QLabel("B"))
        self._combo_b = QComboBox()
        self._combo_b.setFixedHeight(G.INPUT_H)
        self._combo_b.setMinimumWidth(140)
        self._combo_b.activated.connect(self._on_b_activated)
        layout.addWidget(self._combo_b)

        # ---- Mode buttons (mutually exclusive) ----
        # Icon buttons — text labels were ambiguous at small widths
        # and didn't convey the mode visually. Each icon is a small
        # SVG drawn from ``icons.py`` (compare_vert / compare_horiz
        # / opacity). Object-named so the local stylesheet can
        # target ``:checked`` without bleeding into every other
        # QPushButton in the band.
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}
        _icon_color = "#FFE5C0"  # warm cream — readable on the band's bg
        for mode in COMPARE_MODES:
            btn = QPushButton()
            btn.setObjectName("cmpMode")
            btn.setIcon(make_icon(_MODE_ICONS[mode], color=_icon_color))
            btn.setIconSize(QSize(18, 18))
            btn.setCheckable(True)
            btn.setFixedSize(G.INPUT_H + 8, G.INPUT_H)
            btn.setToolTip(_MODE_TOOLTIPS[mode])
            btn.clicked.connect(lambda _checked, m=mode: self.mode_picked.emit(m))
            self._mode_group.addButton(btn)
            self._mode_buttons[mode] = btn
            layout.addWidget(btn)

        # ---- Seam bar (custom progress-style indicator) ----
        self._seam_bar = SeamBar()
        self._seam_bar.set_value(0.5)
        self._seam_bar.seam_changed.connect(self.seam_changed.emit)
        layout.addWidget(self._seam_bar)

        # ---- Solo B toggle (always visible) ----
        # Checkable: when down, ``swap_showing_b`` is True and the
        # compose path returns full B regardless of the blend mode.
        # When up, the picked blend mode + slider apply normally.
        self._swap_btn = QPushButton("A↔B")
        self._swap_btn.setCheckable(True)
        self._swap_btn.setObjectName("cmpMode")
        self._swap_btn.setFixedHeight(G.INPUT_H)
        self._swap_btn.setToolTip(
            "Show full B (override mode) — click again to return to the blend",
        )
        self._swap_btn.clicked.connect(self.swap_toggled.emit)
        layout.addWidget(self._swap_btn)

        # ---- ✕ close ----
        self._close_btn = QPushButton("✕")
        self._close_btn.setFixedSize(G.INPUT_H, G.INPUT_H)
        self._close_btn.setToolTip("Exit compare mode (W)")
        self._close_btn.clicked.connect(self.close_requested.emit)
        layout.addWidget(self._close_btn)

    # ------------------------------------------------------------------ Public API

    def set_available_layers(
        self, options: list[_LayerOption], *,
        a_id: str | None, b_id: str | None,
    ) -> None:
        """Repopulate both dropdowns from the layer stack.

        Called whenever ``layers_changed`` fires. Block signals so
        the rebuild doesn't trigger ``layer_a_picked`` /
        ``layer_b_picked`` emissions for a non-user change.
        """
        for combo, current in (
            (self._combo_a, a_id), (self._combo_b, b_id),
        ):
            combo.blockSignals(True)
            combo.clear()
            for opt in options:
                combo.addItem(opt.name, opt.layer_id)
            if current is not None:
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def set_mode(self, mode: str) -> None:
        """Update the checked mode button without firing ``mode_picked``."""
        btn = self._mode_buttons.get(mode)
        if btn is None:
            return
        btn.blockSignals(True)
        btn.setChecked(True)
        btn.blockSignals(False)

    def set_swap_showing_b(self, on: bool) -> None:
        """Sync the Solo-B button's checked state from outside."""
        self._swap_btn.blockSignals(True)
        self._swap_btn.setChecked(bool(on))
        self._swap_btn.blockSignals(False)

    def set_seam(self, seam: float) -> None:
        """Sync the seam bar with an externally-changed seam (drag in
        viewport, keyboard nudge, session load). Clamped to [0, 1]."""
        self._seam_bar.blockSignals(True)
        self._seam_bar.set_value(seam)
        self._seam_bar.blockSignals(False)

    # ------------------------------------------------------------------ Internals

    def _on_a_activated(self, index: int) -> None:
        layer_id = self._combo_a.itemData(index)
        if isinstance(layer_id, str):
            self.layer_a_picked.emit(layer_id)

    def _on_b_activated(self, index: int) -> None:
        layer_id = self._combo_b.itemData(index)
        if isinstance(layer_id, str):
            self.layer_b_picked.emit(layer_id)
