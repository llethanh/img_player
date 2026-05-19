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

Visual restyle (2026-Q2 redesign): each A/B picker is now a single
horizontal "pill row" (tag pill + layer index + filename + chevron)
that wraps the existing QComboBox so that the dropdown logic /
signal contract stays unchanged. The 3 compare modes live in a
sunken segmented well; the Solo-B / Swap-A↔B buttons keep their
existing signals but use the global ``btnToggle`` / ``btnIcon``
QSS variants.
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
from img_player.ui.theme import C, F, G, H, S

# Sizing knobs for the layer dropdowns. The combo's preferred width
# follows the longest currently-listed name (via Qt's
# ``setMinimumContentsLength``), clamped between a sensible floor
# (so a one-letter name doesn't collapse the combo) and a hard
# ceiling (so a pathological 100-char filename doesn't shove every
# other right-toolbar item off-screen).
_COMBO_MIN_CHARS = 12
_COMBO_MAX_CHARS = 36
_COMBO_MAX_PX = 360


# Tooltips for each mode button — labels live as icons now.
_MODE_TOOLTIPS: dict[str, str] = {
    MODE_VERTICAL: "Vertical split",
    MODE_HORIZONTAL: "Horizontal split",
    MODE_OPACITY: "Opacity blend",
}

# Maps each mode token to the brief's new segmented-control icon set.
_MODE_ICONS: dict[str, str] = {
    MODE_VERTICAL: "compare-vwipe",
    MODE_HORIZONTAL: "compare-hwipe",
    MODE_OPACITY: "compare-opacity",
}


@dataclass(frozen=True)
class _LayerOption:
    """One entry in either A or B dropdown.

    ``index`` is the 1-based stack position used to prefix the display
    name (``"1. shot.exr"``). Mirrors the numbering shown in the layer
    panel rows so the user can map "the layer I clicked in the panel"
    to "the entry in the A/B dropdown" without parsing the filename.
    """

    layer_id: str
    name: str
    index: int = 0


# ============================================================================
# SeamBar — replacement for QSlider, looks like a progress bar with %
# ============================================================================


class SeamBar(QWidget):  # type: ignore[misc]
    """Click/drag horizontal bar showing the seam position 0..1.

    Visually a sunken track with a warm-amber fill from the left to
    the current value and a thin bright thumb stroke at the seam.
    The brief calls for the fill to be ``ACC_TINT_30`` (translucent)
    with a 1 px right border in ``ACC_BRIGHT`` for the seam stroke,
    plus a soft outer glow. Cleaner read than a thumb-on-track slider
    for the A/B-wipe use case and plays better with the rest of the
    design system (no native QSlider chrome to fight).
    """

    seam_changed = Signal(float)

    BAR_W = 140
    BAR_W_MIN = 60
    BAR_H = G.CTRL_BUTTON_H  # 28 — matches the rest of the band

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Shrinkable horizontally so the band collapses gracefully when
        # the menu-bar row gets narrow (otherwise the corner widget
        # overlaps File/Edit/View). Height stays fixed.
        self.setMinimumSize(self.BAR_W_MIN, self.BAR_H)
        self.setMaximumHeight(self.BAR_H)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed,
        )
        self._value = 0.5
        self._dragging = False

    # ---- Sizing -------------------------------------------------------

    def sizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(self.BAR_W, self.BAR_H)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        return QSize(self.BAR_W_MIN, self.BAR_H)

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
        radius = float(G.RADIUS_SM)
        track_rect = rect.adjusted(0.5, 0.5, -0.5, -0.5)

        # Track — sunken well, hairline border. Reads as "below the
        # surface", which is the brief's visual idiom for slider /
        # seam tracks.
        painter.setPen(QPen(C.BORDER_SUB, 1))
        painter.setBrush(QBrush(C.BG_SUNKEN))
        painter.drawRoundedRect(track_rect, radius, radius)

        # Filled portion — translucent accent (ACC_TINT_30) so the
        # "A" letter at the left stays legible across the fill. The
        # right edge is the seam itself; we stroke it 1 px in
        # ACC_BRIGHT so the boundary reads as a hard wipe line.
        fill_w = max(0.0, rect.width() * self._value)
        if fill_w > 0:
            painter.save()
            track_path = QPainterPath()
            track_path.addRoundedRect(track_rect, radius, radius)
            painter.setClipPath(track_path)
            fill_rect = QRectF(rect.left(), rect.top(), fill_w, rect.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(C.ACC_TINT_30))
            painter.drawRect(fill_rect)
            # Seam stroke — bright accent line on the right edge of
            # the fill, plus a soft outer glow one px outside for the
            # "lit" feel the brief calls for.
            seam_x = fill_rect.right()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(C.ACC_GLOW, 3))
            painter.drawLine(
                int(seam_x), int(rect.top()),
                int(seam_x), int(rect.bottom()),
            )
            painter.setPen(QPen(C.ACC_BRIGHT, 1))
            painter.drawLine(
                int(seam_x), int(rect.top()),
                int(seam_x), int(rect.bottom()),
            )
            painter.restore()

        # A / B end-labels — accent at full opacity so the letters
        # stand out from the translucent fill underneath. Same
        # convention as the on-image overlay so the user reads "A is
        # the left bucket, B is the right bucket" everywhere compare
        # mode shows up.
        painter.setPen(QPen(C.ACC_BRIGHT))
        font = painter.font()
        font.setBold(True)
        font.setPixelSize(F.SIZE_PILL)
        painter.setFont(font)
        a_rect = QRectF(rect.left() + 4, rect.top(), 14, rect.height())
        b_rect = QRectF(rect.right() - 18, rect.top(), 14, rect.height())
        painter.drawText(a_rect, Qt.AlignmentFlag.AlignCenter, "A")
        painter.drawText(b_rect, Qt.AlignmentFlag.AlignCenter, "B")


# ============================================================================
# CompareBand — toolbar widget
# ============================================================================


def _build_layer_picker_qss() -> str:
    """QSS for the A / B picker container (tag pill + name combo).

    The picker is a QFrame styled to look like a single bordered
    surface; the QComboBox inside is "naked" (no border / no
    background) so the picker reads as one rectangular control even
    though the dropdown is still a stock QComboBox under the hood.
    """
    return (
        "QFrame#cmpPicker {"
        f"  background: {H.BG_SURFACE};"
        f"  border: 1px solid {H.BORDER_DEF};"
        f"  border-radius: {G.RADIUS_MD}px;"
        "}"
        "QFrame#cmpPicker:hover {"
        f"  border-color: {H.BORDER_STR};"
        "}"
        # Tag pill — small monospace letter with the accent tint, the
        # brief's signature "A / B" cartouche.
        "QLabel#cmpTag {"
        f"  background: {H.ACC_TINT_10};"
        f"  border: 1px solid {H.ACC_BORDER_ON};"
        f"  color: {H.ACC_BRIGHT};"
        f"  border-radius: {G.RADIUS_SM}px;"
        "  padding: 1px 6px;"
        f"  font-family: {F.FAMILY_MONO};"
        f"  font-size: {F.SIZE_MONO_LABEL}px;"
        "  font-weight: 700;"
        "}"
        # Index ("1.", "2.") — secondary mono.
        "QLabel#cmpIdx {"
        f"  color: {H.T_SEC};"
        f"  font-family: {F.FAMILY_MONO};"
        "  font-size: 10.5px;"
        "  font-weight: 500;"
        "  padding: 0;"
        "}"
        # The combo lives inside the picker. Strip its border / bg so
        # the picker reads as one bordered control. The dropdown
        # itself (QAbstractItemView) is styled by the global QSS.
        "QComboBox#cmpCombo {"
        "  background: transparent;"
        "  border: none;"
        f"  color: {H.T_PRI};"
        f"  font-family: {F.FAMILY_MONO};"
        f"  font-size: {F.SIZE_MONO_CODE}px;"
        "  padding: 0 2px 0 4px;"
        "  min-height: 22px;"
        "  max-height: 22px;"
        "}"
        "QComboBox#cmpCombo:hover {"
        "  background: transparent;"
        "  border: none;"
        "}"
        "QComboBox#cmpCombo::drop-down {"
        "  border: none;"
        "  width: 14px;"
        "}"
    )


def _build_mode_well_qss() -> str:
    """QSS for the segmented-control well that holds the mode buttons.

    The container is a sunken frame; the active button picks up the
    accent tint via the global ``btnToggle`` rule (we don't have to
    redefine it here).
    """
    return (
        "QFrame#cmpModeWell {"
        f"  background: {H.BG_SUNKEN};"
        f"  border: 1px solid {H.BORDER_SUB};"
        f"  border-radius: {G.RADIUS_MD}px;"
        "}"
        "QPushButton#cmpMode {"
        "  background: transparent;"
        "  border: 1px solid transparent;"
        f"  border-radius: {G.RADIUS_SM}px;"
        "  padding: 0;"
        "  min-height: 22px;"
        "  max-height: 22px;"
        "  min-width: 32px;"
        "  max-width: 32px;"
        "}"
        "QPushButton#cmpMode:hover {"
        f"  background: {H.BG_HOVER};"
        "}"
        "QPushButton#cmpMode:checked {"
        f"  background: {H.BG_SURFACE};"
        f"  border: 1px solid {H.ACC_BORDER_ON};"
        f"  color: {H.ACC_BRIGHT};"
        "}"
    )


class CompareBand(QFrame):  # type: ignore[misc]
    """``[A│1.│name.####.png ⌄] [⇄] [B│2.│name.####.png ⌄] │ [▌|▌|▒|B] │ A[──●──]B │ [A⇄B]``."""

    # User picked a different layer in dropdown A or B. The app
    # writes the new id into CompareState and triggers a redraw.
    layer_a_picked = Signal(str)
    layer_b_picked = Signal(str)
    # User clicked one of the mode buttons. Carries the mode token
    # (one of :data:`COMPARE_MODES`).
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

    BAND_HEIGHT = G.CTRL_BUTTON_H + 4  # 32 — leaves 2 px breathing top/bottom

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("compareBand")
        self.setFrameShape(QFrame.Shape.NoFrame)
        # Transparent — the band lives inside the right-side toolbar
        # alongside the menu bar (see ``MainWindow._build_menu``), so
        # it should inherit the menu-bar background rather than draw
        # its own raised panel.
        self.setStyleSheet(
            "QFrame#compareBand { background: transparent; }"
            + _build_layer_picker_qss()
            + _build_mode_well_qss()
        )
        self.setFixedHeight(self.BAND_HEIGHT)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(S.S_6)

        # ---- Layer A picker (composite pill) ----
        self._picker_a, self._combo_a = self._build_picker("A")
        self._combo_a.activated.connect(self._on_a_activated)
        layout.addWidget(self._picker_a)

        # ⇄ swap layers button (permute A/B in the dropdowns). Ghost
        # icon button — no background at rest, accent border on
        # hover via the global QSS.
        self._swap_layers_btn = QPushButton()
        self._swap_layers_btn.setObjectName("btnIcon")
        self._swap_layers_btn.setIcon(make_icon("swap-arrows", color=H.T_PRI))
        self._swap_layers_btn.setIconSize(QSize(16, 16))
        self._swap_layers_btn.setToolTip("Swap layers (Ctrl+W)")
        self._swap_layers_btn.clicked.connect(self.swap_layers_requested.emit)
        layout.addWidget(self._swap_layers_btn)

        # ---- Layer B picker ----
        self._picker_b, self._combo_b = self._build_picker("B")
        self._combo_b.activated.connect(self._on_b_activated)
        layout.addWidget(self._picker_b)

        # Thin separator between layer-pickers and mode well.
        layout.addWidget(self._build_separator())

        # ---- Mode buttons (segmented well) ----
        self._mode_well = QFrame()
        self._mode_well.setObjectName("cmpModeWell")
        self._mode_well.setFixedHeight(G.CTRL_BUTTON_H)
        well_layout = QHBoxLayout(self._mode_well)
        well_layout.setContentsMargins(2, 2, 2, 2)
        well_layout.setSpacing(1)

        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_buttons: dict[str, QPushButton] = {}
        for mode in COMPARE_MODES:
            btn = QPushButton()
            btn.setObjectName("cmpMode")
            btn.setIcon(make_icon(_MODE_ICONS[mode], color=H.T_PRI))
            btn.setIconSize(QSize(14, 14))
            btn.setCheckable(True)
            btn.setToolTip(_MODE_TOOLTIPS[mode])
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.clicked.connect(lambda _checked, m=mode: self.mode_picked.emit(m))
            self._mode_group.addButton(btn)
            self._mode_buttons[mode] = btn
            well_layout.addWidget(btn)
        layout.addWidget(self._mode_well)

        # Thin separator between mode well and seam bar.
        layout.addWidget(self._build_separator())

        # ---- Seam bar (custom progress-style indicator) ----
        self._seam_bar = SeamBar()
        self._seam_bar.set_value(0.5)
        self._seam_bar.seam_changed.connect(self.seam_changed.emit)
        layout.addWidget(self._seam_bar)

        # ---- Solo B toggle (always visible) ----
        # Checkable: when down, ``swap_showing_b`` is True and the
        # compose path returns full B regardless of the blend mode.
        # When up, the picked blend mode + slider apply normally.
        # ``btnToggle`` opts into the orange-tint :checked state
        # defined by the global QSS.
        self._swap_btn = QPushButton("A↔B")
        self._swap_btn.setObjectName("btnToggle")
        self._swap_btn.setCheckable(True)
        self._swap_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._swap_btn.setToolTip(
            "Show full B (override mode) — click again to return to the blend",
        )
        self._swap_btn.clicked.connect(self.swap_toggled.emit)
        layout.addWidget(self._swap_btn)

        # NB: the older ✕ close button used to live here. It was
        # removed because clicking the transport's compare toggle
        # (or pressing W) already exits the mode — the ✕ was
        # redundant with those existing surfaces. The
        # ``close_requested`` signal is kept on the class so a
        # future affordance can re-emit it, but the band never
        # fires it on its own anymore.

    # ------------------------------------------------------------------ Helpers

    def _build_picker(self, tag: str) -> tuple[QFrame, QComboBox]:
        """Build one A/B composite picker: ``[tag│idx│name ⌄]``.

        The picker is a QFrame styled like a single bordered surface
        via the ``#cmpPicker`` rule. The combo inside is "naked" (no
        bg / border) and carries the current layer name + a chevron.
        We keep a separate index QLabel that the public ``set_state``
        path can update without re-rendering the whole combo (the
        combo's items already include the "1. " prefix in their text
        for the dropdown row).
        """
        picker = QFrame()
        picker.setObjectName("cmpPicker")
        picker.setFixedHeight(G.CTRL_BUTTON_H)
        h = QHBoxLayout(picker)
        h.setContentsMargins(4, 2, 2, 2)
        h.setSpacing(S.S_4)

        # Tag pill ("A" / "B").
        tag_lbl = QLabel(tag)
        tag_lbl.setObjectName("cmpTag")
        tag_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        h.addWidget(tag_lbl)

        # Combo — naked, expands to fill. The combo's items carry the
        # "1. shot.exr" prefix already (set in ``set_available_layers``)
        # so the current-text reads the whole picker line.
        combo = QComboBox()
        combo.setObjectName("cmpCombo")
        combo.setMinimumContentsLength(_COMBO_MIN_CHARS)
        combo.setMaximumWidth(_COMBO_MAX_PX)
        combo.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed,
        )
        # Custom chevron icon — the bare QSS arrow image isn't
        # readable on the dark surface. We let Qt draw its default
        # arrow but override it in the combo's stylesheet would mean
        # shipping a PNG; instead we tolerate the default 8 px arrow
        # which renders OK on the surface tint. (A future polish pass
        # could swap in the ``chevron-down`` SVG via QProxyStyle.)
        h.addWidget(combo, 1)

        return picker, combo

    def _build_separator(self) -> QWidget:
        """Vertical hairline used between band groups.

        1 px wide × 22 px tall in BORDER_SUB — same recipe as the
        transport bar's :func:`_separator`. Wrapped in a container
        so the surrounding layout has a little breathing room around
        the thin line.
        """
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setFixedHeight(G.CTRL_SEP_H)
        sep.setStyleSheet(f"background:{H.BORDER_SUB};")
        return sep

    # ------------------------------------------------------------------ Public API

    def set_available_layers(
        self, options: list[_LayerOption], *,
        a_id: str | None, b_id: str | None,
    ) -> None:
        """Repopulate both dropdowns from the layer stack.

        Called whenever ``layers_changed`` fires. Block signals so
        the rebuild doesn't trigger ``layer_a_picked`` /
        ``layer_b_picked`` emissions for a non-user change.

        Each combo's ``minimumContentsLength`` is updated to fit the
        longest layer name (clamped to ``[_COMBO_MIN_CHARS,
        _COMBO_MAX_CHARS]``) so a 20-char filename displays fully
        without ellipsis when there's room. Qt's standard layout
        squeezes the combos when the band's parent (the right
        toolbar in ``MainWindow._build_menu``) runs out of horizontal
        space — no manual width clamp on our side.
        """
        # Build the display label up-front so the chars-budget knob
        # below sees the prefixed length, not the bare name — without
        # this a "1. " prefix would just ellipsise into nothing on
        # short-name layers.
        labels = [
            (f"{opt.index}. {opt.name}" if opt.index > 0 else opt.name)
            for opt in options
        ]
        longest = max((len(lbl) for lbl in labels), default=0)
        chars = max(_COMBO_MIN_CHARS, min(_COMBO_MAX_CHARS, longest + 2))
        for combo, current in (
            (self._combo_a, a_id), (self._combo_b, b_id),
        ):
            combo.blockSignals(True)
            combo.clear()
            for opt, label in zip(options, labels):
                combo.addItem(label, opt.layer_id)
            if current is not None:
                idx = combo.findData(current)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            combo.setMinimumContentsLength(chars)
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
