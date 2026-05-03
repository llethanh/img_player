"""The :class:`AnnotationToolbar` — composite widget for the drawing tools.

Provides:

* Pen / eraser tool buttons (mutually exclusive radio behaviour).
* 7-color palette (fixed VFX-curated set, click to select).
* Size slider 1-50 px with live readout.
* Undo / redo buttons.
* Pin button that toggles between **float** and **dock** modes.

The toolbar is a single ``QWidget`` instance whose parent changes
based on the active mode:

* **Float** — parented directly to the GL viewport. Sits above the
  image, semi-transparent. Default position: ``(12, 12)`` from the
  viewport's top-left.
* **Dock** — embedded inside a ``QDockWidget`` anchored to the right
  of the main window. Cohérent with the existing color-panel dock.

The user toggles via the pin button. The active mode and (in float
mode) the position are persisted in :class:`Preferences`.

Slice 3 of the annotations feature — see
``docs/specs/2026-04-27-annotations-design.md``.
"""

from __future__ import annotations

from enum import Enum

from PySide6.QtCore import QPoint, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from img_player.annotate.overlay import ToolKind
from img_player.render.gl_viewport import GLViewport
from img_player.ui.icons import make_icon


class ToolbarMode(Enum):
    """Float (overlay on viewport) or Dock (right-side panel)."""

    FLOAT = "float"
    DOCK = "dock"


# 7 high-contrast review colors. Order matters for keyboard nav (left
# to right, top to bottom) — red is the default and goes first because
# "problem" is the most common review note.
PALETTE: tuple[str, ...] = (
    "#E84A4A",  # red — problem
    "#F5C842",  # yellow — warning
    "#5DC46C",  # green — ok / approved
    "#4A8DE8",  # blue — note / comment
    "#E8901C",  # orange — focus / accent
    "#FFFFFF",  # white
    "#000000",  # black
)
DEFAULT_COLOR = PALETTE[0]
DEFAULT_SIZE = 5.0
MIN_SIZE = 1.0
MAX_SIZE = 30.0

# Ephemeral fade duration presets — 3 fixed values mapped to a
# spec-driven label (court/moyen/long). Natural ascending order:
# left = court (2 s), middle = moyen (5 s, default), right = long (10 s).
# Growing dot sizes reinforce the ordering visually.
# The toolbar holds the integer index; the seconds-mapping lives
# here so a renaming or tuning round only touches one place.
EPHEMERAL_PRESETS_S: tuple[float, ...] = (2.0, 5.0, 10.0)
DEFAULT_EPHEMERAL_PRESET_INDEX = 1

# Pen stabilizer (Lazy Mouse) — 3 discrete strength levels mapped to
# an EMA factor in 0..1. ``factor=0`` = no smoothing (line follows
# cursor exactly). Higher factor = the smoothed cursor position
# catches up more slowly, producing the "trailing line" effect that
# filters hand tremor. The mapping lives here so tuning the strengths
# only touches one constant.
#
# Each tick of the 60 Hz catch-up timer pulls the smoothed point a
# fraction (1 - factor) of the way to the live cursor. So:
#   - level 0: alpha=1.0  → instant catch-up      (no trail)
#   - level 1: alpha=0.5  → half-distance / frame (subtle, ~80 ms trail)
#   - level 2: alpha=0.15 → strong drag           (~500 ms trail, very clean)
PEN_STABILIZER_FACTORS: tuple[float, ...] = (0.0, 0.5, 0.85)
PEN_STABILIZER_LABELS: tuple[str, ...] = ("Off", "Med", "Strong")
DEFAULT_PEN_STABILIZER_LEVEL = 0
# Cyan accent used for the toolbar's outer border when ephemeral
# mode is active — same value as the "blue note" swatch in PALETTE
# so the visual language stays internally consistent.
_EPHEMERAL_ACCENT = "#4A8DE8"


# Visual feedback for the action / mode buttons (pen, eraser, ephemeral
# toggle, undo, redo). Without this the default Qt rendering on a
# semi-transparent panel is almost invisible — the user couldn't tell
# what was selected. Cyan accent matches the rest of v0.4.1.
# - :hover  → soft white wash, hint that the button is clickable.
# - :checked → cyan fill + cyan border, the active tool / mode is
#   unambiguous.
# - :pressed → momentary cyan flash for the non-checkable buttons
#   (undo / redo) so the click registers visually.
# - :disabled → low-contrast grey, drives home that the button
#   doesn't do anything (e.g. eraser greyed in ephemeral mode).
_ACTION_BTN_NAMES = (
    "annotPenBtn",
    "annotEraserBtn",
    "annotEphemeralBtn",
    "annotUndoBtn",
    "annotRedoBtn",
)
_ACTION_BTN_QSS = (
    " ".join(f"QToolButton#{n}" for n in _ACTION_BTN_NAMES).replace(
        " ", ", "
    )
    + " {"
    "  background: transparent;"
    "  border: 1px solid transparent;"
    "  border-radius: 4px;"
    "  padding: 1px;"
    "}"
    + ", ".join(f"QToolButton#{n}:hover" for n in _ACTION_BTN_NAMES)
    + " {"
    "  background: rgba(255, 255, 255, 22);"
    "  border: 1px solid rgba(255, 255, 255, 36);"
    "}"
    + ", ".join(f"QToolButton#{n}:checked" for n in _ACTION_BTN_NAMES)
    + " {"
    f"  background: rgba(74, 141, 232, 70);"
    f"  border: 1px solid {_EPHEMERAL_ACCENT};"
    "}"
    "QToolButton#annotUndoBtn:pressed, QToolButton#annotRedoBtn:pressed {"
    f"  background: rgba(74, 141, 232, 110);"
    f"  border: 1px solid {_EPHEMERAL_ACCENT};"
    "}"
    + ", ".join(f"QToolButton#{n}:disabled" for n in _ACTION_BTN_NAMES)
    + " {"
    "  background: transparent;"
    "  border: 1px solid transparent;"
    "  color: #5A5A5E;"
    "}"
)


class _ColorSwatch(QToolButton):
    """A small square button filled with a single solid color.

    Acts as a checkable radio in the palette group. Renders its color
    via paintEvent + setChecked drawing — Qt's default styling for
    QPushButton/QToolButton would add a border that fights the
    saturated swatch colors at tiny sizes (16-20 px).
    """

    SIZE_PX = 20

    def __init__(self, color_hex: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color_hex = color_hex
        self.setCheckable(True)
        self.setFixedSize(self.SIZE_PX, self.SIZE_PX)
        self.setToolTip(color_hex)
        # Keep focus rectangles off — they look noisy against the
        # tiny saturated swatches.
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    @property
    def color_hex(self) -> str:
        return self._color_hex

    def paintEvent(self, event: object) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        # Fill.
        painter.setBrush(QColor(self._color_hex))
        if self.isChecked():
            # 2 px accent ring around the selected swatch — the only
            # way the user knows which color is active without reading
            # the live preview elsewhere.
            painter.setPen(QColor("#F5A830"))  # ACCENT_BRIGHT
        else:
            # 1 px subtle border so a pure-white swatch doesn't
            # disappear on a near-white panel background.
            painter.setPen(QColor(255, 255, 255, 60))
        # Circular swatches — feels more "color picker"-like than
        # rounded squares and visually distinct from the square tool
        # buttons (pen / eraser) just above.
        painter.drawEllipse(rect)


class _StrokePreview(QWidget):
    """Tiny widget that paints a wavy sample stroke in the current
    brush color and size.

    Sits at the top of the toolbar so the user sees immediate
    feedback when picking a color or moving the size slider — same
    spirit as Photoshop's brush preview, just a flat 80×36 strip.
    """

    _MARGIN = 6
    _AMPLITUDE = 6  # vertical excursion of the wavy line

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = DEFAULT_COLOR
        self._size = DEFAULT_SIZE
        self.setFixedHeight(28)
        self.setMinimumWidth(60)

    def set_color(self, color: str) -> None:
        if color == self._color:
            return
        self._color = color
        self.update()

    def set_size(self, size: float) -> None:
        if size == self._size:
            return
        self._size = size
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Map the stored size (1-50 in image-pixels) to a preview pen
        # width that stays readable in the small strip — clamped so
        # the wave's amplitude is always larger than the stroke.
        pen_w = max(1.5, min(self._size * 0.6, self._AMPLITUDE * 1.6))
        pen = QPen(QColor(self._color))
        pen.setWidthF(pen_w)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        w = self.width()
        h = self.height()
        x0 = self._MARGIN
        x3 = w - self._MARGIN
        y_mid = h / 2.0
        y_top = y_mid - self._AMPLITUDE
        y_bot = y_mid + self._AMPLITUDE
        # Three-loop wave: gives the user a sense of corner / curve
        # behaviour at the picked size + color.
        path = QPainterPath()
        path.moveTo(x0, y_mid)
        third = (x3 - x0) / 3.0
        path.quadTo(x0 + third * 0.5, y_top, x0 + third, y_mid)
        path.quadTo(x0 + third * 1.5, y_bot, x0 + third * 2.0, y_mid)
        path.quadTo(x0 + third * 2.5, y_top, x3, y_mid)
        painter.drawPath(path)


class AnnotationToolbar(QWidget):
    """Composite drawing toolbar — pen / eraser / palette / size / undo / redo / pin."""

    # ------------------------------------------------------------------ Signals

    tool_changed = Signal(object)
    """Emits :class:`ToolKind` when pen / eraser / none becomes active."""

    color_changed = Signal(str)
    """Emits the new hex color when a palette swatch is clicked."""

    size_changed = Signal(float)
    """Emits the new brush size (image-pixels) when the slider moves."""

    undo_requested = Signal()
    redo_requested = Signal()

    clear_requested = Signal()
    """Emits when the user clicks the Clear button — the app removes
    every stroke from the current frame."""

    mode_changed = Signal(object)
    """Emits :class:`ToolbarMode` when the pin toggles float ⇄ dock."""

    floating_pos_changed = Signal(int, int)
    """Emits ``(x, y)`` when the user finishes dragging the toolbar in
    float mode. App.py persists the position to preferences."""

    # ------------------------------------------------------------------ Ephemeral signals (v0.4.1)

    ephemeral_mode_changed = Signal(bool)
    """Emits ``True`` when the user activates ephemeral mode (the 👻
    toggle), ``False`` when they deactivate. App.py routes to
    ``overlay.set_ephemeral_mode``."""

    ephemeral_duration_changed = Signal(float)
    """Emits the new fade duration in seconds (one of the three values
    in :data:`EPHEMERAL_PRESETS_S`) when the user clicks one of the
    three preset dots. App.py routes to ``manager.set_duration``."""

    # ------------------------------------------------------------------ Pen stabilizer (Lazy Mouse)

    pen_stabilizer_level_changed = Signal(int)
    """Emits the new stabilizer level (0/1/2) when the user moves the
    Stabilizer slider. App.py routes to
    ``overlay.set_pen_stabilizer_factor`` and persists to prefs."""

    # ------------------------------------------------------------------ Lifecycle

    def __init__(
        self,
        gl_viewport: GLViewport,
        dock_wrapper: QDockWidget,
        *,
        initial_mode: ToolbarMode = ToolbarMode.FLOAT,
        initial_floating_pos: tuple[int, int] = (12, 12),
        initial_ephemeral_preset: int = DEFAULT_EPHEMERAL_PRESET_INDEX,
        initial_stabilizer_level: int = DEFAULT_PEN_STABILIZER_LEVEL,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._gl_viewport = gl_viewport
        self._dock_wrapper = dock_wrapper
        self._mode: ToolbarMode = initial_mode
        self._floating_pos: tuple[int, int] = initial_floating_pos

        # Active state (the source of truth — rendered widgets reflect
        # this, and external mutations go through the public setters).
        self._current_tool: ToolKind = ToolKind.NONE
        self._current_color: str = DEFAULT_COLOR
        self._current_size: float = DEFAULT_SIZE

        # Ephemeral mode (v0.4.1) — initially off. The preset index
        # comes from preferences (or its default).
        self._ephemeral_mode: bool = False
        if initial_ephemeral_preset not in (0, 1, 2):
            initial_ephemeral_preset = DEFAULT_EPHEMERAL_PRESET_INDEX
        self._ephemeral_preset_index: int = initial_ephemeral_preset

        # Pen stabilizer (Lazy Mouse) — initially off (level 0). The
        # value is restored from prefs on construction.
        if initial_stabilizer_level not in (0, 1, 2):
            initial_stabilizer_level = DEFAULT_PEN_STABILIZER_LEVEL
        self._stabilizer_level: int = initial_stabilizer_level

        # Drag state for moving the toolbar in float mode (we own the
        # title-bar drag because no QWidget gives this for free outside
        # of QDockWidget's float mode).
        self._drag_offset: QPoint | None = None

        self._build_ui()
        self._apply_mode_initial()

    # ------------------------------------------------------------------ Public API

    def mode(self) -> ToolbarMode:
        return self._mode

    def floating_pos(self) -> tuple[int, int]:
        return self._floating_pos

    def current_tool(self) -> ToolKind:
        return self._current_tool

    def current_color(self) -> str:
        return self._current_color

    def current_size(self) -> float:
        return self._current_size

    # ------------------------------------------------------------------ Ephemeral public API (v0.4.1)

    def is_ephemeral_mode(self) -> bool:
        return self._ephemeral_mode

    def ephemeral_preset_index(self) -> int:
        return self._ephemeral_preset_index

    def ephemeral_duration_seconds(self) -> float:
        return EPHEMERAL_PRESETS_S[self._ephemeral_preset_index]

    # ------------------------------------------------------------------ Pen stabilizer public API

    def pen_stabilizer_level(self) -> int:
        """Active stabilizer level: 0 (off), 1 (medium), 2 (strong)."""
        return self._stabilizer_level

    def pen_stabilizer_factor(self) -> float:
        """EMA factor matching the active level — what
        ``overlay.set_pen_stabilizer_factor`` consumes."""
        return PEN_STABILIZER_FACTORS[self._stabilizer_level]

    def set_pen_stabilizer_level(self, level: int, *, emit: bool = True) -> None:
        """Set the stabilizer level (0/1/2) and update the slider UI.

        Silent reject for out-of-range values. ``emit=False`` skips
        the signal — used by the ``__init__``-driven restore from
        prefs so we don't fire a no-op signal at boot.
        """
        if level not in (0, 1, 2) or level == self._stabilizer_level:
            return
        self._stabilizer_level = level
        if hasattr(self, "_stabilizer_slider"):
            self._stabilizer_slider.blockSignals(True)
            self._stabilizer_slider.setValue(level)
            self._stabilizer_slider.blockSignals(False)
            self._stabilizer_label.setText(PEN_STABILIZER_LABELS[level])
        if emit:
            self.pen_stabilizer_level_changed.emit(level)

    def set_ephemeral_mode(self, on: bool, *, emit: bool = True) -> None:
        """Toggle ephemeral mode (the ``👻`` button checked state).

        Side-effects on the toolbar UI:

        * the 👻 button's ``checked`` state mirrors ``on``,
        * the 3-preset bar appears (on) / collapses (off),
        * the toolbar's outer border re-styles (cyan accent on),
        * the pen glyph swaps ``✏️`` ⇄ ``👻``,
        * the eraser is disabled (on) / re-enabled (off).
        * if eraser was the active tool when activating, the active
          tool is auto-switched to ``NONE`` to avoid the inconsistent
          "greyed but checked" state.

        ``emit=False`` is for external callers that want to reflect
        a state change initiated elsewhere (e.g. the ``G`` keyboard
        shortcut going through the app, then back to the toolbar)
        without re-broadcasting.
        """
        on = bool(on)
        if on == self._ephemeral_mode:
            return
        self._ephemeral_mode = on

        # 1. Mirror the button's check state without retriggering its slot.
        self._ephemeral_btn.blockSignals(True)
        try:
            self._ephemeral_btn.setChecked(on)
        finally:
            self._ephemeral_btn.blockSignals(False)

        # 2. Show / hide the 3-preset duration bar.
        self._ephemeral_preset_row.setVisible(on)

        # 3. Re-style the outer border (cyan when on).
        self._apply_mode_styles()

        # 4. Swap the pen glyph + tooltip.
        if on:
            self._pen_btn.setText("👻")
            self._pen_btn.setToolTip(
                "Pen éphémère (P) — clic-glisser, le trait s'effacera tout seul"
            )
        else:
            self._pen_btn.setText("✏️")
            self._pen_btn.setToolTip("Pen (P) — clic-glisser pour dessiner")

        # 5. Eraser availability.
        self._eraser_btn.setEnabled(not on)
        # Inconsistent state guard: eraser checked while greyed.
        if on and self._current_tool == ToolKind.ERASER:
            self.set_current_tool(ToolKind.NONE)

        if emit:
            self.ephemeral_mode_changed.emit(on)

    def set_ephemeral_preset_index(self, index: int, *, emit: bool = True) -> None:
        """Pick one of the 3 fade-duration presets.

        ``index`` ∈ ``{0, 1, 2}`` mapping to ``EPHEMERAL_PRESETS_S``.
        Out-of-range values are silently clamped to the default
        (1 = moyen / 5 s) — same defensive pattern as
        ``set_current_size`` etc.
        """
        if index not in (0, 1, 2):
            index = DEFAULT_EPHEMERAL_PRESET_INDEX
        if index == self._ephemeral_preset_index:
            return
        self._ephemeral_preset_index = index
        # Mirror the radio's check state without firing its slot.
        for i, btn in enumerate(self._ephemeral_preset_btns):
            btn.blockSignals(True)
            try:
                btn.setChecked(i == index)
            finally:
                btn.blockSignals(False)
        if emit:
            self.ephemeral_duration_changed.emit(EPHEMERAL_PRESETS_S[index])

    def set_mode(self, mode: ToolbarMode) -> None:
        """Switch between FLOAT and DOCK. Re-parents the widget."""
        if mode == self._mode:
            return
        self._mode = mode
        if mode == ToolbarMode.FLOAT:
            # Detach from dock, reparent to viewport.
            self._dock_wrapper.setWidget(None)
            self._dock_wrapper.hide()
            self.setParent(self._gl_viewport)
            self.setWindowFlags(Qt.WindowType.SubWindow)
            self.move(*self._floating_pos)
            self.raise_()
            self.show()
            # Belt-and-braces: adjustSize forces the widget to shrink
            # to its sizeHint after show(). Without it, Qt sometimes
            # keeps a leftover larger geometry from the previous
            # parenting (e.g. the dock's vertical fill).
            self.adjustSize()
        else:  # DOCK
            self.setParent(None)
            self.setWindowFlags(Qt.WindowType.Widget)
            self._dock_wrapper.setWidget(self)
            self._dock_wrapper.show()
            self.show()
        self._apply_mode_styles()
        self.mode_changed.emit(mode)

    def set_current_tool(self, tool: ToolKind) -> None:
        """Programmatic tool switch (e.g. from a keyboard shortcut)."""
        if tool == self._current_tool:
            return
        self._current_tool = tool
        # Update button checked states without re-emitting.
        self._pen_btn.blockSignals(True)
        self._eraser_btn.blockSignals(True)
        try:
            self._pen_btn.setChecked(tool == ToolKind.PEN)
            self._eraser_btn.setChecked(tool == ToolKind.ERASER)
        finally:
            self._pen_btn.blockSignals(False)
            self._eraser_btn.blockSignals(False)
        self.tool_changed.emit(tool)

    def set_current_color(self, color_hex: str) -> None:
        if color_hex == self._current_color:
            return
        if color_hex not in PALETTE:
            return  # silently ignore non-palette colors
        self._current_color = color_hex
        for swatch in self._swatches:
            swatch.blockSignals(True)
            try:
                swatch.setChecked(swatch.color_hex == color_hex)
            finally:
                swatch.blockSignals(False)
        # Sync the wavy preview at the top so the user sees the
        # picked color immediately, before they even draw.
        if hasattr(self, "_stroke_preview"):
            self._stroke_preview.set_color(color_hex)
        self.color_changed.emit(color_hex)

    def set_current_size(self, size: float) -> None:
        size = max(MIN_SIZE, min(MAX_SIZE, float(size)))
        if size == self._current_size:
            return
        self._current_size = size
        self._size_slider.blockSignals(True)
        try:
            self._size_slider.setValue(int(round(size)))
        finally:
            self._size_slider.blockSignals(False)
        # Same reason as in set_current_color — keep the preview in
        # sync.
        if hasattr(self, "_stroke_preview"):
            self._stroke_preview.set_size(size)
        self._size_label.setText(f"{int(round(size))}px")
        self.size_changed.emit(size)

    # ------------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        # Outer column. Tight padding so the toolbar matches the
        # reference's compact width.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # --- Pin button (top of the toolbar, alone, centred). Emoji
        # 📌 — the user prefers the colorful native glyph here over
        # a custom monochrome SVG.
        self._pin_btn = QToolButton(self)
        self._pin_btn.setText("📌")
        self._pin_btn.setToolTip("Bascule float ⇄ dock")
        self._pin_btn.setFixedSize(26, 22)
        self._pin_btn.clicked.connect(self._on_pin_clicked)
        layout.addWidget(self._pin_btn, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._separator())

        # --- Ephemeral mode toggle (v0.4.1). Sits right under the pin
        # so the "global mode" controls (📌 dock side, 👻 stroke
        # persistence) cluster at the top — the most-significant
        # behavioural switches share visual real estate.
        self._ephemeral_btn = QToolButton(self)
        self._ephemeral_btn.setObjectName("annotEphemeralBtn")
        self._ephemeral_btn.setText("👻")
        self._ephemeral_btn.setCheckable(True)
        self._ephemeral_btn.setChecked(False)
        self._ephemeral_btn.setFixedSize(26, 22)
        self._ephemeral_btn.setToolTip(
            "Mode éphémère (G) — les traits s'effacent progressivement, "
            "non sauvegardés"
        )
        self._ephemeral_btn.clicked.connect(self._on_ephemeral_btn_clicked)
        layout.addWidget(
            self._ephemeral_btn, alignment=Qt.AlignmentFlag.AlignHCenter
        )

        # --- Duration presets [● ● ●]: 3 dots of increasing size.
        # Wrapped in a single row widget so we can show()/hide() it
        # as a unit when the ephemeral mode flips.
        self._ephemeral_preset_row = self._build_ephemeral_preset_row()
        # Default visibility: hidden (mode starts off).
        self._ephemeral_preset_row.setVisible(False)
        layout.addWidget(self._ephemeral_preset_row)
        layout.addWidget(self._separator())

        # --- Stroke preview (sample wavy stroke in current color/size)
        self._stroke_preview = _StrokePreview(self)
        self._stroke_preview.set_color(self._current_color)
        self._stroke_preview.set_size(self._current_size)
        layout.addWidget(self._stroke_preview)

        # --- Pen + eraser as a horizontal pair (matches the user's
        # reference layout: tools side-by-side rather than stacked).
        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        tool_row.setContentsMargins(0, 0, 0, 0)

        # ✏️ + 🧽 native emojis — same reasoning as for the pin: the
        # user prefers the colorful glyphs (pencil + sponge) over
        # custom monochrome SVGs. The variation selector U+FE0F on
        # the pencil forces emoji presentation; sponge is emoji-by
        # -default.
        self._pen_btn = QToolButton(self)
        self._pen_btn.setObjectName("annotPenBtn")
        self._pen_btn.setText("✏️")
        self._pen_btn.setToolTip("Pen (P) — clic-glisser pour dessiner")
        self._pen_btn.setCheckable(True)
        self._pen_btn.setFixedSize(30, 28)
        self._pen_btn.clicked.connect(self._on_pen_clicked)

        self._eraser_btn = QToolButton(self)
        self._eraser_btn.setObjectName("annotEraserBtn")
        self._eraser_btn.setText("🧽")
        self._eraser_btn.setToolTip(
            "Eraser (E) — clic sur un trait pour le supprimer"
        )
        self._eraser_btn.setCheckable(True)
        self._eraser_btn.setFixedSize(30, 28)
        self._eraser_btn.clicked.connect(self._on_eraser_clicked)

        # QButtonGroup handles the radio (mutex) behaviour, but we
        # also want "click an active button to deactivate" — Qt's
        # group doesn't do that. We handle it manually in the slots.
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(False)
        self._tool_group.addButton(self._pen_btn)
        self._tool_group.addButton(self._eraser_btn)

        tool_row.addStretch(1)
        tool_row.addWidget(self._pen_btn)
        tool_row.addWidget(self._eraser_btn)
        tool_row.addStretch(1)
        layout.addLayout(tool_row)
        layout.addWidget(self._separator())

        # --- Size: label + slider (sits BELOW the tool pair, per the
        # user's reference layout).
        size_caption = QLabel("Size", self)
        size_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        size_caption.setStyleSheet(
            "color: #E4E4E6; font-size: 11px; font-weight: 500;"
        )
        layout.addWidget(size_caption)

        self._size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._size_slider.setRange(int(MIN_SIZE), int(MAX_SIZE))
        self._size_slider.setValue(int(self._current_size))
        self._size_slider.valueChanged.connect(self._on_size_slider)
        layout.addWidget(self._size_slider)

        self._size_label = QLabel(f"{int(self._current_size)}px", self)
        self._size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._size_label.setStyleSheet("color: #8A8A8E; font-size: 10px;")
        layout.addWidget(self._size_label)
        layout.addWidget(self._separator())

        # --- Pen stabilizer (Lazy Mouse) ----------------------------
        # 3-position discrete slider mirroring the Size pattern: title
        # caption + horizontal slider + dynamic value label below. The
        # slider snaps to integer steps so the user always lands on
        # one of the three configured strengths (Off / Med / Strong).
        # We use a slider rather than 3 dots so the UI doesn't read
        # as a duplicate of the ephemeral fade preset row.
        stab_caption = QLabel("Stabilizer", self)
        stab_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stab_caption.setStyleSheet(
            "color: #E4E4E6; font-size: 11px; font-weight: 500;"
        )
        layout.addWidget(stab_caption)

        self._stabilizer_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._stabilizer_slider.setRange(0, 2)
        self._stabilizer_slider.setSingleStep(1)
        self._stabilizer_slider.setPageStep(1)
        self._stabilizer_slider.setTickInterval(1)
        self._stabilizer_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._stabilizer_slider.setValue(self._stabilizer_level)
        self._stabilizer_slider.valueChanged.connect(self._on_stabilizer_slider)
        layout.addWidget(self._stabilizer_slider)

        self._stabilizer_label = QLabel(
            PEN_STABILIZER_LABELS[self._stabilizer_level], self,
        )
        self._stabilizer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stabilizer_label.setStyleSheet(
            "color: #8A8A8E; font-size: 10px;"
        )
        layout.addWidget(self._stabilizer_label)
        layout.addWidget(self._separator())

        # --- Color palette: 7 swatches in a 2-column grid (4 rows, the
        # last with a single swatch). Round shape was set in PR #42.
        self._swatches: list[_ColorSwatch] = []
        for color in PALETTE:
            sw = _ColorSwatch(color, self)
            sw.setChecked(color == self._current_color)
            sw.clicked.connect(
                lambda _checked=False, c=color: self._on_swatch_clicked(c)
            )
            self._swatches.append(sw)
        self._palette_grid = self._build_palette_grid()
        layout.addWidget(self._palette_grid)
        layout.addWidget(self._separator())

        # --- Undo / redo (horizontal pair).
        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        action_row.setContentsMargins(0, 0, 0, 0)

        self._undo_btn = QToolButton(self)
        self._undo_btn.setObjectName("annotUndoBtn")
        self._undo_btn.setIcon(make_icon("undo"))
        self._undo_btn.setIconSize(QSize(16, 16))
        self._undo_btn.setToolTip("Undo (Ctrl+Z)")
        self._undo_btn.setFixedSize(28, 24)
        self._undo_btn.clicked.connect(self.undo_requested.emit)

        self._redo_btn = QToolButton(self)
        self._redo_btn.setObjectName("annotRedoBtn")
        self._redo_btn.setIcon(make_icon("redo"))
        self._redo_btn.setIconSize(QSize(16, 16))
        self._redo_btn.setToolTip("Redo (Ctrl+Y)")
        self._redo_btn.setFixedSize(28, 24)
        self._redo_btn.clicked.connect(self.redo_requested.emit)

        action_row.addStretch(1)
        action_row.addWidget(self._undo_btn)
        action_row.addWidget(self._redo_btn)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        layout.addWidget(self._separator())

        # --- Clear button — distinct visual (subtle red tint) so the
        # user understands it's a destructive action vs. the neutral
        # tool buttons above. Removes every stroke on the current
        # frame; each removal is undoable so an accidental click
        # walks back stroke-by-stroke with Ctrl+Z.
        self._clear_btn = QPushButton("Clear", self)
        self._clear_btn.setToolTip(
            "Supprime toutes les annotations sur la frame courante"
        )
        self._clear_btn.setStyleSheet(
            "QPushButton {"
            "  background: rgba(232, 74, 74, 0.16);"  # subtle red tint
            "  border: 1px solid rgba(232, 74, 74, 0.55);"
            "  color: #E4E4E6;"
            "  padding: 4px 10px;"
            "  border-radius: 4px;"
            "  font-size: 11px;"
            "}"
            "QPushButton:hover {"
            "  background: rgba(232, 74, 74, 0.32);"
            "  border-color: rgba(232, 74, 74, 0.85);"
            "}"
            "QPushButton:pressed {"
            "  background: rgba(232, 74, 74, 0.45);"
            "}"
        )
        self._clear_btn.clicked.connect(self.clear_requested.emit)
        layout.addWidget(self._clear_btn)

        # Compact width — sized to fit the pen + eraser pair side by
        # side with minimal padding, matching the user's reference.
        # 2 × 30 (buttons) + 6 (spacing) + 12 (margins) = 78.
        self.setFixedWidth(78)
        # Height shrinks to the layout's natural size (no addStretch
        # filler, no expand). Without this, the float-mode toolbar
        # would fill the entire viewport vertically because Qt gives
        # a SubWindow-flagged QWidget the parent's available area
        # by default.
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)

    def _build_palette_grid(self) -> QWidget:
        """Lay out the 7 swatches as a 2-column grid (4 rows; last
        row carries a single swatch).

        Vertical orientation matches the rest of the redesigned
        toolbar — the older 4-3 horizontal grid felt cramped next to
        the now-paired pen / eraser row.
        """
        wrapper = QWidget(self)
        grid_layout = QVBoxLayout(wrapper)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(4)
        for row_start in range(0, len(self._swatches), 2):
            row = QHBoxLayout()
            row.setSpacing(4)
            row.addStretch(1)
            for sw in self._swatches[row_start:row_start + 2]:
                row.addWidget(sw)
            row.addStretch(1)
            grid_layout.addLayout(row)
        return wrapper

    def _build_ephemeral_preset_row(self) -> QWidget:
        """Build the ``[● ● ●]`` 3-preset duration row.

        Three checkable QToolButtons — one bullet glyph each, growing
        in size — wrapped in an exclusive QButtonGroup. The active
        index is initialised from ``self._ephemeral_preset_index``.

        Width-wise the row sits inside the 78 px toolbar — the dots
        + a tiny inter-button gap fits comfortably without changing
        the toolbar's overall width.
        """
        wrapper = QWidget(self)
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        # Three buttons : "moyen" (default, leftmost), then "court"
        # and "long" flanking. Each glyph is rendered at a different
        # point size so the gradient hints at the duration without
        # needing the tooltip. Visual position MATCHES value order
        # in EPHEMERAL_PRESETS_S — index 0 first, etc.
        labels = (
            ("●", 9, "Court · ~2 s — fade rapide pour gestes brefs"),
            ("●", 12, "Moyen · ~5 s — usage courant (défaut)"),
            ("●", 16, "Long · ~10 s — pour expliquer en plusieurs phrases"),
        )

        self._ephemeral_preset_btns: list[QToolButton] = []
        self._ephemeral_preset_group = QButtonGroup(self)
        self._ephemeral_preset_group.setExclusive(True)

        row.addStretch(1)
        for i, (glyph, point_size, tooltip) in enumerate(labels):
            btn = QToolButton(wrapper)
            btn.setText(glyph)
            btn.setCheckable(True)
            btn.setChecked(i == self._ephemeral_preset_index)
            btn.setToolTip(tooltip)
            btn.setFixedSize(20, 18)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            # Color-on-select : the active preset's glyph turns cyan
            # (the same accent as the toolbar's ephemeral border) so
            # which one is picked reads instantly. ``:checked`` is
            # the QSS pseudo-class that maps to QAbstractButton's
            # checked state.
            btn.setStyleSheet(
                f"QToolButton {{"
                f"  font-size: {point_size}px;"
                f"  padding: 0;"
                f"  color: #8A8A8E;"  # muted grey when not selected
                f"  background: transparent;"
                f"  border: none;"
                f"}}"
                f"QToolButton:checked {{"
                f"  color: {_EPHEMERAL_ACCENT};"  # cyan when selected
                f"  background: rgba(74, 141, 232, 36);"  # subtle cyan halo
                f"  border-radius: 4px;"
                f"}}"
                f"QToolButton:hover {{"
                f"  color: #C0C0C4;"
                f"}}"
                f"QToolButton:checked:hover {{"
                f"  color: {_EPHEMERAL_ACCENT};"
                f"}}"
            )
            # Closure capture: bind ``i`` at definition time, not at
            # call time, otherwise every lambda would emit index 2.
            btn.clicked.connect(
                lambda _checked=False, idx=i: self._on_ephemeral_preset_clicked(idx)
            )
            self._ephemeral_preset_btns.append(btn)
            self._ephemeral_preset_group.addButton(btn)
            row.addWidget(btn)
        row.addStretch(1)
        return wrapper

    def _separator(self) -> QFrame:
        line = QFrame(self)
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Plain)
        line.setStyleSheet("color: #28282C;")  # BORDER_SUBTLE
        line.setMaximumHeight(1)
        return line

    def _apply_mode_initial(self) -> None:
        """Apply the parenting + styles for the initial mode at boot
        (without going through the change-emitting path)."""
        # Set parent according to initial mode.
        if self._mode == ToolbarMode.FLOAT:
            self.setParent(self._gl_viewport)
            self.setWindowFlags(Qt.WindowType.SubWindow)
            self.move(*self._floating_pos)
            self.raise_()
            self.adjustSize()  # see set_mode for why
        else:  # DOCK
            self._dock_wrapper.setWidget(self)
        self._apply_mode_styles()

    def _apply_mode_styles(self) -> None:
        """Apply the visual style for the current mode (translucent
        background in float, opaque in dock).

        v0.4.1 also tints the outer border cyan when ephemeral mode
        is active — the strongest possible signal that "everything
        you draw now disappears", visible regardless of where on the
        toolbar the user's eye lands.
        """
        # Border colour: cyan accent in ephemeral mode, default greys
        # otherwise. Same accent in both float and dock mode so the
        # mode is recognisable regardless of toolbar parenting.
        if self._mode == ToolbarMode.FLOAT:
            border = (
                f"2px solid {_EPHEMERAL_ACCENT}"
                if self._ephemeral_mode
                else "1px solid rgba(56, 56, 60, 220)"
            )
            # Semi-transparent dark panel — 70 % opaque so the image
            # behind shows through enough to keep spatial context
            # without making the toolbar's silhouette dissolve into
            # the viewport.
            self.setStyleSheet(
                "AnnotationToolbar {"
                "  background: rgba(36, 36, 40, 178);"  # 178/255 ≈ 70%
                f"  border: {border};"
                "  border-radius: 8px;"
                "}"
                + _ACTION_BTN_QSS
            )
            # WA_StyledBackground is REQUIRED for a custom QWidget
            # subclass to honor a QSS background rule on the widget
            # itself (without it, QSS only styles children).
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self.setAutoFillBackground(False)
            # Cursor indicates draggable in float mode.
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            # Opaque, matches BG_RAISED — looks like the rest of the
            # dock pattern. In ephemeral mode we still apply the cyan
            # border so the visual signal is consistent across modes.
            border_rule = (
                f"  border: 2px solid {_EPHEMERAL_ACCENT};"
                if self._ephemeral_mode
                else ""
            )
            self.setStyleSheet(
                "AnnotationToolbar {"
                "  background: #242428;"  # BG_RAISED
                f"{border_rule}"
                "}"
                + _ACTION_BTN_QSS
            )
            # Same reason as in float mode: ensures the QSS background
            # actually paints on the toolbar itself (not just the
            # children).
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
            self.setAutoFillBackground(True)
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.Window, QColor("#242428"))
            self.setPalette(palette)
            self.unsetCursor()

    # ------------------------------------------------------------------ Slots

    def _on_pin_clicked(self) -> None:
        new_mode = (
            ToolbarMode.DOCK
            if self._mode == ToolbarMode.FLOAT
            else ToolbarMode.FLOAT
        )
        self.set_mode(new_mode)

    def _on_pen_clicked(self) -> None:
        new_tool = ToolKind.NONE if self._current_tool == ToolKind.PEN else ToolKind.PEN
        self.set_current_tool(new_tool)

    def _on_eraser_clicked(self) -> None:
        new_tool = (
            ToolKind.NONE if self._current_tool == ToolKind.ERASER else ToolKind.ERASER
        )
        self.set_current_tool(new_tool)

    def _on_swatch_clicked(self, color_hex: str) -> None:
        # Don't let the user "uncheck" a color — there must always be one
        # active. Re-clicking the active one is a no-op.
        if color_hex == self._current_color:
            # Re-assert the checked state to undo Qt's auto-toggle.
            for sw in self._swatches:
                if sw.color_hex == color_hex:
                    sw.setChecked(True)
            return
        self.set_current_color(color_hex)

    def _on_size_slider(self, value: int) -> None:
        self.set_current_size(float(value))

    def _on_stabilizer_slider(self, value: int) -> None:
        # The slider is range 0..2, single-step 1, so ``value`` is
        # already a valid level. ``set_pen_stabilizer_level`` updates
        # the label and emits ``pen_stabilizer_level_changed``; the
        # blockSignals dance protects against re-entry when the
        # setter syncs the slider back.
        self.set_pen_stabilizer_level(value)

    # ------------------------------------------------------------------ Ephemeral slots (v0.4.1)

    def _on_ephemeral_btn_clicked(self) -> None:
        """User clicked the 👻 toggle. Mirror UI + emit the change.

        Note: ``QToolButton.clicked`` fires with the *new* checked
        state already applied by Qt's checkable machinery. We pull
        it from the button rather than from a parameter to keep
        ``set_ephemeral_mode`` as the one place state actually
        flips.
        """
        new_state = self._ephemeral_btn.isChecked()
        # The setter handles the case where the state is already in
        # sync (no-op) — important because we just synthesised a
        # click; Qt has already toggled the button's checked state.
        # Force a delta via the internal flag check.
        if new_state == self._ephemeral_mode:
            # Qt and our flag are out-of-sync — force-apply.
            self._ephemeral_mode = not new_state
        self.set_ephemeral_mode(new_state, emit=True)

    def _on_ephemeral_preset_clicked(self, index: int) -> None:
        """User clicked one of the 3 preset dots. Mirror + emit."""
        self.set_ephemeral_preset_index(index, emit=True)

    # ------------------------------------------------------------------ Drag in float mode

    def mousePressEvent(self, event: QMouseEvent) -> None:
        # In float mode, clicking on the toolbar's background (not on a
        # button) starts a drag. Buttons receive their own events first
        # so they don't trigger this.
        if self._mode == ToolbarMode.FLOAT and event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.position().toPoint()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and self._mode == ToolbarMode.FLOAT:
            # Translate from screen coords to viewport coords. event.globalPosition
            # gives screen, gl_viewport.mapFromGlobal converts.
            global_pos = event.globalPosition().toPoint()
            viewport_pos = self._gl_viewport.mapFromGlobal(global_pos)
            new_pos = viewport_pos - self._drag_offset
            # Clamp into the viewport so the toolbar can't be dragged off-screen.
            max_x = max(0, self._gl_viewport.width() - self.width())
            max_y = max(0, self._gl_viewport.height() - self.height())
            x = max(0, min(max_x, new_pos.x()))
            y = max(0, min(max_y, new_pos.y()))
            self.move(x, y)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_offset is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = None
            self._floating_pos = (self.x(), self.y())
            self.floating_pos_changed.emit(self.x(), self.y())
            event.accept()
            return
        super().mouseReleaseEvent(event)
