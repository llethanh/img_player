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

from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPalette
from PySide6.QtWidgets import (
    QButtonGroup,
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from img_player.annotate.overlay import ToolKind
from img_player.render.gl_viewport import GLViewport


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
MAX_SIZE = 50.0


class _ColorSwatch(QToolButton):
    """A small square button filled with a single solid color.

    Acts as a checkable radio in the palette group. Renders its color
    via paintEvent + setChecked drawing — Qt's default styling for
    QPushButton/QToolButton would add a border that fights the
    saturated swatch colors at tiny sizes (16-20 px).
    """

    SIZE_PX = 18

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

    mode_changed = Signal(object)
    """Emits :class:`ToolbarMode` when the pin toggles float ⇄ dock."""

    floating_pos_changed = Signal(int, int)
    """Emits ``(x, y)`` when the user finishes dragging the toolbar in
    float mode. App.py persists the position to preferences."""

    # ------------------------------------------------------------------ Lifecycle

    def __init__(
        self,
        gl_viewport: GLViewport,
        dock_wrapper: QDockWidget,
        *,
        initial_mode: ToolbarMode = ToolbarMode.FLOAT,
        initial_floating_pos: tuple[int, int] = (12, 12),
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
        self._size_label.setText(f"{int(round(size))}px")
        self.size_changed.emit(size)

    # ------------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        # Outer column.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # --- Pin button (top of the toolbar)
        self._pin_btn = QToolButton(self)
        self._pin_btn.setText("📌")
        self._pin_btn.setToolTip("Bascule float ⇄ dock")
        self._pin_btn.setFixedSize(28, 24)
        self._pin_btn.clicked.connect(self._on_pin_clicked)
        layout.addWidget(self._pin_btn)
        layout.addWidget(self._separator())

        # --- Pen / eraser
        tool_row = QVBoxLayout()
        tool_row.setSpacing(4)

        self._pen_btn = QToolButton(self)
        # ✏️ = U+270F LOWER RIGHT PENCIL + U+FE0F variation selector.
        # The selector forces emoji presentation; without it the pencil
        # falls back to the monochrome text glyph on most systems.
        self._pen_btn.setText("✏️")
        self._pen_btn.setToolTip("Pen (P) — clic-glisser pour dessiner")
        self._pen_btn.setCheckable(True)
        self._pen_btn.setFixedSize(36, 28)
        self._pen_btn.clicked.connect(self._on_pen_clicked)

        self._eraser_btn = QToolButton(self)
        # 🧽 = U+1F9FD SPONGE. Native emoji rendering by default — no
        # selector needed. Visually distinct from the pen and reads
        # "remove / clean" universally.
        self._eraser_btn.setText("🧽")
        self._eraser_btn.setToolTip("Eraser (E) — clic sur un trait pour le supprimer")
        self._eraser_btn.setCheckable(True)
        self._eraser_btn.setFixedSize(36, 28)
        self._eraser_btn.clicked.connect(self._on_eraser_clicked)

        # QButtonGroup handles the radio (mutex) behaviour, but we
        # also want "click an active button to deactivate" — Qt's
        # group doesn't do that. We handle it manually in the slots.
        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(False)  # we manage exclusivity
        self._tool_group.addButton(self._pen_btn)
        self._tool_group.addButton(self._eraser_btn)

        tool_row.addWidget(self._pen_btn)
        tool_row.addWidget(self._eraser_btn)
        layout.addLayout(tool_row)
        layout.addWidget(self._separator())

        # --- Color palette: 7 swatches. In float mode we lay them out
        # in a 4-3 grid (compact); in dock mode they go in a single
        # column (more vertical room available).
        self._palette_layout = QHBoxLayout()
        self._palette_layout.setSpacing(4)
        self._swatches: list[_ColorSwatch] = []
        for color in PALETTE:
            sw = _ColorSwatch(color, self)
            sw.setChecked(color == self._current_color)
            sw.clicked.connect(lambda _checked=False, c=color: self._on_swatch_clicked(c))
            self._swatches.append(sw)
        self._palette_grid = self._build_palette_grid()
        layout.addWidget(self._palette_grid)
        layout.addWidget(self._separator())

        # --- Size slider
        self._size_slider = QSlider(Qt.Orientation.Horizontal, self)
        self._size_slider.setRange(int(MIN_SIZE), int(MAX_SIZE))
        self._size_slider.setValue(int(self._current_size))
        self._size_slider.setFixedWidth(80)
        self._size_slider.valueChanged.connect(self._on_size_slider)
        self._size_label = QLabel(f"{int(self._current_size)}px", self)
        self._size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._size_label.setStyleSheet("color: #8A8A8E; font-size: 10px;")  # TEXT_SECONDARY

        size_box = QVBoxLayout()
        size_box.setSpacing(2)
        size_box.addWidget(self._size_slider)
        size_box.addWidget(self._size_label)
        layout.addLayout(size_box)
        layout.addWidget(self._separator())

        # --- Undo / redo
        undo_row = QVBoxLayout()
        undo_row.setSpacing(4)
        self._undo_btn = QToolButton(self)
        self._undo_btn.setText("↶")
        self._undo_btn.setToolTip("Undo (Ctrl+Z)")
        self._undo_btn.setFixedSize(36, 24)
        self._undo_btn.clicked.connect(self.undo_requested.emit)
        self._redo_btn = QToolButton(self)
        self._redo_btn.setText("↷")
        self._redo_btn.setToolTip("Redo (Ctrl+Y)")
        self._redo_btn.setFixedSize(36, 24)
        self._redo_btn.clicked.connect(self.redo_requested.emit)
        undo_row.addWidget(self._undo_btn)
        undo_row.addWidget(self._redo_btn)
        layout.addLayout(undo_row)

        layout.addStretch(1)

        # Auto-sized — let Qt compute. We set a fixed width minimum so
        # the toolbar doesn't collapse weirdly in dock mode.
        self.setMinimumWidth(80)

    def _build_palette_grid(self) -> QWidget:
        """Lay out the 7 swatches as a 4-3 grid for vertical compactness."""
        wrapper = QWidget(self)
        grid_layout = QVBoxLayout(wrapper)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.setSpacing(4)

        row1 = QHBoxLayout()
        row1.setSpacing(4)
        for sw in self._swatches[:4]:
            row1.addWidget(sw)
        row1.addStretch(1)

        row2 = QHBoxLayout()
        row2.setSpacing(4)
        for sw in self._swatches[4:]:
            row2.addWidget(sw)
        row2.addStretch(1)

        grid_layout.addLayout(row1)
        grid_layout.addLayout(row2)
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
        else:  # DOCK
            self._dock_wrapper.setWidget(self)
        self._apply_mode_styles()

    def _apply_mode_styles(self) -> None:
        """Apply the visual style for the current mode (translucent
        background in float, opaque in dock)."""
        if self._mode == ToolbarMode.FLOAT:
            # Semi-transparent dark panel — the spec calls for ~92 %
            # alpha so the image just shows through enough to keep
            # spatial context.
            self.setStyleSheet(
                "AnnotationToolbar {"
                "  background: rgba(36, 36, 40, 235);"  # 235/255 ≈ 92%
                "  border: 1px solid rgba(56, 56, 60, 220);"
                "  border-radius: 8px;"
                "}"
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
            self.setAutoFillBackground(False)
            # Cursor indicates draggable in float mode.
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            # Opaque, matches BG_RAISED — looks like the rest of the dock pattern.
            self.setStyleSheet(
                "AnnotationToolbar {"
                "  background: #242428;"  # BG_RAISED
                "}"
            )
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
