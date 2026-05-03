"""Container around the GL viewport.

Stacks decorative / interactive overlay widgets on top of the GL
viewport via ``QStackedLayout`` in ``StackAll`` mode. The overlay
slot starts with the corner :class:`BracketsOverlay` from the design
charter; it's the same architecture annotation tools (V3) will plug
into without touching GL code.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QStackedLayout, QWidget

from img_player.render.gl_viewport import GLViewport
from img_player.ui.brackets_overlay import BracketsOverlay
from img_player.ui.drop_zone import (
    DropOverlay,
    REPLACE_ACCENT,
    install_file_drop_zone,
)


class ViewerWidget(QWidget):  # type: ignore[misc]
    """GL viewport + decorative brackets overlay (and future annotation slot)."""

    # File(s) / folder(s) dropped on the viewer — the user wants to
    # replace the currently loaded sequence. Same destination as
    # File → Open. Carries a list because a single drop can include
    # multiple folders / files; the picker resolves the choice.
    replace_requested = Signal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._gl = GLViewport()
        # Decorative L-brackets in the four corners. Transparent to
        # mouse events — clicks fall through to the GL widget so drag
        # & drop of sequences keeps working.
        self._overlay = BracketsOverlay(self)

        layout = QStackedLayout(self)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._gl)
        layout.addWidget(self._overlay)

        # Drop zone with a "REPLACE" overlay shown during drag-over.
        # Sits as a child of ``self`` (not in the stacked layout) so
        # we can ``raise_()`` it to the absolute top during a drag —
        # the brackets overlay already lives in the stack and would
        # otherwise paint on top of the drop hint.
        self._drop_overlay = DropOverlay("REPLACE", REPLACE_ACCENT, self)
        install_file_drop_zone(
            self, self._drop_overlay,
            lambda paths: self.replace_requested.emit(paths),
        )

        # Info label, top-right corner overlay. Always visible
        # regardless of the side-panel state — earlier placement in
        # the side panel disappeared whenever the user hid the
        # right-hand dock. Child of ``self`` (not in the stacked
        # layout) so we can position it absolutely.
        self._info_label = QLabel("", self)
        self._info_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
        )
        self._info_label.setStyleSheet(
            "QLabel {"
            "  background: rgba(0, 0, 0, 140);"
            "  color: #C8C8C8;"
            "  padding: 4px 8px;"
            "  border-radius: 4px;"
            "  font-family: 'Consolas', 'Menlo', monospace;"
            "  font-size: 11px;"
            "}"
        )
        self._info_label.hide()
        self._info_label.raise_()

    @property
    def gl(self) -> GLViewport:
        return self._gl

    @property
    def overlay(self) -> BracketsOverlay:
        return self._overlay

    def set_info_text(self, text: str) -> None:
        """Display ``text`` in the corner overlay just outside the
        image's top-right corner. Empty string hides the label.

        Only shown when the image is letterboxed / pillarboxed —
        i.e. the widget has empty padding to the right of the
        image. When the image fills the widget edge-to-edge there's
        nowhere "outside the image" to put the label so we hide it.
        """
        if not text:
            self._info_label.hide()
            return
        self._info_label.setText(text)
        self._info_label.adjustSize()
        if self._reposition_info_label():
            self._info_label.show()
            self._info_label.raise_()
        else:
            self._info_label.hide()

    def _reposition_info_label(self) -> bool:
        """Place the label above the image's top edge, right-aligned
        with the image's right edge. The label is rigidly anchored
        to the image — it never gets nudged onto the image to stay
        in view. When the user pans the image past the viewport's
        edge the label slides off too, getting naturally clipped at
        the widget's bounds by Qt's child-widget clipping. We only
        hide it when it would render zero pixels (= entirely off
        the widget), so the user sees a progressively shrinking
        readout instead of a sudden disappearance.

        Returns ``False`` only when no image is loaded or when the
        computed position is fully off-widget."""
        img_w, img_h = self._gl.image_size()
        if img_w <= 0 or img_h <= 0:
            return False
        factor, pan_x, pan_y = self._gl.current_transform()
        win_w = self.width()
        win_h = self.height()
        displayed_w = img_w * factor
        displayed_h = img_h * factor
        rect_left = (win_w - displayed_w) / 2.0 + pan_x
        rect_top = (win_h - displayed_h) / 2.0 + pan_y
        rect_right = rect_left + displayed_w
        label_w = self._info_label.width()
        label_h = self._info_label.height()
        margin = 6
        # Right-align with the image's right edge, sit just above
        # the image's top edge — strictly outside the image. No
        # clamps to widget bounds: we'd rather have the label
        # naturally clipped at the viewport edge than push it back
        # onto the image content.
        x = rect_right - label_w
        y = rect_top - label_h - margin
        # Hide only when nothing would be visible at all (label fully
        # past the widget's edge in either axis).
        if x + label_w <= 0 or x >= win_w:
            return False
        if y + label_h <= 0 or y >= win_h:
            return False
        self._info_label.move(int(x), int(y))
        return True

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        # Keep the drop-overlay sized with the widget while it's
        # visible (unusual case — drag-over during a window resize —
        # but trivial to support).
        if self._drop_overlay.isVisible():
            self._drop_overlay.setGeometry(self.rect())
        if self._info_label.isVisible():
            self._reposition_info_label()
