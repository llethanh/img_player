"""Container around the GL viewport.

Stacks decorative / interactive overlay widgets on top of the GL
viewport via ``QStackedLayout`` in ``StackAll`` mode. The overlay
slot starts with the corner :class:`BracketsOverlay` from the design
charter; it's the same architecture annotation tools (V3) will plug
into without touching GL code.
"""

from __future__ import annotations

from PySide6.QtWidgets import QStackedLayout, QWidget

from img_player.render.gl_viewport import GLViewport
from img_player.ui.brackets_overlay import BracketsOverlay


class ViewerWidget(QWidget):  # type: ignore[misc]
    """GL viewport + decorative brackets overlay (and future annotation slot)."""

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

    @property
    def gl(self) -> GLViewport:
        return self._gl

    @property
    def overlay(self) -> BracketsOverlay:
        return self._overlay
