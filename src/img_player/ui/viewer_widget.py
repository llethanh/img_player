"""Container around the GL viewport.

Today it just wraps :class:`GLViewport`; tomorrow (V3 — annotations) it will
add overlay layers on top of the image without touching the GL code.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QStackedLayout, QWidget

from img_player.render.gl_viewport import GLViewport


class ViewerWidget(QWidget):  # type: ignore[misc]
    """GL viewport + reserved overlay slot."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._gl = GLViewport()

        # Reserved overlay: empty in V1 but kept Z-above the GL widget so
        # future annotation widgets can sit there without interfering with
        # the GL redraw path.
        self._overlay = QWidget(self)
        self._overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        layout = QStackedLayout(self)
        layout.setStackingMode(QStackedLayout.StackingMode.StackAll)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._gl)
        layout.addWidget(self._overlay)

    @property
    def gl(self) -> GLViewport:
        return self._gl
