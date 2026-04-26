"""Decorative L-shaped brackets in the four corners of the viewport.

Why a custom QWidget rather than QSS or images: the brackets are pure
1-pixel geometry and we want them to scale with the viewport
automatically. Painting them in a transparent overlay is cheaper than
shipping image assets, scales perfectly, and doesn't fight the QSS
cascade that styles every other widget.

The overlay is stacked above the ``GLViewport`` via ``QStackedLayout``
in ``StackAll`` mode (cf. :class:`ViewerWidget`). It sets
``WA_TransparentForMouseEvents`` so clicks and drag-and-drop fall
through to the GL widget below — the user never even notices it's
there beyond the visual treatment.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import QWidget


class BracketsOverlay(QWidget):  # type: ignore[misc]
    """Paints four 90° corner brackets, ``BRACKET_SIZE`` × ``BRACKET_SIZE``,
    inset by ``BRACKET_INSET`` from each edge."""

    # Geometry — kept as class attributes so a future user / variant can
    # tweak them via subclassing without touching internal code paths.
    BRACKET_SIZE = 20
    BRACKET_INSET = 20
    BRACKET_WIDTH = 1
    # rgba(255, 255, 255, 30) ≈ 0.12 alpha — matches the mockup. White
    # over the deep viewport bg (#141416) reads as a charcoal-on-grey
    # accent without grabbing attention.
    BRACKET_COLOR = QColor(255, 255, 255, 30)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Critical: drop mouse events through to whatever's underneath
        # in the QStackedLayout (the GLViewport, which receives drag &
        # drop events for sequence loading).
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Also critical: don't let Qt paint the widget's own background
        # (which the global QSS would set to BG_BASE / BG_RAISED). With
        # this attribute, only the explicit drawLine calls in paintEvent
        # produce pixels — everything else stays transparent and the
        # GLViewport below shows through.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

    # ------------------------------------------------------------------ Paint

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        pen = QPen(self.BRACKET_COLOR, self.BRACKET_WIDTH)
        pen.setCosmetic(True)
        painter.setPen(pen)

        size = self.BRACKET_SIZE
        inset = self.BRACKET_INSET
        # Coordinates run from 0 to width-1 inclusive — using `width`
        # itself for the right edge would paint just outside the widget
        # and the rightmost / bottom brackets would be invisible.
        right = self.width() - 1
        bottom = self.height() - 1

        # Don't draw if the widget is too small to host the geometry —
        # avoids overlapping brackets when the user yanks a window
        # tiny.
        if right + 1 < 2 * (inset + size) or bottom + 1 < 2 * (inset + size):
            return

        # Each bracket is a vertical line + horizontal line meeting at
        # the corner. We draw the *inner* corner of each L `inset` px
        # from the widget edge.
        # Top-left
        painter.drawLine(inset, inset, inset + size, inset)
        painter.drawLine(inset, inset, inset, inset + size)
        # Top-right
        painter.drawLine(right - inset, inset, right - inset - size, inset)
        painter.drawLine(right - inset, inset, right - inset, inset + size)
        # Bottom-left
        painter.drawLine(inset, bottom - inset, inset + size, bottom - inset)
        painter.drawLine(inset, bottom - inset, inset, bottom - inset - size)
        # Bottom-right
        painter.drawLine(
            right - inset, bottom - inset, right - inset - size, bottom - inset
        )
        painter.drawLine(
            right - inset, bottom - inset, right - inset, bottom - inset - size
        )

        painter.end()
