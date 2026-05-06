"""Container around the GL viewport.

Stacks decorative / interactive overlay widgets on top of the GL
viewport via ``QStackedLayout`` in ``StackAll`` mode. The overlay
slot starts with the corner :class:`BracketsOverlay` from the design
charter; it's the same architecture annotation tools (V3) will plug
into without touching GL code.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QStackedLayout, QWidget

from img_player.render.gl_viewport import GLViewport
from img_player.ui.brackets_overlay import BracketsOverlay
from img_player.ui.drop_zone import (
    REPLACE_ACCENT,
    DropOverlay,
    install_file_drop_zone,
)
from img_player.ui.info_band import InfoBand


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

        # Bottom HUD — image dims / fps / local layer frame / global
        # timeline frame. Child of ``self`` (not in the stacked
        # layout) so we can position it absolutely flush with the
        # bottom edge of the viewer (= just above the timeline panel
        # in the main layout). The image-dimensions readout that
        # used to live as a separate top-right corner label was
        # retired — duplicated info in two places, the band reads
        # better.
        self._info_band = InfoBand(self)
        self._info_band.raise_()

    @property
    def gl(self) -> GLViewport:
        return self._gl

    @property
    def overlay(self) -> BracketsOverlay:
        return self._overlay

    @property
    def info_band(self) -> InfoBand:
        return self._info_band

    def _reposition_info_band(self) -> None:
        """Pin the info band to the bottom edge of the viewer, full
        width. Visible / hidden state isn't touched here — the
        caller controls it."""
        h = self._info_band.height()
        self._info_band.setGeometry(0, self.height() - h, self.width(), h)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        # Keep the drop-overlay sized with the widget while it's
        # visible (unusual case — drag-over during a window resize —
        # but trivial to support).
        if self._drop_overlay.isVisible():
            self._drop_overlay.setGeometry(self.rect())
        self._reposition_info_band()
