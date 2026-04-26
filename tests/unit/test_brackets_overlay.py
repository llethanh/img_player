"""Tests for the corner-brackets overlay over the viewport."""

from __future__ import annotations

import pytest
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QImage

from img_player.ui.brackets_overlay import BracketsOverlay


@pytest.fixture
def overlay(qtbot) -> BracketsOverlay:  # type: ignore[no-untyped-def]
    w = BracketsOverlay()
    qtbot.addWidget(w)
    w.resize(640, 360)
    return w


class TestBasicWidget:
    def test_constructs_without_crashing(self, overlay: BracketsOverlay) -> None:
        assert overlay is not None
        assert overlay.size() == QSize(640, 360)

    def test_is_transparent_to_mouse_events(self, overlay: BracketsOverlay) -> None:
        # The crucial property — clicks fall through to the GL widget
        # below in the QStackedLayout. Without this, drag-and-drop of
        # sequences would land on the bracket overlay and never reach
        # the actual viewport.
        assert overlay.testAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )


class TestPaintGeometry:
    """Render the overlay into a QImage and inspect pixels.

    We don't try to be pixel-exact — Qt's painter has subpixel
    alignment quirks that depend on platform and DPR. We just check
    the high-level invariants: corners have *something* drawn, centre
    is fully transparent, and the brackets actually paint pixels of
    the configured colour.
    """

    def _render(self, overlay: BracketsOverlay) -> QImage:
        # Force a paint into an off-screen image of the exact size,
        # so we can introspect any pixel.
        image = QImage(overlay.size(), QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.transparent)
        overlay.render(image)
        return image

    def test_centre_pixel_is_transparent(self, overlay: BracketsOverlay) -> None:
        img = self._render(overlay)
        cx, cy = img.width() // 2, img.height() // 2
        # Centre is far from any bracket — alpha must be 0.
        assert img.pixelColor(cx, cy).alpha() == 0

    def test_corners_have_painted_pixels(self, overlay: BracketsOverlay) -> None:
        # Find a pixel that *should* be in each bracket: at the inner
        # corner of the L, exactly inset px from the edge.
        img = self._render(overlay)
        inset = BracketsOverlay.BRACKET_INSET
        w, h = img.width(), img.height()
        # Top-left corner of the L is at (inset, inset).
        for px, py in [
            (inset, inset),
            (w - inset - 1, inset),
            (inset, h - inset - 1),
            (w - inset - 1, h - inset - 1),
        ]:
            color = img.pixelColor(px, py)
            assert color.alpha() > 0, (
                f"expected a painted pixel at ({px}, {py}) for the bracket corner, "
                f"got fully transparent"
            )

    def test_too_small_widget_skips_paint(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """A widget tinier than 2 × (inset + size) should refuse to paint
        rather than overlap brackets in the middle."""
        small = BracketsOverlay()
        qtbot.addWidget(small)
        small.resize(20, 20)  # way smaller than 2 × (20 + 20) = 80
        img = self._render(small)
        # No bracket should have been drawn — the whole image is
        # transparent.
        for x in (0, 5, 10, 15, 19):
            for y in (0, 5, 10, 15, 19):
                assert img.pixelColor(x, y).alpha() == 0
