"""Smoke tests for MainWindow's two-block status bar.

We don't try to render the rich-text dots here (that's covered by the
status_format tests). All we want is: the labels exist, set_status()
still works (backwards compat with all the existing call sites), and
status_right accepts rich-text HTML.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from img_player.color.ocio_manager import OCIOManager
from img_player.ui.main_window import MainWindow


@pytest.fixture
def main_window(qtbot) -> MainWindow:  # type: ignore[no-untyped-def]
    # OCIOManager is heavy (loads the built-in config). We pass a mock
    # so the window builds quickly — none of the methods we exercise
    # below touch OCIO state.
    ocio = MagicMock(spec=OCIOManager)
    ocio.list_colorspaces.return_value = ["scene_linear", "sRGB"]
    ocio.list_displays.return_value = ["sRGB"]
    ocio.list_views.return_value = ["ACES 1.0 SDR-video"]
    ocio.default_display.return_value = "sRGB"
    ocio.default_view.return_value = "ACES 1.0 SDR-video"
    ocio.role.return_value = "scene_linear"
    window = MainWindow(ocio)
    qtbot.addWidget(window)
    return window


class TestStatusBarLabels:
    def test_labels_exist(self, main_window: MainWindow) -> None:
        assert main_window.status_left is not None
        assert main_window.status_right is not None

    def test_set_status_routes_to_left(self, main_window: MainWindow) -> None:
        main_window.set_status("Loaded SH0010_Rendered_RGB.####.exr (90 frames)")
        assert "Loaded SH0010_Rendered_RGB" in main_window.status_left.text()

    def test_status_right_accepts_rich_text(self, main_window: MainWindow) -> None:
        # Setting rich-text HTML should not raise; QLabel's RichText
        # mode (configured at construction) renders the markup.
        main_window.status_right.setText(
            "<span style='color:#38B464'>●</span> cache 42/90"
        )
        # Qt strips the markup when reading .text(); we just check the
        # textual payload made it.
        assert "cache 42/90" in main_window.status_right.text()
