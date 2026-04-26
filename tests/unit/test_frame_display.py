"""Tests for the editable frame / timecode field."""

from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from img_player.ui.frame_display import FrameDisplay


@pytest.fixture
def display(qtbot) -> FrameDisplay:  # type: ignore[no-untyped-def]
    fd = FrameDisplay()
    qtbot.addWidget(fd)
    fd.show()
    return fd


class TestSetFrame:
    def test_initial_text_is_zero_padded(self, display: FrameDisplay) -> None:
        display.set_frame(42)
        assert display.text() == "0042"

    def test_set_frame_updates_text_even_with_focus(self, display: FrameDisplay) -> None:
        # Regression: an external frame change (timeline scrub, J/L,
        # playback) must update the display even when the user has
        # the cursor in the field — *as long as they haven't typed*.
        display.set_frame(5)
        display.setFocus(Qt.FocusReason.OtherFocusReason)
        display.set_frame(99)
        assert display.text() == "0099"

    def test_set_frame_does_not_clobber_typing(
        self, display: FrameDisplay,
    ) -> None:
        # Regression: while the user is mid-typing, external frame
        # changes (e.g. the playback loop ticking 24 times per second)
        # must not erase what they're typing — otherwise they can
        # never finish entering a value.
        display.set_frame(5)
        display.setFocus(Qt.FocusReason.OtherFocusReason)
        QTest.keyClicks(display, "42")
        # ``set_frame`` from playback fires while the user has the
        # field dirty: it must *not* clobber the typed text.
        display.set_frame(6)
        assert "42" in display.text()


class TestEnterCommits:
    def test_typing_then_return_emits_signal(
        self, display: FrameDisplay, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        display.set_frame(5)
        # Click to focus, clear, type a new value, press Enter.
        display.setFocus(Qt.FocusReason.OtherFocusReason)
        with qtbot.waitSignal(
            display.frame_seek_requested, timeout=1000,
        ) as blocker:
            display.clear()
            QTest.keyClicks(display, "42")
            QTest.keyClick(display, Qt.Key.Key_Return)
        assert blocker.args == [42]

    def test_enter_with_invalid_text_restores_previous(
        self, display: FrameDisplay,
    ) -> None:
        display.set_frame(7)
        display.setFocus(Qt.FocusReason.OtherFocusReason)
        display.clear()
        QTest.keyClicks(display, "abc")
        QTest.keyClick(display, Qt.Key.Key_Return)
        # Garbage rejected, the field shows the last good frame.
        assert display.text() == "0007"

    def test_focus_loss_also_commits(
        self, display: FrameDisplay, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        # editingFinished fires both on Return and on focus-out.
        display.set_frame(10)
        display.setFocus(Qt.FocusReason.OtherFocusReason)
        with qtbot.waitSignal(
            display.frame_seek_requested, timeout=1000,
        ) as blocker:
            display.clear()
            QTest.keyClicks(display, "55")
            display.clearFocus()
        assert blocker.args == [55]


class TestTcMode:
    def test_typing_a_timecode_in_tc_mode_seeks_correctly(
        self, display: FrameDisplay, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        display.set_fps(24.0)
        display.set_display_mode("tc")
        display.set_frame(0)
        display.setFocus(Qt.FocusReason.OtherFocusReason)
        with qtbot.waitSignal(
            display.frame_seek_requested, timeout=1000,
        ) as blocker:
            display.clear()
            QTest.keyClicks(display, "00:00:01:00")
            QTest.keyClick(display, Qt.Key.Key_Return)
        # 1 second at 24 fps = frame 24
        assert blocker.args == [24]
