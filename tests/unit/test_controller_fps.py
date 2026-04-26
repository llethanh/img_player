"""Tests for PlayerController.effective_fps — the live playback metric.

We don't drive the QTimer here. Instead we exercise the property
directly by feeding the rolling timestamp deque, because what we care
about is the *math* of the rolling average — not Qt's timer plumbing,
which is well-covered elsewhere.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from img_player.cache.frame_cache import FrameCache
from img_player.player.controller import PlayerController


@pytest.fixture
def controller(qtbot) -> PlayerController:  # type: ignore[no-untyped-def]
    """A bare controller with a mocked cache. qtbot ensures a QApplication exists."""
    cache = MagicMock(spec=FrameCache)
    return PlayerController(cache)


def _set_playing(controller: PlayerController, playing: bool = True) -> None:
    """Force is_playing without going through play() (which needs a sequence)."""
    controller._state = replace(controller._state, is_playing=playing)  # noqa: SLF001


class TestEffectiveFps:
    def test_returns_none_when_not_playing(self, controller: PlayerController) -> None:
        # Even with samples in the deque, paused → None (matches the
        # status bar's "— fps" rendering).
        controller._tick_timestamps.extend(  # noqa: SLF001
            [1000.0, 1000.04, 1000.08]
        )
        assert controller.effective_fps() is None

    def test_returns_none_with_one_sample(self, controller: PlayerController) -> None:
        _set_playing(controller)
        controller._tick_timestamps.append(1000.0)  # noqa: SLF001
        assert controller.effective_fps() is None

    def test_returns_none_with_zero_samples(self, controller: PlayerController) -> None:
        _set_playing(controller)
        # Deque is empty
        assert controller.effective_fps() is None

    def test_returns_target_with_uniform_24fps(self, controller: PlayerController) -> None:
        """24 samples at 1/24 s apart should yield ~24 fps."""
        _set_playing(controller)
        base = 1000.0
        period = 1.0 / 24.0
        for i in range(24):
            controller._tick_timestamps.append(base + i * period)  # noqa: SLF001
        fps = controller.effective_fps()
        assert fps is not None
        assert abs(fps - 24.0) < 0.1

    def test_returns_target_with_uniform_60fps(self, controller: PlayerController) -> None:
        _set_playing(controller)
        base = 5000.0
        period = 1.0 / 60.0
        # Deque maxlen=24, so the most recent 24 will be kept.
        for i in range(48):
            controller._tick_timestamps.append(base + i * period)  # noqa: SLF001
        fps = controller.effective_fps()
        assert fps is not None
        assert abs(fps - 60.0) < 0.1

    def test_returns_half_target_when_lagging(self, controller: PlayerController) -> None:
        """24 samples spaced 2/24 s apart → effective is 12 fps even though
        the user wanted 24."""
        _set_playing(controller)
        base = 1000.0
        period = 2.0 / 24.0  # twice as slow as a 24 fps target
        for i in range(24):
            controller._tick_timestamps.append(base + i * period)  # noqa: SLF001
        fps = controller.effective_fps()
        assert fps is not None
        assert abs(fps - 12.0) < 0.1


class TestRollingWindowReset:
    def test_pause_clears_deque(self, controller: PlayerController) -> None:
        _set_playing(controller)
        controller._tick_timestamps.extend(range(24))  # noqa: SLF001
        # pause() early-returns if not is_playing — set state then call
        controller.pause()
        assert len(controller._tick_timestamps) == 0  # noqa: SLF001

    def test_seek_clears_deque(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """seek() needs a real sequence to do anything; we simulate the
        path by attaching a Mock sequence and asserting the deque is
        cleared after the call."""
        cache = MagicMock(spec=FrameCache)
        controller = PlayerController(cache)

        # Fake a loaded sequence so seek() doesn't early-return.
        seq = MagicMock()
        seq.first_frame = 1
        seq.last_frame = 100
        controller._sequence = seq  # noqa: SLF001
        controller._state = replace(controller._state, in_frame=None, out_frame=None)  # noqa: SLF001
        controller._tick_timestamps.extend(range(24))  # noqa: SLF001

        controller.seek(50)
        assert len(controller._tick_timestamps) == 0  # noqa: SLF001
