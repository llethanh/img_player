"""Tests for the controller's prefetch scheduling.

We exercise the *priority* shape of what gets submitted, not the
real cache decode path — that's covered by ``test_frame_cache``.
A ``MagicMock`` stands in for the cache, and we inspect the calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pathlib import Path

from img_player.cache.frame_cache import FrameCache
from img_player.player.controller import PlayerController
from img_player.sequence.models import FrameInfo, SequenceInfo


def _make_sequence(first: int = 1, last: int = 90) -> SequenceInfo:
    """Build a SequenceInfo with synthetic FrameInfo stubs.

    The controller never reads pixel data — it only consults the
    frame range and forwards requests to the cache — so a stub
    path per frame is plenty.
    """
    frames = tuple(
        FrameInfo(path=Path(f"/fake/render.{f:04d}.exr"), frame_number=f)
        for f in range(first, last + 1)
    )
    return SequenceInfo(
        base_name="render.",
        extension=".exr",
        directory=Path("/fake"),
        padding=4,
        frames=frames,
        channel_names=("R", "G", "B", "A"),
    )


@pytest.fixture
def controller_with_cache(qtbot) -> tuple[PlayerController, MagicMock]:  # type: ignore[no-untyped-def]
    cache = MagicMock(spec=FrameCache)
    ctl = PlayerController(cache)
    return ctl, cache


def _frames_requested(cache: MagicMock) -> dict[int, int]:
    """Map of frame → priority across all ``cache.request`` calls.

    Later submissions overwrite earlier ones in the dict, which is
    fine for our priority assertions: we just want to know what
    priority a given frame ended up at.
    """
    out: dict[int, int] = {}
    for call in cache.request.call_args_list:
        # signature: request(frame, priority=...)
        if call.args:
            frame = call.args[0]
        else:
            frame = call.kwargs["frame"]
        priority = call.kwargs.get("priority", 0)
        if len(call.args) >= 2:
            priority = call.args[1]
        out[frame] = priority
    return out


class TestFullSequencePrefetch:
    def test_load_schedules_every_frame(
        self, controller_with_cache: tuple[PlayerController, MagicMock],
    ) -> None:
        ctl, cache = controller_with_cache
        seq = _make_sequence(first=1001, last=1090)
        ctl.load_sequence(seq)

        scheduled = _frames_requested(cache)
        # Every frame in the sequence range must be scheduled at
        # least once — otherwise the user can end up staring at a
        # permanent gap on the cache bar.
        assert set(scheduled.keys()) == set(range(1001, 1091))

    def test_seek_reschedules_far_frames(
        self, controller_with_cache: tuple[PlayerController, MagicMock],
    ) -> None:
        # The actual user bug: load → seek to the end →
        # frames in the middle (between the previous prefetch
        # high-water mark and the new playhead) must still be
        # scheduled, otherwise they never come back.
        ctl, cache = controller_with_cache
        seq = _make_sequence(first=1001, last=1090)
        ctl.load_sequence(seq)
        cache.reset_mock()

        ctl.seek(1085)
        scheduled = _frames_requested(cache)
        # The whole sequence must be re-planned, not just the
        # close window around 1085.
        assert set(scheduled.keys()) == set(range(1001, 1091))

    def test_close_frames_get_lowest_priority(
        self, controller_with_cache: tuple[PlayerController, MagicMock],
    ) -> None:
        ctl, cache = controller_with_cache
        seq = _make_sequence(first=1, last=200)
        ctl.load_sequence(seq)
        cache.reset_mock()

        ctl.seek(100)
        scheduled = _frames_requested(cache)
        # The playhead frame is priority 0 — the worker pool
        # serves the smallest priority first, so this is the
        # frame that gets decoded immediately.
        assert scheduled[100] == 0
        # Frame 110 is 10 ahead of the playhead → priority 10.
        assert scheduled[110] == 10
        # Frame 90 is 10 behind → penalised by the behind-multiplier.
        # We don't pin the exact constant in the test (it's tunable),
        # but it must beat any forward frame at the same delta.
        assert scheduled[90] > scheduled[110]

    def test_reverse_play_inverts_priority(
        self, controller_with_cache: tuple[PlayerController, MagicMock],
    ) -> None:
        ctl, cache = controller_with_cache
        seq = _make_sequence(first=1, last=200)
        ctl.load_sequence(seq)
        ctl.set_direction(-1)
        cache.reset_mock()

        ctl.seek(100)
        scheduled = _frames_requested(cache)
        # In reverse play, frames *behind* the playhead (lower
        # frame numbers) are now the "ahead" ones — they should
        # decode first.
        assert scheduled[90] < scheduled[110]
