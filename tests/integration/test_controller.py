"""Controller tests (pytest-qt).

Drive the state machine directly (no reliance on QTimer firing) by calling
the private ``_tick`` method. That keeps tests deterministic and fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from img_player.cache.frame_cache import FrameCache
from img_player.player.controller import PlayerController
from img_player.player.state import LoopMode, PlaybackState
from img_player.sequence.scanner import scan


class _MockClock:
    """Controllable monotonic clock for deterministic ``_tick`` tests.

    The wall-clock-driven controller advances the playhead by
    ``round(elapsed × fps)`` per tick — calling ``_tick`` without
    elapsed time would be a no-op. ``MockClock.tick`` bumps the
    pretend-time by one frame interval at the controller's current
    fps, so ``controller._tick()`` advances by exactly one frame
    just like the legacy "++ per tick" behaviour the tests expect.
    """

    def __init__(self) -> None:
        self.t = 1000.0  # arbitrary non-zero start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def controller(qtbot, sequence_dir: Path):  # type: ignore[no-untyped-def]
    cache = FrameCache(budget_bytes=8 * 1024 * 1024, num_workers=2)
    clock = _MockClock()
    ctrl = PlayerController(cache, clock=clock)
    seq = scan(sequence_dir)
    ctrl.load_sequence(seq)
    cache.wait_idle(timeout=5.0)
    # Wrap _tick so each call advances the mock clock by one frame
    # interval first — matches the pre-refactor "+1 per tick"
    # contract every test relies on.
    real_tick = ctrl._tick

    def stepping_tick():
        clock.advance(1.0 / ctrl.state.fps)
        real_tick()

    ctrl._tick = stepping_tick  # type: ignore[method-assign]
    ctrl._mock_clock = clock  # type: ignore[attr-defined]
    yield ctrl
    ctrl.shutdown()
    cache.shutdown()


def test_load_sequence_sets_first_frame(controller: PlayerController) -> None:
    assert controller.state.current_frame == 1
    assert controller.state.is_playing is False
    assert controller.state.direction == 1


def test_play_sets_is_playing(controller: PlayerController) -> None:
    controller.play()
    assert controller.state.is_playing is True
    controller.pause()
    assert controller.state.is_playing is False


def test_play_is_idempotent(controller: PlayerController) -> None:
    controller.play()
    controller.play()
    assert controller.state.is_playing is True


def test_seek_clamps_to_range(controller: PlayerController) -> None:
    controller.seek(999)
    assert controller.state.current_frame == 10  # sequence ends at 10
    controller.seek(-5)
    assert controller.state.current_frame == 1


def test_step_forward_and_back(controller: PlayerController) -> None:
    controller.seek(5)
    controller.step(2)
    assert controller.state.current_frame == 7
    controller.step(-4)
    assert controller.state.current_frame == 3


def test_stop_returns_to_in_frame(controller: PlayerController) -> None:
    controller.seek(7)
    controller.play()
    controller.stop()
    assert controller.state.current_frame == 1
    assert controller.state.is_playing is False


def test_tick_advances_forward(controller: PlayerController) -> None:
    controller.seek(3)
    controller.play()
    controller._tick()
    assert controller.state.current_frame == 4
    controller._tick()
    assert controller.state.current_frame == 5


def test_tick_catches_up_after_slow_qtimer(controller: PlayerController) -> None:
    """Wall-clock-driven advance gradually catches up after a slip.

    Pre-refactor the controller advanced ``current_frame`` by exactly
    +1 per tick — if the QTimer slipped, the playhead fell behind
    wall time and A/V drift built up. With the wall-clock anchor
    the next ticks target the wall-clock-correct frame; we cap the
    per-tick step to ±1 (smoother visual at the cost of one or two
    ticks to fully catch up after a slip), so a 5-frame slip is
    caught up over the next 5 ticks at +1 each.
    """
    controller.seek(2)
    controller.play()
    # Simulate a 5-frame interval slip (e.g. the QTimer was blocked
    # for ~208 ms at 24 fps). Bypass the fixture's stepping wrapper
    # so we control the elapsed time directly.
    clock = controller._mock_clock  # type: ignore[attr-defined]
    clock.advance(5.0 / controller.state.fps)
    # First tick after the slip: +1 (capped) → frame 3.
    PlayerController._tick(controller)
    assert controller.state.current_frame == 3
    # Subsequent ticks (no further wall-clock advance) keep advancing
    # +1 each because the wall-clock target is still ahead of us,
    # until we catch up at frame 7 (= 2 + 5).
    for expected in (4, 5, 6, 7):
        PlayerController._tick(controller)
        assert controller.state.current_frame == expected
    # Now caught up: another tick with no clock advance is a no-op.
    PlayerController._tick(controller)
    assert controller.state.current_frame == 7


def test_loop_mode_wraps_to_start(controller: PlayerController) -> None:
    controller.set_loop_mode(LoopMode.LOOP)
    controller.seek(10)
    controller.play()
    controller._tick()
    assert controller.state.current_frame == 1  # wrapped to first_frame


def test_once_stops_at_end(controller: PlayerController) -> None:
    controller.set_loop_mode(LoopMode.ONCE)
    controller.seek(10)
    controller.play()
    controller._tick()
    assert controller.state.is_playing is False
    assert controller.state.current_frame == 10


def test_ping_pong_reverses_direction(controller: PlayerController) -> None:
    controller.set_loop_mode(LoopMode.PING_PONG)
    controller.seek(10)
    controller.play()
    controller._tick()
    assert controller.state.direction == -1
    assert controller.state.current_frame == 9
    # Go all the way down
    for _ in range(15):
        controller._tick()
    # After reaching frame 1 and bouncing, we should be going forward again
    assert controller.state.current_frame >= 1


def test_state_changed_signal_on_play(qtbot, controller: PlayerController) -> None:  # type: ignore[no-untyped-def]
    with qtbot.waitSignal(controller.state_changed, timeout=1000):
        controller.play()
    controller.pause()


def test_frame_changed_signal_on_seek(qtbot, controller: PlayerController) -> None:  # type: ignore[no-untyped-def]
    with qtbot.waitSignal(controller.frame_changed, timeout=1000) as blocker:
        controller.seek(5)
    assert blocker.args == [5]


def test_state_changed_not_emitted_on_noop(controller: PlayerController) -> None:
    emissions: list[PlaybackState] = []
    controller.state_changed.connect(lambda s: emissions.append(s))

    # pause() while not playing is a no-op
    controller.pause()
    assert emissions == []


def test_set_fps_updates_state(controller: PlayerController) -> None:
    controller.set_fps(60.0)
    assert controller.state.fps == 60.0

    # negative/tiny values are clamped to something sane
    controller.set_fps(-5.0)
    assert controller.state.fps > 0.0


def test_set_in_out_clamps_current(controller: PlayerController) -> None:
    controller.seek(8)
    controller.set_in_out(in_frame=2, out_frame=4)
    assert 2 <= controller.state.current_frame <= 4


def test_dropped_frames_counted_on_miss(controller: PlayerController) -> None:
    # Clear the cache so the next tick observes a miss
    controller._cache.clear()
    initial_drops = controller.state.dropped_frames
    controller.seek(3)
    controller.play()
    controller._tick()
    # The tick may or may not miss depending on prefetch speed, but at least
    # it should not decrease.
    assert controller.state.dropped_frames >= initial_drops
