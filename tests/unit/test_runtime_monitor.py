"""Tests for ``perf.runtime_monitor.RuntimeMonitor`` (slice 5).

The monitor is a 1 Hz watchdog that auto-corrects mid-playback on
two signals: cache hit rate and swap usage. We don't want to actually
wait one second per tick in tests — that would make the suite slow
and flaky. Instead we drive ``_tick()`` directly and feed mocked
controller / cache state to exercise each branch.

Three groups:

* state transitions — play_started / play_stopped lifecycle and
  the 1 Hz timer being correctly armed;
* cache hit rate branch — threshold detection, prefetch shrink at
  5 s, ``playback_struggle`` warning at 10 s, recovery, and the
  no-spam latch;
* swap pressure branch — delta from the play-start baseline,
  cache shrink + ``memory_pressure`` warning, and the no-spam
  latch.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from img_player.cache.frame_cache import FrameCache
from img_player.perf.runtime_monitor import RuntimeMonitor
from img_player.player.controller import PlayerController
from img_player.player.state import PlaybackState

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cache(qtbot) -> MagicMock:  # type: ignore[no-untyped-def]
    """A mocked FrameCache with a writable budget attribute."""
    c = MagicMock(spec=FrameCache)
    # The monitor reads `_budget` directly (private but documented as
    # a coordination point). MagicMock won't auto-create non-spec'd
    # attributes, so we set it explicitly.
    c._budget = 8 * 1024**3
    return c


@pytest.fixture
def controller(qtbot) -> MagicMock:  # type: ignore[no-untyped-def]
    """A mocked controller that the monitor talks to."""
    c = MagicMock(spec=PlayerController)
    # ``state_changed`` is a Qt signal on the real class — MagicMock
    # provides a connect/emit-compatible stub by default.
    c.cache_hit_rate.return_value = 1.0  # default: all hits, healthy
    c.get_prefetch_ahead.return_value = 64
    return c


@pytest.fixture
def monitor(controller: MagicMock, cache: MagicMock, qtbot) -> RuntimeMonitor:  # type: ignore[no-untyped-def]
    return RuntimeMonitor(controller, cache)


def _start_playing(monitor: RuntimeMonitor, swap_gb: float = 0.0) -> None:
    """Drive the monitor into "playing" state with a known swap baseline."""
    import img_player.perf.runtime_monitor as mod
    # Replace the module-level swap reader so we control what
    # _on_play_started captures as the baseline. The default 0.0 GB
    # mirrors a fresh boot.
    mod._read_swap_used_gb = lambda: swap_gb
    state = PlaybackState(is_playing=True, current_frame=1)
    monitor._on_state_changed(state)


def _set_swap(value_gb: float) -> None:
    """Update the mocked swap reader for the rest of the test."""
    import img_player.perf.runtime_monitor as mod
    mod._read_swap_used_gb = lambda: value_gb


# ============================================================================
# State transitions
# ============================================================================


class TestStateTransitions:
    def test_timer_not_armed_at_construction(self, monitor: RuntimeMonitor) -> None:
        assert monitor.is_running() is False

    def test_play_started_arms_timer(self, monitor: RuntimeMonitor) -> None:
        _start_playing(monitor)
        assert monitor.is_running() is True

    def test_play_stopped_disarms_timer(self, monitor: RuntimeMonitor) -> None:
        _start_playing(monitor)
        state = PlaybackState(is_playing=False)
        monitor._on_state_changed(state)
        assert monitor.is_running() is False

    def test_consecutive_play_started_is_idempotent(
        self, monitor: RuntimeMonitor,
    ) -> None:
        """A second ``state_changed`` while already playing must not
        re-capture the swap baseline (which would erase any growth
        we'd accumulated during the session)."""
        _start_playing(monitor, swap_gb=0.5)
        baseline_first = monitor._swap_at_play_start
        # Re-emit with the same is_playing flag.
        _set_swap(2.0)
        monitor._on_state_changed(PlaybackState(is_playing=True, current_frame=1))
        assert monitor._swap_at_play_start == baseline_first


# ============================================================================
# Cache hit rate branch
# ============================================================================


class TestCacheHitRate:
    def test_above_threshold_does_nothing(
        self, monitor: RuntimeMonitor, controller: MagicMock,
    ) -> None:
        """Healthy hit rate → no shrink, no emit."""
        _start_playing(monitor)
        controller.cache_hit_rate.return_value = 0.95
        monitor._tick()
        controller.set_prefetch_ahead.assert_not_called()

    def test_below_threshold_brief_does_not_shrink(
        self, monitor: RuntimeMonitor, controller: MagicMock,
    ) -> None:
        """A single sub-threshold tick is just noise — don't act."""
        _start_playing(monitor)
        controller.cache_hit_rate.return_value = 0.5
        monitor._tick()
        controller.set_prefetch_ahead.assert_not_called()

    def test_sustained_low_rate_shrinks_prefetch(
        self, monitor: RuntimeMonitor, controller: MagicMock,
    ) -> None:
        """After 5 s of sub-threshold rate, the monitor reduces the
        prefetch window by 25 %."""
        import time as time_mod
        _start_playing(monitor)
        controller.cache_hit_rate.return_value = 0.5
        controller.get_prefetch_ahead.return_value = 64

        # Simulate the 5-second window having elapsed by setting
        # _hit_rate_low_since to 5.5 s ago.
        monitor._hit_rate_low_since = time_mod.monotonic() - 5.5
        monitor._tick()
        # 64 * 0.75 = 48
        controller.set_prefetch_ahead.assert_called_once_with(48)

    def test_sustained_low_rate_only_shrinks_once(
        self, monitor: RuntimeMonitor, controller: MagicMock,
    ) -> None:
        """The shrink is one-shot per session — repeated low-rate
        ticks must not zero the prefetch out."""
        import time as time_mod
        _start_playing(monitor)
        controller.cache_hit_rate.return_value = 0.5
        controller.get_prefetch_ahead.return_value = 64
        monitor._hit_rate_low_since = time_mod.monotonic() - 5.5
        monitor._tick()
        monitor._tick()
        monitor._tick()
        # set_prefetch_ahead called at most once
        assert controller.set_prefetch_ahead.call_count == 1

    def test_long_sustained_low_rate_emits_struggle(
        self, monitor: RuntimeMonitor, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        import time as time_mod
        _start_playing(monitor)
        monitor._controller.cache_hit_rate.return_value = 0.5
        monitor._hit_rate_low_since = time_mod.monotonic() - 11.0

        with qtbot.waitSignal(monitor.playback_struggle, timeout=500) as blocker:
            monitor._tick()
        assert "irrégulière" in blocker.args[0].lower()

    def test_struggle_emitted_only_once_per_session(
        self, monitor: RuntimeMonitor, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """A second tick after the warning has fired must not re-emit."""
        import time as time_mod
        _start_playing(monitor)
        monitor._controller.cache_hit_rate.return_value = 0.5
        monitor._hit_rate_low_since = time_mod.monotonic() - 11.0

        emitted: list[str] = []
        monitor.playback_struggle.connect(emitted.append)
        monitor._tick()
        monitor._tick()
        assert len(emitted) == 1

    def test_recovery_clears_low_since_but_keeps_prefetch_shrink(
        self, monitor: RuntimeMonitor, controller: MagicMock,
    ) -> None:
        """When the rate recovers, the timer resets, but the prefetch
        window stays shrunk for the session (no auto-grow)."""
        import time as time_mod
        _start_playing(monitor)
        controller.cache_hit_rate.return_value = 0.5
        controller.get_prefetch_ahead.return_value = 64
        monitor._hit_rate_low_since = time_mod.monotonic() - 5.5
        monitor._tick()  # shrinks
        # Now the rate recovers.
        controller.cache_hit_rate.return_value = 0.95
        monitor._tick()
        assert monitor._hit_rate_low_since is None
        # No follow-up call to set_prefetch_ahead (which would grow it back).
        assert controller.set_prefetch_ahead.call_count == 1


# ============================================================================
# Swap pressure branch
# ============================================================================


class TestSwapPressure:
    def test_no_growth_does_nothing(
        self, monitor: RuntimeMonitor, cache: MagicMock,
    ) -> None:
        _start_playing(monitor, swap_gb=0.5)
        # Swap unchanged.
        monitor._tick()
        cache.shrink_budget.assert_not_called()

    def test_small_growth_below_threshold_does_nothing(
        self, monitor: RuntimeMonitor, cache: MagicMock,
    ) -> None:
        _start_playing(monitor, swap_gb=0.5)
        # Grew by 100 MB only — below the 500 MB threshold.
        _set_swap(0.6)
        monitor._tick()
        cache.shrink_budget.assert_not_called()

    def test_growth_above_threshold_shrinks_cache_and_warns(
        self, monitor: RuntimeMonitor, cache: MagicMock, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        _start_playing(monitor, swap_gb=0.5)
        _set_swap(1.5)  # +1 GB during playback
        with qtbot.waitSignal(monitor.memory_pressure, timeout=500) as blocker:
            monitor._tick()
        # 8 GB * 0.75 = 6 GB
        cache.shrink_budget.assert_called_once_with(int(8 * 1024**3 * 0.75))
        assert "mémoire" in blocker.args[0].lower()

    def test_memory_pressure_emitted_only_once(
        self, monitor: RuntimeMonitor, cache: MagicMock, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        _start_playing(monitor, swap_gb=0.5)
        _set_swap(1.5)
        emitted: list[str] = []
        monitor.memory_pressure.connect(emitted.append)
        monitor._tick()
        # Even if swap keeps growing, we don't spam.
        _set_swap(3.0)
        monitor._tick()
        assert len(emitted) == 1
        assert cache.shrink_budget.call_count == 1
