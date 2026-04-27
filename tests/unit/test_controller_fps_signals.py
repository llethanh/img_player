"""Tests for the live-metric Qt signals on PlayerController (slice 5).

The controller exposes two signals for the UI status bar to consume:

* :pyattr:`PlayerController.effective_fps_changed` — emits the rolling
  effective FPS at most once a second while playing.
* :pyattr:`PlayerController.cache_hit_rate_changed` — emits the
  rolling cache-hit rate, same throttle.

We don't drive the QTimer / event loop here — we exercise the
``_tick`` body directly with a controller in a known state. This
keeps the tests fast and removes Qt timing flake.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from img_player.cache.frame_cache import FrameCache
from img_player.player.controller import PlayerController


@pytest.fixture
def controller(qtbot) -> PlayerController:  # type: ignore[no-untyped-def]
    """A bare controller with a mocked cache + a synthetic sequence."""
    cache = MagicMock(spec=FrameCache)
    cache.contains.return_value = True
    cache.get.return_value = None
    ctl = PlayerController(cache)
    # Force the controller into "playing" with a tiny sequence so _tick runs.
    seq = MagicMock()
    seq.first_frame = 1
    seq.last_frame = 100
    seq.frame_count = 100
    ctl._sequence = seq
    ctl._state = replace(ctl._state, is_playing=True, current_frame=1, direction=1)
    ctl._cache = cache
    return ctl


# ============================================================================
# Helpers
# ============================================================================


def _force_metric_throttle_open(ctl: PlayerController) -> None:
    """Trick the throttle into thinking the last emit was long ago,
    so the next ``_tick`` will actually emit. Used between simulated
    ticks in the tests below."""
    ctl._last_metric_emit = 0.0


# ============================================================================
# Cache hit rate rolling window
# ============================================================================


class TestCacheHitRate:
    def test_returns_none_with_too_few_samples(self, controller: PlayerController) -> None:
        """Below 4 samples, the rate is too noisy to report."""
        controller._tick_hits.extend([True, True, True])
        assert controller.cache_hit_rate() is None

    def test_returns_full_hit_rate(self, controller: PlayerController) -> None:
        controller._tick_hits.extend([True] * 24)
        assert controller.cache_hit_rate() == pytest.approx(1.0)

    def test_returns_zero_on_all_misses(self, controller: PlayerController) -> None:
        controller._tick_hits.extend([False] * 24)
        assert controller.cache_hit_rate() == pytest.approx(0.0)

    def test_returns_partial_hit_rate(self, controller: PlayerController) -> None:
        # 18 hits, 6 misses out of 24 → 0.75
        controller._tick_hits.extend([True] * 18 + [False] * 6)
        assert controller.cache_hit_rate() == pytest.approx(0.75, abs=0.01)


# ============================================================================
# Prefetch ahead getter / setter
# ============================================================================


class TestPrefetchAhead:
    def test_default_matches_class_constant(self, controller: PlayerController) -> None:
        assert controller.get_prefetch_ahead() == PlayerController.PREFETCH_AHEAD

    def test_setter_propagates(self, controller: PlayerController) -> None:
        controller.set_prefetch_ahead(32)
        assert controller.get_prefetch_ahead() == 32

    def test_setter_floors_at_four(self, controller: PlayerController) -> None:
        controller.set_prefetch_ahead(1)
        assert controller.get_prefetch_ahead() == 4

    def test_setter_handles_zero(self, controller: PlayerController) -> None:
        controller.set_prefetch_ahead(0)
        assert controller.get_prefetch_ahead() == 4


# ============================================================================
# Signal emission via _tick
# ============================================================================


class TestMetricSignals:
    def test_throttle_blocks_back_to_back_emits(
        self, controller: PlayerController, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """Two ticks in quick succession should yield at most ONE emit
        of each signal, because of the 1-second throttle."""
        # Pre-seed the deques so the rolling metrics return real values.
        import time as time_mod
        now = time_mod.monotonic()
        controller._tick_timestamps.extend(
            [now - 0.5, now - 0.4, now - 0.3, now - 0.2, now - 0.1, now]
        )
        controller._tick_hits.extend([True] * 6)
        # First tick: simulate that we just emitted (recent _last_metric_emit).
        controller._last_metric_emit = now - 0.1

        # Run _tick — should NOT emit because <1 s since last emit.
        with qtbot.assertNotEmitted(controller.effective_fps_changed):
            controller._tick()

    def test_emit_after_throttle_window(
        self, controller: PlayerController, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """If more than 1 s elapsed since the last emit, the next tick
        should emit BOTH metric signals."""
        # Pre-seed enough history so both metrics return values.
        import time as time_mod
        now = time_mod.monotonic()
        controller._tick_timestamps.extend(
            [now - 1.0 + i * 0.04 for i in range(10)]
        )
        controller._tick_hits.extend([True] * 10)
        _force_metric_throttle_open(controller)

        with qtbot.waitSignal(controller.effective_fps_changed, timeout=500):
            with qtbot.waitSignal(controller.cache_hit_rate_changed, timeout=500):
                controller._tick()

    def test_emits_zero_fps_silenced_until_two_samples(
        self, controller: PlayerController, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """A single tick alone produces ``effective_fps()=None`` so the
        signal must not fire — emitting NaN to the UI would be worse
        than emitting nothing."""
        _force_metric_throttle_open(controller)
        # Only one timestamp will be in the deque after _tick (since
        # we start empty and _tick appends one).
        with qtbot.assertNotEmitted(controller.effective_fps_changed):
            controller._tick()
