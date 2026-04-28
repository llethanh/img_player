"""Tests for :class:`img_player.annotate.ephemeral.EphemeralStrokeManager`.

Level 2 of the ephemeral feature's testing strategy (spec §8.2). We
monkeypatch ``time.monotonic`` in the ``ephemeral`` module so virtual
time can be advanced deterministically — without sleeping the test
suite.

Three groups:

* lifecycle — add / kill_last / clear_all + auto start/stop of timer;
* alpha snapshot — ``live_strokes_with_alpha`` filters expired,
  preserves order, respects ``set_duration``;
* tick behaviour — ``_on_tick`` sweeps expired and self-stops.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from img_player.annotate.ephemeral import EphemeralStrokeManager
from img_player.annotate.stroke import Stroke


# ============================================================================
# Fixtures
# ============================================================================


def _stroke(color: str = "#FF0000") -> Stroke:
    """Quick stroke factory — geometry doesn't matter for these tests."""
    return Stroke(points=((0.0, 0.0), (10.0, 10.0)), color=color, size=5.0)


class _Clock:
    """Mutable clock the tests advance manually."""

    def __init__(self, t: float = 0.0) -> None:
        self.now = t

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Clock]:
    """Patch ``time.monotonic`` *as imported by the ephemeral module*.

    The manager's ``add()`` reads ``time.monotonic()`` to stamp birth,
    and ``_on_tick`` reads it to compute current alpha. Patching the
    module-local reference (not the global ``time`` module) is the
    cleanest way: any other code in the test process keeps the real
    clock.
    """
    c = _Clock()
    monkeypatch.setattr(
        "img_player.annotate.ephemeral.time.monotonic", c
    )
    yield c


@pytest.fixture
def manager(qtbot, clock: _Clock) -> EphemeralStrokeManager:  # type: ignore[no-untyped-def]
    """Fresh manager with the default 5 s duration and a clock at t=0."""
    m = EphemeralStrokeManager()
    qtbot.addWidget(m) if False else None  # QObject, not QWidget — qtbot is for the event loop
    return m


# ============================================================================
# Lifecycle
# ============================================================================


class TestLifecycle:
    def test_starts_idle(self, manager: EphemeralStrokeManager) -> None:
        """Fresh manager has no strokes and an idle timer."""
        assert manager.has_live_strokes() is False
        assert manager._stroke_count() == 0
        assert manager._is_timer_active() is False

    def test_add_starts_timer_and_emits_repaint(
        self, manager: EphemeralStrokeManager, qtbot
    ) -> None:
        """First ``add()`` wakes the timer and signals a repaint."""
        with qtbot.waitSignal(manager.repaint_needed, timeout=200):
            manager.add(_stroke())
        assert manager._is_timer_active() is True
        assert manager._stroke_count() == 1

    def test_consecutive_adds_keep_timer_running(
        self, manager: EphemeralStrokeManager
    ) -> None:
        """Multiple adds don't restart the timer — it stays active throughout."""
        manager.add(_stroke())
        manager.add(_stroke())
        manager.add(_stroke())
        assert manager._is_timer_active() is True
        assert manager._stroke_count() == 3

    def test_kill_last_removes_youngest(
        self, manager: EphemeralStrokeManager, clock: _Clock
    ) -> None:
        """``kill_last`` pops the most recently added stroke."""
        manager.add(_stroke(color="#A1A1A1"))
        clock.advance(0.5)
        manager.add(_stroke(color="#B2B2B2"))
        assert manager.kill_last() is True
        # The remaining stroke is the older one (#A1).
        live = manager.live_strokes_with_alpha(now_ts=clock.now)
        assert len(live) == 1
        assert live[0][0].color == "#A1A1A1"

    def test_kill_last_on_empty_returns_false(
        self, manager: EphemeralStrokeManager
    ) -> None:
        """No-op + signals nothing changed."""
        assert manager.kill_last() is False

    def test_kill_last_stops_timer_when_list_empties(
        self, manager: EphemeralStrokeManager
    ) -> None:
        manager.add(_stroke())
        assert manager._is_timer_active() is True
        manager.kill_last()
        assert manager._is_timer_active() is False

    def test_clear_all_returns_count_and_stops_timer(
        self, manager: EphemeralStrokeManager
    ) -> None:
        manager.add(_stroke())
        manager.add(_stroke())
        manager.add(_stroke())
        assert manager.clear_all() == 3
        assert manager._stroke_count() == 0
        assert manager._is_timer_active() is False

    def test_clear_all_on_empty_returns_zero(
        self, manager: EphemeralStrokeManager
    ) -> None:
        assert manager.clear_all() == 0


# ============================================================================
# Alpha snapshot
# ============================================================================


class TestLiveStrokesWithAlpha:
    def test_returns_tuple_immutable(
        self, manager: EphemeralStrokeManager, clock: _Clock
    ) -> None:
        """Read API returns a tuple so callers can't mutate internal state."""
        manager.add(_stroke())
        result = manager.live_strokes_with_alpha(now_ts=clock.now)
        assert isinstance(result, tuple)

    def test_insertion_order_preserved(
        self, manager: EphemeralStrokeManager, clock: _Clock
    ) -> None:
        """Order of (stroke, alpha) tuples follows insertion order so
        the overlay paints earlier strokes underneath."""
        manager.add(_stroke(color="#A1A1A1"))
        clock.advance(0.1)
        manager.add(_stroke(color="#B2B2B2"))
        clock.advance(0.1)
        manager.add(_stroke(color="#C3C3C3"))
        live = manager.live_strokes_with_alpha(now_ts=clock.now)
        assert [s.color for s, _ in live] == ["#A1A1A1", "#B2B2B2", "#C3C3C3"]

    def test_filters_out_expired(
        self, manager: EphemeralStrokeManager, clock: _Clock
    ) -> None:
        """Strokes whose alpha is 0 are filtered from the snapshot
        even before the timer sweeps them."""
        manager.set_duration(2.0)
        manager.add(_stroke(color="#0D0D0D"))  # "OLD"
        clock.advance(1.0)
        manager.add(_stroke(color="#9E9E9E"))  # "NEW"
        # OLD is now 1s old (alpha 0.5), NEW is 0s old (alpha 1.0).
        # Advance another 1.5s: OLD is 2.5s old → expired.
        clock.advance(1.5)
        live = manager.live_strokes_with_alpha(now_ts=clock.now)
        # Only NEW should remain visible (1.5s/2.0s = 0.25 alpha).
        assert len(live) == 1
        assert live[0][0].color == "#9E9E9E"
        assert live[0][1] == pytest.approx(0.25)

    def test_alpha_reflects_age(
        self, manager: EphemeralStrokeManager, clock: _Clock
    ) -> None:
        """Linear fade — at half-duration, alpha is 0.5."""
        manager.set_duration(4.0)
        manager.add(_stroke())
        clock.advance(2.0)
        live = manager.live_strokes_with_alpha(now_ts=clock.now)
        assert live[0][1] == pytest.approx(0.5)

    def test_set_duration_recomputes_alpha_for_live(
        self, manager: EphemeralStrokeManager, clock: _Clock
    ) -> None:
        """Changing duration mid-life updates the alpha of in-flight strokes."""
        manager.set_duration(10.0)
        manager.add(_stroke())
        clock.advance(2.0)
        # At 10s duration, age 2s → alpha 0.8
        live_long = manager.live_strokes_with_alpha(now_ts=clock.now)
        assert live_long[0][1] == pytest.approx(0.8)
        # Shrink to 5s — age 2s → alpha 0.6
        manager.set_duration(5.0)
        live_short = manager.live_strokes_with_alpha(now_ts=clock.now)
        assert live_short[0][1] == pytest.approx(0.6)

    def test_uses_real_monotonic_when_now_omitted(
        self, manager: EphemeralStrokeManager
    ) -> None:
        """Production callers don't pass ``now_ts`` — the manager
        reads the (possibly patched) ``time.monotonic`` itself."""
        manager.add(_stroke())
        # We don't advance the clock fixture, so age == 0 → alpha 1.0
        live = manager.live_strokes_with_alpha()
        assert len(live) == 1
        assert live[0][1] == pytest.approx(1.0)


# ============================================================================
# Tick behaviour
# ============================================================================


class TestTick:
    def test_tick_sweeps_expired(
        self, manager: EphemeralStrokeManager, clock: _Clock
    ) -> None:
        """The internal sweep removes strokes past their duration."""
        manager.set_duration(2.0)
        manager.add(_stroke(color="#A1A1A1"))
        clock.advance(1.0)
        manager.add(_stroke(color="#B2B2B2"))
        clock.advance(1.5)
        # A1 is 2.5s old → expired; B2 is 1.5s old → still visible.
        manager._on_tick()
        assert manager._stroke_count() == 1
        live = manager.live_strokes_with_alpha(now_ts=clock.now)
        assert live[0][0].color == "#B2B2B2"

    def test_tick_self_stops_when_empty(
        self, manager: EphemeralStrokeManager, clock: _Clock
    ) -> None:
        """When a tick clears the last stroke, the timer is stopped
        so we don't poll while idle."""
        manager.set_duration(1.0)
        manager.add(_stroke())
        assert manager._is_timer_active() is True
        clock.advance(2.0)
        manager._on_tick()
        assert manager._stroke_count() == 0
        assert manager._is_timer_active() is False

    def test_tick_emits_repaint(
        self, manager: EphemeralStrokeManager, qtbot
    ) -> None:
        """Every tick triggers a repaint so the alpha animates smoothly."""
        manager.add(_stroke())
        with qtbot.waitSignal(manager.repaint_needed, timeout=200):
            manager._on_tick()


# ============================================================================
# Duration setter
# ============================================================================


class TestSetDuration:
    def test_default_duration_is_5s(
        self, manager: EphemeralStrokeManager
    ) -> None:
        """Boot default — the spec's "moyen" preset."""
        assert manager.duration() == pytest.approx(5.0)

    def test_set_duration_updates_value(
        self, manager: EphemeralStrokeManager
    ) -> None:
        manager.set_duration(2.0)
        assert manager.duration() == pytest.approx(2.0)
        manager.set_duration(10.0)
        assert manager.duration() == pytest.approx(10.0)

    def test_set_duration_rejects_non_float(
        self, manager: EphemeralStrokeManager
    ) -> None:
        """Defensive: a non-numeric input shouldn't crash the call."""
        manager.set_duration(5.0)
        manager.set_duration("not a number")  # type: ignore[arg-type]
        assert manager.duration() == pytest.approx(5.0)
