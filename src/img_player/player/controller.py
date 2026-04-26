"""Playback orchestration: a QObject that drives the cache from a QTimer.

Emits Qt signals when the current frame or playback state changes. UI
widgets subscribe to ``frame_changed`` (for rendering) and ``state_changed``
(for transport buttons / status display).
"""

from __future__ import annotations

import gc
import logging
import time
from collections import deque
from dataclasses import replace
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from img_player.bench import recorder
from img_player.cache.frame_cache import FrameCache
from img_player.player.state import LoopMode, PlaybackState
from img_player.sequence.models import SequenceInfo

log = logging.getLogger(__name__)

# Rolling window of tick timestamps used to compute the live effective fps.
# 24 samples ~= 1 second at 24 fps — the right bin for "is playback
# smooth right now". Bigger would smooth too much (slow to react to a
# stutter), smaller would be twitchy.
_TICK_WINDOW = 24


class PlayerController(QObject):  # type: ignore[misc]  # mypy: QObject is Any
    """Drives a :class:`FrameCache` to emit frames at the requested FPS.

    Not a renderer — just the control layer. Downstream consumers listen to
    ``frame_changed`` and pull the array out of the cache themselves.
    """

    PREFETCH_AHEAD = 64
    PREFETCH_BEHIND = 8

    frame_changed = Signal(int)
    state_changed = Signal(object)  # emits PlaybackState

    def __init__(self, cache: FrameCache, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._cache = cache
        self._sequence: SequenceInfo | None = None
        self._state = PlaybackState()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        # Rolling window of monotonic timestamps captured at each tick.
        # Drives :meth:`effective_fps` so the UI can show what the play
        # loop is actually delivering, independent of the target FPS
        # combo. Cleared on play / pause / seek so the metric reflects
        # the *current* playback, not stale data from a previous run.
        self._tick_timestamps: deque[float] = deque(maxlen=_TICK_WINDOW)

    # ------------------------------------------------------------------ Properties

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def sequence(self) -> SequenceInfo | None:
        return self._sequence

    def effective_fps(self) -> float | None:
        """Rolling-average effective playback FPS.

        Returns ``None`` when not enough samples are available — i.e. not
        currently playing, or fewer than two ticks have fired since the
        last play/pause/seek. Otherwise returns the average rate over the
        last ``_TICK_WINDOW`` ticks.

        This is the metric the status bar shows to the user, and it's
        what the bench harness independently measures from outside.
        """
        if not self._state.is_playing:
            return None
        if len(self._tick_timestamps) < 2:
            return None
        span = self._tick_timestamps[-1] - self._tick_timestamps[0]
        if span <= 0:
            return None
        return (len(self._tick_timestamps) - 1) / span

    # ------------------------------------------------------------------ Commands

    def load_sequence(self, sequence: SequenceInfo) -> None:
        self._timer.stop()
        self._sequence = sequence
        self._cache.attach(sequence)
        first = sequence.first_frame
        self._update(
            current_frame=first,
            is_playing=False,
            direction=1,
            in_frame=None,
            out_frame=None,
            dropped_frames=0,
        )
        self._cache.set_current_frame(first)
        self._cache.request_range(first, first + self.PREFETCH_AHEAD, direction=1)
        self.frame_changed.emit(first)

    def play(self) -> None:
        if self._sequence is None or self._state.is_playing:
            return
        # GC tweak: collect *now* (so we start clean), freeze long-lived
        # objects (Qt widgets, OCIO config, cached frames) so the GC stops
        # walking them, then disable the cyclic collector entirely while
        # playing. This kills the p99 paint spikes that show up as 200 ms
        # frame stalls in the baseline. We re-enable on pause() — the GC
        # then runs once and catches anything that piled up.
        gc.collect()
        gc.freeze()
        gc.disable()

        # Reset the FPS rolling window so the metric reflects this
        # playback session, not the previous one (which may have ended
        # at a wildly different rate).
        self._tick_timestamps.clear()

        self._update(is_playing=True)
        self._timer.start(self._interval_ms())

    def pause(self) -> None:
        if not self._state.is_playing:
            return
        self._timer.stop()
        # Restore GC. unfreeze() moves the frozen set back to the regular
        # generations so they can eventually be collected; enable() turns
        # the cyclic collector back on; collect() does one immediate pass
        # to catch the playback-time garbage we deferred.
        gc.unfreeze()
        gc.enable()
        gc.collect()
        # Drop the rolling window so effective_fps() returns None until
        # the next play() — paused players don't have a meaningful fps.
        self._tick_timestamps.clear()
        self._update(is_playing=False)

    def stop(self) -> None:
        """Pause and jump back to the in-frame (or sequence start)."""
        self.pause()
        if self._sequence is None:
            return
        self.seek(self._effective_in_frame())

    def seek(self, frame: int) -> None:
        if self._sequence is None:
            return
        clamped = self._clamp(frame)
        self._cache.clear_pending()
        self._update(current_frame=clamped)
        self._cache.set_current_frame(clamped)
        self._prefetch_from(clamped, self._state.direction)
        # Seeking discards the rolling FPS window — pre-seek samples
        # don't reflect the post-seek decode pressure (we're likely to
        # land on a non-cached frame which slows the next few ticks).
        self._tick_timestamps.clear()
        self._try_render(count_misses=False)

    def step(self, delta: int) -> None:
        if self._sequence is None:
            return
        self.seek(self._state.current_frame + delta)

    def set_fps(self, fps: float) -> None:
        self._update(fps=max(0.1, fps))
        if self._state.is_playing:
            self._timer.setInterval(self._interval_ms())

    def set_loop_mode(self, mode: LoopMode) -> None:
        self._update(loop_mode=mode)

    def set_direction(self, direction: int) -> None:
        """Set playback direction: +1 forward, -1 backward."""
        d = 1 if direction >= 0 else -1
        self._update(direction=d)
        self._cache.set_direction(d)

    def set_in_out(self, in_frame: int | None, out_frame: int | None) -> None:
        self._update(in_frame=in_frame, out_frame=out_frame)
        if self._sequence is not None:
            clamped = self._clamp(self._state.current_frame)
            if clamped != self._state.current_frame:
                self._update(current_frame=clamped)
                self._cache.set_current_frame(clamped)
                self.frame_changed.emit(clamped)

    def shutdown(self) -> None:
        self._timer.stop()

    # ------------------------------------------------------------------ Tick

    def _tick(self) -> None:
        if self._sequence is None or not self._state.is_playing:
            return
        # Stamp the tick *first* so effective_fps() reflects the actual
        # cadence of the QTimer, not the time we spend below in advance/
        # prefetch logic. The status bar reads this every 500 ms.
        self._tick_timestamps.append(time.monotonic())
        next_frame, next_dir, should_stop = self._advance()

        # The playhead always moves forward at the configured FPS —
        # otherwise the user sees no sign of playback while the cache
        # catches up. On a cache miss we count a dropped frame and let
        # the UI pick the nearest available frame to display.
        self._update(current_frame=next_frame, direction=next_dir)
        self._cache.set_current_frame(next_frame)
        self._cache.set_direction(next_dir)
        self._prefetch_from(next_frame, next_dir)
        cache_hit = self._cache.contains(next_frame)
        if not cache_hit:
            self._update(dropped_frames=self._state.dropped_frames + 1)
        # Bench hook: record the tick decision (no-op when bench is off).
        if recorder.is_enabled():
            recorder.record_tick(
                requested_frame=next_frame,
                cache_hit=cache_hit,
                pending_decodes=self._cache._pool.pending(),  # noqa: SLF001
            )
        self.frame_changed.emit(next_frame)
        if should_stop:
            self.pause()

    def _advance(self) -> tuple[int, int, bool]:
        """Return (next_frame, next_direction, should_pause_after)."""
        lo = self._effective_in_frame()
        hi = self._effective_out_frame()
        cur = self._state.current_frame
        d = self._state.direction
        tentative = cur + d

        if d > 0 and tentative > hi:
            if self._state.loop_mode == LoopMode.ONCE:
                return (hi, 1, True)
            if self._state.loop_mode == LoopMode.LOOP:
                return (lo, 1, False)
            return (max(hi - 1, lo), -1, False)  # PING_PONG
        if d < 0 and tentative < lo:
            if self._state.loop_mode == LoopMode.ONCE:
                return (lo, -1, True)
            if self._state.loop_mode == LoopMode.LOOP:
                return (hi, -1, False)
            return (min(lo + 1, hi), 1, False)  # PING_PONG
        return (tentative, d, False)

    # ------------------------------------------------------------------ Helpers

    def _try_render(self, *, count_misses: bool) -> None:
        frame_n = self._state.current_frame
        arr = self._cache.get(frame_n)
        if arr is None and count_misses:
            self._update(dropped_frames=self._state.dropped_frames + 1)
        self.frame_changed.emit(frame_n)

    def _prefetch_from(self, frame: int, direction: int) -> None:
        if direction >= 0:
            self._cache.request_range(frame, frame + self.PREFETCH_AHEAD, direction=1)
        else:
            self._cache.request_range(frame - self.PREFETCH_AHEAD, frame, direction=-1)

    def _interval_ms(self) -> int:
        return max(1, round(1000.0 / self._state.fps))

    def _effective_in_frame(self) -> int:
        assert self._sequence is not None
        return (
            self._state.in_frame if self._state.in_frame is not None else self._sequence.first_frame
        )

    def _effective_out_frame(self) -> int:
        assert self._sequence is not None
        return (
            self._state.out_frame
            if self._state.out_frame is not None
            else self._sequence.last_frame
        )

    def _clamp(self, frame: int) -> int:
        return max(self._effective_in_frame(), min(self._effective_out_frame(), frame))

    def _update(self, **kwargs: Any) -> None:
        new_state = replace(self._state, **kwargs)
        if new_state != self._state:
            self._state = new_state
            self.state_changed.emit(new_state)
