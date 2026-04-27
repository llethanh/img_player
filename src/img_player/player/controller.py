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

    # Throttle rate for the live metric signals (effective_fps_changed /
    # cache_hit_rate_changed). The status bar only needs to refresh
    # ~once a second; emitting at every tick (24 Hz) would be wasted
    # work for the UI layout system.
    _METRIC_EMIT_INTERVAL_S = 1.0

    frame_changed = Signal(int)
    state_changed = Signal(object)  # emits PlaybackState
    # Live metric signals consumed by the UI status bar (slice 5).
    # Emitted at most once a second while playing; cleared / silent
    # while paused or before enough samples have accumulated.
    effective_fps_changed = Signal(float)
    cache_hit_rate_changed = Signal(float)

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
        # Parallel rolling window of cache-hit booleans so we can compute
        # an instantaneous hit rate over the same window as
        # ``effective_fps``. The runtime monitor (slice 5) reads this to
        # detect "the prefetch isn't keeping up".
        self._tick_hits: deque[bool] = deque(maxlen=_TICK_WINDOW)
        # Last monotonic clock at which we emitted the live-metric signals.
        # Throttles the emit rate to ``_METRIC_EMIT_INTERVAL_S`` so the
        # status bar refreshes at human pace, not at tick frequency.
        self._last_metric_emit: float = 0.0
        # Mutable mirror of PREFETCH_AHEAD that the runtime monitor can
        # shrink mid-playback when the cache hit rate drops. Initialised
        # to the class default so existing call sites stay correct.
        self._prefetch_ahead: int = type(self).PREFETCH_AHEAD

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

    def cache_hit_rate(self) -> float | None:
        """Rolling cache-hit rate over the same window as effective_fps.

        Differs from ``cache.stats().hits / .misses`` (which is
        process-lifetime): this returns the rate over the last
        ``_TICK_WINDOW`` ticks only, so the runtime monitor can
        detect a *recent* prefetch shortfall without being averaged
        out by the long history.

        Returns ``None`` until at least 4 ticks have accumulated —
        below that, the rate is too noisy to act on.
        """
        if len(self._tick_hits) < 4:
            return None
        return sum(1 for h in self._tick_hits if h) / len(self._tick_hits)

    def get_prefetch_ahead(self) -> int:
        """Current prefetch window size. Mirrors ``PREFETCH_AHEAD`` until
        the runtime monitor decides to shrink it mid-playback."""
        return self._prefetch_ahead

    def set_prefetch_ahead(self, value: int) -> None:
        """Adjust the prefetch window at runtime.

        Called by the runtime monitor when ``cache_hit_rate`` drops
        below the threshold for a sustained window: a smaller
        prefetch front means the workers focus on what we'll actually
        display next instead of pre-decoding far frames the cache
        can't hold anyway. Floors at 4 — below that we'd be one
        cache miss away from a permanent stall.
        """
        self._prefetch_ahead = max(4, int(value))

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
        # Schedule the *whole* sequence to be filled in the background,
        # prioritised by distance from the playhead. The close window
        # decodes first (the worker pool is a min-heap on priority);
        # everything else trickles in afterwards. This way the user
        # eventually gets a fully-cached sequence even if they never
        # play through it — and they don't see unexplained gaps on
        # the cache bar after seeking past the close window.
        self._prefetch_full_sequence(first, 1)
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
        self._tick_hits.clear()
        self._last_metric_emit = 0.0

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
        self._tick_hits.clear()
        self._last_metric_emit = 0.0
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
        # Re-plan the entire sequence's prefetch, prioritised by
        # distance from the new playhead. ``clear_pending`` just
        # dropped the old queue, including any "middle of the
        # sequence" frames that were waiting their turn — without
        # this re-plan they'd never come back, and the user would
        # see a permanent gap on the cache bar between the previous
        # prefetch's high-water mark and the new playhead.
        self._prefetch_full_sequence(clamped, self._state.direction)
        # Seeking discards the rolling FPS window — pre-seek samples
        # don't reflect the post-seek decode pressure (we're likely to
        # land on a non-cached frame which slows the next few ticks).
        self._tick_timestamps.clear()
        self._tick_hits.clear()
        self._last_metric_emit = 0.0
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

    def play_direction(self, direction: int) -> None:
        """Direction-aware play / pause.

        The semantics most VFX viewers (Nuke, Hiero, Resolve) implement
        for a clicked direction button:

        * Already playing in that direction → pause.
        * Playing in the *other* direction → flip without stopping.
        * Paused → set the direction and start playing.

        ``Space`` / ``K`` shortcuts go through :meth:`pause` and
        :meth:`play` directly — they're the older direction-agnostic
        toggle.
        """
        d = 1 if direction >= 0 else -1
        if self._state.is_playing and self._state.direction == d:
            self.pause()
            return
        self.set_direction(d)
        if not self._state.is_playing:
            self.play()

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
        # Mirror the hit/miss into the rolling window the runtime
        # monitor reads via cache_hit_rate(). Same window size as the
        # fps deque, so both metrics describe the same time slice.
        self._tick_hits.append(cache_hit)
        if not cache_hit:
            self._update(dropped_frames=self._state.dropped_frames + 1)
        # Bench hook: record the tick decision (no-op when bench is off).
        if recorder.is_enabled():
            recorder.record_tick(
                requested_frame=next_frame,
                cache_hit=cache_hit,
                pending_decodes=self._cache._pool.pending(),
            )
        self.frame_changed.emit(next_frame)

        # Throttled live metrics emit (slice 5). Runs at most once a
        # second so the status bar redraws at human pace. We keep the
        # throttle inside _tick instead of using a separate QTimer to
        # avoid one more timer object on the event loop hot path.
        now = time.monotonic()
        if now - self._last_metric_emit >= self._METRIC_EMIT_INTERVAL_S:
            self._last_metric_emit = now
            fps = self.effective_fps()
            if fps is not None:
                self.effective_fps_changed.emit(fps)
            hr = self.cache_hit_rate()
            if hr is not None:
                self.cache_hit_rate_changed.emit(hr)

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
        """Tick-time prefetch: just the close window.

        Called on every playback tick (24+ Hz). Keep it cheap — only
        schedule what we need imminently. The full-sequence fill is
        handled by :meth:`_prefetch_full_sequence` on seek and load.
        """
        if direction >= 0:
            self._cache.request_range(frame, frame + self._prefetch_ahead, direction=1)
        else:
            self._cache.request_range(frame - self._prefetch_ahead, frame, direction=-1)

    # Frames *behind* the playhead in the current play direction are
    # only revisited on a loop wrap, so we treat them as significantly
    # less urgent than frames ahead. Mirrors the eviction-scoring rule
    # in :class:`FrameCache`. 4× was picked empirically: enough that
    # forward frames always win worker time, not so much that distant
    # behind-frames never get scheduled.
    _BEHIND_PRIORITY_PENALTY = 4

    def _prefetch_full_sequence(self, frame: int, direction: int) -> None:
        """Schedule every frame in the sequence, prioritised by
        signed distance from ``frame``.

        The worker pool is a min-heap on priority — frames closer to
        the playhead (small priority numbers) decode first; far
        frames trickle in once the close window is drained.
        ``FrameCache.request`` dedups against already-cached and
        already-pending frames, so calling this on every seek is
        cheap (one lock + dict lookup per frame) and idempotent.

        Without this, a user who scrubs past the close-window
        boundary mid-prefetch leaves a permanent gap in the middle
        of the cache bar: the old queued tasks were dropped by
        ``clear_pending``, and ``_prefetch_from`` only re-requests
        the new close window. The gap is exactly what the user
        reported on the timeline.
        """
        if self._sequence is None:
            return
        seq = self._sequence
        d = 1 if direction >= 0 else -1
        for f in range(seq.first_frame, seq.last_frame + 1):
            delta = (f - frame) * d
            priority = delta if delta >= 0 else (-delta * self._BEHIND_PRIORITY_PENALTY)
            self._cache.request(f, priority=priority)

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
