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
from img_player.cache.master_frame_cache import MasterFrameCache
from img_player.player.state import LoopMode, PlaybackState
from img_player.sequence.models import SequenceInfo

# Both cache classes present the same public surface (attach,
# detach, request, get, contains, set_current_frame, …) — the
# controller accepts either, so v1.0 can swap to MasterFrameCache
# without disturbing the existing FrameCache-based tests.
CacheLike = FrameCache | MasterFrameCache

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

    def __init__(
        self,
        cache: CacheLike,
        parent: QObject | None = None,
        *,
        clock: "callable" | None = None,
    ) -> None:
        super().__init__(parent)
        self._cache = cache
        self._sequence: SequenceInfo | None = None
        self._state = PlaybackState()
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.timeout.connect(self._tick)
        # Wall-clock provider. Defaults to ``time.monotonic`` so playback
        # advances at real-world rate independent of QTimer jitter.
        # Tests inject a controllable mock clock so ``_tick`` is
        # deterministic without sleeping. Earlier the controller
        # advanced ``current_frame`` by exactly +1 per tick — that
        # made the master clock = "tick count" and any QTimer drift
        # leaked straight into A/V desync. With a wall-clock anchor,
        # the playhead targets the frame the wall clock says we
        # should be on, and the audio (which plays at wall clock
        # natively via PortAudio) stays aligned by construction.
        self._clock: "callable" = clock if clock is not None else time.monotonic
        # Anchor for the current play burst. ``None`` when paused or
        # not yet anchored. ``play()`` and every range-/rate-changing
        # call (seek, set_fps, set_direction) reset it so subsequent
        # ticks measure elapsed time from the new anchor.
        self._play_start_clock: float | None = None
        self._play_start_frame: int = 0
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
        # Navigable master-frame range override. ``None`` falls back to
        # the loaded sequence's first/last frame (single-layer legacy
        # behaviour). When the multi-layer stack is active, ``app.py``
        # pushes the broad master range here so scrubbing isn't capped
        # by the first-loaded layer's bounds — without this the user
        # can't move the playhead past the first layer even if a later
        # layer extends further.
        self._navigable_range: tuple[int, int] | None = None

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

    def _sync_loop_range_to_cache(self) -> None:
        """Push the active playback range + loop flag down to the cache.

        The cache uses these to switch its eviction scoring to ring
        distance when LOOP is on (see
        ``MasterFrameCache._evict_if_over_budget``). Called whenever
        anything that could change the effective range or the loop
        mode flips: ``load_sequence``, ``seek``, ``set_loop_mode``,
        ``set_in_out``, ``set_navigable_range``. Cheap (a lock + 3
        attribute writes) and idempotent — safe to over-call.

        No-op when no sequence is loaded; the cache treats absent /
        invalid bounds as ``loop_enabled=False``.
        """
        if not hasattr(self._cache, "set_loop_range"):
            return
        if self._sequence is None:
            self._cache.set_loop_range(None, None, False)
            return
        lo = self._effective_in_frame()
        hi = self._effective_out_frame()
        enabled = self._state.loop_mode == LoopMode.LOOP and hi > lo
        self._cache.set_loop_range(lo, hi, enabled)

    def replan_prefetch(self) -> None:
        """Re-schedule a full-sequence prefetch from the current playhead.

        Public hook for the app's ``_refresh_after_stack_change`` so a
        LayerStack mutation (add / remove / reorder, eye toggle,
        offset / trim drag, channel switch) replays the same
        priority-ranked submit pass that ``seek`` and ``load_sequence``
        run. Without this the multi-layer cache wipe (driven by
        ``MasterFrameCache._on_layers_changed``) leaves the cache empty
        outside the close window — the user then sees idle gaps on the
        timeline cache bar that only fill once playback rolls the
        playhead through them.

        No-op when no sequence is attached. Idempotent: ``request``
        dedup-rejects frames already cached or pending.
        """
        if self._sequence is None:
            return
        # Refresh loop hints first — a stack mutation can change the
        # broad master range (= the active loop range) and the cache
        # needs the updated bounds before the new prefetch pass picks
        # up its loop-aware priorities.
        self._sync_loop_range_to_cache()
        self._prefetch_full_sequence(
            self._state.current_frame, self._state.direction,
        )

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
        # Tell the cache about the loop range BEFORE the first
        # prefetch pass — without this the wave-1 ring-distance
        # priorities (computed below) and the eviction scoring (run
        # at decode-store time) disagree, and frames near ``first_f``
        # can still be evicted in a tight-budget multi-layer load.
        self._sync_loop_range_to_cache()
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
        # If the user parked the cursor outside the in/out range
        # while paused (allowed since v0.5.2), snap it back inside
        # NOW so playback starts on a frame that's actually part of
        # the loop. Choose the direction-appropriate boundary so a
        # forward press lands on the in-point and a reverse press
        # lands on the out-point.
        cur = self._state.current_frame
        lo = self._effective_in_frame()
        hi = self._effective_out_frame()
        if cur < lo or cur > hi:
            target = lo if self._state.direction >= 0 else hi
            self.seek(target)
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
        # Anchor the wall clock for this play burst. Subsequent ticks
        # compute target_frame = anchor_frame + round(direction × elapsed × fps),
        # so the playhead stays in lockstep with wall time even if
        # the QTimer is jittery.
        self._play_start_clock = self._clock()
        self._play_start_frame = self._state.current_frame

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
        # Drop the wall-clock anchor — the next play() reseeds it.
        self._play_start_clock = None
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
        # User-initiated seeks (timeline scrub, ←/→ keys, frame
        # input box) are free to land OUTSIDE the in/out range —
        # only the sequence's absolute first/last act as hard
        # bounds. The in/out range is a *playback* constraint, not
        # a *navigation* one (matches Nuke / RV behaviour and was
        # the v0.5.2 user request). Playback itself still respects
        # the range via ``_advance``; if the user starts playback
        # while parked outside it, ``_tick`` snaps back in.
        clamped = self._clamp_to_sequence(frame)
        self._cache.clear_pending()
        self._update(current_frame=clamped)
        self._cache.set_current_frame(clamped)
        # Re-sync loop hints — the seek may have crossed in/out
        # boundaries that affect the effective range, and the next
        # prefetch wave + every subsequent eviction round needs the
        # right bounds to score frames correctly.
        self._sync_loop_range_to_cache()
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
        # Re-anchor the wall clock if we're playing — the seek moved
        # the playhead arbitrarily; without a re-anchor the next tick
        # would compute target = old_anchor_frame + round(elapsed × fps)
        # and yank the playhead back toward the pre-seek position.
        if self._state.is_playing:
            self._play_start_clock = self._clock()
            self._play_start_frame = clamped
        self._try_render(count_misses=False)

    def step(self, delta: int) -> None:
        if self._sequence is None:
            return
        self.seek(self._state.current_frame + delta)

    def set_fps(self, fps: float) -> None:
        self._update(fps=max(0.1, fps))
        if self._state.is_playing:
            self._timer.setInterval(self._interval_ms())
            # Re-anchor: the conversion ``elapsed × fps`` would
            # produce an inconsistent target if we kept the old
            # anchor across a rate change.
            self._play_start_clock = self._clock()
            self._play_start_frame = self._state.current_frame

    def set_loop_mode(self, mode: LoopMode) -> None:
        self._update(loop_mode=mode)
        # Cache eviction scoring depends on whether LOOP is active —
        # flip the ring-distance mode in lockstep so the next budget
        # eviction round sees the matching geometry.
        self._sync_loop_range_to_cache()

    def set_direction(self, direction: int) -> None:
        """Set playback direction: +1 forward, -1 backward."""
        d = 1 if direction >= 0 else -1
        self._update(direction=d)
        self._cache.set_direction(d)
        # Re-anchor: direction flip means target = anchor + direction
        # × elapsed × fps reads the wrong way without a fresh anchor.
        if self._state.is_playing:
            self._play_start_clock = self._clock()
            self._play_start_frame = self._state.current_frame

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
        # No auto-snap of the current frame anymore — the user is
        # allowed to park the cursor outside the in/out range.
        # Playback ``_tick`` is the only place that snaps, and only
        # when the user actually presses play.
        self._update(in_frame=in_frame, out_frame=out_frame)
        # In/out markers are the loop's effective bounds — push them
        # to the cache so ring-distance eviction scopes to the new
        # playback range.
        self._sync_loop_range_to_cache()

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
        # Wall-clock-driven advance. ``play()`` and every range-/rate-
        # changing call seed the anchor; the target frame is the one
        # the wall clock says we should be on by now. This eliminates
        # QTimer-jitter drift that used to leak straight into A/V
        # desync (audio plays at PortAudio wall rate by construction;
        # if the video tick advances by exactly +1 per QTimer firing
        # without referring to wall time, any timer slippage shows up
        # as cumulative desync).
        if self._play_start_clock is None:
            self._play_start_clock = self._clock()
            self._play_start_frame = self._state.current_frame
        elapsed = self._clock() - self._play_start_clock
        d = self._state.direction
        target_offset = int(round(d * elapsed * self._state.fps))
        wall_target = self._play_start_frame + target_offset
        cur = self._state.current_frame
        # Idle tick: clock fired before a full frame interval elapsed.
        # Skip the advance path so we don't burn cache requests on
        # no-op transitions.
        if wall_target == cur:
            self._maybe_emit_metrics()
            return
        # Cap the per-tick step to ±1 frame. The wall-clock target
        # still corrects long-term drift (a "behind" tick stays
        # behind until the next idle tick lets the wall clock
        # advance the target past us — the playhead catches up
        # gradually instead of jumping multiple frames at once,
        # which reads as visual stutter even though it's
        # technically more accurate). For genuine multi-frame
        # slips (GC pause, GUI redraw stall) the catch-up plays
        # out over the next few ticks at +1/-1 each — perceptually
        # smoother and within a few frames of true wall time.
        if wall_target > cur:
            tentative = cur + 1
        else:
            tentative = cur - 1
        next_frame, next_dir, should_stop = self._advance(tentative)

        # Cache-bound playback: stall the playhead instead of running
        # ahead of the cache fill bar. Per user feedback ("qd on lance
        # la lecture, je voudrais que le curseur n'aille pas plus vite
        # que la barre de chargement"). The previous behaviour
        # (always advance, show nearest cached frame as fallback) had
        # a confusing UX — the cursor lies about progress.
        #
        # The exception is ``should_stop`` (ONCE mode reaching the end):
        # we always honour the stop, even if the final frame isn't
        # cached, otherwise we'd loop forever waiting for a frame at a
        # boundary the user explicitly asked us to stop on.
        # Multi-layer gap: no visible layer covers ``next_frame``, so
        # the cache will *never* decode anything for it. Advancing
        # through is the right move — the viewport's frame-changed
        # handler clears to black on uncovered frames, and the user
        # sees the playhead glide across the gap at normal speed
        # rather than freezing every time. Without this opt-out the
        # cache-bound stall (added for "cursor doesn't run ahead of
        # the cache fill") fires forever on a frame that's structurally
        # uncacheable.
        is_gap = (
            hasattr(self._cache, "is_gap_frame")
            and self._cache.is_gap_frame(next_frame)
        )
        if not should_stop and not is_gap and not self._cache.contains(next_frame):
            # Stay on the current frame. Re-issue the prefetch so the
            # worker pool keeps loading toward the would-be next
            # frame. Direction matters for the prefetch heuristic.
            self._cache.set_current_frame(self._state.current_frame)
            self._cache.set_direction(next_dir)
            self._prefetch_from(self._state.current_frame, next_dir)
            # Loop-wrap stall: at end-of-range, ``_advance`` returns
            # ``next_frame = lo`` (start of range), but ``_prefetch_from``
            # above only queues ``current_frame ± window`` — i.e. around
            # ``hi``, NOT ``lo``. Result: ``lo`` is never (re)requested
            # and the stall is infinite — playback freezes at the end
            # instead of looping. Explicitly queue ``next_frame`` at
            # priority 0 so the worker pool decodes it ASAP. Cheap and
            # idempotent (FrameCache.request dedups).
            self._cache.request(next_frame, priority=0)
            # Record the stall in the rolling window — the runtime
            # monitor (slice 5) reads this to surface "Lecture
            # difficile" warnings if stalls dominate.
            self._tick_hits.append(False)
            # No frame_changed emit — current_frame didn't move, so
            # downstream consumers (timeline cursor, frame display,
            # annotation overlay) can stay where they are.

            # Bench hook: log the stall as a missed tick.
            if recorder.is_enabled():
                recorder.record_tick(
                    requested_frame=next_frame,
                    cache_hit=False,
                    pending_decodes=self._cache._pool.pending(),
                )
            # Live metric emit still runs — status bar's effective
            # fps drops naturally when stalls dominate.
            self._maybe_emit_metrics()
            return

        # Normal advance — the next frame is cached (or we're forced
        # by should_stop).
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
            # Reachable only via should_stop on a missed boundary
            # frame; counting it preserves the dropped_frames
            # semantic (= "we displayed something we shouldn't have").
            self._update(dropped_frames=self._state.dropped_frames + 1)
        # Bench hook: record the tick decision (no-op when bench is off).
        if recorder.is_enabled():
            recorder.record_tick(
                requested_frame=next_frame,
                cache_hit=cache_hit,
                pending_decodes=self._cache._pool.pending(),
            )
        self.frame_changed.emit(next_frame)

        self._maybe_emit_metrics()

        if should_stop:
            self.pause()

    def _maybe_emit_metrics(self) -> None:
        """Throttled live metrics emit (slice 5). Runs at most once
        a second so the status bar redraws at human pace. Extracted
        from ``_tick`` so it can run on both the advance path and
        the stall path."""
        now = time.monotonic()
        if now - self._last_metric_emit >= self._METRIC_EMIT_INTERVAL_S:
            self._last_metric_emit = now
            fps = self.effective_fps()
            if fps is not None:
                self.effective_fps_changed.emit(fps)
            hr = self.cache_hit_rate()
            if hr is not None:
                self.cache_hit_rate_changed.emit(hr)

    def _advance(self, tentative: int | None = None) -> tuple[int, int, bool]:
        """Apply range / loop semantics to a target frame.

        Returns ``(next_frame, next_direction, should_pause_after)``.
        ``tentative`` is the wall-clock-derived target the new
        :meth:`_tick` computes; absent (legacy single-step callers)
        defaults to ``current_frame + direction`` for a one-frame
        advance.
        """
        lo = self._effective_in_frame()
        hi = self._effective_out_frame()
        cur = self._state.current_frame
        d = self._state.direction
        if tentative is None:
            tentative = cur + d

        if d > 0 and tentative > hi:
            if self._state.loop_mode == LoopMode.ONCE:
                return (hi, 1, True)
            if self._state.loop_mode == LoopMode.LOOP:
                # Wrap around — re-anchor the wall clock so the next
                # tick measures elapsed from ``lo``, not from the
                # pre-wrap position. Without this, a long run
                # accumulates seconds of "phantom" elapsed time and
                # the post-wrap target rockets past ``lo``.
                if self._play_start_clock is not None:
                    self._play_start_clock = self._clock()
                    self._play_start_frame = lo
                return (lo, 1, False)
            return (max(hi - 1, lo), -1, False)  # PING_PONG
        if d < 0 and tentative < lo:
            if self._state.loop_mode == LoopMode.ONCE:
                return (lo, -1, True)
            if self._state.loop_mode == LoopMode.LOOP:
                if self._play_start_clock is not None:
                    self._play_start_clock = self._clock()
                    self._play_start_frame = hi
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
        # Prefetch covers the full navigable range — with multi-layer
        # stacks the master range can extend past the controller's
        # held sequence, and frames beyond it would otherwise never
        # decode in the background.
        bounds = self._nav_bounds()
        if bounds is not None:
            first_f, last_f = bounds
        else:
            first_f, last_f = seq.first_frame, seq.last_frame
        # In LOOP mode, prioritise frames by their **ring distance**
        # forward from the playhead instead of signed distance with
        # a behind-penalty. Otherwise the wrap target (``first_f``)
        # is queued at priority ``range_len * BEHIND_PENALTY``
        # (= bottom of the worker pool), so when ``_tick`` stalls at
        # ``last_f`` waiting for the wrap, the dedup-rejected high-
        # priority re-request stays stuck behind every other backlog
        # frame and the loop visibly never fires.
        loop_on = (
            self._state.loop_mode == LoopMode.LOOP
            and last_f > first_f
        )
        ring_size = last_f - first_f + 1 if loop_on else 0
        for f in range(first_f, last_f + 1):
            if loop_on:
                if d >= 0:
                    priority = (f - frame) % ring_size
                else:
                    priority = (frame - f) % ring_size
            else:
                delta = (f - frame) * d
                priority = (
                    delta if delta >= 0
                    else (-delta * self._BEHIND_PRIORITY_PENALTY)
                )
            self._cache.request(f, priority=priority)

    def _interval_ms(self) -> int:
        return max(1, round(1000.0 / self._state.fps))

    def _effective_in_frame(self) -> int:
        assert self._sequence is not None
        if self._state.in_frame is not None:
            return self._state.in_frame
        bounds = self._nav_bounds()
        return bounds[0] if bounds is not None else self._sequence.first_frame

    def _effective_out_frame(self) -> int:
        assert self._sequence is not None
        if self._state.out_frame is not None:
            return self._state.out_frame
        bounds = self._nav_bounds()
        return (
            bounds[1] if bounds is not None else self._sequence.last_frame
        )

    def _clamp(self, frame: int) -> int:
        return max(self._effective_in_frame(), min(self._effective_out_frame(), frame))

    def set_navigable_range(self, first: int, last: int) -> None:
        """Override the navigable master-frame range.

        Pass ``(-1, -1)`` (or any reversed pair) to reset to the
        sequence's own bounds. Used by ``app.py`` after every
        :class:`LayerStack` mutation so scrubbing covers the union of
        all layers, not just the one the controller happens to hold a
        reference to.
        """
        if last < first:
            self._navigable_range = None
        else:
            self._navigable_range = (int(first), int(last))
        # The navigable range IS the loop range when no in/out marker
        # is set — refresh the cache so eviction scoring updates with
        # the new bounds. Important for multi-layer stacks: the broad
        # master range is set here on every stack mutation, and the
        # cache wouldn't otherwise pick up the change.
        self._sync_loop_range_to_cache()

    def _nav_bounds(self) -> tuple[int, int] | None:
        """Effective (first, last) navigable bounds, or ``None`` when
        neither override nor sequence is set."""
        if self._navigable_range is not None:
            return self._navigable_range
        if self._sequence is not None:
            return (self._sequence.first_frame, self._sequence.last_frame)
        return None

    def _clamp_to_sequence(self, frame: int) -> int:
        """Clamp ``frame`` to the navigable bounds, ignoring any
        in/out range. Used by ``seek`` so the user can scrub freely
        outside the playback range — the in/out is a constraint on
        PLAYBACK, not navigation."""
        bounds = self._nav_bounds()
        if bounds is None:
            return frame
        first, last = bounds
        return max(first, min(last, frame))

    def _update(self, **kwargs: Any) -> None:
        new_state = replace(self._state, **kwargs)
        if new_state != self._state:
            self._state = new_state
            self.state_changed.emit(new_state)
