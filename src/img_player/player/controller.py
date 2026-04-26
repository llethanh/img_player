"""Playback orchestration: a QObject that drives the cache from a QTimer.

Emits Qt signals when the current frame or playback state changes. UI
widgets subscribe to ``frame_changed`` (for rendering) and ``state_changed``
(for transport buttons / status display).
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal

from img_player.bench import recorder
from img_player.cache.frame_cache import FrameCache
from img_player.player.state import LoopMode, PlaybackState
from img_player.sequence.models import SequenceInfo

log = logging.getLogger(__name__)


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

    # ------------------------------------------------------------------ Properties

    @property
    def state(self) -> PlaybackState:
        return self._state

    @property
    def sequence(self) -> SequenceInfo | None:
        return self._sequence

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
        self._update(is_playing=True)
        self._timer.start(self._interval_ms())

    def pause(self) -> None:
        if not self._state.is_playing:
            return
        self._timer.stop()
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
        self._update(direction=1 if direction >= 0 else -1)

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
        next_frame, next_dir, should_stop = self._advance()

        # The playhead always moves forward at the configured FPS —
        # otherwise the user sees no sign of playback while the cache
        # catches up. On a cache miss we count a dropped frame and let
        # the UI pick the nearest available frame to display.
        self._update(current_frame=next_frame, direction=next_dir)
        self._cache.set_current_frame(next_frame)
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
