"""Thread-safe RAM cache of decoded frames, with async prefetch and LRU-by-distance eviction."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from img_player.bench import recorder
from img_player.cache.worker_pool import WorkerPool
from img_player.io.reader import FrameReadError, read_frame
from img_player.sequence.models import SequenceInfo

log = logging.getLogger(__name__)

_DEFAULT_BUDGET_BYTES = 8 * 1024**3  # 8 GB
_DEFAULT_NUM_WORKERS = 4
# Eviction multiplier for frames that lie *behind* the playhead in the
# current playback direction. They cost more because we'll only revisit
# them after a full loop wrap — so we throw them out first to free space
# for what's coming up next.
_BEHIND_PLAYHEAD_PENALTY = 3.0


@dataclass(frozen=True)
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    decode_errors: int = 0
    bytes_used: int = 0
    bytes_budget: int = 0
    frames_cached: int = 0


class FrameCache:
    """Decoded-frame cache with async prefetch.

    Stores ``np.ndarray`` keyed by frame number. Worker threads decode
    frames in the background via :mod:`OpenImageIO`. When the cache size
    exceeds the budget, frames furthest from the current playhead are
    evicted first.

    Not thread-safe for attach/shutdown (those are lifecycle ops called
    from the Qt main thread). get / request / put are fully thread-safe.
    """

    def __init__(
        self,
        budget_bytes: int = _DEFAULT_BUDGET_BYTES,
        num_workers: int = _DEFAULT_NUM_WORKERS,
    ) -> None:
        self._budget = budget_bytes
        self._lock = threading.RLock()
        self._frames: dict[int, np.ndarray] = {}
        self._bytes_used = 0
        self._sequence: SequenceInfo | None = None
        self._paths_by_frame: dict[int, Path] = {}
        self._current_frame = 0
        # +1 = forward play, -1 = reverse. Used to skew eviction so frames
        # ahead of the playhead are preserved over frames already behind.
        self._direction = 1
        # Active channel selection. ``None`` means "let the reader pick
        # its sensible default" (R/G/B/A when available). A list like
        # ``["Z"]`` or ``["N.X"]`` means "decode only this channel" —
        # the reader broadcasts a single-channel readout to RGB so the
        # GL viewport renders it as monochrome.
        self._active_channels: list[str] | None = None
        # Bumped every time the channel selection (or sequence, via
        # ``attach``) changes. Each in-flight decode captures the epoch
        # under which it started; if the epoch has moved on by the time
        # the worker tries to store its result, we drop it. Without
        # this, a channel switch issued mid-decode lets the worker
        # store a stale-channel array into the freshly-cleared cache —
        # the cache bar then shows a phantom run of frames that decode
        # to the *previous* channel.
        self._epoch = 0
        self._pool = WorkerPool(num_workers=num_workers, name="decode")

        # Counters (guarded by _lock)
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._decode_errors = 0

    # ------------------------------------------------------------------ Lifecycle

    def attach(self, sequence: SequenceInfo) -> None:
        """Switch to a new sequence; clears cached frames and pending requests."""
        self._pool.clear()
        with self._lock:
            self._frames.clear()
            self._bytes_used = 0
            self._sequence = sequence
            self._paths_by_frame = {f.frame_number: f.path for f in sequence.frames}
            self._current_frame = sequence.first_frame
            # Any decode that was in flight against the previous
            # sequence is now meaningless — fence them out.
            self._epoch += 1

    def clear(self) -> None:
        """Remove cached frames and drop pending decodes. Sequence stays attached."""
        self._pool.clear()
        with self._lock:
            self._frames.clear()
            self._bytes_used = 0
            # Workers currently mid-decode can't be cancelled; bump
            # the epoch so their store-time check drops the result.
            self._epoch += 1

    def clear_pending(self) -> int:
        """Drop only the pending decode queue (keep cached frames)."""
        return self._pool.clear()

    def shutdown(self) -> None:
        """Stop the worker pool. The cache must not be used after this."""
        self._pool.shutdown()

    # ------------------------------------------------------------------ API

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                decode_errors=self._decode_errors,
                bytes_used=self._bytes_used,
                bytes_budget=self._budget,
                frames_cached=len(self._frames),
            )

    def set_current_frame(self, frame: int) -> None:
        """Inform the cache of the playhead position (used for eviction scoring)."""
        with self._lock:
            self._current_frame = frame

    def set_direction(self, direction: int) -> None:
        """+1 forward / -1 reverse. Tells eviction which side of the playhead
        is "future" (cheap to evict on the past side)."""
        with self._lock:
            self._direction = 1 if direction >= 0 else -1

    def set_channels(self, channels: list[str] | None) -> None:
        """Switch which channels are decoded for subsequent frames.

        Currently-cached frames belong to the *previous* channel set,
        so we drop them — they were decoded with different OIIO
        ``read_image`` parameters and would re-display the wrong
        channels until naturally evicted. The pending decode queue
        is dropped too; the controller will request fresh prefetches
        once it sees the next frame_changed.

        Pass ``None`` to revert to the reader's default (R/G/B/A).
        """
        self._pool.clear()
        with self._lock:
            self._active_channels = list(channels) if channels else None
            self._frames.clear()
            self._bytes_used = 0
            # Any decode currently in flight was started against the
            # previous channel selection. ``pool.clear()`` only drops
            # *queued* tasks; running workers continue to completion.
            # We can't cancel their ``read_frame`` call, but we can
            # make them drop the result at store time by bumping the
            # epoch (see ``_decode_and_store``).
            self._epoch += 1

    def get(self, frame: int) -> np.ndarray | None:
        """Non-blocking fetch. Returns None if the frame is not cached."""
        with self._lock:
            self._current_frame = frame
            arr = self._frames.get(frame)
            if arr is not None:
                self._hits += 1
                return arr
            self._misses += 1
            return None

    def contains(self, frame: int) -> bool:
        with self._lock:
            return frame in self._frames

    def cached_frames(self) -> frozenset[int]:
        """Snapshot of the currently cached frame numbers (thread-safe)."""
        with self._lock:
            return frozenset(self._frames.keys())

    def request(self, frame: int, priority: int = 0) -> bool:
        """Enqueue async decode. Returns True if submitted, False if already
        cached or already pending."""
        with self._lock:
            if self._sequence is None:
                return False
            if frame in self._frames:
                return False
            path = self._paths_by_frame.get(frame)
            if path is None:
                return False
        return self._pool.submit(priority, frame, lambda: self._decode_and_store(frame, path))

    def request_range(self, start: int, end: int, direction: int = 1) -> None:
        """Prefetch frames from `start` to `end` (inclusive).

        ``direction`` only selects the iteration order (so earlier-in-direction
        frames get lower priority numbers and thus decode first). Out-of-sequence
        bounds are clamped.
        """
        if self._sequence is None:
            return
        lo = min(start, end)
        hi = max(start, end)
        lo = max(lo, self._sequence.first_frame)
        hi = min(hi, self._sequence.last_frame)
        if lo > hi:
            return
        frames = range(lo, hi + 1) if direction >= 0 else range(hi, lo - 1, -1)
        for i, f in enumerate(frames):
            self.request(f, priority=i)

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until the worker pool has nothing left to do. For tests."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._pool.pending() == 0:
                return True
            time.sleep(0.005)
        return False

    # ------------------------------------------------------------------ Internals

    def _decode_and_store(self, frame: int, path: Path) -> None:
        # Bench hook: time the decode itself, not the lock contention or store.
        # The recorder check is fast (single attribute load) so we keep it
        # outside the timing window only when disabled.
        bench_enabled = recorder.is_enabled()
        t_start = time.monotonic() if bench_enabled else 0.0
        # Capture the active channels under the lock — set_channels can
        # be called from the Qt thread while a decode is in flight,
        # but we want this single decode to use a stable selection.
        # The epoch is captured at the same moment so we can detect at
        # store time whether the world has shifted under us.
        with self._lock:
            channels = list(self._active_channels) if self._active_channels else None
            epoch = self._epoch
        try:
            arr = read_frame(path, channels=channels)
        except FrameReadError as err:
            log.warning("decode failed frame=%d path=%s: %s", frame, path, err)
            with self._lock:
                self._decode_errors += 1
            return
        if bench_enabled:
            recorder.record_decode(
                frame=frame,
                decode_ms=(time.monotonic() - t_start) * 1000.0,
                nbytes=int(arr.nbytes),
            )

        with self._lock:
            # The epoch moved while we were decoding — channels were
            # switched (or sequence re-attached, or cache cleared).
            # Storing the array now would re-pollute the freshly
            # cleared cache with a frame decoded under the *previous*
            # selection, and the cache bar would show a phantom run
            # the user can't account for. Drop it.
            if epoch != self._epoch:
                return
            if frame in self._frames:
                return  # raced with another worker; keep the existing one
            self._frames[frame] = arr
            self._bytes_used += arr.nbytes
            self._evict_if_over_budget()

    def shrink_budget(self, new_bytes: int) -> None:
        """Reduce the budget at runtime and force an immediate eviction.

        Used by the runtime memory-pressure monitor (slice 5) to give
        memory back to the OS when ``psutil.swap_memory().used`` rises
        during playback. Atomic under the existing lock — concurrent
        decodes that finish during this call will respect the new
        budget when they run their store-time eviction.

        **Never grows.** Once the cache has been shrunk for the
        session, it stays shrunk; the user gets a chance to close
        other apps and restart for a roomier budget. Auto-grow would
        oscillate under bursty memory pressure.
        """
        with self._lock:
            if new_bytes >= self._budget:
                return
            self._budget = max(0, new_bytes)
            self._evict_if_over_budget()

    def _evict_if_over_budget(self) -> None:
        """Must be called with _lock held.

        Scoring rule: distance from the playhead, with frames *behind* the
        playhead in the current play direction multiplied by
        ``_BEHIND_PLAYHEAD_PENALTY``. The frame with the highest score is
        evicted first. This keeps the prefetch window in front of the head
        intact even when budget is tight, and prevents the case where a
        cache miss pushes us to re-decode a frame we just played.
        """
        if self._bytes_used <= self._budget:
            return
        cur = self._current_frame
        d = self._direction
        penalty = _BEHIND_PLAYHEAD_PENALTY

        def score(f: int) -> float:
            # Signed delta along the play direction: positive = ahead, negative = behind.
            delta = (f - cur) * d
            if delta < 0:
                return -delta * penalty
            return float(delta)

        # Highest score = furthest in eviction priority order.
        by_priority = sorted(self._frames.keys(), key=score, reverse=True)
        for f in by_priority:
            if self._bytes_used <= self._budget:
                break
            arr = self._frames.pop(f)
            self._bytes_used -= arr.nbytes
            self._evictions += 1
