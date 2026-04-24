"""Thread-safe RAM cache of decoded frames, with async prefetch and LRU-by-distance eviction."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from img_player.cache.worker_pool import WorkerPool
from img_player.io.reader import FrameReadError, read_frame
from img_player.sequence.models import SequenceInfo

log = logging.getLogger(__name__)

_DEFAULT_BUDGET_BYTES = 8 * 1024**3  # 8 GB
_DEFAULT_NUM_WORKERS = 4


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

    def clear(self) -> None:
        """Remove cached frames and drop pending decodes. Sequence stays attached."""
        self._pool.clear()
        with self._lock:
            self._frames.clear()
            self._bytes_used = 0

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
        try:
            arr = read_frame(path)
        except FrameReadError as err:
            log.warning("decode failed frame=%d path=%s: %s", frame, path, err)
            with self._lock:
                self._decode_errors += 1
            return

        with self._lock:
            if frame in self._frames:
                return  # raced with another worker; keep the existing one
            self._frames[frame] = arr
            self._bytes_used += arr.nbytes
            self._evict_if_over_budget()

    def _evict_if_over_budget(self) -> None:
        """Must be called with _lock held."""
        if self._bytes_used <= self._budget:
            return
        by_distance = sorted(
            self._frames.keys(),
            key=lambda f: abs(f - self._current_frame),
            reverse=True,
        )
        for f in by_distance:
            if self._bytes_used <= self._budget:
                break
            arr = self._frames.pop(f)
            self._bytes_used -= arr.nbytes
            self._evictions += 1
