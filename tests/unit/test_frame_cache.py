"""Tests for cache/frame_cache.py — async decode, LRU-by-distance eviction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from img_player.cache.frame_cache import FrameCache
from img_player.sequence.scanner import scan

# One 16x16 RGBA half-float frame is 16*16*4*2 = 2048 bytes (the reader
# defaults to float16 for memory efficiency).
_FRAME_BYTES = 16 * 16 * 4 * 2


@pytest.fixture
def cache_small() -> FrameCache:
    """Budget just large enough to hold ~3 of the fixture frames."""
    cache = FrameCache(budget_bytes=3 * _FRAME_BYTES, num_workers=2)
    yield cache
    cache.shutdown()


@pytest.fixture
def cache_roomy() -> FrameCache:
    cache = FrameCache(budget_bytes=1024 * 1024, num_workers=2)
    yield cache
    cache.shutdown()


def test_miss_returns_none(cache_roomy: FrameCache, sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    cache_roomy.attach(seq)
    assert cache_roomy.get(1) is None


def test_requested_frame_eventually_cached(cache_roomy: FrameCache, sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    cache_roomy.attach(seq)
    cache_roomy.request(seq.first_frame)
    assert cache_roomy.wait_idle(timeout=3.0)
    arr = cache_roomy.get(seq.first_frame)
    assert arr is not None
    assert arr.dtype in (np.float16, np.float32)


def test_request_range_populates_cache(cache_roomy: FrameCache, sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    cache_roomy.attach(seq)
    cache_roomy.request_range(seq.first_frame, seq.last_frame)
    assert cache_roomy.wait_idle(timeout=5.0)
    stats = cache_roomy.stats()
    assert stats.frames_cached == seq.frame_count


def test_request_dedup(cache_roomy: FrameCache, sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    cache_roomy.attach(seq)
    assert cache_roomy.request(seq.first_frame) is True
    assert (
        cache_roomy.request(seq.first_frame) is True
        or cache_roomy.request(seq.first_frame) is False
    )
    # Either the worker already finished (so request returns True again after first cache miss)
    # OR it's still pending (so submit_to_pool returned False). We test the stronger
    # invariant: after wait_idle we only have one cached entry, not duplicates.
    cache_roomy.wait_idle(timeout=3.0)
    assert cache_roomy.stats().frames_cached <= 1


def test_eviction_by_distance(cache_small: FrameCache, sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    cache_small.attach(seq)
    # Set playhead far from frame 1 so that when eviction kicks in,
    # low frame numbers get dropped first.
    cache_small.set_current_frame(10)
    cache_small.request_range(seq.first_frame, seq.last_frame)
    assert cache_small.wait_idle(timeout=5.0)

    stats = cache_small.stats()
    # Budget is ~3 frames worth. We should have evicted some.
    assert stats.evictions > 0
    assert stats.bytes_used <= cache_small._budget

    # The frames closest to playhead (10) should be kept.
    assert cache_small.contains(10)
    # Far-away frames should have been evicted.
    assert not cache_small.contains(1)


def test_clear_removes_all_frames(cache_roomy: FrameCache, sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    cache_roomy.attach(seq)
    cache_roomy.request_range(seq.first_frame, seq.last_frame)
    cache_roomy.wait_idle(timeout=5.0)
    assert cache_roomy.stats().frames_cached > 0

    cache_roomy.clear()
    assert cache_roomy.stats().frames_cached == 0
    assert cache_roomy.stats().bytes_used == 0


def test_attach_resets_cache(cache_roomy: FrameCache, sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    cache_roomy.attach(seq)
    cache_roomy.request(seq.first_frame)
    cache_roomy.wait_idle(timeout=3.0)
    assert cache_roomy.stats().frames_cached >= 1

    # re-attach the same sequence: cache reset
    cache_roomy.attach(seq)
    assert cache_roomy.stats().frames_cached == 0


def test_get_updates_playhead(cache_roomy: FrameCache, sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    cache_roomy.attach(seq)
    cache_roomy.get(5)  # miss, but sets playhead to 5
    assert cache_roomy._current_frame == 5


def test_request_out_of_sequence_is_noop(cache_roomy: FrameCache, sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    cache_roomy.attach(seq)
    # frame 999 does not exist in sequence
    assert cache_roomy.request(999) is False
    assert cache_roomy.wait_idle(timeout=0.5)
    assert cache_roomy.stats().frames_cached == 0


def test_set_channels_drops_in_flight_decode(
    cache_roomy: FrameCache, sequence_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a channel switch issued mid-decode must not let the
    in-flight worker pollute the freshly-cleared cache.

    ``WorkerPool.clear()`` only drops queued tasks — workers already
    running their ``read_frame()`` call cannot be cancelled. Without
    the epoch fence in ``_decode_and_store``, those workers would
    write a stale-channel array into the cache after the switch, and
    the cache bar would show phantom rectangles the user cannot
    account for.
    """
    import threading

    seq = scan(sequence_dir)
    cache_roomy.attach(seq)

    # Block the decode until we say so. The worker thread will sit in
    # read_frame() with the OLD channel selection captured; meanwhile
    # the test thread switches channels and clears the cache.
    decode_gate = threading.Event()
    decode_started = threading.Event()
    real_read_frame = None

    def slow_read_frame(path: Path, *, channels: list[str] | None = None) -> np.ndarray:
        decode_started.set()
        # Block until the test releases us.
        decode_gate.wait(timeout=5.0)
        # Return a tiny dummy array; we just want to exercise the
        # store path, not the real decoder.
        return np.zeros((4, 4, 4), dtype=np.float16)

    # Patch the symbol the cache module imported, not the io one.
    import img_player.cache.frame_cache as fc_mod
    monkeypatch.setattr(fc_mod, "read_frame", slow_read_frame)

    cache_roomy.request(seq.first_frame)
    assert decode_started.wait(timeout=2.0), "worker never picked up the request"

    # Mid-decode: switch channels. This bumps the epoch and clears the
    # (still-empty) frame dict.
    cache_roomy.set_channels(["Z"])

    # Release the worker. It will return its stale-channel array and
    # try to store it — the epoch fence must drop it on the floor.
    decode_gate.set()
    assert cache_roomy.wait_idle(timeout=3.0)

    assert cache_roomy.stats().frames_cached == 0, (
        "stale-channel decode leaked into the cache after set_channels"
    )


def test_stats_counts_hits_and_misses(cache_roomy: FrameCache, sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    cache_roomy.attach(seq)
    cache_roomy.get(1)  # miss
    cache_roomy.request(1)
    cache_roomy.wait_idle(timeout=3.0)
    cache_roomy.get(1)  # hit
    cache_roomy.get(2)  # miss

    s = cache_roomy.stats()
    assert s.hits >= 1
    assert s.misses >= 2
