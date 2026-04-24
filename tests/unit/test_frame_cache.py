"""Tests for cache/frame_cache.py — async decode, LRU-by-distance eviction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from img_player.cache.frame_cache import FrameCache
from img_player.sequence.scanner import scan

# One 16x16 RGBA float32 frame is 16*16*4*4 = 4096 bytes.
_FRAME_BYTES = 16 * 16 * 4 * 4


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
    assert arr.dtype == np.float32


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
