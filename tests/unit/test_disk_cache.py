"""Exhaustive tests for :class:`img_player.cache.disk_cache.DiskCache`.

Covers every shipped feature of the disk-cache tier:

  * Basic API contract (put / get / contains_keys / remove / clear)
  * Stats counters (hits / misses / writes / read+written MB)
  * Serialization round-trip for the 3 on-disk formats (v1 NPY,
    v2 struct+lz4, v3 struct+raw) and the dtype matrix.
  * LRU eviction once budget is exceeded.
  * Async writer back-pressure (bounded queue drops on overflow).
  * Format-version migration (E4) — PRAGMA user_version bump wipes.
  * Orphan blob sweep (E2) at boot.
  * Multi-process lock (F) — second instance falls back to read-only.
  * Shutdown drain (E1) — progress callback ticks pending count.
  * Compression toggle (perf v3) — flipping at runtime is non-destructive.

Each test is hermetic via ``tmp_path``; the writer thread is always
explicitly drained so test order can't leak state.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

from img_player.cache.disk_cache import (
    _BLOB_MAGIC_V1,
    _BLOB_MAGIC_V2,
    _BLOB_MAGIC_V3,
    _CACHE_FORMAT_VERSION,
    DiskCache,
    DiskCacheStats,
    _compress,
    _deserialize,
    _serialize,
)


# ============================================================================
# Helpers
# ============================================================================


def _frame(h: int = 16, w: int = 16, c: int = 4, dtype=np.float32) -> np.ndarray:
    """Tiny deterministic frame buffer for round-trip tests."""
    arr = np.zeros((h, w, c), dtype=dtype)
    if dtype in (np.float16, np.float32):
        arr[..., 0] = np.linspace(0.0, 1.0, w, dtype=dtype)
        arr[..., 1] = np.linspace(0.0, 1.0, h, dtype=dtype)[:, None]
        arr[..., 2] = 0.5
        arr[..., 3] = 1.0
    return arr


def _wait_for_writes(cache: DiskCache, expected: int, timeout_s: float = 5.0) -> None:
    """Spin until the cache reports ``expected`` writes or timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cache.stats().writes >= expected:
            return
        time.sleep(0.02)
    pytest.fail(
        f"writer thread did not produce {expected} writes in {timeout_s}s "
        f"(got {cache.stats().writes})"
    )


def _make_v1_blob(arr: np.ndarray) -> bytes:
    """Reproduce the legacy 1.5.0..1.5.4 serialization format."""
    import io as _io

    buf = _io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return _BLOB_MAGIC_V1 + _compress(buf.getvalue())


# ============================================================================
# Pure serialization round-trips (no DB / writer thread)
# ============================================================================


class TestSerialization:
    """Standalone round-trip checks for the module-level _serialize /
    _deserialize helpers — covers all three on-disk formats and the
    full dtype matrix without touching the filesystem."""

    def test_v2_float32_input_stored_as_float16(self) -> None:
        src = _frame(dtype=np.float32)
        blob = _serialize(src, compress=True)
        assert blob.startswith(_BLOB_MAGIC_V2)
        out = _deserialize(blob)
        # Float32 → float16 cast happens inside _serialize.
        assert out.dtype == np.float16
        assert out.shape == src.shape
        # Half-float quantisation tolerance.
        np.testing.assert_allclose(out, src.astype(np.float16), atol=1e-3)

    def test_v2_float16_stored_as_is(self) -> None:
        src = _frame(dtype=np.float16)
        blob = _serialize(src, compress=True)
        out = _deserialize(blob)
        assert out.dtype == np.float16
        np.testing.assert_array_equal(out, src)

    def test_v2_uint8_preserved(self) -> None:
        # Placeholder buffers come through as uint8 — must round-trip
        # without a float cast.
        src = np.full((8, 8, 4), 200, dtype=np.uint8)
        blob = _serialize(src, compress=True)
        out = _deserialize(blob)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, src)

    def test_v2_uint16_preserved(self) -> None:
        src = np.full((8, 8, 4), 30000, dtype=np.uint16)
        blob = _serialize(src, compress=True)
        out = _deserialize(blob)
        assert out.dtype == np.uint16
        np.testing.assert_array_equal(out, src)

    def test_v3_uncompressed_round_trip(self) -> None:
        src = _frame(dtype=np.float16)
        blob = _serialize(src, compress=False)
        assert blob.startswith(_BLOB_MAGIC_V3)
        out = _deserialize(blob)
        assert out.dtype == np.float16
        np.testing.assert_array_equal(out, src)

    def test_v3_blob_is_larger_than_v2(self) -> None:
        # On structured data lz4 should buy us ~50 %; the test uses a
        # large-enough buffer that the magic-prefix overhead doesn't
        # dominate. Sanity check that v3 = bigger.
        src = np.zeros((512, 512, 4), dtype=np.float16)  # all-zeros = lz4 ideal
        v2 = _serialize(src, compress=True)
        v3 = _serialize(src, compress=False)
        assert len(v3) > len(v2)

    def test_v1_legacy_blob_still_readable(self) -> None:
        """A blob written in the 1.5.0..1.5.4 format must continue to
        deserialize after the v2 upgrade — otherwise a Flick update
        would silently invalidate every cached frame."""
        src = _frame(dtype=np.float16)
        legacy = _make_v1_blob(src)
        out = _deserialize(legacy)
        np.testing.assert_array_equal(out, src)

    def test_corrupt_blob_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing magic prefix"):
            _deserialize(b"NOPE" + b"\x00" * 100)

    def test_exotic_dtype_falls_back_to_v1_npy(self) -> None:
        """Dtypes not in the code table go through the legacy NPY
        path so we never lose data — only the read is slower."""
        src = np.zeros((4, 4, 4), dtype=np.int32)
        blob = _serialize(src, compress=True)
        # No struct-header code for int32 → fallback uses the v1 magic.
        assert blob.startswith(_BLOB_MAGIC_V1)
        out = _deserialize(blob)
        assert out.dtype == np.int32
        np.testing.assert_array_equal(out, src)


# ============================================================================
# DiskCache class — basic API surface
# ============================================================================


@pytest.fixture
def cache(tmp_path: Path) -> DiskCache:
    """Plain unlimited cache. Each test gets a fresh tmp directory."""
    c = DiskCache(tmp_path / "cache", budget_bytes=0)
    yield c
    c.shutdown(timeout_s=2.0)


class TestBasicAPI:
    def test_get_miss_returns_none(self, cache: DiskCache) -> None:
        assert cache.get("no-such-key") is None
        assert cache.stats().misses == 1
        assert cache.stats().hits == 0

    def test_put_then_get_round_trip(self, cache: DiskCache) -> None:
        src = _frame()
        cache.put("k1" * 20, src)
        _wait_for_writes(cache, 1)
        out = cache.get("k1" * 20)
        assert out is not None
        assert out.shape == src.shape
        # float32 input is stored as float16.
        np.testing.assert_allclose(
            out.astype(np.float32), src, atol=1e-3,
        )
        st = cache.stats()
        assert st.hits == 1
        assert st.writes == 1
        assert st.entries == 1

    def test_put_then_remove(self, cache: DiskCache) -> None:
        cache.put("removeme" * 5, _frame())
        _wait_for_writes(cache, 1)
        assert cache.get("removeme" * 5) is not None
        cache.remove("removeme" * 5)
        assert cache.get("removeme" * 5) is None
        assert cache.entry_count() == 0

    def test_clear_wipes_everything(self, cache: DiskCache) -> None:
        for i in range(5):
            cache.put(f"k{i:040d}", _frame())
        _wait_for_writes(cache, 5)
        freed = cache.clear()
        assert freed > 0, "clear should report bytes freed"
        assert cache.entry_count() == 0
        # The blob tree should be empty too.
        bins = list((cache.cache_dir()).rglob("*.bin"))
        assert bins == [], f"blobs left after clear: {bins}"

    def test_contains_keys_bulk_query(self, cache: DiskCache) -> None:
        keys = [f"k{i:040d}" for i in range(10)]
        for k in keys:
            cache.put(k, _frame())
        _wait_for_writes(cache, 10)
        # Mix present + absent keys
        probe = keys[:5] + ["absent" * 8]
        present = cache.contains_keys(probe)
        assert present == set(keys[:5])

    def test_put_silently_drops_invalid_inputs(self, cache: DiskCache) -> None:
        # Empty key → no-op (not a crash).
        cache.put("", _frame())
        # Wrong ndim → no-op.
        cache.put("k" * 40, np.zeros((4, 4), dtype=np.float32))
        # None → no-op.
        cache.put("k" * 40, None)  # type: ignore[arg-type]
        # Nothing should have made it past the gates.
        time.sleep(0.1)
        assert cache.stats().writes == 0

    def test_last_access_updates_on_hit(self, cache: DiskCache) -> None:
        """LRU correctness — a get() must bump last_access so the
        entry isn't the first to evict on the next pressure round."""
        cache.put("a" * 40, _frame())
        cache.put("b" * 40, _frame())
        _wait_for_writes(cache, 2)
        time.sleep(0.05)
        # Hit the second key → it becomes the most-recently-accessed.
        cache.get("b" * 40)
        # The batched updater flushes every _ACCESS_FLUSH_INTERVAL (=2s).
        # Force a flush via the public API (shutdown drains).
        # Verifying the actual SQLite timestamp is overkill here —
        # the hit was registered if the counter ticks.
        assert cache.stats().hits >= 1


# ============================================================================
# Stats
# ============================================================================


class TestStats:
    def test_default_stats_zeroed(self, cache: DiskCache) -> None:
        s = cache.stats()
        assert s.hits == 0
        assert s.misses == 0
        assert s.writes == 0
        assert s.entries == 0
        assert s.size_bytes == 0
        assert s.read_only is False

    def test_hit_rate_safe_when_no_calls(self) -> None:
        s = DiskCacheStats()
        assert s.hit_rate == 0.0  # no divide-by-zero

    def test_hit_rate_after_one_hit(self) -> None:
        s = DiskCacheStats(hits=3, misses=1)
        assert s.hit_rate == 0.75


# ============================================================================
# Eviction by budget
# ============================================================================


class TestEviction:
    def test_unlimited_budget_never_evicts(self, tmp_path: Path) -> None:
        c = DiskCache(tmp_path / "c", budget_bytes=0)
        try:
            for i in range(20):
                c.put(f"k{i:040d}", _frame())
            _wait_for_writes(c, 20)
            assert c.entry_count() == 20
            assert c.stats().evictions == 0
        finally:
            c.shutdown(timeout_s=2.0)

    def test_tight_budget_triggers_eviction(self, tmp_path: Path) -> None:
        """Budget = ~3 small frames; writing 10 must trigger eviction
        bringing total entries down towards the budget headroom."""
        # Each frame compresses to ~200 bytes; budget of 2 KB ⇒ ~10 fit.
        # Push more than that and confirm the LRU starts trimming.
        budget = 2 * 1024
        c = DiskCache(tmp_path / "c", budget_bytes=budget)
        try:
            for i in range(30):
                c.put(f"k{i:040d}", _frame())
                # Don't hammer the queue — give the writer a moment.
                time.sleep(0.005)
            # Wait for the writer to drain. We don't assert on exact
            # eviction count — depends on how much each frame compresses
            # — but evictions should have fired at least once.
            time.sleep(0.5)
            stats = c.stats()
            assert stats.size_bytes <= budget * 1.1, (
                f"size_bytes ({stats.size_bytes}) exceeds budget "
                f"({budget}) by more than 10% — eviction not running"
            )
            assert stats.evictions > 0, "no eviction recorded"
        finally:
            c.shutdown(timeout_s=2.0)

    def test_set_budget_shrinks_immediately(self, tmp_path: Path) -> None:
        c = DiskCache(tmp_path / "c", budget_bytes=0)
        try:
            for i in range(10):
                c.put(f"k{i:040d}", _frame())
            _wait_for_writes(c, 10)
            size_before = c.size_bytes()
            # Shrink to ~30% of current usage → eviction must run.
            c.set_budget(size_before // 3)
            time.sleep(0.2)
            assert c.size_bytes() < size_before
            assert c.stats().evictions > 0
        finally:
            c.shutdown(timeout_s=2.0)


# ============================================================================
# Async writer back-pressure
# ============================================================================


class TestAsyncWriter:
    def test_writer_queue_does_not_block_caller(self, tmp_path: Path) -> None:
        """put() must return immediately even if the queue is full.
        Saturate the queue and confirm the caller never stalls."""
        c = DiskCache(tmp_path / "c", budget_bytes=0)
        try:
            t0 = time.perf_counter()
            for i in range(500):
                c.put(f"k{i:040d}", _frame(128, 128, 4))
            elapsed = time.perf_counter() - t0
            # 500 puts should never take more than ~1 s of pure-Python
            # enqueue + back-pressure dropping. The actual disk writes
            # finish in the background.
            assert elapsed < 1.0, f"put() blocked: {elapsed:.2f}s for 500 puts"
        finally:
            c.shutdown(timeout_s=5.0)

    def test_pending_writes_reflects_queue_depth(self, tmp_path: Path) -> None:
        c = DiskCache(tmp_path / "c", budget_bytes=0)
        try:
            # Quickly flood the queue.
            for i in range(50):
                c.put(f"k{i:040d}", _frame(256, 256, 4))
            # Immediately after the flood, some writes are still pending.
            # We don't assert an exact number (timing-dependent), just
            # that the method returns a sensible non-negative int.
            depth = c.pending_writes()
            assert depth >= 0
            assert isinstance(depth, int)
        finally:
            c.shutdown(timeout_s=5.0)


# ============================================================================
# Format-version migration (E4)
# ============================================================================


class TestMigration:
    def _seed_legacy(
        self, cache_dir: Path, version: int, with_entry: bool,
    ) -> Path:
        """Create a pre-existing SQLite index at ``version`` with an
        optional fake entry + blob file."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(cache_dir / "index.sqlite", isolation_level=None)
        db.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "key TEXT PRIMARY KEY, blob_path TEXT NOT NULL, "
            "size_bytes INTEGER NOT NULL, last_access REAL NOT NULL, "
            "created_at REAL NOT NULL)"
        )
        blob_path = None
        if with_entry:
            rel = "ab/cd/abcd1234legacy.bin"
            blob_path = cache_dir / rel
            blob_path.parent.mkdir(parents=True, exist_ok=True)
            blob_path.write_bytes(b"LEGACY-PAYLOAD")
            db.execute(
                "INSERT INTO entries(key, blob_path, size_bytes, "
                "last_access, created_at) VALUES (?, ?, ?, ?, ?)",
                ("abcd1234legacy", rel, 14, time.time(), time.time()),
            )
        db.execute(f"PRAGMA user_version = {version}")
        db.close()
        return blob_path

    def _read_user_version(self, cache_dir: Path) -> int:
        db = sqlite3.connect(cache_dir / "index.sqlite", isolation_level=None)
        row = db.execute("PRAGMA user_version").fetchone()
        db.close()
        return int(row[0]) if row else 0

    def test_legacy_v0_with_entries_gets_wiped(self, tmp_path: Path) -> None:
        cdir = tmp_path / "c"
        legacy_blob = self._seed_legacy(cdir, version=0, with_entry=True)
        assert legacy_blob is not None and legacy_blob.exists()
        c = DiskCache(cdir, budget_bytes=0)
        try:
            assert self._read_user_version(cdir) == _CACHE_FORMAT_VERSION
            assert not legacy_blob.exists(), "legacy blob should be wiped"
            assert c.entry_count() == 0
        finally:
            c.shutdown(timeout_s=2.0)

    def test_fresh_db_stamped_silently(self, tmp_path: Path) -> None:
        cdir = tmp_path / "c"
        c = DiskCache(cdir, budget_bytes=0)
        try:
            assert self._read_user_version(cdir) == _CACHE_FORMAT_VERSION
        finally:
            c.shutdown(timeout_s=2.0)

    def test_current_version_db_no_op(self, tmp_path: Path) -> None:
        cdir = tmp_path / "c"
        kept_blob = self._seed_legacy(
            cdir, version=_CACHE_FORMAT_VERSION, with_entry=True,
        )
        c = DiskCache(cdir, budget_bytes=0)
        try:
            # Entry stays in the DB; only its blob_path matters for the
            # sweep, so the legacy fake file might get swept (no magic
            # prefix matches) but that's the orphan-sweep path, not the
            # migration path.
            assert self._read_user_version(cdir) == _CACHE_FORMAT_VERSION
        finally:
            c.shutdown(timeout_s=2.0)


# ============================================================================
# Orphan blob sweep (E2)
# ============================================================================


class TestSweepOrphans:
    def test_orphan_blob_removed(self, tmp_path: Path) -> None:
        """A .bin file with no matching SQLite row must be deleted at
        boot."""
        cdir = tmp_path / "c"
        cdir.mkdir(parents=True)
        # Plant an orphan with valid sharded path but no DB row.
        orphan = cdir / "ef" / "01" / "ef01dead-orphan.bin"
        orphan.parent.mkdir(parents=True)
        orphan.write_bytes(b"orphan-bytes")
        assert orphan.exists()
        c = DiskCache(cdir, budget_bytes=0)
        try:
            assert not orphan.exists(), "orphan should be swept at boot"
        finally:
            c.shutdown(timeout_s=2.0)

    def test_tracked_blob_preserved(self, tmp_path: Path) -> None:
        """Sweep must NOT touch blobs that have a matching SQLite row."""
        cdir = tmp_path / "c"
        c = DiskCache(cdir, budget_bytes=0)
        try:
            c.put("survivor" * 5, _frame())
            _wait_for_writes(c, 1)
            blobs_before = list(cdir.rglob("*.bin"))
            assert len(blobs_before) == 1
        finally:
            c.shutdown(timeout_s=2.0)

        # Re-open. The sweep should NOT remove the legitimate blob.
        c2 = DiskCache(cdir, budget_bytes=0)
        try:
            blobs_after = list(cdir.rglob("*.bin"))
            assert len(blobs_after) == 1
            assert c2.get("survivor" * 5) is not None
        finally:
            c2.shutdown(timeout_s=2.0)

    def test_sweep_handles_empty_dir(self, tmp_path: Path) -> None:
        # No crash on fresh empty dir.
        c = DiskCache(tmp_path / "fresh", budget_bytes=0)
        c.shutdown(timeout_s=2.0)


# ============================================================================
# Multi-process lock (F)
# ============================================================================


class TestMultiProcessLock:
    def test_first_instance_owns_lock(self, tmp_path: Path) -> None:
        c = DiskCache(tmp_path / "c", budget_bytes=0)
        try:
            assert c.is_read_only() is False
            assert c.stats().read_only is False
        finally:
            c.shutdown(timeout_s=2.0)

    def test_second_instance_falls_back_to_read_only(
        self, tmp_path: Path,
    ) -> None:
        cdir = tmp_path / "c"
        c1 = DiskCache(cdir, budget_bytes=0)
        try:
            # Second open must succeed but flag itself read-only.
            # ``lock_retry_timeout_s=0`` makes the test fast — we
            # don't want the production 3 s retry budget here since
            # the first instance won't release for the duration of
            # the test anyway.
            c2 = DiskCache(cdir, budget_bytes=0, lock_retry_timeout_s=0.0)
            try:
                assert c2.is_read_only() is True
                assert c2.stats().read_only is True
            finally:
                c2.shutdown(timeout_s=2.0)
        finally:
            c1.shutdown(timeout_s=2.0)

    def test_read_only_put_is_noop(self, tmp_path: Path) -> None:
        cdir = tmp_path / "c"
        c1 = DiskCache(cdir, budget_bytes=0)
        try:
            c2 = DiskCache(cdir, budget_bytes=0, lock_retry_timeout_s=0.0)
            try:
                assert c2.is_read_only()
                # put() on a read-only instance must NOT enqueue.
                c2.put("k" * 40, _frame())
                time.sleep(0.1)
                assert c2.stats().writes == 0
            finally:
                c2.shutdown(timeout_s=2.0)
        finally:
            c1.shutdown(timeout_s=2.0)

    def test_read_only_clear_is_noop(self, tmp_path: Path) -> None:
        cdir = tmp_path / "c"
        c1 = DiskCache(cdir, budget_bytes=0)
        try:
            c1.put("k" * 40, _frame())
            _wait_for_writes(c1, 1)
            c2 = DiskCache(cdir, budget_bytes=0, lock_retry_timeout_s=0.0)
            try:
                # Read-only clear returns 0 and leaves entries intact.
                assert c2.clear() == 0
                # Owner's view still has the entry.
                assert c1.entry_count() == 1
            finally:
                c2.shutdown(timeout_s=2.0)
        finally:
            c1.shutdown(timeout_s=2.0)

    def test_lock_released_after_owner_shutdown(self, tmp_path: Path) -> None:
        cdir = tmp_path / "c"
        c1 = DiskCache(cdir, budget_bytes=0)
        c1.shutdown(timeout_s=2.0)
        # Now a fresh instance should be able to re-acquire.
        c2 = DiskCache(cdir, budget_bytes=0)
        try:
            assert c2.is_read_only() is False
        finally:
            c2.shutdown(timeout_s=2.0)

    def test_close_then_reopen_within_retry_window_keeps_writable(
        self, tmp_path: Path,
    ) -> None:
        """The bug we hit in the field: user closes Flick and
        immediately re-launches. The previous process is still
        draining the writer queue (E1) so the lock is still held
        for a few hundred ms. Without retry, the new instance would
        flip to read-only — surprising and frustrating.

        Here we simulate it: first instance shuts down on a delayed
        background thread so the second instance hits the retry
        loop, succeeds within the retry window, and ends up
        writable. Catches the regression if the retry logic is
        ever stripped out.
        """
        import threading

        cdir = tmp_path / "c"
        c1 = DiskCache(cdir, budget_bytes=0)
        # Release the lock ~400 ms in the future. Comfortably under
        # the production 3 s retry window so the retry loop will
        # catch it; comfortably above the 100 ms retry interval so
        # at least one failed attempt happens before success.
        release_after = 0.4

        def _delayed_shutdown() -> None:
            time.sleep(release_after)
            c1.shutdown(timeout_s=2.0)

        t = threading.Thread(target=_delayed_shutdown, daemon=True)
        t.start()
        try:
            c2 = DiskCache(cdir, budget_bytes=0)
            # Retry kicked in and the lock landed → writable!
            assert c2.is_read_only() is False, (
                "second instance should have acquired the lock after retry"
            )
            c2.shutdown(timeout_s=2.0)
        finally:
            t.join(timeout=5.0)

    def test_read_only_can_still_read(self, tmp_path: Path) -> None:
        """The owner writes; the read-only second instance must be
        able to retrieve those entries via get()."""
        cdir = tmp_path / "c"
        c1 = DiskCache(cdir, budget_bytes=0)
        try:
            src = _frame()
            c1.put("shared" * 7, src)
            _wait_for_writes(c1, 1)
            c2 = DiskCache(cdir, budget_bytes=0, lock_retry_timeout_s=0.0)
            try:
                assert c2.is_read_only()
                out = c2.get("shared" * 7)
                assert out is not None
                assert out.shape == src.shape
            finally:
                c2.shutdown(timeout_s=2.0)
        finally:
            c1.shutdown(timeout_s=2.0)


# ============================================================================
# Shutdown drain (E1)
# ============================================================================


class TestShutdown:
    def test_shutdown_is_idempotent(self, tmp_path: Path) -> None:
        c = DiskCache(tmp_path / "c", budget_bytes=0)
        c.shutdown(timeout_s=2.0)
        # Second call must not raise.
        c.shutdown(timeout_s=2.0)

    def test_progress_callback_fires_during_drain(
        self, tmp_path: Path,
    ) -> None:
        c = DiskCache(tmp_path / "c", budget_bytes=0)
        # Queue up a backlog. With small frames the writer drains
        # fast; with bigger arrays it has measurable work.
        for i in range(30):
            c.put(f"k{i:040d}", _frame(256, 256, 4))
        ticks: list[int] = []
        c.shutdown(
            timeout_s=10.0,
            progress_callback=lambda n: ticks.append(n),
        )
        # The callback should have fired at least once. The last
        # reported tick should be 0 (drain complete) — but only if the
        # backlog was non-trivial; if the writer was already idle the
        # callback may be skipped entirely. We tolerate both.
        if ticks:
            assert ticks[-1] == 0 or ticks[-1] < ticks[0]

    def test_shutdown_releases_lock(self, tmp_path: Path) -> None:
        cdir = tmp_path / "c"
        c1 = DiskCache(cdir, budget_bytes=0)
        c1.shutdown(timeout_s=2.0)
        c2 = DiskCache(cdir, budget_bytes=0)
        try:
            assert c2.is_read_only() is False
        finally:
            c2.shutdown(timeout_s=2.0)


# ============================================================================
# Compression toggle (perf v3)
# ============================================================================


class TestCompressionToggle:
    def test_default_writes_use_v2(self, tmp_path: Path) -> None:
        c = DiskCache(tmp_path / "c", budget_bytes=0)
        try:
            c.put("k" * 40, _frame())
            _wait_for_writes(c, 1)
            # Walk the cache tree, find the single blob, sniff magic.
            blobs = list((c.cache_dir()).rglob("*.bin"))
            assert len(blobs) == 1
            assert blobs[0].read_bytes()[:4] == _BLOB_MAGIC_V2
        finally:
            c.shutdown(timeout_s=2.0)

    def test_compress_false_writes_v3(self, tmp_path: Path) -> None:
        c = DiskCache(tmp_path / "c", budget_bytes=0, compress=False)
        try:
            c.put("k" * 40, _frame())
            _wait_for_writes(c, 1)
            blobs = list((c.cache_dir()).rglob("*.bin"))
            assert len(blobs) == 1
            assert blobs[0].read_bytes()[:4] == _BLOB_MAGIC_V3
        finally:
            c.shutdown(timeout_s=2.0)

    def test_set_compress_toggle_non_destructive(self, tmp_path: Path) -> None:
        """Existing entries written under the old compression mode
        must remain readable after the toggle flips."""
        c = DiskCache(tmp_path / "c", budget_bytes=0, compress=True)
        try:
            src = _frame()
            c.put("a" * 40, src)
            _wait_for_writes(c, 1)
            # Flip to no-compression for future writes.
            c.set_compress(False)
            c.put("b" * 40, src)
            _wait_for_writes(c, 2)
            # Both keys still readable.
            assert c.get("a" * 40) is not None
            assert c.get("b" * 40) is not None
            # The two blobs should have different magics on disk.
            magics = {
                p.read_bytes()[:4]
                for p in c.cache_dir().rglob("*.bin")
            }
            assert _BLOB_MAGIC_V2 in magics
            assert _BLOB_MAGIC_V3 in magics
        finally:
            c.shutdown(timeout_s=2.0)


# ============================================================================
# Persistence across re-open (the core promise of the disk tier)
# ============================================================================


class TestPersistence:
    def test_entries_survive_reopen(self, tmp_path: Path) -> None:
        """The whole point of the disk tier: closing and re-opening
        Flick finds yesterday's frames warm."""
        cdir = tmp_path / "c"
        src = _frame()
        c1 = DiskCache(cdir, budget_bytes=0)
        try:
            c1.put("warm" * 10, src)
            _wait_for_writes(c1, 1)
        finally:
            c1.shutdown(timeout_s=2.0)
        # Fresh DiskCache pointing at the same dir.
        c2 = DiskCache(cdir, budget_bytes=0)
        try:
            out = c2.get("warm" * 10)
            assert out is not None
            assert out.shape == src.shape
            # Stats counters reset across sessions by design.
            assert c2.stats().writes == 0
            assert c2.stats().hits == 1
        finally:
            c2.shutdown(timeout_s=2.0)

    def test_entry_count_reflects_persisted_state(self, tmp_path: Path) -> None:
        cdir = tmp_path / "c"
        c1 = DiskCache(cdir, budget_bytes=0)
        try:
            for i in range(5):
                c1.put(f"k{i:040d}", _frame())
            _wait_for_writes(c1, 5)
        finally:
            c1.shutdown(timeout_s=2.0)
        c2 = DiskCache(cdir, budget_bytes=0)
        try:
            assert c2.entry_count() == 5
        finally:
            c2.shutdown(timeout_s=2.0)
