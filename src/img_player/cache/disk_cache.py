"""On-disk frame cache tier — half-float lz4 blobs + SQLite LRU index.

Sits between :class:`MasterFrameCache` (RAM) and ``read_frame`` (source
decode + OCIO input prep) as a third tier. The goal: re-opening the
same shot tomorrow finds yesterday's frames warm without re-decoding
the EXRs.

Why these choices
-----------------

* **Half-float blobs** — frames are stored as ``float16`` rather than
  ``float32``. 2× smaller on disk; the dtype is restored to float32 on
  read (free) so consumers see no difference. Half-float covers HDR up
  to 65 504 with sub-percent precision in the 0..1 range — well beyond
  what a viewer needs to display.

* **lz4 compression** — fastest mainstream compressor (~3 GB/s decode
  on a modern CPU). A 4K float16 frame compresses to ~25 MB and
  decompresses in ~5 ms. Compared to a fresh EXR decode (~300 ms) this
  is a 50× win on slow sources. Falls back to zlib if ``lz4`` isn't
  importable — the cache still works, just slower.

* **SQLite for the index** — atomic, crash-safe (WAL mode),
  transactional bulk operations for LRU eviction. One column carries
  ``last_access`` (UNIX epoch) so trimming to budget is a single
  indexed query.

* **Async writes** — :meth:`put` enqueues to a dedicated writer
  thread; the caller never blocks on disk I/O. Useful when eviction
  happens during playback: the RAM is freed immediately, the blob
  lands a few hundred ms later.

* **Sharded blob layout** — ``<cache_dir>/<hash[:2]>/<hash[2:4]>/<hash>.bin``
  to avoid 100k files in a single directory (lethal on NTFS). Each
  blob's existence is the source of truth; SQLite is rebuilt-able by
  scanning the tree if it's ever lost.

Key contract
------------

Keys are opaque strings (callers compute them — typically a
SHA-1 of ``(canonical_path, mtime, channel_set, alpha_flags)``).
The cache doesn't interpret them; it just maps key → blob. This
keeps the cache decoupled from layer / session semantics.

Threading
---------

* :meth:`get` and :meth:`put` are thread-safe.
* :meth:`put` is fire-and-forget — actual disk write happens on the
  writer thread. The ndarray passed in **must not be mutated** by the
  caller after the call returns; the cache assumes the frame is
  immutable once cached (same contract as ``MasterFrameCache._frames``).
* SQLite access is serialised via a single connection + lock.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import queue
import sqlite3
import struct
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DiskCacheStats:
    """Snapshot of the disk-cache counters.

    Updated continuously by ``get()`` / ``put()`` / writer thread;
    sampled via :meth:`DiskCache.stats` for the Preferences readout
    and any future telemetry / debug overlay. Counters reset to 0
    on process restart (the SQLite index doesn't persist them — they
    describe runtime behaviour, not on-disk state).
    """

    hits: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    errors: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    entries: int = 0
    size_bytes: int = 0
    budget_bytes: int = 0
    # True when another Flick instance already holds the cache-dir
    # lock — writes are no-ops, only reads work. The UI uses this to
    # show a "(read-only — second instance)" badge in Preferences so
    # the user understands why the write counters never tick.
    read_only: bool = False

    @property
    def hit_rate(self) -> float:
        """Fraction of ``get()`` calls that landed a hit. ``0.0`` when
        no calls have been made yet (avoids a divide-by-zero in the
        Preferences readout's percentage formatter)."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total


def default_cache_dir() -> Path:
    """Resolve the platform-default disk-cache location.

    * **Windows** — ``%LOCALAPPDATA%\\img_player\\disk_cache\\`` (the
      same root the log file uses).
    * **macOS**   — ``~/Library/Caches/img_player/disk_cache/``.
    * **Linux**   — ``$XDG_CACHE_HOME/img_player/disk_cache/`` falling
      back to ``~/.cache/img_player/disk_cache/``.

    Called from :class:`img_player.app.ImgPlayerApp` when the user
    hasn't overridden the path in Preferences. The directory itself
    is created on first :class:`DiskCache` construction; this helper
    only computes the location.
    """
    # Canonical location is centralised in :mod:`img_player.app_paths`
    # so renames (img_player → FlickPlayer in v1.5.9) live in one
    # place rather than being duplicated per-feature.
    from img_player.app_paths import disk_cache_default_dir
    return disk_cache_default_dir()

log = logging.getLogger(__name__)

# Try lz4 first (the fast path), fall back to stdlib zlib.
try:
    import lz4.frame as _lz4_frame  # type: ignore[import-untyped]
    _HAS_LZ4 = True
except ImportError:  # pragma: no cover — only on stripped envs
    _lz4_frame = None
    _HAS_LZ4 = False
    log.warning(
        "lz4 not available; disk cache will use stdlib zlib (~3× slower). "
        "Add ``lz4`` to environment.yml to enable the fast path.",
    )

import zlib

# Magic prefix written at the start of every blob so a corrupt /
# orphaned file can be identified by its first 4 bytes when sweeping
# the cache root.
#
# Two versions are recognised on read:
#
#   * ``FCD1`` — 1.5.0..1.5.4 format. Payload is
#     ``lz4(np.save(arr))``. Slow on read (npy header parse, ~1-2 ms
#     per blob) but compatible with older caches the user already
#     has on disk.
#   * ``FCD2`` — 1.5.5+ format. Payload is ``lz4(header + raw_bytes)``
#     with a 16-byte fixed-layout header packed via :mod:`struct`.
#     Skips the ``np.save`` overhead — relevant for HD/4K where it's
#     a measurable fraction of the read budget.
#
# Writes always use ``FCD2``; the legacy path is kept only so the
# user doesn't lose their warm cache the day they upgrade. ``FCD1``
# blobs fade naturally via LRU eviction as new writes replace them.
_BLOB_MAGIC_V1 = b"FCD1"
_BLOB_MAGIC_V2 = b"FCD2"
# v3 = same struct-packed header as v2 but the payload is NOT
# lz4-compressed. Trades ~2× more disk space for skipping the lz4
# decode step on every read — worth it on fast NVMe where I/O is
# essentially free and lz4 decode time (~5 ms / 4K frame) becomes
# the bottleneck.
_BLOB_MAGIC_V3 = b"FCD3"
# Backwards-compat alias for callers that imported ``_BLOB_MAGIC``
# from this module. v2 is the default for writes.
_BLOB_MAGIC = _BLOB_MAGIC_V2

# Dtype codes for the v2 header. Stable across cache format versions:
# adding a new dtype must use a fresh code (never reassign existing
# ones — would silently misread stored blobs).
_DTYPE_CODES: dict[np.dtype, int] = {
    np.dtype(np.float16): 0,
    np.dtype(np.float32): 1,
    np.dtype(np.uint8): 2,
    np.dtype(np.uint16): 3,
}
_CODE_TO_DTYPE: dict[int, np.dtype] = {v: k for k, v in _DTYPE_CODES.items()}

# v2 header — 16 bytes, little-endian:
#   B uint8  dtype code (see ``_DTYPE_CODES``)
#   B uint8  ndim (only 3 today; future-proofed)
#   H uint16 reserved (=0; align dim0 to 4-byte boundary)
#   I uint32 dim0 (height)
#   I uint32 dim1 (width)
#   I uint32 dim2 (channels)
_V2_HEADER_FMT = "<BBHIII"
_V2_HEADER_SIZE = struct.calcsize(_V2_HEADER_FMT)  # = 16

# Current on-disk format / key version, stored as SQLite
# ``PRAGMA user_version``. Bump alongside any change that invalidates
# existing blobs (key schema, serialisation layout, …). When the cache
# opens an index whose ``user_version`` is below this, it auto-clears
# the cache once at boot rather than letting the user discover stale
# pixels and reach for the "Clear cache now" button manually.
#
# History:
#   * v0 — initial 1.5.0 ship; PRAGMA never set.
#   * v1 — bogus key schema (live-state instead of submit-state).
#   * v2 — corrected key schema, lz4 + half-float blobs (current).
_CACHE_FORMAT_VERSION = 2


# Boot-time lock retry budget. Covers the common "user closes Flick
# then immediately re-launches" case: the previous process is still
# draining its disk-cache writer queue (up to E1's 10 s shutdown
# budget) and hasn't released the lock yet. Without retry, the new
# instance gives up after 1 attempt and falls back to read-only —
# a 3-second wait at boot is invisible to the user but generously
# covers a typical close→relaunch round-trip. If the lock is still
# held after that, we assume a real second instance and go read-only.
_LOCK_RETRY_TIMEOUT_S = 3.0
_LOCK_RETRY_INTERVAL_S = 0.1


def _try_acquire_lock(  # type: ignore[no-untyped-def]
    lock_path: Path,
    retry_timeout_s: float = _LOCK_RETRY_TIMEOUT_S,
):
    """Acquire an exclusive file lock cross-platform with brief retry.

    Used to detect a second Flick instance sharing the same cache
    directory. Returns the open file handle on success (caller keeps
    it alive for the duration of the lock) or ``None`` if another
    process actually holds it after ``retry_timeout_s`` seconds.

    The retry loop handles the common case where the user closes
    Flick and immediately re-launches: the previous process is still
    inside its shutdown drain (which can take up to 10 s for E1's
    writer flush). Without retry, the new instance would fall back
    to read-only every time — surprising and frustrating since the
    user thought they cleanly closed the app.

    On Windows we use :func:`msvcrt.locking` with ``LK_NBLCK``; on
    POSIX :func:`fcntl.flock` with ``LOCK_EX | LOCK_NB``. Both APIs
    release the lock when the file handle is closed, so the caller
    only has to remember to ``close()`` at shutdown.
    """
    deadline = time.monotonic() + max(0.0, retry_timeout_s)
    first_attempt = True
    while True:
        try:
            fh = open(lock_path, "ab")  # noqa: SIM115 — handle held by caller
        except OSError as err:
            log.warning("DiskCache: could not open lock file %s (%s)", lock_path, err)
            return None
        try:
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover — non-Windows
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Lock is currently held by someone else. Close our handle
            # (would otherwise leak) and either retry or give up.
            try:
                fh.close()
            except OSError:
                pass
            if time.monotonic() >= deadline:
                return None
            if first_attempt:
                # One-shot informational log so the user sees what's
                # happening if a previous instance is still draining.
                log.info(
                    "DiskCache: lock held by another instance — "
                    "waiting up to %.1f s for it to release...",
                    retry_timeout_s,
                )
                first_attempt = False
            time.sleep(_LOCK_RETRY_INTERVAL_S)
            continue
        return fh


def _compress(payload: bytes) -> bytes:
    if _HAS_LZ4:
        return _lz4_frame.compress(payload, compression_level=1)
    return zlib.compress(payload, level=1)


def _decompress(blob: bytes) -> bytes:
    if _HAS_LZ4:
        return _lz4_frame.decompress(blob)
    return zlib.decompress(blob)


def _serialize(arr: np.ndarray, compress: bool = True) -> bytes:
    """Convert ndarray → bytes blob (v2 / v3 format).

    Float arrays are cast to ``float16`` for storage (2× space win).
    Other dtypes (uint8 for placeholders, uint16 for half-float-already
    inputs) are stored as-is.

    Layout written:
      * ``compress=True``  → ``FCD2`` + ``lz4(header + raw_bytes)``
      * ``compress=False`` → ``FCD3`` + ``header + raw_bytes``

    The struct-packed header (16 bytes) replaces ``np.save``'s NPY
    format, saving ~5 ms per HD read by skipping the ``literal_eval``
    of the NPY header string and the implicit memcpy that ``np.load``
    does.

    Falls back to the v1 NPY-based path for dtypes we don't have a
    code for in :data:`_DTYPE_CODES`. That keeps the cache writable
    even if someone hands it an exotic dtype, at the cost of the
    slower read path for those entries.
    """
    # Avoid a copy when the array is already float16; ``copy=False``
    # short-circuits when the dtype already matches.
    if arr.dtype == np.float32:
        arr = arr.astype(np.float16, copy=False)
    dtype_code = _DTYPE_CODES.get(arr.dtype)
    if dtype_code is None or arr.ndim != 3:
        # Exotic dtype or non-3D shape → fall back to NPY format.
        # ``allow_pickle=False`` is a sanity guard — we only round-trip
        # plain arrays here; if a pickled object ever sneaks in we'd
        # rather fail loud than silently load arbitrary code on read.
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        return _BLOB_MAGIC_V1 + _compress(buf.getvalue())
    # Ensure C-contiguous so ``tobytes()`` matches the natural
    # ``frombuffer`` layout on read. Already-contiguous arrays
    # (the overwhelming common case) hit the no-op branch inside
    # ``ascontiguousarray``.
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    h, w, c = arr.shape
    header = struct.pack(_V2_HEADER_FMT, dtype_code, arr.ndim, 0, h, w, c)
    payload = header + arr.tobytes()
    if compress:
        return _BLOB_MAGIC_V2 + _compress(payload)
    return _BLOB_MAGIC_V3 + payload


def _deserialize(blob: bytes) -> np.ndarray:
    """Inverse of :func:`_serialize`. Returns the array in its **native
    storage dtype** — float16 for frames written by :func:`_serialize`,
    other dtypes preserved as-is.

    The consumer pipeline (GL viewport, ``CompareDecoder``) handles
    float16 natively — GL uses ``GL_HALF_FLOAT`` for the texture
    upload, no conversion needed. Returning float16 directly skips a
    ~30 ms astype copy per 4K frame (HD = ~3 ms) AND keeps the RAM
    cache half as wide, so the same budget fits 2× more frames when
    they come from disk.

    Auto-routes between v1 (legacy NPY) and v2 (raw header) on the
    magic prefix. The returned ``ndarray`` from the v2 path is a view
    over the decompressed bytes (``writeable=False``); the composite
    builder never mutates contributor buffers in place (it always
    allocates fresh accumulators via ``+`` and ``copy()``) so this
    is safe and avoids a 4K-frame memcpy on every disk hit.
    """
    if blob.startswith(_BLOB_MAGIC_V2) or blob.startswith(_BLOB_MAGIC_V3):
        is_compressed = blob.startswith(_BLOB_MAGIC_V2)
        if is_compressed:
            raw = _decompress(blob[len(_BLOB_MAGIC_V2):])
        else:
            # v3: raw payload, no lz4 step. Slice off the magic prefix
            # and hand the rest straight to ``frombuffer``. On fast
            # NVMe this saves ~5 ms / 4K read vs the v2 lz4 path.
            raw = blob[len(_BLOB_MAGIC_V3):]
        dtype_code, ndim, _reserved, h, w, c = struct.unpack_from(
            _V2_HEADER_FMT, raw, 0,
        )
        dtype = _CODE_TO_DTYPE.get(dtype_code)
        if dtype is None:
            raise ValueError(
                f"DiskCache v2/v3 blob has unknown dtype code {dtype_code}; "
                "blob written by a newer Flick version?",
            )
        # ``frombuffer`` is a view over the bytes object — no copy,
        # but read-only. ``ndim`` is captured in the header for
        # future flexibility; today only 3D is written.
        arr = np.frombuffer(raw, dtype=dtype, offset=_V2_HEADER_SIZE)
        if ndim == 3:
            return arr.reshape(h, w, c)
        # Future 2D / 4D paths land here if we ever stop gating on
        # ``ndim == 3`` in ``put()``. For now this branch is dead.
        return arr.reshape((h, w, c)[:ndim])  # pragma: no cover
    if blob.startswith(_BLOB_MAGIC_V1):
        # Legacy 1.5.0..1.5.4 path — NPY format. Kept for backwards
        # compat so an upgrade doesn't invalidate the warm cache.
        # New writes always use v2; v1 entries fade via LRU.
        raw = _decompress(blob[len(_BLOB_MAGIC_V1):])
        return np.load(io.BytesIO(raw), allow_pickle=False)
    raise ValueError(
        f"DiskCache blob missing magic prefix (got {blob[:4]!r}); "
        "file is corrupt or pre-dates 1.5.0.",
    )


# Sentinel for the writer thread's shutdown signal — comparing identity
# in the queue drain is cleaner than checking a flag inside the dequeued
# task tuple.
_SHUTDOWN_SENTINEL = object()


class DiskCache:
    """Persistent secondary cache for decoded frame buffers.

    Construct once at app startup, pass into :class:`MasterFrameCache`.
    Survives across sessions — opening the same shot tomorrow finds
    yesterday's frames available without re-decoding.

    Parameters
    ----------
    cache_dir
        Where to store the SQLite index + blob tree. Created (with
        parents) if missing.
    budget_bytes
        Soft upper bound on total disk usage. After each write the
        oldest (least-recently-accessed) entries are evicted until the
        total is ≤ 85 % of the budget (hysteresis avoids thrashing
        when the budget is tight). ``0`` = unlimited (eviction
        disabled; the cache only grows when the user explicitly
        clears it).
    """

    # When evicting to make room, trim down to this fraction of the
    # budget so a single new write doesn't immediately re-trigger
    # eviction on the next put. 85 % gives ~15 % headroom before the
    # next eviction wave.
    _EVICT_TARGET_RATIO = 0.85
    # Batch interval for ``last_access`` UPDATE flushes — see
    # ``__init__`` for the why. 2 s feels imperceptible at LRU
    # granularity (eviction happens on the order of minutes / hours).
    _ACCESS_FLUSH_INTERVAL = 2.0

    def __init__(
        self,
        cache_dir: Path,
        budget_bytes: int = 0,
        compress: bool = True,
        lock_retry_timeout_s: float = _LOCK_RETRY_TIMEOUT_S,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._budget_bytes = max(0, int(budget_bytes))
        # Compression toggle for new writes. The reader auto-detects
        # via magic prefix so flipping this at runtime (via
        # :meth:`set_compress`) doesn't invalidate the warm cache —
        # old entries stay readable, new entries follow the new flag.
        self._compress = bool(compress)

        # ---- Multi-process lock (F) ----------------------------------
        # Acquire <cache_dir>/.lock so a second Flick instance can't
        # mutate this cache concurrently. SQLite WAL handles overlapping
        # reads/writes safely, but the side-channel blob files don't —
        # two writer threads could race on the same blob path and leave
        # a truncated file. If the lock is already held, we fall back
        # to read-only mode: queries (get / contains_keys) still work,
        # but puts / writes / clears become no-ops with a debug log.
        self._lock_file = _try_acquire_lock(
            self._cache_dir / ".lock",
            retry_timeout_s=lock_retry_timeout_s,
        )
        self._read_only = self._lock_file is None
        if self._read_only:
            log.warning(
                "DiskCache at %s: another instance holds the lock — "
                "running read-only (writes will be dropped)",
                self._cache_dir,
            )

        # ---- SQLite index --------------------------------------------
        db_path = self._cache_dir / "index.sqlite"
        # ``check_same_thread=False`` because both the main thread
        # (lookups) and the writer thread touch the connection. All
        # access is guarded by ``self._db_lock``.
        self._db = sqlite3.connect(
            str(db_path), check_same_thread=False, isolation_level=None,
        )
        self._db_lock = threading.Lock()
        # WAL mode = readers don't block the writer and vice versa.
        # Crash-safe by SQLite's normal guarantees.
        with self._db_lock:
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA synchronous = NORMAL")
            self._db.execute(
                """
                CREATE TABLE IF NOT EXISTS entries (
                    key TEXT PRIMARY KEY,
                    blob_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    last_access REAL NOT NULL,
                    created_at REAL NOT NULL
                )
                """,
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_last_access "
                "ON entries(last_access)",
            )

        # ---- Format-version migration (E4) ---------------------------
        # Before spinning up the writer thread, check the on-disk
        # format version. If it's behind the current code, the blobs
        # are guaranteed stale (bogus keys or obsolete serialisation),
        # so we wipe them once at boot. The user sees a single INFO
        # log line — no dialog interrupt, no manual "Clear cache now"
        # needed after a Flick update that bumps the format.
        # Skipped in read-only mode: the owning instance will run the
        # migration when it boots; we just consume whatever's there.
        if not self._read_only:
            self._migrate_if_needed()

        # ---- Orphan-blob sweep (E2) ----------------------------------
        # Scan the cache tree for ``.bin`` files that don't appear in
        # the SQLite index. Sources of orphans:
        #   * Pre-1.5.1 blobs left over from the wrong-key bug (now
        #     dropped from the index by E4's migration, but the files
        #     themselves stay until LRU eviction touches them).
        #   * A user manually deleted ``index.sqlite`` while keeping
        #     the cache root (unlikely, but cheap to handle).
        #   * Crash mid-write where the file landed but the INSERT
        #     never committed.
        # Done at init (writer thread not started yet) so there's no
        # race with concurrent writes touching the same files.
        # Skipped in read-only mode — sweeping would race with the
        # owning instance's writer.
        if not self._read_only:
            self._sweep_orphans()

        # ---- Async writer --------------------------------------------
        # Bounded queue so a runaway producer (e.g. eviction storm)
        # doesn't OOM the process. When full, ``put`` falls back to
        # dropping the oldest enqueued task — the disk cache is a
        # best-effort persistence layer, not a correctness-critical
        # path.
        self._write_queue: queue.Queue = queue.Queue(maxsize=128)
        # ---- Batched last_access updates ----------------------------
        # Each ``get()`` would otherwise fire one UPDATE per hit; under
        # 4-worker decode contention that's a steady stream of SQLite
        # write transactions through the single ``_db_lock``, each
        # costing ~3-5 ms (lock + WAL append + fsync inside SQLite's
        # NORMAL sync mode). Batching: in-memory dict accumulates the
        # latest timestamp per key, the writer thread flushes the
        # batch every :attr:`_ACCESS_FLUSH_INTERVAL` seconds.
        # Cost trade-off: LRU eviction sees timestamps up to ~2 s
        # stale, which is fine — eviction is rare and the precision
        # isn't safety-critical.
        self._pending_access: dict[str, float] = {}
        self._pending_access_lock = threading.Lock()
        self._last_access_flush = time.monotonic()
        self._shutdown = threading.Event()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="DiskCacheWriter",
            daemon=True,
        )
        # In read-only mode the writer thread has nothing to do —
        # ``put`` short-circuits before enqueueing, and we don't want
        # the batched ``last_access`` flushes to compete with the
        # owning instance for the WAL. ``shutdown`` still works (the
        # thread checks ``self._shutdown`` so the join falls through).
        if not self._read_only:
            self._writer_thread.start()

        # Cached total size — updated atomically alongside SQLite
        # writes so callers don't pay an aggregate query per ``put``.
        # Initialised by scanning the DB once at startup.
        with self._db_lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(size_bytes), 0) FROM entries"
            ).fetchone()
        self._total_bytes = int(row[0]) if row else 0

        # ---- Runtime counters --------------------------------------
        # Simple int increments — GIL guarantees atomicity for the
        # ``+= 1`` patterns we use (no lock needed). Surfaced via
        # :meth:`stats` for the Preferences readout / debug telemetry.
        # Reset to 0 on process restart by design — these are
        # session-scoped runtime counters, not on-disk state.
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._evictions = 0
        self._errors = 0
        self._bytes_read = 0
        self._bytes_written = 0

        log.info(
            "DiskCache ready at %s (budget=%s, entries=%d, used=%d MB, "
            "compressor=%s)",
            self._cache_dir,
            "unlimited" if self._budget_bytes == 0 else f"{self._budget_bytes} B",
            self._entry_count_unlocked(),
            self._total_bytes // (1024 * 1024),
            "lz4" if _HAS_LZ4 else "zlib",
        )

    # ------------------------------------------------------------------ Public API

    def get(self, key: str) -> np.ndarray | None:
        """Synchronous read. Returns ``None`` on miss.

        On hit, the blob is decompressed (~3-8 ms at HD) and the
        entry's ``last_access`` timestamp is updated so LRU eviction
        keeps recently-used frames around. Hot path — gates a decode
        on miss, so latency matters.
        """
        if not key:
            return None
        with self._db_lock:
            row = self._db.execute(
                "SELECT blob_path FROM entries WHERE key = ?", (key,),
            ).fetchone()
        if row is None:
            self._misses += 1
            return None
        blob_path = self._cache_dir / row[0]
        try:
            blob = blob_path.read_bytes()
        except OSError as err:
            # File vanished between DB lookup and read (manual
            # ``rmdir``, antivirus quarantine, …). Sweep the orphaned
            # index entry so a future ``put`` doesn't believe the
            # entry already exists.
            log.warning(
                "DiskCache blob missing at %s (%s); dropping index entry",
                blob_path, err,
            )
            self._errors += 1
            self._misses += 1
            self._remove_internal(key)
            return None
        try:
            arr = _deserialize(blob)
        except Exception:  # pragma: no cover — only on corrupted files
            log.exception(
                "DiskCache failed to deserialize blob at %s; dropping",
                blob_path,
            )
            self._errors += 1
            self._misses += 1
            self._remove_internal(key)
            return None
        self._hits += 1
        self._bytes_read += len(blob)
        # Stash the access timestamp for batched flush by the writer
        # thread (see ``_ACCESS_FLUSH_INTERVAL``). The dict overwrite
        # collapses multiple reads of the same key in the window into
        # one UPDATE — bonus.
        with self._pending_access_lock:
            self._pending_access[key] = time.time()
        return arr

    def put(self, key: str, arr: np.ndarray) -> None:
        """Queue an array for async disk write.

        Returns immediately. The actual serialization + write happens
        on the writer thread. The caller **must not mutate ``arr``**
        after this call — the writer reads it directly.

        Drops the put silently when:
          * ``key`` is empty
          * The array is not 3-D (we only cache HxWxC frame buffers)
          * The disk cache has been shut down
          * The write queue is full (back-pressure — better to skip a
            disk-cache opportunity than block the eviction path)
        """
        if not key or arr is None:
            return
        if arr.ndim != 3:
            return
        if self._shutdown.is_set():
            return
        if self._read_only:
            # Another instance owns the lock; we can read its blobs
            # but must not write to avoid corrupting its index /
            # blob files. Drop silently — cache stays correct via the
            # natural cold-cache fallback (source decode).
            return
        try:
            self._write_queue.put_nowait((key, arr))
        except queue.Full:
            # Full queue == we're falling behind on writes. The frame
            # stays cached only in RAM and (probably) gets evicted
            # before we catch up. Acceptable; the disk cache is
            # opportunistic, not a guarantee.
            log.debug("DiskCache write queue full; dropping put for %s", key[:16])

    def contains_keys(self, keys) -> set[str]:  # type: ignore[no-untyped-def]
        """Bulk-existence query — return the subset of ``keys`` that
        are present in the index.

        One SQLite query (split into ~900-key chunks to stay under
        SQLite's default ``IN`` parameter limit) instead of N
        sequential ``get()`` calls. Used by
        :meth:`img_player.cache.master_frame_cache.MasterFrameCache.disk_available_master_frames`
        at session-load to pre-paint the timeline cache bar so the
        user sees "this shot is warm on disk" before they even scrub.

        Does NOT touch ``last_access`` (= bulk existence is not the
        same intent as an actual read). Eviction will treat unread
        but pre-detected entries as cold, which is the desired
        behaviour for a probe.
        """
        unique = list({k for k in keys if k})
        if not unique:
            return set()
        out: set[str] = set()
        # SQLite's ``SQLITE_MAX_VARIABLE_NUMBER`` is 999 by default —
        # chunk well under that so the prepared statement is accepted.
        CHUNK = 900
        with self._db_lock:
            for i in range(0, len(unique), CHUNK):
                chunk = unique[i:i + CHUNK]
                placeholders = ",".join(["?"] * len(chunk))
                rows = self._db.execute(
                    f"SELECT key FROM entries WHERE key IN ({placeholders})",
                    chunk,
                ).fetchall()
                out.update(r[0] for r in rows)
        return out

    def remove(self, key: str) -> None:
        """Synchronous delete. No-op if the entry doesn't exist."""
        if not key:
            return
        if self._read_only:
            # See :meth:`put` — another instance owns the cache,
            # we must not mutate the index.
            return
        self._remove_internal(key)

    def clear(self) -> int:
        """Wipe every entry. Returns the number of bytes freed.

        Synchronous (the user clicked "Clear cache" and is waiting for
        the disk to free up). The writer thread is paused via a
        flag-check inside the writer loop so a pending write doesn't
        re-create files mid-clear.

        Read-only mode no-ops: blowing away the index while another
        instance is writing to it is the worst kind of cache
        corruption. The user gets the "another instance" warning at
        boot, that's the right place to address it.
        """
        if self._read_only:
            log.warning(
                "DiskCache: clear() requested in read-only mode — ignored "
                "(another Flick instance owns this cache directory)"
            )
            return 0
        # Drain pending writes first so nothing gets re-created after
        # we've removed it.
        self._drain_pending_writes(timeout_s=1.0)
        freed = 0
        with self._db_lock:
            rows = self._db.execute(
                "SELECT blob_path, size_bytes FROM entries",
            ).fetchall()
            for rel, size in rows:
                blob_path = self._cache_dir / rel
                try:
                    blob_path.unlink(missing_ok=True)
                    freed += int(size)
                except OSError as err:
                    log.warning(
                        "DiskCache clear: failed to remove %s (%s)",
                        blob_path, err,
                    )
            self._db.execute("DELETE FROM entries")
            self._total_bytes = 0
        # Best-effort directory cleanup — the sharded sub-dirs left
        # empty after the unlinks above are pruned so a subsequent
        # ``ls`` doesn't show a fan of empty stubs. Errors ignored;
        # the tree is auto-recreated on the next put.
        self._prune_empty_dirs()
        log.info("DiskCache cleared %d bytes (%d entries)", freed, len(rows))
        return freed

    def size_bytes(self) -> int:
        """Current total disk usage (sum of all stored blob sizes)."""
        return self._total_bytes

    def entry_count(self) -> int:
        with self._db_lock:
            return self._entry_count_unlocked()

    def set_budget(self, budget_bytes: int) -> None:
        """Update the budget. ``0`` = unlimited.

        Triggers an immediate eviction round if the new budget is
        below current usage — frees disk on demand when the user
        shrinks the limit. Read-only no-ops the eviction (it would
        mutate) but still records the budget locally so the UI
        reflects the user's choice.
        """
        new = max(0, int(budget_bytes))
        if new == self._budget_bytes:
            return
        self._budget_bytes = new
        if self._read_only:
            return
        if new > 0 and self._total_bytes > new:
            self._evict_to_budget()

    def budget_bytes(self) -> int:
        return self._budget_bytes

    def cache_dir(self) -> Path:
        return self._cache_dir

    def stats(self) -> DiskCacheStats:
        """Snapshot of the runtime counters + on-disk metrics.

        Used by Preferences > Disk cache for the live readout. Cheap
        — counters are plain ints already in memory, the entry-count
        is a single SQLite COUNT(*) which the index hits in O(1).
        """
        return DiskCacheStats(
            hits=self._hits,
            misses=self._misses,
            writes=self._writes,
            evictions=self._evictions,
            errors=self._errors,
            bytes_read=self._bytes_read,
            bytes_written=self._bytes_written,
            entries=self.entry_count(),
            size_bytes=self._total_bytes,
            budget_bytes=self._budget_bytes,
            read_only=self._read_only,
        )

    def is_read_only(self) -> bool:
        """True iff another instance owns the cache lock. UI uses this."""
        return self._read_only

    def set_compress(self, compress: bool) -> None:
        """Toggle lz4 compression for **new** writes. Idempotent.

        Existing entries keep their on-disk format (v2 lz4 or v3 raw);
        the reader auto-routes via the blob's magic prefix so the
        switch is non-destructive — no need to clear the cache when
        the user changes their mind about the speed-vs-space trade.
        """
        self._compress = bool(compress)

    def pending_writes(self) -> int:
        """Approximate queue depth of frames waiting to be written.

        Lets callers (e.g. app-exit path) decide whether to show a
        "flushing disk cache" indicator. ``queue.qsize`` is documented
        as approximate but for our display purpose it's plenty.
        """
        try:
            return self._write_queue.qsize()
        except NotImplementedError:  # pragma: no cover — macOS edge
            return 0

    def shutdown(
        self,
        timeout_s: float = 10.0,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Stop the writer thread and close the DB.

        Called from :meth:`MasterFrameCache.shutdown` at app exit.
        Best-effort flush of pending writes within ``timeout_s``;
        anything still queued after the timeout is dropped silently
        (better than a hanging exit).

        The default 10 s budget is sized for a worst-case queue of
        ~150 frames at ~50 ms / blob — anything beyond that the user
        was probably aware they were churning the cache and a few
        dropped frames re-decode on next open. ``progress_callback``
        is invoked every ~100 ms during the drain with the current
        pending count so a UI can show a flushing indicator; called
        from the same thread that called :meth:`shutdown`.
        """
        if self._shutdown.is_set():
            return
        # Drain + join only matter when the writer thread is actually
        # running. In read-only mode the queue is always empty and
        # the thread was never started — calling .join() on a fresh
        # Thread instance raises RuntimeError.
        if not self._read_only:
            self._drain_pending_writes(
                timeout_s=timeout_s,
                progress_callback=progress_callback,
            )
            self._shutdown.set()
            try:
                self._write_queue.put_nowait(_SHUTDOWN_SENTINEL)
            except queue.Full:
                pass
            self._writer_thread.join(timeout=timeout_s)
        else:
            self._shutdown.set()
        with self._db_lock:
            try:
                self._db.close()
            except sqlite3.Error:
                pass
        # Release the cross-process lock by closing the file handle.
        # The OS auto-releases the byte-range lock on close; this also
        # lets a second instance acquire the cache after we exit.
        if self._lock_file is not None:
            try:
                self._lock_file.close()
            except OSError:
                pass
            self._lock_file = None

    # ------------------------------------------------------------------ Internals

    def _migrate_if_needed(self) -> None:
        """Auto-wipe the cache when the on-disk format version is behind.

        Reads ``PRAGMA user_version``; if below
        :data:`_CACHE_FORMAT_VERSION` AND the index actually has stale
        entries, drop all blobs + rows. New installs (version = 0,
        zero entries) are stamped silently to the current version.

        Called from :meth:`__init__` *before* the writer thread starts,
        so there's no need to drain a queue — we can mutate the DB
        and unlink blobs directly under the existing ``_db_lock``.
        """
        with self._db_lock:
            row = self._db.execute("PRAGMA user_version").fetchone()
            current_version = int(row[0]) if row else 0
            if current_version >= _CACHE_FORMAT_VERSION:
                return
            entry_count = self._entry_count_unlocked()

            if entry_count == 0:
                # Fresh DB (or one previously cleared by the user).
                # Just stamp the current version and move on; no log
                # noise on first-launch installs.
                self._db.execute(
                    f"PRAGMA user_version = {_CACHE_FORMAT_VERSION}"
                )
                return

            # Real migration: pre-existing entries from an older
            # format. Wipe blobs + rows in one transaction equivalent
            # (PRAGMA + autocommit). The list of blob_paths is read
            # before the DELETE so we still know which files to unlink.
            rows = self._db.execute(
                "SELECT blob_path, size_bytes FROM entries",
            ).fetchall()
            freed = 0
            for rel, size in rows:
                blob_path = self._cache_dir / rel
                try:
                    blob_path.unlink(missing_ok=True)
                    freed += int(size)
                except OSError as err:
                    log.warning(
                        "DiskCache migration: failed to remove %s (%s)",
                        blob_path, err,
                    )
            self._db.execute("DELETE FROM entries")
            self._db.execute(
                f"PRAGMA user_version = {_CACHE_FORMAT_VERSION}"
            )
        # Best-effort: drop the now-empty shard sub-dirs so a fresh
        # ``ls`` shows a tidy root. Done outside the DB lock; harmless
        # if it races with anything (the dirs will just be recreated
        # by the next put).
        try:
            self._prune_empty_dirs()
        except Exception:  # pragma: no cover — defensive
            log.exception("DiskCache migration: dir cleanup failed")
        log.info(
            "DiskCache migrated v%d → v%d (wiped %d entries, %d MB)",
            current_version,
            _CACHE_FORMAT_VERSION,
            entry_count,
            freed // (1024 * 1024),
        )

    def _sweep_orphans(self) -> None:
        """Delete ``.bin`` files in the cache tree not referenced by SQLite.

        Called from :meth:`__init__` before the writer thread starts.
        On a typical 50 GB cache (≈2000 4K frames) the walk takes <1s
        on SSD; for HD-only caches with 20k entries we've measured
        ~2-3 s — still well below the user's perceptual budget for an
        app launch. If it ever becomes a hot spot we can move the
        scan to a background thread, but for now the simplicity wins.

        Errors during unlink are logged and ignored — a leftover
        orphan is harmless (it just wastes disk space until the next
        sweep) so we never let a permission glitch take down the boot.
        """
        t0 = time.monotonic()
        # Build the reference set from SQLite — relative paths exactly
        # as stored in ``entries.blob_path`` (forward-slash POSIX
        # form). We normalise the on-disk paths to match.
        with self._db_lock:
            rows = self._db.execute("SELECT blob_path FROM entries").fetchall()
        known = {row[0] for row in rows}

        # Walk the tree. ``os.walk`` is a touch faster than Path.rglob
        # for our 2-level shard layout and avoids constructing Path
        # objects we'd immediately string-ify.
        removed = 0
        freed = 0
        scanned = 0
        for dirpath, _dirnames, filenames in os.walk(self._cache_dir):
            for name in filenames:
                if not name.endswith(".bin"):
                    continue
                scanned += 1
                full = os.path.join(dirpath, name)
                # Recompute the same relative form that
                # ``_blob_path_for`` produces — POSIX-style separators.
                rel = os.path.relpath(full, self._cache_dir).replace(os.sep, "/")
                if rel in known:
                    continue
                try:
                    size = os.path.getsize(full)
                    os.remove(full)
                    removed += 1
                    freed += size
                except OSError as err:
                    log.warning(
                        "DiskCache sweep: failed to remove orphan %s (%s)",
                        full, err,
                    )

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        if removed > 0:
            log.info(
                "DiskCache swept %d orphan blob(s) (%d MB) "
                "out of %d scanned in %d ms",
                removed,
                freed // (1024 * 1024),
                scanned,
                elapsed_ms,
            )
            # Clean up shard sub-dirs that became empty as a result.
            try:
                self._prune_empty_dirs()
            except Exception:  # pragma: no cover — defensive
                log.exception("DiskCache sweep: dir cleanup failed")
        elif scanned > 0:
            # Quiet success path — debug only so a clean launch
            # doesn't spam INFO.
            log.debug(
                "DiskCache sweep: %d blobs scanned, no orphans (%d ms)",
                scanned, elapsed_ms,
            )

    def _entry_count_unlocked(self) -> int:
        row = self._db.execute("SELECT COUNT(*) FROM entries").fetchone()
        return int(row[0]) if row else 0

    def _blob_path_for(self, key: str) -> Path:
        """Sharded relative path inside ``cache_dir``. ``ab/cd1234.../hash.bin``."""
        return Path(key[:2]) / key[2:4] / f"{key}.bin"

    def _writer_loop(self) -> None:
        """Worker thread loop — drain the queue, serialize, write.

        Also responsible for periodically flushing the batched
        ``last_access`` updates (see :attr:`_pending_access`). The
        flush check happens between queue drains so a steady write
        stream doesn't starve the access-update flush.
        """
        while not self._shutdown.is_set():
            try:
                task = self._write_queue.get(timeout=0.5)
            except queue.Empty:
                self._maybe_flush_access()
                continue
            if task is _SHUTDOWN_SENTINEL:
                break
            try:
                key, arr = task
                self._write_one(key, arr)
            except Exception:  # pragma: no cover — defensive
                log.exception("DiskCache writer failed on task")
            self._maybe_flush_access()
        # Final flush on shutdown so a pending batch doesn't get lost
        # when the user closes the app right after a scrub session.
        self._maybe_flush_access(force=True)

    def _maybe_flush_access(self, force: bool = False) -> None:
        """Flush batched ``last_access`` updates if the interval has
        elapsed, or unconditionally when ``force`` (= shutdown)."""
        if not force:
            now = time.monotonic()
            if now - self._last_access_flush < self._ACCESS_FLUSH_INTERVAL:
                return
            self._last_access_flush = now
        with self._pending_access_lock:
            updates = list(self._pending_access.items())
            self._pending_access.clear()
        if not updates:
            return
        try:
            with self._db_lock:
                # ``executemany`` runs a single transaction with N
                # parameterised UPDATEs — much cheaper than N
                # individual transactions, and avoids per-statement
                # WAL append overhead.
                self._db.executemany(
                    "UPDATE entries SET last_access = ? WHERE key = ?",
                    [(t, k) for k, t in updates],
                )
        except sqlite3.Error:
            log.exception("DiskCache batched access flush failed (non-fatal)")

    def _write_one(self, key: str, arr: np.ndarray) -> None:
        """Serialize + write a single (key, ndarray) → disk + update DB."""
        if self._shutdown.is_set():
            return
        rel = self._blob_path_for(key)
        abs_path = self._cache_dir / rel
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            blob = _serialize(arr, compress=self._compress)
        except Exception:  # pragma: no cover — defensive
            log.exception("DiskCache serialize failed for key %s", key[:16])
            return
        # Write to a sibling tmp file then rename — atomic-ish on
        # Windows, atomic on POSIX. Avoids corrupt half-written blobs
        # on a crash or power-cut during write.
        tmp_path = abs_path.with_suffix(abs_path.suffix + ".tmp")
        try:
            tmp_path.write_bytes(blob)
            tmp_path.replace(abs_path)
        except OSError as err:
            log.warning("DiskCache write failed for %s: %s", abs_path, err)
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            return
        now = time.time()
        size = len(blob)
        with self._db_lock:
            # UPSERT — replace any existing entry so the disk usage
            # tracker doesn't double-count when the same key is
            # written twice (rare but possible: same signature
            # decoded by two threads concurrently).
            existing = self._db.execute(
                "SELECT size_bytes FROM entries WHERE key = ?", (key,),
            ).fetchone()
            if existing is not None:
                self._total_bytes -= int(existing[0])
            self._db.execute(
                """
                INSERT INTO entries(key, blob_path, size_bytes,
                                    last_access, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    blob_path = excluded.blob_path,
                    size_bytes = excluded.size_bytes,
                    last_access = excluded.last_access
                """,
                (key, str(rel).replace("\\", "/"), size, now, now),
            )
            self._total_bytes += size
        self._writes += 1
        self._bytes_written += size
        # Eviction outside the lock — _evict_to_budget acquires it
        # internally per batch to keep critical sections short.
        if self._budget_bytes > 0 and self._total_bytes > self._budget_bytes:
            self._evict_to_budget()

    def _remove_internal(self, key: str) -> None:
        with self._db_lock:
            row = self._db.execute(
                "SELECT blob_path, size_bytes FROM entries WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return
            blob_path = self._cache_dir / row[0]
            size = int(row[1])
            self._db.execute("DELETE FROM entries WHERE key = ?", (key,))
            self._total_bytes -= size
        try:
            blob_path.unlink(missing_ok=True)
        except OSError as err:
            log.debug("DiskCache unlink failed for %s: %s", blob_path, err)

    def _evict_to_budget(self) -> None:
        """LRU-trim down to ``_EVICT_TARGET_RATIO`` of the budget."""
        if self._budget_bytes <= 0:
            return
        target = int(self._budget_bytes * self._EVICT_TARGET_RATIO)
        if self._total_bytes <= target:
            return
        # Pull the oldest entries one batch at a time so we don't load
        # the full table into memory on a 50 GB cache.
        batch_size = 64
        evicted = 0
        bytes_freed = 0
        while self._total_bytes > target:
            with self._db_lock:
                rows = self._db.execute(
                    "SELECT key FROM entries "
                    "ORDER BY last_access ASC LIMIT ?",
                    (batch_size,),
                ).fetchall()
            if not rows:
                break
            for (key,) in rows:
                size_before = self._total_bytes
                self._remove_internal(key)
                evicted += 1
                bytes_freed += size_before - self._total_bytes
                if self._total_bytes <= target:
                    break
        if evicted:
            self._evictions += evicted
            log.info(
                "DiskCache evicted %d entries (%d MB) to fit budget",
                evicted, bytes_freed // (1024 * 1024),
            )

    def _drain_pending_writes(
        self,
        timeout_s: float,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Block until the write queue is empty or ``timeout_s`` elapses.

        If ``progress_callback`` is supplied it's fired every ~100 ms
        with the current queue depth — the UI can use that to show a
        flushing indicator that ticks down to zero.
        """
        deadline = time.monotonic() + timeout_s
        last_cb = 0.0
        while time.monotonic() < deadline:
            if progress_callback is not None:
                now = time.monotonic()
                if now - last_cb >= 0.1:
                    try:
                        progress_callback(self._write_queue.qsize())
                    except Exception:  # pragma: no cover — UI cb should never throw
                        log.exception("disk-cache drain progress callback failed")
                    last_cb = now
            if self._write_queue.empty():
                # The queue can become empty while the writer is mid-
                # task; give the writer thread a moment to finish.
                time.sleep(0.05)
                if self._write_queue.empty():
                    if progress_callback is not None:
                        try:
                            progress_callback(0)
                        except Exception:  # pragma: no cover
                            pass
                    return
            else:
                time.sleep(0.05)

    def _prune_empty_dirs(self) -> None:
        """Remove empty shard sub-dirs left over after a clear. Best-effort."""
        try:
            for sub in self._cache_dir.iterdir():
                if not sub.is_dir() or sub.name in (".",):
                    continue
                # Two levels deep (`ab/cd/`).
                for sub2 in list(sub.iterdir()):
                    if sub2.is_dir():
                        try:
                            sub2.rmdir()
                        except OSError:
                            pass
                try:
                    sub.rmdir()
                except OSError:
                    pass
        except OSError:
            pass
