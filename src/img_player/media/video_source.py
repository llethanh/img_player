"""Synchronous PyAV video decoder with seek-then-decode-forward strategy.

A :class:`VideoSource` wraps an ``av.container.Container`` for one
video file and answers ``frame_at_time(t) -> ndarray`` queries. It is
intentionally **synchronous**: a higher-level component (the future
``VideoDecoderRunner``) will hold one of these on a worker thread and
push results to the cache. Keeping the decoder itself synchronous
makes seek semantics testable without QThread infrastructure.

Strategy:

* **Forward decode (cheap)**: when the requested time is at or after
  the last decoded frame, just keep pulling packets — long-GOP H.264
  / H.265 streams are decode-forward friendly, this is the hot path.
* **Backward seek (costly)**: when the requested time is before the
  last decoded frame, ``container.seek()`` to the requested PTS
  (FFmpeg lands at the keyframe at or before, "backward" mode), then
  decode-forward to find the frame whose PTS bracket the target.

A single-frame cache (last decoded frame + its PTS) makes scrubbing
within a frame's display duration a no-op and stops the decoder from
re-decoding the same frame on every paint event.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from fractions import Fraction
from pathlib import Path

import numpy as np

from img_player.media.video_probe import VideoMetadata, probe_video

log = logging.getLogger(__name__)


# Default RAM budget for the per-source frame cache. Tunable via
# ``Preferences.video_cache_budget_mb``.
#
# Cached frames are stored as **float32 RGBA** rather than uint8 — that's
# 4× heavier per frame but lets cache hits skip the
# ``astype(float32) * (1/255)`` pass in ``decode_at`` (= 23 ms on a
# 1440p frame, completely dominating any cache-hit savings if the cast
# stayed downstream). With float32 cache:
#   1440p frame = 2560×1440×4×4 = 57.6 MB   →  ~70 frames in 4 GB
#   1080p frame = 1920×1080×4×4 = 31.6 MB   →  ~128 frames in 4 GB
#    720p frame =  1280×720×4×4 = 14.0 MB   →  ~280 frames in 4 GB
# Most VFX dailies are 30-90 s of 720p/1080p — fits end-to-end after
# the prefetch worker finishes its sweep.
DEFAULT_VIDEO_CACHE_BUDGET_BYTES: int = 4 * 1024 * 1024 * 1024


class VideoSource:
    """Open a video file once, answer ``frame_at_time(t)`` queries.

    Not thread-safe — one instance per worker thread. ``close()`` (or
    the context manager protocol) releases the container; calling
    ``frame_at_time`` after close raises.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        cache_budget_bytes: int = DEFAULT_VIDEO_CACHE_BUDGET_BYTES,
        prefetch: bool = True,
    ) -> None:
        self._path = Path(path)
        self._meta = probe_video(self._path)
        if not self._meta.has_video:
            raise ValueError(f"No video stream in {self._path}")
        # --- RAM frame cache --------------------------------------
        # OpenRV-style: every successfully decoded frame goes into an
        # LRU cache keyed by integer frame index. Re-visits (= scrub
        # backward, loop playback, A/B compare) hit cache and skip
        # the seek-then-decode-forward cost entirely. Bounded by a
        # configurable budget; oldest frames get evicted first.
        #
        # Thread-safety: holds an RLock so a future prefetch worker
        # can write to the cache while the foreground decoder reads.
        self._frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._frame_cache_bytes: int = 0
        self._frame_cache_budget: int = max(0, int(cache_budget_bytes))
        self._cache_lock = threading.RLock()

        # Imported lazily so module import is light — same pattern as
        # in ``video_probe`` for headless / non-video code paths.
        import av  # type: ignore[import-untyped]

        # ``metadata_errors='replace'`` mirrors the probe path —
        # QuickTime files with latin1-encoded tags would otherwise
        # raise UnicodeDecodeError on open.
        self._container = av.open(str(self._path), metadata_errors="replace")
        # Pick the first (= primary) video stream. Multi-track files
        # are rare in VFX delivery; if needed later, expose a stream
        # index parameter.
        self._stream = self._container.streams.video[0]
        # ``thread_type="AUTO"`` lets FFmpeg pick FRAME/SLICE threading
        # per codec — H.264 typically defaults to slice, ProRes to frame.
        # Either is well-suited to a player workload (low latency on
        # individual decodes, cores stay busy on long-GOP streams).
        self._stream.thread_type = "AUTO"

        # Cache of the most recently decoded frame: (pts_seconds,
        # ndarray). Holding one frame is enough — repeated paints of
        # the same time hit this cache, while a forward step always
        # advances past it. Bigger caches (an LRU) duplicate what the
        # MasterFrameCache already provides at the layer level.
        self._last_pts: float | None = None
        self._last_frame: np.ndarray | None = None
        # Fast-seek mode is toggled on by the app while the user is
        # actively scrubbing the timeline. In that window we land on
        # the nearest keyframe ≤ ``t`` and **return it directly**
        # without decode-forwarding to bracket the exact target —
        # 1-3 ms per seek instead of 5-15 ms. On scrub release the
        # app turns this off and re-issues a precise request so the
        # final landing frame is exact.
        self._fast_seek: bool = False
        # Persistent decoder generator — PyAV's ``container.decode()``
        # returns a generator that pulls packets from the demuxer; once
        # exhausted (EOF or interrupted via ``break``/``return``), we
        # need a fresh one. Recreated on seek and on EOF; survives
        # across ``frame_at_time`` calls so a sequence of forward queries
        # decodes through the stream without redundant seeks.
        self._decoder = self._container.decode(self._stream)

        # --- Background prefetch worker --------------------------
        # OpenRV-style behaviour: as soon as the file opens, a
        # dedicated thread decodes forward from t=0 and pushes every
        # frame into the cache. The user sees the cache fill up
        # "all at once" (~6× real-time for AV1 at 1440p) — by the
        # time they hit Play, the first few seconds are RAM-hot and
        # playback / scrubs feel instant.
        #
        # The worker uses its OWN PyAV container so it doesn't
        # contend with the foreground ``self._container``. PyAV
        # is not thread-safe per-container, but separate
        # containers reading the same file is the standard pattern
        # (it's how FFmpeg handles concurrent readers too).
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_stop = threading.Event()
        self._prefetch_max_cached_idx: int = -1
        if prefetch and self._frame_cache_budget > 0:
            self._prefetch_thread = threading.Thread(
                target=self._prefetch_loop,
                name=f"VideoPrefetch[{self._path.name}]",
                daemon=True,
            )
            self._prefetch_thread.start()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> VideoMetadata:
        return self._meta

    @property
    def fps(self) -> Fraction:
        # The probe always populates ``fps`` for files that have a
        # video stream, but the type system can't see that — assert
        # at the access point so the failure mode is obvious if a
        # truly weird container slips through.
        assert self._meta.fps is not None, "VideoSource requires a known fps"
        return self._meta.fps

    @property
    def duration_seconds(self) -> float:
        # Same logic as ``fps``: probe + metadata invariant on open.
        if self._meta.duration_seconds is not None:
            return self._meta.duration_seconds
        # Fallback: derive from frame_count and fps when the container
        # didn't report a clean duration.
        if self._meta.frame_count is not None and self._meta.fps is not None:
            return self._meta.frame_count / float(self._meta.fps)
        raise RuntimeError(f"Cannot determine duration of {self._path}")

    @property
    def width(self) -> int:
        assert self._meta.width is not None
        return self._meta.width

    @property
    def height(self) -> int:
        assert self._meta.height is not None
        return self._meta.height

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    @staticmethod
    def _frame_to_rgba_f32(frame) -> np.ndarray:  # type: ignore[no-untyped-def]
        """PyAV frame → (H, W, 4) float32 normalised [0,1].

        The float32 conversion is rolled in here so cached frames
        are display-ready: a cache hit returns immediately without
        the ~23 ms ``astype(float32) * (1/255)`` pass that used to
        live in :func:`video_renderer.decode_at`. Trade-off is 4×
        RAM per cached frame — see the budget defaults in this
        module's preamble for the math.
        """
        rgba_u8 = frame.to_ndarray(format="rgba")
        return rgba_u8.astype(np.float32, copy=False) * (1.0 / 255.0)

    def frame_at_time(self, t_seconds: float) -> np.ndarray:
        """Return the frame whose presentation interval contains ``t``.

        Output is an ``(H, W, 4)`` ``float32`` RGBA ndarray with
        values in [0, 1] — display-ready, no further conversion in
        ``decode_at``. The OCIO input transform, when wired, will
        operate on this same array shape and dtype.

        Clamps ``t`` to ``[0, duration)``: requesting before-start
        returns the first frame, after-end returns the last decoded
        frame on file (matches the "freeze on tail" behaviour of
        common players).
        """
        if self._container is None:
            raise RuntimeError("VideoSource is closed")

        # Clamp to the valid range so ``seek`` doesn't bounce off the
        # end and the caller doesn't have to special-case boundaries.
        t = max(0.0, t_seconds)
        if self._meta.duration_seconds is not None:
            # Stay strictly less than duration so the last frame's PTS
            # (≤ duration - 1/fps) is reachable.
            t = min(t, max(0.0, self._meta.duration_seconds - 1e-9))

        # --- RAM cache lookup ------------------------------------
        # Compute the target frame index and check the LRU cache
        # before doing any seek / decode work. A hit is ~free
        # (dict lookup + ndarray return), a miss costs only a
        # second-of-microseconds extra over the legacy single-frame
        # cache. The cache fills as the user plays through the
        # video; subsequent re-passes (loop, scrub-back, A/B
        # compare) become instant.
        target_idx = int(round(t * float(self.fps)))
        cached = self._cache_get(target_idx)
        if cached is not None:
            # Keep the legacy single-frame cache in sync so other
            # code paths that read ``_last_frame`` still work.
            self._last_pts = target_idx / float(self.fps)
            self._last_frame = cached
            return cached

        # Cache hit: same display interval as the last decoded frame.
        # The "interval" check uses 1/fps as the upper bound — within
        # that window, the same frame is the correct answer.
        if self._last_frame is not None and self._last_pts is not None:
            interval = 1.0 / float(self.fps)
            if self._last_pts <= t < self._last_pts + interval:
                return self._last_frame

        interval = 1.0 / float(self.fps)

        # Fast-seek (scrub) path — runs **before** the
        # forward/backward asymmetry check. Otherwise a forward drag
        # falls into the natural decode-forward loop and pays the
        # full 5-15 ms per frame, while backward goes through the
        # cheap seek+keyframe path: scrub forward feels twice as
        # laggy as scrub backward. Seeking unconditionally in fast
        # mode keeps the cost uniform (~1-3 ms) in both directions.
        # Visual fidelity: keyframe-resolution scrub — within a GOP
        # the displayed frame stays the same I-frame until the user
        # crosses the next keyframe. Trade-off accepted; release
        # fires a precise re-decode at the final frame.
        if self._fast_seek:
            self._seek_to(t)
            try:
                frame = next(self._decoder)
            except StopIteration:
                if self._last_frame is not None:
                    return self._last_frame
                raise RuntimeError(
                    f"No decodable frame found at t={t} in {self._path}"
                ) from None
            pts = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None
                else 0.0
            )
            target_frame = self._frame_to_rgba_f32(frame)
            self._last_pts = pts
            self._last_frame = target_frame
            self._cache_put(int(round(pts * float(self.fps))), target_frame)
            return target_frame

        # Precise path: only seek when the request lands before the
        # current decode position. Forward steps stay on the cheap
        # decode-forward generator. Tolerance ½ a frame interval so
        # float jitter in PTS arithmetic doesn't trigger spurious
        # seeks.
        needs_seek = (
            self._last_pts is None
            or t < self._last_pts - 0.5 * interval
        )
        if needs_seek:
            self._seek_to(t)

        # Decode-forward until we land on a frame whose PTS interval
        # contains ``t``, or we run out of frames. The ``_decoder``
        # generator persists across calls so consecutive forward
        # queries don't re-seek; it is only recreated on seek (above)
        # or after EOF (below).
        retried = False
        while True:
            try:
                frame = next(self._decoder)
            except StopIteration:
                # Generator exhausted — end of stream. Return the last
                # frame we have on file (freeze-on-tail), or, if we've
                # truly never decoded anything for this source, raise.
                if self._last_frame is not None:
                    return self._last_frame
                raise RuntimeError(
                    f"No decodable frame found at t={t} in {self._path}"
                ) from None
            except Exception as exc:  # PyAV EOFError, codec hiccups
                # PyAV occasionally raises ``av.error.EOFError`` on the
                # final packet of a short clip — treat as end of
                # stream, same recovery as StopIteration above. We
                # prefer the typed isinstance check (robust to PyAV
                # version bumps) and keep a string-match fallback for
                # exotic builds where the class hierarchy changed.
                is_eof = False
                try:
                    import av.error  # noqa: PLC0415 — lazy, optional
                    is_eof = isinstance(exc, av.error.EOFError)
                except (ImportError, AttributeError):  # pragma: no cover
                    pass
                if not is_eof:
                    is_eof = (
                        "End of file" in str(exc) or "EOF" in type(exc).__name__
                    )
                if is_eof:
                    if self._last_frame is not None:
                        return self._last_frame
                    raise RuntimeError(
                        f"No decodable frame found at t={t} in {self._path}"
                    ) from exc
                raise

            pts = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None
                else 0.0
            )
            if pts <= t < pts + interval:
                target_frame = self._frame_to_rgba_f32(frame)
                self._last_pts = pts
                self._last_frame = target_frame
                self._cache_put(int(round(pts * float(self.fps))), target_frame)
                return target_frame
            if pts > t + interval:
                # Overshot — the right frame was before this one and
                # we've already consumed it. Re-seek once and try
                # again. ``retried`` guards against infinite loops if
                # the seek lands past ``t`` (can happen when ``t`` is
                # before the first keyframe).
                if retried:
                    target_frame = self._frame_to_rgba_f32(frame)
                    self._last_pts = pts
                    self._last_frame = target_frame
                    self._cache_put(int(round(pts * float(self.fps))), target_frame)
                    return target_frame
                self._seek_to(t)
                retried = True
                continue
            # pts < t — keep decoding forward until we bracket the
            # target. The frame we're about to discard cost us the
            # decode work already; lazy-convert + cache it so a
            # later forward step (= playback) finds it ready.
            passing_idx = int(round(pts * float(self.fps)))
            with self._cache_lock:
                if passing_idx not in self._frame_cache:
                    # Only convert + store if it isn't already there
                    # — avoids re-allocating on a re-traversed range.
                    passing_arr = self._frame_to_rgba_f32(frame)
                    self._cache_put(passing_idx, passing_arr)

    def set_fast_seek(self, enabled: bool) -> None:
        """Toggle the keyframe-only scrub shortcut. See ``_fast_seek``
        in ``__init__`` for the tradeoff. Cheap setter — no decode
        side-effects; just flips the flag the next ``frame_at_time``
        reads."""
        self._fast_seek = bool(enabled)

    def _seek_to(self, t_seconds: float) -> None:
        """Seek the demuxer to the keyframe at or before ``t``.

        Uses ``container.seek`` rather than ``stream.seek`` because
        the container-level call respects the ``backward`` flag
        across all streams (audio + video stay aligned), while
        per-stream seek can leave audio dangling.
        """
        # ``container.seek`` takes microseconds in AV_TIME_BASE units.
        target_us = int(t_seconds * 1_000_000)
        # ``backward=True`` is the keyframe-at-or-before semantic.
        # ``any_frame=False`` forces it to a keyframe (otherwise the
        # decoder would have to re-derive the keyframe boundary).
        self._container.seek(target_us, backward=True, any_frame=False)
        # ``seek`` flushes the decoder, but the per-frame state we
        # cache (last_pts) becomes stale until the next decode lands
        # — invalidate it so the bracketing logic above doesn't
        # short-circuit on a now-incorrect cache.
        self._last_pts = None
        self._last_frame = None
        # Recreate the persistent decoder generator — the previous one
        # is bound to the demuxer state pre-seek and would yield
        # stale packets (or raise) post-seek.
        self._decoder = self._container.decode(self._stream)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        # Stop the prefetch worker BEFORE tearing down the container —
        # the worker has its own container, but we still want a clean
        # exit so the thread joins quickly and doesn't trail past the
        # close.
        self._prefetch_stop.set()
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            # Daemon thread so process exit doesn't hang on us, but
            # join briefly so resources release deterministically.
            self._prefetch_thread.join(timeout=1.0)
        self._prefetch_thread = None

        if self._container is not None:
            self._container.close()
            self._container = None  # type: ignore[assignment]
            self._last_frame = None
            self._last_pts = None
            # Drop the RAM cache too — closing typically means we're
            # about to swap layers or shut down. Holding onto a
            # couple of GB of decoded frames for a video the user
            # no longer cares about would defeat the rest of the
            # cache budget.
            with self._cache_lock:
                self._frame_cache.clear()
                self._frame_cache_bytes = 0
                self._prefetch_max_cached_idx = -1

    # ------------------------------------------------------------------
    # Background prefetch
    # ------------------------------------------------------------------

    def _prefetch_loop(self) -> None:
        """Sequentially decode frames into the cache.

        Runs in its own thread with its own PyAV container so the
        foreground decoder isn't blocked. Stops on:

        * Stop event set (``close()`` / ``shutdown``)
        * EOF reached on the prefetch stream
        * Cache budget exhausted — at that point we exit and let the
          foreground decoder manage what stays cached (= LRU
          eviction follows real usage, not prefetch order)
        """
        import av  # noqa: PLC0415

        try:
            container = av.open(str(self._path), metadata_errors="replace")
        except Exception:  # noqa: BLE001
            log.debug(
                "[video-prefetch] failed to open second container for %s",
                self._path,
            )
            return

        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            fps = float(self.fps)
            for frame in container.decode(stream):
                if self._prefetch_stop.is_set():
                    return
                pts = (
                    float(frame.pts * frame.time_base)
                    if frame.pts is not None
                    else 0.0
                )
                idx = int(round(pts * fps))
                # Skip frames already cached (e.g. foreground decoder
                # already pulled them) — saves the YUV → RGBA
                # conversion cost.
                with self._cache_lock:
                    if idx in self._frame_cache:
                        if idx > self._prefetch_max_cached_idx:
                            self._prefetch_max_cached_idx = idx
                        continue
                arr = self._frame_to_rgba_f32(frame)
                self._cache_put(idx, arr)
                with self._cache_lock:
                    if idx > self._prefetch_max_cached_idx:
                        self._prefetch_max_cached_idx = idx
                    # Stop prefetching once we'd be evicting our own
                    # work. Cheap: integer compare against the
                    # current total. The foreground decoder takes
                    # over once the user crosses the prefetch
                    # frontier.
                    near_budget = (
                        self._frame_cache_bytes
                        > self._frame_cache_budget * 0.95
                    )
                if near_budget:
                    log.debug(
                        "[video-prefetch] %s budget filled at frame %d",
                        self._path.name, idx,
                    )
                    return
        except Exception:  # noqa: BLE001
            # Bad packet / codec hiccup / EOF — done.
            log.debug(
                "[video-prefetch] %s ended with exception",
                self._path.name, exc_info=True,
            )
        finally:
            try:
                container.close()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------
    # RAM frame cache
    # ------------------------------------------------------------------

    def _cache_get(self, idx: int) -> np.ndarray | None:
        """LRU-bump + lookup. Returns ``None`` on miss."""
        if self._frame_cache_budget <= 0:
            return None
        with self._cache_lock:
            arr = self._frame_cache.get(idx)
            if arr is not None:
                # Move to end = most-recently used.
                self._frame_cache.move_to_end(idx)
            return arr

    def _cache_put(self, idx: int, arr: np.ndarray) -> None:
        """Store ``arr`` for ``idx``. Idempotent (re-puts bump LRU
        and don't double-count budget). Evicts the oldest entries
        when the new total would exceed budget."""
        if self._frame_cache_budget <= 0 or arr is None:
            return
        nbytes = int(arr.nbytes)
        with self._cache_lock:
            existing = self._frame_cache.get(idx)
            if existing is not None:
                # Re-put with same data — just bump LRU position.
                self._frame_cache.move_to_end(idx)
                return
            self._frame_cache[idx] = arr
            self._frame_cache_bytes += nbytes
            # Evict from the front (= least-recently used) until
            # we're back under budget.
            while (
                self._frame_cache_bytes > self._frame_cache_budget
                and len(self._frame_cache) > 1
            ):
                _, dropped = self._frame_cache.popitem(last=False)
                self._frame_cache_bytes -= int(dropped.nbytes)

    def cache_stats(self) -> dict[str, int]:
        """``{frames, bytes, budget, max_cached_idx, contiguous_to}``
        — for UI cache-fill bars / debug. ``max_cached_idx`` is the
        highest frame index the prefetch worker has reached;
        ``contiguous_to`` is the highest frame index for which every
        index from 0 onward is in the cache (= the bar's blue-tip
        position the user sees in OpenRV-style timelines). Cheap,
        doesn't fault any work."""
        with self._cache_lock:
            # Contiguous-from-zero is straightforward when prefetch
            # has been running sequentially: scan forward from 0
            # until we hit a gap. Bounded by max_cached_idx so we
            # never walk further than necessary.
            contig = -1
            cache = self._frame_cache
            for i in range(0, self._prefetch_max_cached_idx + 1):
                if i in cache:
                    contig = i
                else:
                    break
            return {
                "frames": len(cache),
                "bytes": self._frame_cache_bytes,
                "budget": self._frame_cache_budget,
                "max_cached_idx": self._prefetch_max_cached_idx,
                "contiguous_to": contig,
            }

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Defensive — explicit ``close`` is preferred (the GC timing
        # of FFmpeg containers can hold file handles longer than
        # expected on Windows).
        try:
            self.close()
        except Exception:
            pass
