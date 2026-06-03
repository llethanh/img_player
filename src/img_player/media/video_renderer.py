"""Owns open :class:`VideoSource` decoders keyed by layer id.

The renderer-side counterpart to ``Layer.from_video``: when a video
layer is added to the stack the manager opens its decoder lazily on
first frame access; when the layer is removed the manager closes the
file handle so we don't leak across session loads.

Decoding runs on **per-layer worker threads** wrapped in
:class:`_ThreadedDecoder`. The main / Qt thread sends frame requests
through a small command queue and either gets a cached result
immediately (when the worker has prefetched the upcoming frame —
the play-time hot path) or waits briefly on a sync decode (scrub /
seek). VideoSource is intentionally single-threaded; serialising
every access through the worker means we never have two threads
fighting the decoder generator.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from img_player.media.video_source import VideoSource

log = logging.getLogger(__name__)


# How many frames ahead of the current target the worker tries to
# keep decoded in the cache. 2 is enough: at 24 fps the worker has
# ~83 ms to produce the next frame, far more than typical H.264
# decode-forward (5–15 ms). Bigger windows just bloat memory.
_PREFETCH_WINDOW = 2

# Maximum cached frames. The worker prunes once this is exceeded —
# a ring centred on the playhead so backward scrubs within this
# window stay free of seek overhead. 32 keeps roughly a second of
# 30 fps content live, which covers most "drag back to compare"
# motions during review without re-seeking. At HD that's ~200 MB
# per video layer (32 × 6 MB); 4K caps to ~770 MB — still well
# under the main image-sequence cache budget.
_CACHE_CAPACITY = 32


class _ThreadedDecoder:
    """Wrap a :class:`VideoSource` on a dedicated worker thread.

    Public surface is one method — ``get(t_seconds)`` — used by the
    manager. Internally, the worker continuously prefetches the next
    frames ahead of the most-recently-requested time; ``get`` returns
    the cached result instantly when available and falls back to a
    blocking sync request when the worker has nothing yet (first
    frame after a fresh open / large seek).

    Not thread-safe to share across multiple managers — one decoder
    per layer.
    """

    def __init__(self, path: Path) -> None:
        self._source = VideoSource(path)
        self._lock = threading.Lock()
        # Keyed by ``round(t * fps)`` so float jitter doesn't miss
        # otherwise-identical hits. The worker writes; ``get`` reads.
        self._cache: dict[int, np.ndarray] = {}
        # Most-recently-requested frame index — drives prefetch direction.
        self._target_idx: int | None = None
        # Pending sync request from the main thread when the cache
        # missed. Worker services these before any prefetch work so
        # the GUI never waits behind speculative decodes.
        self._sync_request_t: float | None = None
        self._sync_result: np.ndarray | None = None
        self._sync_done = threading.Event()
        # Wake the worker when there's new work (a sync request, a
        # new target, or shutdown).
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name=f"video-decoder-{path.name}", daemon=True,
        )
        self._thread.start()

    @property
    def fps(self) -> float:
        return float(self._source.fps)

    def set_fast_seek(self, enabled: bool) -> None:
        """Toggle the keyframe-only scrub mode + drop the frame cache.

        We clear ``self._cache`` on **every** transition because fast
        results are temporally approximate (keyframe ≤ target rather
        than exact) — leaving them under their target-idx key would
        let a precise request after scrub release return a stale
        keyframe instead of re-decoding. Cheap to flush: the worker's
        prefetch fills it again within a few ticks.
        """
        with self._lock:
            self._source.set_fast_seek(enabled)
            self._cache.clear()

    def get(self, t_seconds: float) -> np.ndarray:
        """Return the frame at ``t_seconds`` (RGB uint8).

        Hot path (play tick where the worker already prefetched the
        target): instant dict lookup, no thread sync. Cold path
        (scrub, seek, first-frame-after-open): submit a sync request
        and block until the worker has decoded.
        """
        idx = self._idx_for(t_seconds)
        with self._lock:
            arr = self._cache.get(idx)
            self._target_idx = idx
        # Wake the worker so it queues the next prefetch (or services
        # this sync request below if we missed).
        self._wake.set()
        if arr is not None:
            return arr
        # Cache miss — escalate to a sync request.
        with self._lock:
            self._sync_request_t = t_seconds
            self._sync_done.clear()
        self._wake.set()
        # 0.5 s is generous — even a backward seek + decode through
        # a long GOP usually finishes in <100 ms. If we hit the
        # timeout the user sees a held frame; better than a UI freeze.
        if not self._sync_done.wait(timeout=0.5):
            log.warning("[video] decode timeout at t=%.3f", t_seconds)
            # Last-resort fallback: try the cache once more in case
            # the worker landed just after we gave up.
            with self._lock:
                fallback = self._cache.get(idx)
            if fallback is not None:
                return fallback
            # Nothing available — return a black frame the same
            # shape as the most recent cached one if any, else raise.
            with self._lock:
                if self._cache:
                    sample = next(iter(self._cache.values()))
                    return np.zeros_like(sample)
            raise RuntimeError("video decode timeout with no fallback")
        with self._lock:
            # Decoded buffers are RGBA uint8 since v1.8.2 (was RGB);
            # the fallback shape needs to match so callers don't
            # crash on a shape mismatch downstream.
            return self._sync_result if self._sync_result is not None \
                else np.zeros((1, 1, 4), dtype=np.uint8)

    def _idx_for(self, t_seconds: float) -> int:
        return round(t_seconds * self.fps)

    def _t_for(self, idx: int) -> float:
        # +half-frame so the requested time falls firmly inside the
        # frame's display interval (matches VideoSource's bracket
        # check: ``pts <= t < pts + interval``).
        return (idx + 0.5) / self.fps

    def _loop(self) -> None:
        while not self._stop.is_set():
            # Block until there's something to do. The 50 ms wake-up
            # makes the worker re-evaluate prefetch after a target
            # change even if no explicit ``set`` happened.
            self._wake.wait(timeout=0.05)
            self._wake.clear()
            if self._stop.is_set():
                break
            # Service any pending sync request first — the main
            # thread is waiting on it.
            with self._lock:
                req_t = self._sync_request_t
                self._sync_request_t = None
            if req_t is not None:
                try:
                    arr = self._source.frame_at_time(req_t)
                except Exception:
                    log.exception("[video] sync decode failed")
                    arr = None
                with self._lock:
                    self._sync_result = arr
                    if arr is not None:
                        self._cache[self._idx_for(req_t)] = arr
                        self._prune_locked()
                self._sync_done.set()
                continue
            # Prefetch: decode the next 1-2 frames ahead of target.
            with self._lock:
                target = self._target_idx
                if target is None:
                    continue
                missing = [
                    target + k for k in range(_PREFETCH_WINDOW + 1)
                    if (target + k) not in self._cache
                ]
            if not missing:
                continue
            next_idx = missing[0]
            try:
                arr = self._source.frame_at_time(self._t_for(next_idx))
            except Exception:
                log.exception("[video] prefetch decode failed at idx=%d", next_idx)
                continue
            with self._lock:
                self._cache[next_idx] = arr
                self._prune_locked()
                # If new work arrived during the decode, loop again
                # to service it instead of sleeping.
                if self._target_idx != target or self._sync_request_t is not None:
                    self._wake.set()

    def _prune_locked(self) -> None:
        """Trim the cache to ``_CACHE_CAPACITY`` keeping the window
        around the current target. Caller holds ``self._lock``."""
        if len(self._cache) <= _CACHE_CAPACITY:
            return
        target = self._target_idx if self._target_idx is not None else 0
        # Sort by distance from target, drop the farthest.
        keys = sorted(self._cache.keys(), key=lambda k: abs(k - target))
        for k in keys[_CACHE_CAPACITY:]:
            del self._cache[k]

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=1.0)
        try:
            self._source.close()
        except Exception:
            log.exception("[video] error closing VideoSource")
        with self._lock:
            self._cache.clear()


class VideoSourceManager:
    """Pool of decoders keyed by layer id.

    Each entry is a :class:`_ThreadedDecoder` that owns one
    :class:`VideoSource` on a worker thread + a small frame cache.
    Opens are lazy: the first ``decode_at`` for a layer spawns the
    worker.
    """

    def __init__(self) -> None:
        self._decoders: dict[str, _ThreadedDecoder] = {}
        # Latched scrub state — new decoders opened mid-scrub inherit
        # the flag so they don't decode-forward on their first frame.
        self._fast_seek: bool = False

    # Backwards-compat alias used by tests written against the old
    # ``_sources`` attribute name. New code should use ``_decoders``.
    #
    # .. deprecated:: v1.5.13
    #    Scheduled for removal in v1.7. Tests should refer to
    #    ``_decoders`` directly.
    @property
    def _sources(self) -> dict[str, _ThreadedDecoder]:  # pragma: no cover — back-compat shim
        return self._decoders

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def get_or_open(self, layer_id: str, path: Path) -> _ThreadedDecoder:
        """Return the threaded decoder for ``layer_id``, opening it
        if absent. Distinct ``layer_id``s map to distinct decoders
        even if they point at the same file (independent per-layer
        scrub / play state).
        """
        dec = self._decoders.get(layer_id)
        if dec is None:
            dec = _ThreadedDecoder(path)
            # New decoders inherit the manager's scrub state so the
            # first frame request after a layer add during an active
            # drag doesn't pay the precise-decode cost.
            if self._fast_seek:
                dec.set_fast_seek(True)
            self._decoders[layer_id] = dec
        return dec

    def set_fast_seek_all(self, enabled: bool) -> None:
        """Fan the fast-seek (keyframe-only) scrub mode across every
        open decoder. Called by the app when the timeline mouse drag
        starts (``True``) and ends (``False``). Cheap fanout — the
        per-decoder setter flips the bool *and* clears its frame
        cache so an approximate scrub frame doesn't leak into the
        precise post-release request.
        """
        self._fast_seek = bool(enabled)
        for dec in self._decoders.values():
            try:
                dec.set_fast_seek(enabled)
            except Exception:
                log.exception(
                    "[video] failed to push fast_seek=%s to decoder", enabled,
                )

    def close(self, layer_id: str) -> None:
        """Close the decoder for ``layer_id`` if any. No-op if absent."""
        dec = self._decoders.pop(layer_id, None)
        if dec is not None:
            try:
                dec.close()
            except Exception:
                log.exception("error closing decoder for layer %s", layer_id)

    def shutdown(self) -> None:
        """Close every open decoder. Called on app exit / session swap."""
        for layer_id in list(self._decoders.keys()):
            self.close(layer_id)

    # ------------------------------------------------------------------
    # Decode + format conversion
    # ------------------------------------------------------------------

    def decode_at(
        self, layer_id: str, path: Path, t_seconds: float,
    ) -> np.ndarray:
        """Return the frame at time ``t`` as ``(H, W, 4) float32`` RGBA.

        The viewport's ``set_frame`` accepts HxWx3 or HxWx4 in float
        precision; we pad to RGBA1 so the OCIO + alpha-composite
        pipeline downstream doesn't have to special-case 3-channel
        sources. uint8 → float32 conversion divides by 255 so the
        display path treats the values as already-normalised display
        colour (no OCIO input transform applied for now — proper
        colour-managed path arrives once we surface the FFmpeg
        color-primaries / transfer enum at the OCIO input picker).

        Since v1.8.2 the conversion to float32 happens inside
        :class:`VideoSource` (see ``_frame_to_rgba_f32``) so cached
        frames return display-ready and ``decode_at`` is a pass-
        through on the hot path. Per-frame cost on a 1440p cache
        hit drops from ~26 ms (the cast pass) to ~3 ms (just the
        thread sync) — that's what makes scrub-back / loop feel
        truly instant rather than just "fast".
        """
        dec = self.get_or_open(layer_id, path)
        return dec.get(t_seconds)
