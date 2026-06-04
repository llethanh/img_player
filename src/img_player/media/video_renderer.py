"""Owns open :class:`VideoSource` decoders keyed by layer id.

The renderer-side counterpart to ``Layer.from_video``: when a video
layer is added to the stack the manager opens its decoder lazily on
first frame access; when the layer is removed the manager closes the
file handle so we don't leak across session loads.

Decoding is synchronous from the Qt thread's perspective: each
:class:`_ThreadedDecoder` is a thin wrapper around a
:class:`VideoSource` that calls ``frame_at_time`` directly. The
source itself maintains a multi-hundred-frame LRU + a background
prefetch thread, so cache hits return in microseconds and never
race a separate worker (the v1.5–v1.8.2 design had a per-layer
worker thread that contended with the source's prefetch and could
time out at 500 ms — see :class:`_ThreadedDecoder` docstring).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from img_player.media.video_source import VideoSource

log = logging.getLogger(__name__)


class _ThreadedDecoder:
    """Thin synchronous wrapper around :class:`VideoSource`.

    History — there used to be a per-layer worker thread + small
    dict cache + sync_request/sync_done dance here. The intent was
    a 2-frame prefetch window the main Qt thread could rely on for
    instant cache hits.

    Since v1.8.3 ``VideoSource`` owns its OWN background prefetch
    worker (which streams the whole near-playhead window into a
    multi-hundred-frame float32 LRU). The local worker became
    strictly redundant — worse, it CONTENDED with the source-level
    prefetch (both ran ``frame_at_time`` against the same
    container's seek state), and on cache misses it forced the main
    thread to wait on a 500 ms sync_done event. When that timed
    out, decode_at returned a 1×1 black ndarray (visible to the
    user as a black flicker / freeze).

    The replacement is a pure pass-through: every call to ``get(t)``
    runs ``self._source.frame_at_time(t)`` directly on the calling
    thread. The source's LRU is the only cache. Cache hits return
    in ~0.03 ms (a dict lookup). Cache misses fall back to a real
    seek + decode in ~10-30 ms on AV1 1440p — no timeout, no black
    frame.

    Single-threaded access is still enforced because the Qt main
    thread is the only caller for a given decoder. The source's
    own prefetch container runs in parallel on a separate
    container, so the main thread and the prefetch never share the
    same PyAV decoder state.

    Not thread-safe to share across multiple managers — one
    decoder per layer.
    """

    def __init__(self, path: Path, *, cache_budget_bytes: int | None = None) -> None:
        if cache_budget_bytes is not None:
            self._source = VideoSource(
                path, cache_budget_bytes=cache_budget_bytes,
            )
        else:
            self._source = VideoSource(path)

    @property
    def fps(self) -> float:
        return float(self._source.fps)

    def set_fast_seek(self, enabled: bool) -> None:
        """Toggle the keyframe-only scrub shortcut on the underlying
        source. No local cache to flush — the source's LRU handles
        scrub-back correctness on its own (a precise re-decode
        request post-release populates the exact frame into the
        same LRU bucket the keyframe occupied)."""
        self._source.set_fast_seek(enabled)

    def get(self, t_seconds: float) -> np.ndarray:
        """Return the frame at ``t_seconds`` as RGBA float32 in [0, 1].

        Pure synchronous proxy to ``VideoSource.frame_at_time``: the
        source has its own LRU and prefetch worker, so cache hits
        are zero-cost and cache misses fall back to a real seek +
        decode rather than the 500 ms sync_done timeout that the
        old worker layer imposed.

        Notify the source's prefetch thread of the playhead before
        the lookup so the prefetcher steers ahead even on a string
        of hits (where ``frame_at_time`` itself wouldn't need to
        update the playhead).
        """
        self._source.notify_playback_position(t_seconds)
        return self._source.frame_at_time(t_seconds)

    def close(self) -> None:
        try:
            self._source.close()
        except Exception:
            log.exception("[video] error closing VideoSource")


class VideoSourceManager:
    """Pool of decoders keyed by layer id.

    Each entry is a :class:`_ThreadedDecoder` that owns one
    :class:`VideoSource` on a worker thread + a small frame cache.
    Opens are lazy: the first ``decode_at`` for a layer spawns the
    worker.
    """

    def __init__(
        self,
        *,
        source_cache_budget_bytes: int | None = None,
        cache_budget_provider: Callable[[], int | None] | None = None,
    ) -> None:
        self._decoders: dict[str, _ThreadedDecoder] = {}
        # Per-source LRU cache budget plumbed into every
        # ``_ThreadedDecoder`` we open. None = use VideoSource's
        # default (8 GB).
        #
        # Two configuration modes:
        #   * ``source_cache_budget_bytes`` — fixed value, locked at
        #     construction. Used by tests and any caller that doesn't
        #     need live tuning.
        #   * ``cache_budget_provider`` — callable resolved on every
        #     ``get_or_open`` call. The App uses this to read the
        #     live ``Preferences.video_cache_budget_gb`` so a change
        #     in the Preferences dialog applies the next time the
        #     user opens a video layer (no restart required).
        #
        # If both are provided the provider wins; if neither, the
        # source falls back to its module-level default.
        self._source_cache_budget_bytes: int | None = source_cache_budget_bytes
        self._cache_budget_provider: Callable[[], int | None] | None = (
            cache_budget_provider
        )
        # Latched scrub state — new decoders opened mid-scrub inherit
        # the flag so they don't decode-forward on their first frame.
        self._fast_seek: bool = False

    def _resolve_cache_budget(self) -> int | None:
        """Return the cache budget that the next open should use.

        Reads the live provider if set, otherwise the value latched at
        construction. Defensive against a misbehaving provider:
        anything raising falls back to the latched value rather than
        propagating into a video open failure.
        """
        if self._cache_budget_provider is not None:
            try:
                return self._cache_budget_provider()
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "[video] cache_budget_provider raised; "
                    "falling back to latched value %r",
                    self._source_cache_budget_bytes,
                )
        return self._source_cache_budget_bytes

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

        On open the cache budget is **re-resolved from the live
        provider** (if one was passed at manager construction), so a
        Preferences-dialog tweak to ``video_cache_budget_gb`` lands
        the next time the user opens a video layer — no restart
        needed.
        """
        dec = self._decoders.get(layer_id)
        if dec is None:
            dec = _ThreadedDecoder(
                path,
                cache_budget_bytes=self._resolve_cache_budget(),
            )
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

        Implementation:
          1. ``dec.get(t)`` forwards to ``VideoSource.frame_at_time(t)``
             since v1.8.3 (the per-layer worker thread + small dict
             cache was retired — see ``_ThreadedDecoder`` docstring
             for the reason).
          2. ``VideoSource`` keeps its own background prefetch thread
             filling a multi-hundred-frame float32 LRU; cache hits
             return in ~0.03 ms with no main-thread cast work.
          3. Cache misses fall back to a real seek + decode on the
             calling thread (10-30 ms on AV1 1440p) — no 500 ms
             sync_done timeout that would surface as a black 1×1
             ndarray.
        """
        dec = self.get_or_open(layer_id, path)
        return dec.get(t_seconds)
