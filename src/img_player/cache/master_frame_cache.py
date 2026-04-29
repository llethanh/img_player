"""Master-frame-keyed RAM cache for multi-layer playback.

Sibling of :class:`~img_player.cache.frame_cache.FrameCache` but
addresses the multi-layer model: the cache is keyed on
**master-frame indices** (= the user-visible timeline coordinates),
and the decoder resolves each master frame to a concrete
``(layer, source_frame)`` via a :class:`LayerStack` at decode time.

Why a sibling rather than a refactor of ``FrameCache``? The cache
class is already well-tested and tuned for the single-sequence
path (eviction scoring, missing-frame placeholders, epoch races
with the worker pool). Mutating it for multi-layer would risk
regressing the single-layer behaviour during the v1.0 transition.
``MasterFrameCache`` mirrors its public surface while baking in
the LayerStack resolution; the live app will switch between the
two during v1.0 phase 2b.

Cache invalidation rules (driven by :class:`LayerStack` signals):

* ``layers_changed`` (add / remove / reorder) → nuclear ``clear()``.
* ``visibility_changed(id)`` → invalidate every master-frame the
  toggled layer covers. (Q8: only the topmost visible is cached, so
  hiding the topmost reveals what's below — different decode.)
* ``layer_modified(id)`` for offset / trim / channel changes →
  invalidate the layer's master-frame range.

The class hooks these signals itself, so callers wire it once to a
LayerStack and then forget about invalidation.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from img_player.cache.missing_placeholder import get_missing_placeholder
from img_player.cache.worker_pool import WorkerPool
from img_player.io.reader import FrameReadError, read_frame
from img_player.layers import Layer, LayerStack

log = logging.getLogger(__name__)


_DEFAULT_BUDGET_BYTES = 8 * 1024**3
_DEFAULT_NUM_WORKERS = 4
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


class MasterFrameCache:
    """RAM cache keyed on master-timeline frames, resolved via a LayerStack."""

    def __init__(
        self,
        stack: LayerStack,
        budget_bytes: int = _DEFAULT_BUDGET_BYTES,
        num_workers: int = _DEFAULT_NUM_WORKERS,
    ) -> None:
        self._stack = stack
        self._budget = budget_bytes
        self._lock = threading.RLock()
        self._frames: dict[int, np.ndarray] = {}
        self._missing: set[int] = set()
        self._bytes_used = 0
        self._current_frame = 0
        self._direction = 1
        # Per-layer path / mtime index. Built lazily on first
        # request() for a given layer + cleared / refreshed via
        # ``_on_layers_changed``. The dict layouts:
        #   _path_index[layer_id][source_frame] -> Path
        #   _mtime_index[layer_id][source_frame] -> mtime float
        # Indexing on a dict is O(1) — vs scanning ``layer.sequence.frames``
        # at every request, which is O(n) per frame and quadratic
        # over a full prefetch range.
        self._path_index: dict[str, dict[int, object]] = {}
        self._mtime_index: dict[str, dict[int, float]] = {}
        # Bumped on every invalidation so workers in flight drop
        # their results when the world has moved on (channel change,
        # visibility flip, layer reorder, …).
        self._epoch = 0
        self._pool = WorkerPool(num_workers=num_workers, name="decode-master")

        # Counters
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._decode_errors = 0

        # Hook the stack so invalidation happens automatically.
        self._stack.layers_changed.connect(self._on_layers_changed)
        self._stack.visibility_changed.connect(self._on_visibility_changed)
        self._stack.layer_modified.connect(self._on_layer_modified)

    # ------------------------------------------------------------------ Lifecycle

    def shutdown(self) -> None:
        """Stop the worker pool. Must be called before app exit."""
        self._pool.shutdown()

    def clear(self) -> None:
        """Drop every cached frame + bump epoch so in-flight decodes
        get discarded at store time."""
        self._pool.clear()
        with self._lock:
            self._frames.clear()
            self._missing.clear()
            self._bytes_used = 0
            self._epoch += 1

    # ------------------------------------------------------------------ Compat shims

    def attach(self, sequence) -> None:  # type: ignore[no-untyped-def]
        """Drop-in replacement for :meth:`FrameCache.attach`.

        Replaces the LayerStack's contents with a single Layer
        wrapping ``sequence`` at ``offset = sequence.first_frame`` so
        master-frame indices line up with the source frame numbers
        one-to-one. Existing layers are removed.

        The mutation goes through the stack so listeners (LayerPanel,
        cache itself) get the standard ``layers_changed`` signal —
        no special-case wiring needed.
        """
        from img_player.layers import Layer
        for existing in self._stack.layers():
            self._stack.remove(existing.id)
        layer = Layer.from_sequence(sequence, offset=sequence.first_frame)
        self._stack.add(layer)

    def detach(self) -> None:
        """Drop-in replacement for :meth:`FrameCache.detach`. Empties
        the LayerStack so subsequent decodes see no layers."""
        for existing in self._stack.layers():
            self._stack.remove(existing.id)

    def set_channels(self, channels) -> None:  # type: ignore[no-untyped-def]
        """Drop-in replacement for :meth:`FrameCache.set_channels`.

        Updates the focused layer's ``channel_selection`` (or the
        first layer if none is focused). Triggers
        ``layer_modified`` → cache invalidation for that layer's
        master range. ``channels`` is either ``None`` (default
        RGB(A) reader) or a list of names.
        """
        from img_player.sequence.channels import (
            ChannelGroup, ChannelSelection,
        )
        focused = self._stack.focused() or (
            self._stack.layers()[0] if self._stack else None
        )
        if focused is None:
            return
        if channels is None:
            sel = None
        else:
            cs = tuple(channels)
            label = (
                "RGB" if cs == ("R", "G", "B")
                else "RGBA" if cs == ("R", "G", "B", "A")
                else cs[0] if len(cs) == 1
                else " / ".join(cs)
            )
            sel = ChannelSelection(active=ChannelGroup(label, cs))
        self._stack.update(focused.id, channel_selection=sel)

    def clear_pending(self) -> int:
        """Drop the worker pool's pending decode queue. Mirrors
        :meth:`FrameCache.clear_pending` for controller compat."""
        return self._pool.clear()

    def reload(self, new_sequence) -> tuple[int, int, int]:  # type: ignore[no-untyped-def]
        """Drop-in replacement for :meth:`FrameCache.reload`.

        For a single-layer stack (the typical post-attach state),
        this re-mints the focused layer with the fresh
        ``new_sequence`` and refreshes path / mtime indexes,
        invalidating only the slots whose mtime changed since the
        last index. Returns ``(kept, dropped, missing)`` for the
        status bar.

        Multi-layer reload is **not** supported via this entry point
        — the caller would need to know which layer to refresh. For
        v1.0 this is fine: reload is bound to a single sequence.
        """
        # Find the layer whose sequence directory matches the reload
        # target. Falls back to the focused layer.
        target = None
        for layer in self._stack.layers():
            if layer.sequence.directory == new_sequence.directory \
                    and layer.sequence.base_name == new_sequence.base_name:
                target = layer
                break
        if target is None:
            target = self._stack.focused() or (
                self._stack.layers()[0] if self._stack else None
            )
        if target is None:
            return (0, 0, 0)

        kept = dropped = 0
        old_mtimes = self._mtime_index.get(target.id, {})
        new_mtimes = {fi.frame_number: fi.mtime for fi in new_sequence.frames}
        new_paths = {fi.frame_number: fi.path for fi in new_sequence.frames}
        # Diff per source frame: same mtime → keep; different / now-missing → drop.
        with self._lock:
            for source_frame in set(old_mtimes) | set(new_mtimes):
                master_frame = target.offset + (source_frame - target.layer_in)
                old_mt = old_mtimes.get(source_frame, 0.0)
                new_mt = new_mtimes.get(source_frame, 0.0)
                if old_mt == new_mt and source_frame in new_paths:
                    if master_frame in self._frames \
                            and master_frame not in self._missing:
                        kept += 1
                    continue
                # Drop the cached buffer (if any) so the next
                # request decodes fresh.
                arr = self._frames.pop(master_frame, None)
                if arr is not None:
                    if master_frame not in self._missing:
                        self._bytes_used -= arr.nbytes
                        dropped += 1
                self._missing.discard(master_frame)
            self._epoch += 1
        # Rebuild the index against the refreshed sequence. Mutate
        # the layer dataclass directly (no stack.update call) so the
        # ``layer_modified`` signal doesn't fire — that signal would
        # invalidate the entire layer range, defeating the
        # mtime-based "kept" frames we just preserved.
        target.sequence = new_sequence
        target.layer_in = new_sequence.first_frame
        target.layer_out = new_sequence.last_frame
        self._path_index[target.id] = new_paths
        self._mtime_index[target.id] = new_mtimes
        missing = sum(
            1 for f in self._missing
            if target.master_start <= f <= target.master_end
        )
        return (kept, dropped, missing)

    # ------------------------------------------------------------------ Public read API

    def get(self, master_frame: int) -> np.ndarray | None:
        """Non-blocking fetch. ``None`` when the frame is not cached.

        Updates the playhead position so the next eviction round
        scores frames against this center.
        """
        with self._lock:
            self._current_frame = master_frame
            arr = self._frames.get(master_frame)
            if arr is not None:
                self._hits += 1
                return arr
            self._misses += 1
            return None

    def contains(self, master_frame: int) -> bool:
        with self._lock:
            return master_frame in self._frames

    def cached_frames(self) -> frozenset[int]:
        """Snapshot of currently cached master-frame indices."""
        with self._lock:
            return frozenset(self._frames.keys())

    def missing_frames(self) -> frozenset[int]:
        """Master frames whose decode failed (file missing /
        unreadable). They hold a checkerboard placeholder so
        playback doesn't stall."""
        with self._lock:
            return frozenset(self._missing)

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

    # ------------------------------------------------------------------ Public request API

    def request(self, master_frame: int, priority: int = 0) -> bool:
        """Enqueue an async decode. ``False`` when the frame is
        already cached or no layer covers this master frame."""
        with self._lock:
            if master_frame in self._frames:
                return False
        layer = self._stack.topmost_visible_at(master_frame)
        if layer is None:
            # Empty region — nothing to decode. The viewer paints
            # black for these master frames.
            return False
        source_frame = layer.source_frame_at(master_frame)
        self._ensure_index(layer)
        path = self._path_index[layer.id].get(source_frame)
        if path is None:
            # Layer covers this master frame but the source has a
            # hole there (sparse sequence). Pre-mark missing.
            with self._lock:
                placeholder = get_missing_placeholder(
                    layer.sequence.width or 512,
                    layer.sequence.height or 512,
                )
                self._frames[master_frame] = placeholder
                self._missing.add(master_frame)
            return False
        # Capture the layer + channels at submit time so the worker
        # decodes against a stable selection even if the user toggles
        # the menu mid-flight.
        channels = self._channels_for(layer)
        ph_w = layer.sequence.width or 512
        ph_h = layer.sequence.height or 512
        return self._pool.submit(
            priority,
            master_frame,
            lambda: self._decode_and_store(
                master_frame, path, channels, ph_w, ph_h,
            ),
        )

    def request_range(
        self, start: int, end: int, direction: int = 1,
    ) -> None:
        """Pre-fetch master frames in ``[start, end]`` (inclusive).

        ``direction`` only controls the iteration order — earlier-
        in-direction frames get lower priority numbers and decode
        first. Out-of-range bounds are clamped to the stack's
        master range so we don't queue work for empty regions.
        """
        if not self._stack:
            return
        m_first, m_last = self._stack.master_range()
        lo = max(min(start, end), m_first)
        hi = min(max(start, end), m_last)
        if lo > hi:
            return
        frames = range(lo, hi + 1) if direction >= 0 else range(hi, lo - 1, -1)
        for i, f in enumerate(frames):
            self.request(f, priority=i)

    def set_current_frame(self, master_frame: int) -> None:
        """Inform the cache of the playhead position (used for eviction
        scoring)."""
        with self._lock:
            self._current_frame = master_frame

    def set_direction(self, direction: int) -> None:
        """+1 forward / -1 reverse — drives the eviction skew."""
        with self._lock:
            self._direction = 1 if direction >= 0 else -1

    def shrink_budget(self, new_bytes: int) -> None:
        """Reduce the budget at runtime + force an immediate eviction.

        Mirrors the single-layer cache's runtime-monitor hook.
        Never grows back: once shrunk, stays shrunk for the session.
        """
        with self._lock:
            if new_bytes >= self._budget:
                return
            self._budget = max(0, new_bytes)
            self._evict_if_over_budget()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until the worker pool has nothing left to do. For tests."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._pool.pending() == 0:
                return True
            time.sleep(0.005)
        return False

    # ------------------------------------------------------------------ Stack signals → invalidation

    def _on_layers_changed(self) -> None:
        """Composition mutated → drop everything + rebuild indexes.

        Add / remove / reorder are rare enough that nuclear is
        acceptable. We also pre-mark missing frames here so the
        timeline cache bar lights up red on holes immediately
        rather than after the first prefetch wave reaches them.
        """
        self.clear()
        # Drop indexes for layers that no longer exist; rebuild for
        # layers that do. Cheap relative to the cache clear above.
        live_ids = {layer.id for layer in self._stack.layers()}
        for stale_id in list(self._path_index.keys()):
            if stale_id not in live_ids:
                self._path_index.pop(stale_id, None)
                self._mtime_index.pop(stale_id, None)
        # Eager pre-mark missing frames for every layer so the
        # timeline cache-bar shows holes immediately. Iterating each
        # layer's master range here is O(N) over total master frames,
        # acceptable for usual sequence sizes (thousands of frames).
        self._pre_mark_missing()

    def _pre_mark_missing(self) -> None:
        """Mark every master frame whose source has no file on disk
        as a missing-placeholder slot. Iterates through every layer
        in the stack so deeper layers (= covered by a hidden top one)
        still get their holes flagged should the user toggle
        visibility later.
        """
        for layer in self._stack.layers():
            self._ensure_index(layer)
            paths = self._path_index[layer.id]
            ph_w = layer.sequence.width or 512
            ph_h = layer.sequence.height or 512
            placeholder = get_missing_placeholder(ph_w, ph_h)
            for master_frame in range(layer.master_start, layer.master_end + 1):
                if master_frame in self._frames:
                    continue  # already populated (shouldn't happen post-clear)
                source_frame = layer.source_frame_at(master_frame)
                if paths.get(source_frame) is None:
                    # Sparse hole on this layer. Only mark missing if
                    # NO higher layer covers this master frame with a
                    # real path — otherwise the displayed pixel comes
                    # from the layer above and isn't actually missing.
                    topmost = self._stack.topmost_visible_at(master_frame)
                    if topmost is None or topmost.id == layer.id:
                        with self._lock:
                            self._frames[master_frame] = placeholder
                            self._missing.add(master_frame)

    def _on_visibility_changed(self, layer_id: str) -> None:
        """The toggled layer's master-frame region needs re-decode
        (different topmost-visible)."""
        layer = self._stack.find(layer_id)
        if layer is None:
            return
        self._invalidate_master_range(layer.master_start, layer.master_end)

    def _on_layer_modified(self, layer_id: str) -> None:
        """Trim / offset / channel change on a layer.

        We can't tell which field moved without diffing — invalidate
        the layer's current master-frame range. If the user just
        bumped exposure (no decode change), this is wasteful but
        correct; can be tightened later by emitting field-specific
        signals.
        """
        layer = self._stack.find(layer_id)
        if layer is None:
            return
        self._invalidate_master_range(layer.master_start, layer.master_end)

    # ------------------------------------------------------------------ Internals

    def _invalidate_master_range(self, first: int, last: int) -> None:
        """Drop cached frames in ``[first, last]`` + bump epoch so
        in-flight decodes for that range don't sneak back in."""
        if first > last:
            return
        with self._lock:
            for f in list(self._frames.keys()):
                if first <= f <= last:
                    arr = self._frames.pop(f)
                    if f not in self._missing:
                        self._bytes_used -= arr.nbytes
                    self._missing.discard(f)
            self._epoch += 1

    def _ensure_index(self, layer: Layer) -> None:
        """Lazy-build the path + mtime indexes for ``layer``.

        Called from ``request()`` so brand-new layers don't pay any
        indexing cost until they're actually accessed (matters when
        the user loads N layers at once but only plays through one
        of them — the others stay un-indexed).
        """
        if layer.id in self._path_index:
            return
        self._path_index[layer.id] = {
            fi.frame_number: fi.path for fi in layer.sequence.frames
        }
        self._mtime_index[layer.id] = {
            fi.frame_number: fi.mtime for fi in layer.sequence.frames
        }

    @staticmethod
    def _channels_for(layer: Layer) -> list[str] | None:
        """Per-layer channel selection → flat list for OIIO. ``None``
        defers to the reader's default (R/G/B/A)."""
        sel = layer.channel_selection
        if sel is None:
            return None
        union = list(sel.union_channels())
        return union or None

    def _decode_and_store(
        self,
        master_frame: int,
        path,
        channels: list[str] | None,
        placeholder_w: int,
        placeholder_h: int,
    ) -> None:
        """Worker entry point — runs on a decode thread.

        ``placeholder_w / _h`` are captured at submit time from the
        layer that was the topmost-visible-then; they're used only
        if the decode fails so the placeholder matches the expected
        resolution and the GL viewport doesn't have to rescale a
        random size.
        """
        with self._lock:
            epoch = self._epoch
        try:
            arr = read_frame(path, channels=channels)
        except FrameReadError as err:
            log.warning(
                "decode failed master=%d path=%s: %s",
                master_frame, path, err,
            )
            placeholder = get_missing_placeholder(placeholder_w, placeholder_h)
            with self._lock:
                self._decode_errors += 1
                if epoch != self._epoch:
                    return
                if master_frame in self._frames:
                    return
                self._frames[master_frame] = placeholder
                self._missing.add(master_frame)
            return

        with self._lock:
            if epoch != self._epoch:
                return  # invalidated mid-decode — drop
            if master_frame in self._frames:
                return  # raced — keep existing
            self._frames[master_frame] = arr
            self._bytes_used += arr.nbytes
            self._evict_if_over_budget()

    def _evict_if_over_budget(self) -> None:
        """Distance-from-playhead eviction with a behind-the-playhead
        penalty (= we evict frames the user just played first)."""
        if self._bytes_used <= self._budget:
            return
        cur = self._current_frame
        d = self._direction
        penalty = _BEHIND_PLAYHEAD_PENALTY

        def score(f: int) -> float:
            delta = (f - cur) * d
            if delta < 0:
                return -delta * penalty
            return float(delta)

        by_priority = sorted(self._frames.keys(), key=score, reverse=True)
        for f in by_priority:
            if self._bytes_used <= self._budget:
                break
            arr = self._frames.pop(f)
            if f not in self._missing:
                self._bytes_used -= arr.nbytes
            self._missing.discard(f)
            self._evictions += 1
