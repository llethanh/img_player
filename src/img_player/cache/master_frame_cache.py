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


def _ensure_rgba(arr: np.ndarray) -> np.ndarray:
    """Pad a 1- or 3-channel buffer to 4 channels with full opacity.

    The compositing path always works in RGBA so the alpha is
    explicit and the per-pixel maths can stay simple. If the source
    happens to be RGB-only (no alpha file), we treat the layer as
    fully opaque — a sensible fallback that mirrors how Nuke / RV
    handle the case ("missing alpha = solid").
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC array, got shape {arr.shape}")
    if arr.shape[2] == 4:
        return arr
    if arr.shape[2] == 3:
        a = np.ones(
            (arr.shape[0], arr.shape[1], 1), dtype=arr.dtype,
        )
        return np.concatenate([arr, a], axis=2)
    if arr.shape[2] == 1:
        rgb = np.broadcast_to(arr, (arr.shape[0], arr.shape[1], 3))
        a = np.ones(
            (arr.shape[0], arr.shape[1], 1), dtype=arr.dtype,
        )
        return np.ascontiguousarray(np.concatenate([rgb, a], axis=2))
    raise ValueError(f"Unsupported channel count: {arr.shape[2]}")


def _force_alpha_one(arr: np.ndarray) -> np.ndarray:
    """Set every alpha to 1.0 — used for "opaque floor" layers in the
    composite path so the layers below them are masked, regardless of
    the source's actual A channel."""
    out = arr.copy()
    out[..., 3:4] = np.ones_like(out[..., 3:4])
    return out


def _premultiply(arr: np.ndarray) -> np.ndarray:
    """Multiply RGB by alpha so straight-alpha buffers can feed the
    same over operator that premult buffers do."""
    out = arr.copy()
    a = out[..., 3:4]
    out[..., :3] = out[..., :3] * a
    return out


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
        # Loop-aware eviction (set by the controller on
        # ``set_loop_range``). When ``_loop_enabled``, the eviction
        # scoring switches from signed-distance to **ring distance**:
        # frames near ``_loop_lo`` are treated as "imminent" once the
        # playhead approaches ``_loop_hi``, instead of being charged
        # the behind-playhead penalty. Without this, the wrap target
        # (``lo``) has the highest score in ``_evict_if_over_budget``
        # and is evicted the moment the worker finishes decoding it —
        # the playhead then stalls forever at ``hi`` because every
        # decode of ``lo`` is immediately undone.
        self._loop_lo: int | None = None
        self._loop_hi: int | None = None
        self._loop_enabled: bool = False
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
        # Last-known master range per layer. Compared against the
        # post-mutation range when ``layer_modified`` fires so the
        # cache knows to invalidate frames the layer USED to cover
        # but no longer does — e.g. trimming a top-layer's
        # ``layer_out`` shrinks its master_end, and the master
        # frames in the now-uncovered tail must be re-decoded
        # (revealing the layer underneath, or going to black).
        self._last_known_range: dict[str, tuple[int, int]] = {}
        # Last-known per-layer state for pinpointing what changed in
        # ``_on_layer_modified``. Tuple = (offset, layer_in,
        # layer_out, channel_selection). When only ``offset`` differs
        # AND the stack has a single layer, we *re-index* the cached
        # frames (master_frame F → F + Δ) instead of dropping them —
        # decoding is the expensive part, and the pixel data is
        # identical between the two offsets.
        self._last_known_state: dict[str, tuple] = {}
        # Per-frame contributor chain snapshot — captures which layers
        # produced each cached frame's pixels (top→bottom in stack
        # order, walking until an opaque floor). Used by
        # ``_on_layers_changed`` to detect which frames are still
        # valid after a reorder / add / remove: when the chain at a
        # given master frame is unchanged, the cached pixels stay
        # accurate and we can skip the re-decode. Without this,
        # every reorder triggered a nuclear ``clear()`` and the user
        # saw a noticeable freeze + grey cache bar while every frame
        # came back from disk.
        self._chain_snapshot: dict[int, tuple[str, ...]] = {}
        # Bumped on every invalidation so workers in flight drop
        # their results when the world has moved on (channel change,
        # visibility flip, layer reorder, …).
        self._epoch = 0
        # NB: alpha-compositing and the premult/straight convention
        # both used to live on the cache as global flags. They moved
        # to per-layer fields (``Layer.alpha_composite`` /
        # ``Layer.alpha_is_straight``) so a stack can mix conventions
        # without forcing a single mode for the whole composite.
        # ``_channels_for`` reads ``alpha_composite`` to decide
        # whether to force A in the decoded selection;
        # ``_decode_composited_and_store`` reads
        # ``alpha_is_straight`` per contributor.
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
        """Stop the worker pool **and flush every cached frame**.

        Without the explicit flush the decoded numpy arrays sit in
        ``_frames`` until the Python GC tears down the cache object —
        which on Windows can stall the process exit for a noticeable
        second or two when the cache holds millions of small RGBA
        buffers. Dropping the dict here makes the exit snappy and
        gives the OS the RAM back deterministically. Idempotent: a
        second call after shutdown sees an empty dict and no-ops.
        """
        self._pool.shutdown()
        with self._lock:
            self._frames.clear()
            self._missing.clear()
            self._chain_snapshot.clear()
            self._path_index.clear()
            self._mtime_index.clear()
            self._last_known_range.clear()
            self._last_known_state.clear()
            self._bytes_used = 0

    def clear(self) -> None:
        """Drop every cached frame + bump epoch so in-flight decodes
        get discarded at store time."""
        self._pool.clear()
        with self._lock:
            self._frames.clear()
            self._missing.clear()
            self._chain_snapshot.clear()
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
        # Batch so the multi-step replace (remove every existing
        # layer + add the new one) collapses to a single undo entry.
        # Without this, Ctrl+Z after a drop-replace would only revert
        # the last sub-step, leaving the user with an unexpected
        # half-state.
        with self._stack.batch():
            for existing in self._stack.layers():
                self._stack.remove(existing.id)
            layer = Layer.from_sequence(sequence, offset=sequence.first_frame)
            self._stack.add(layer)

    def detach(self) -> None:
        """Drop-in replacement for :meth:`FrameCache.detach`. Empties
        the LayerStack so subsequent decodes see no layers."""
        with self._stack.batch():
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
                self._chain_snapshot.pop(master_frame, None)
            self._epoch += 1
        # Rebuild the index against the refreshed sequence. Mutate
        # the layer dataclass directly (no stack.update call) so the
        # ``layer_modified`` signal doesn't fire — that signal would
        # invalidate the entire layer range, defeating the
        # mtime-based "kept" frames we just preserved.
        #
        # IMPORTANT: preserve the user's offset / layer_in / layer_out.
        # Earlier revisions re-anchored offset/in/out to the new
        # sequence's first_frame on every reload so a single-layer
        # ``master_frame == source_frame`` invariant held. That broke
        # any layer the user had moved or trimmed: a Ctrl+R after
        # nudging a layer would teleport it back to its initial
        # position, and (worse) the cached frames stored under the
        # OLD master keys would now decode against the NEW layer
        # geometry — wrong source_frame mapping → wrong pixels on
        # screen. Keeping offset/in/out as-is means the diff loop's
        # master keys stay coherent with the live layer.
        #
        # If the new sequence shrank past the existing trim, the
        # out-of-range frames simply turn into missing-placeholders
        # via the cache's normal sparse-hole path; the user can
        # extend layer_in / layer_out via the bar's drag handles to
        # uncover the rest. Same NLE convention as Premiere /
        # Resolve "media offline" placeholders.
        target.sequence = new_sequence
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

    def is_gap_frame(self, master_frame: int) -> bool:
        """True when this cache will never decode for this master frame.

        Two cases qualify as "gap" from the controller's perspective:

        1. **No visible layer covers the frame** — the multi-layer
           void between two clips, or before/after the only layer.
           The viewport paints black; the playhead must advance
           through (otherwise it freezes at the void edge).
        2. **The topmost-visible layer is video** — video layers
           bypass this cache entirely (pixels come from
           :class:`VideoSourceManager`), so ``cache.contains`` will
           never be true for them. Without this carve-out, the
           controller's "stall when not cached" guard fires forever
           and play() doesn't advance the playhead through video
           clips.

        The shared semantic is "the cache has nothing to say about
        this frame" — the viewport's frame_changed handler is what
        produces pixels (gap placeholder for case 1, video decode
        for case 2).
        """
        top = self._stack.topmost_visible_at(master_frame)
        if top is None:
            return True
        if getattr(top, "is_video", False):
            return True
        return False

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
        # Per-layer compositing: collect visible layers covering this
        # master frame top→bottom, take them until we hit one with
        # ``alpha_composite=False`` (= opaque floor that masks
        # everything below). If the topmost is itself opaque we
        # short-circuit to the single-decode fast path — no point
        # walking a plan of one layer through the composite worker.
        #
        # Video-backed layers are excluded here: video pixels come
        # from :class:`VideoSourceManager` (PyAV), not from this
        # OIIO-driven cache. Including them would point the path
        # index at the .mp4 container and the OIIO reader would log
        # noisy "unsupported format" errors on every prefetch tick.
        visible = [
            layer for layer in self._stack.layers()
            if layer.visible and layer.covers(master_frame)
            and not getattr(layer, "is_video", False)
        ]
        if not visible:
            return False
        topmost = visible[0]
        # Contact-sheet selections demand the union of every checked
        # tile's channels (potentially 6+ on a multi-AOV pass). The
        # composite path normalises every contributor to RGBA via
        # ``_ensure_rgba`` and rejects anything else (= the user-
        # reported ``Unsupported channel count: 7`` crash). For a
        # single-layer stack with a contact-sheet selection the
        # composite is structurally a no-op anyway (nothing to blend
        # against); detour through the single-decode fast path so the
        # full union buffer reaches ``compose_contact_sheet`` intact.
        is_contact_sheet_selection = (
            topmost.channel_selection is not None
            and topmost.channel_selection.is_contact_sheet
        )
        single_layer = len(visible) == 1
        if not topmost.alpha_composite:
            # Topmost is opaque — single-layer fast path.
            layer = topmost
        elif single_layer and is_contact_sheet_selection:
            # Single-layer contact-sheet — fast path, full union buffer.
            layer = topmost
        else:
            plan_layers: list = []
            for layer in visible:
                plan_layers.append(layer)
                if not layer.alpha_composite:
                    break  # opaque floor — stop
            if len(plan_layers) > 1 or plan_layers[0].alpha_composite:
                return self._submit_composite(
                    master_frame, plan_layers, priority,
                )
            layer = plan_layers[0]
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
                self._record_frame_chain(master_frame)
            return False
        # Capture the layer + channels at submit time so the worker
        # decodes against a stable selection even if the user toggles
        # the menu mid-flight.
        channels = self._channels_for(layer)
        ph_w = layer.sequence.width or 512
        ph_h = layer.sequence.height or 512
        # When ``alpha_composite=False`` the layer is treated as
        # opaque — strip A from the cached buffer so the shader sees
        # alpha=1 everywhere and no checker shows. Without this the
        # source's alpha leaks into the texture and the user sees
        # checker behind transparent regions even though they
        # explicitly disabled the layer's transparency mode.
        strip_alpha = not layer.alpha_composite
        return self._pool.submit(
            priority,
            master_frame,
            lambda: self._decode_and_store(
                master_frame, path, channels, ph_w, ph_h,
                strip_alpha=strip_alpha,
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

    def set_loop_range(
        self, lo: int | None, hi: int | None, enabled: bool,
    ) -> None:
        """Inform the cache of the playback loop range.

        When ``enabled`` is True, ``_evict_if_over_budget`` switches
        to **ring distance** scoring: a frame's distance from the
        playhead wraps around at ``hi → lo`` so the wrap target
        stays cheap to keep instead of being the first thing evicted.

        Pass ``enabled=False`` (or ``None`` bounds) to revert to the
        signed-distance scoring used outside loop playback (ONCE
        mode, scrub, in/out range with non-loop semantics).
        """
        with self._lock:
            if not enabled or lo is None or hi is None or hi <= lo:
                self._loop_lo = None
                self._loop_hi = None
                self._loop_enabled = False
                return
            self._loop_lo = int(lo)
            self._loop_hi = int(hi)
            self._loop_enabled = True

    def shrink_budget(self, new_bytes: int) -> None:
        """Reduce the budget at runtime + force an immediate eviction.

        Mirrors the single-layer cache's runtime-monitor hook.
        Never grows back via this entry point: once shrunk, stays
        shrunk for the runtime-monitor's autonomic loop. Use
        :meth:`set_budget` for the explicit per-session re-tune
        (which is allowed to grow when the user has freed RAM
        between Flick launch and project open).
        """
        with self._lock:
            if new_bytes >= self._budget:
                return
            self._budget = max(0, new_bytes)
            self._evict_if_over_budget()

    def set_budget(self, new_bytes: int) -> None:
        """Set the budget to an explicit value, growing or shrinking.

        Forces an eviction if the new value is below current usage.
        Used by the per-session re-tune (``app._retune_for_current_ram``):
        when the user opens a new project, we re-snapshot
        ``RuntimeState`` and call this to follow ambient RAM
        availability — closing Chrome between Flick launch and
        opening a project gives the next project a roomier cache,
        without the oscillation risk of a continuous grow-back loop.

        Distinct from :meth:`shrink_budget`: that one is the
        runtime-monitor's one-way safety valve and intentionally
        refuses to grow.
        """
        with self._lock:
            self._budget = max(0, int(new_bytes))
            self._evict_if_over_budget()

    def wait_idle(self, timeout: float = 5.0) -> bool:
        """Block until the worker pool has nothing left to do. For tests."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._pool.pending() == 0:
                return True
            time.sleep(0.005)
        return False

    def _record_frame_chain(self, master_frame: int) -> None:
        """Store the current contributor chain for ``master_frame``.

        Called next to every ``self._frames[mf] = ...`` insertion
        point so ``_on_layers_changed`` always has an accurate "old
        chain" to compare against. Cheap (a list walk + dict write
        under the existing lock); a no-op overhead per decode.
        """
        self._chain_snapshot[master_frame] = self._compute_chain_at(master_frame)

    def _compute_chain_at(self, master_frame: int) -> tuple[str, ...]:
        """Return the ordered tuple of contributor layer ids that
        would produce the pixels at ``master_frame`` under the
        current stack state.

        Walks visible layers top→bottom, collecting ids until the
        first ``alpha_composite=False`` (opaque floor) or the bottom
        of the stack. Empty tuple when no layer covers the frame —
        that's a "no coverage" gap, the viewer paints black.

        Used by ``_on_layers_changed`` to compare per-frame chains
        before / after a stack mutation. Cheap to call: a few
        attribute reads + a list walk per layer in the stack.
        """
        chain: list[str] = []
        for layer in self._stack.layers():
            if not layer.visible or not layer.covers(master_frame):
                continue
            chain.append(layer.id)
            if not layer.alpha_composite:
                # Opaque floor — anything below is masked, the chain
                # ends here.
                break
        return tuple(chain)

    # ------------------------------------------------------------------ Stack signals → invalidation

    def _on_layers_changed(self) -> None:
        """Composition mutated → invalidate selectively.

        For each cached master frame, recompute the contributor
        chain under the new stack state and compare with the chain
        recorded at decode time. Drop only frames whose chain
        actually changed; the rest stay cached and the user sees no
        freeze on reorders that don't affect their pixels (e.g.
        swapping two layers below an opaque-floor topmost, or
        moving an invisible layer).

        Add / remove are uniformly handled by the same comparison:
        adding a top layer changes the chain at every frame it
        covers; removing a layer that was contributing changes the
        chain at every frame in its old reach. Frames untouched by
        the mutation keep their pixels.

        We still ``pool.clear()`` and bump the epoch to discard any
        in-flight decode whose result might no longer match the
        new state (cheap correctness over salvaging a few hundred
        ms of in-flight work). The dropped pending re-fires through
        the controller's ``replan_prefetch`` immediately after.
        """
        # Compute new chains for every currently-cached frame and
        # gather the set to drop.
        with self._lock:
            cached_frames = list(self._frames.keys())
        frames_to_drop: list[int] = []
        new_chains: dict[int, tuple[str, ...]] = {}
        for f in cached_frames:
            new_chain = self._compute_chain_at(f)
            old_chain = self._chain_snapshot.get(f)
            new_chains[f] = new_chain
            if old_chain != new_chain:
                frames_to_drop.append(f)

        # Drop pending decodes (their target chain may also have
        # changed) and bump the epoch so in-flight workers' results
        # land in the bin instead of polluting the cache.
        self._pool.clear()
        with self._lock:
            for f in frames_to_drop:
                arr = self._frames.pop(f, None)
                if arr is not None and f not in self._missing:
                    self._bytes_used -= arr.nbytes
                self._missing.discard(f)
                self._chain_snapshot.pop(f, None)
            self._epoch += 1
            # Refresh chain snapshot for kept frames so the next
            # ``_on_layers_changed`` compares against the current
            # state, not a stale one.
            for f, chain in new_chains.items():
                if f in self._frames:
                    self._chain_snapshot[f] = chain

        # Drop indexes for layers that no longer exist; rebuild for
        # layers that do. Cheap relative to a per-frame eviction.
        live_ids = {layer.id for layer in self._stack.layers()}
        for stale_id in list(self._path_index.keys()):
            if stale_id not in live_ids:
                self._path_index.pop(stale_id, None)
                self._mtime_index.pop(stale_id, None)
                self._last_known_range.pop(stale_id, None)
        # Snapshot the current master ranges + per-layer state so
        # the next ``layer_modified`` can diff against this baseline.
        for layer in self._stack.layers():
            self._last_known_range[layer.id] = (
                layer.master_start, layer.master_end,
            )
            self._last_known_state[layer.id] = (
                layer.offset, layer.layer_in, layer.layer_out,
                layer.channel_selection,
                layer.alpha_composite, layer.alpha_is_straight,
            )
        # Drop state for layers that no longer exist.
        for stale_id in list(self._last_known_state.keys()):
            if stale_id not in {l.id for l in self._stack.layers()}:
                self._last_known_state.pop(stale_id, None)
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
                            self._record_frame_chain(master_frame)

    def _on_visibility_changed(self, layer_id: str) -> None:
        """The toggled layer's master-frame region needs re-decode
        (different topmost-visible)."""
        layer = self._stack.find(layer_id)
        if layer is None:
            return
        self._invalidate_master_range(layer.master_start, layer.master_end)

    def _on_layer_modified(self, layer_id: str) -> None:
        """Trim / offset / channel change on a layer.

        Two paths:

        * **Pure offset shift** (single-layer stack, no trim/channel
          change). The decoded pixels are identical between the old
          and new offsets — only their master-frame keys change. We
          re-index the cache by ``Δ = new_offset - old_offset``
          rather than dropping and re-decoding. Big win when the
          user nudges a layer left/right on the timeline: dragging
          across hundreds of frames used to invalidate the entire
          range and trigger a full re-decode wave.

        * **Anything else** (trim, channel selection, multi-layer
          offset shift) — invalidate the union of old and new master
          ranges, same as before. The previous range matters when
          the layer shrunk: the now-uncovered tail / head holds
          stale pixels that must come from whatever's underneath in
          the stack.
        """
        layer = self._stack.find(layer_id)
        if layer is None:
            return
        prev_start, prev_end = self._last_known_range.get(
            layer_id, (layer.master_start, layer.master_end),
        )
        new_start = layer.master_start
        new_end = layer.master_end
        prev_state = self._last_known_state.get(layer_id)
        new_state = (
            layer.offset, layer.layer_in, layer.layer_out,
            layer.channel_selection,
            layer.alpha_composite, layer.alpha_is_straight,
        )

        # Detect pure-offset shift. Only safe when:
        #   * trim (in/out) and channel selection are unchanged
        #   * a single layer is loaded — otherwise the cached pixels
        #     at old master frames may have come from a different
        #     layer (whichever was topmost-visible at decode time)
        #     and shifting them would scramble the composition.
        is_offset_only = (
            prev_state is not None
            and prev_state[0] != new_state[0]            # offset differs
            and prev_state[1:] == new_state[1:]          # rest equal
            and len(self._stack) == 1
        )
        if is_offset_only:
            shift = new_state[0] - prev_state[0]
            self._shift_cached_frames(prev_start, prev_end, shift)
            self._last_known_range[layer_id] = (new_start, new_end)
            self._last_known_state[layer_id] = new_state
            return

        # Cover the union so any frame the layer USED to or NOW does
        # cover gets re-decoded.
        invalidate_first = min(prev_start, new_start)
        invalidate_last = max(prev_end, new_end)
        self._invalidate_master_range(invalidate_first, invalidate_last)
        self._last_known_range[layer_id] = (new_start, new_end)
        self._last_known_state[layer_id] = new_state

    # ------------------------------------------------------------------ Internals

    def _shift_cached_frames(self, old_first: int, old_last: int, shift: int) -> None:
        """Re-key cached frames in ``[old_first, old_last]`` by ``shift``.

        Used for the pure-offset-shift fast path: instead of dropping
        every cached frame and re-decoding (= seconds of work for a
        100-frame layer), we just move each entry from master key F
        to F + shift. The decoded pixel buffer is unchanged — only
        the timeline coordinate it answers to.

        Bumps the epoch so any in-flight decode (which captured the
        OLD master_frame at submit time) gets dropped at store time
        rather than landing under a now-stale key. The lost in-flight
        work is much smaller than what we'd otherwise re-decode, so
        the trade-off is favourable.
        """
        if shift == 0 or old_first > old_last:
            return
        with self._lock:
            # Snapshot the entries to move; iterate later to avoid
            # mutating the dict while iterating it.
            to_move: list[tuple[int, np.ndarray, bool]] = []
            for f in list(self._frames.keys()):
                if old_first <= f <= old_last:
                    arr = self._frames.pop(f)
                    is_missing = f in self._missing
                    self._missing.discard(f)
                    # Drop the old chain snapshot too — re-recorded
                    # at the new key below.
                    self._chain_snapshot.pop(f, None)
                    to_move.append((f, arr, is_missing))
            # Apply the shift. If a destination key is already
            # occupied (shouldn't happen with a single-layer stack,
            # but be defensive) the existing entry wins and the
            # shifted one is dropped — that frame will simply be
            # re-decoded by the next prefetch wave.
            for old_f, arr, is_missing in to_move:
                new_f = old_f + shift
                if new_f in self._frames:
                    if not is_missing:
                        self._bytes_used -= arr.nbytes
                    continue
                self._frames[new_f] = arr
                self._record_frame_chain(new_f)
                if is_missing:
                    self._missing.add(new_f)
            self._epoch += 1

    def _invalidate_master_range(self, first: int, last: int) -> None:
        """Drop cached frames in ``[first, last]`` + bump epoch so
        in-flight decodes for that range don't sneak back in.

        Skips the epoch bump when no frames are *currently cached*
        in the range. Important for the fresh-load case:

        1. ``cache.attach`` adds a layer; controller queues 100
           decodes against the new layer's channels (== ``None`` →
           reader default RGBA).
        2. ``restore_channel_state`` then writes the prefs-saved
           ``ChannelSelection`` onto the layer, firing
           ``layer_modified``.

        Without the skip, step 2 invalidates an empty range AND
        bumps the epoch — the 100 in-flight decodes from step 1
        store their results, hit the epoch mismatch, and drop. The
        viewport stays black until a second prefetch wave decodes
        again. With the skip, the in-flight decodes succeed; their
        captured channels (default RGBA) are at worst slightly
        broader than the prefs-saved selection (typically RGB),
        which the compose path handles fine.

        For the legitimate user-driven case (channel toggle while
        frames are cached), ``to_drop`` is non-empty so we still
        bump the epoch and discard the stale workers.
        """
        if first > last:
            return
        with self._lock:
            # Skip placeholder slots (entries in ``_missing``) — they
            # represent "this source file is gone from disk", which a
            # layer-state change (channel selection, trim, offset on
            # another layer) doesn't undo. Dropping them would just
            # cost a re-decode that lands on the same placeholder,
            # AND it would force an epoch bump that ghosts every
            # in-flight wave-1 decode for this range. The wave-1
            # workers' keys stay in the pool's ``_pending`` set until
            # they finish, so the next ``request_range`` (wave 2)
            # silently dedup-rejects re-submissions for those keys —
            # the very frames closest to the playhead end up missing
            # from the cache forever (= viewer black at startup with
            # sparse-source sequences).
            to_drop = [
                f for f in self._frames
                if first <= f <= last and f not in self._missing
            ]
            if not to_drop:
                return
            for f in to_drop:
                arr = self._frames.pop(f)
                self._bytes_used -= arr.nbytes
                self._chain_snapshot.pop(f, None)
            self._epoch += 1

    def _broadest_layer_size(self) -> tuple[int, int]:
        """Largest ``(width, height)`` across the stack's layers.

        Used as the missing-placeholder size when the gap is a true
        no-coverage hole (no layer reaches this master frame) — picking
        the broadest layer keeps the placeholder visually consistent
        with whatever else is on screen during playback. Defaults to
        512² when the stack is empty or every layer reports zero size."""
        widths: list[int] = []
        heights: list[int] = []
        for layer in self._stack.layers():
            w = layer.sequence.width or 0
            h = layer.sequence.height or 0
            if w > 0 and h > 0:
                widths.append(w)
                heights.append(h)
        if not widths:
            return (512, 512)
        return (max(widths), max(heights))

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

    def _channels_for(self, layer: Layer) -> list[str] | None:
        """Per-layer channel selection → flat list for OIIO. ``None``
        defers to the reader's default (R/G/B/A).

        When the layer has ``alpha_composite=True`` AND its source
        actually has an A channel, A is appended to the explicit
        selection so the over operator has alpha data to work with
        even if the user's channel menu is locked on "RGB" / "Y" /
        etc. Per-layer flag means a stack can mix opaque and
        compositing layers without a single global toggle forcing
        the wrong choice.
        """
        sel = layer.channel_selection
        if sel is None:
            base: list[str] | None = None
        else:
            union = list(sel.union_channels())
            base = union or None
        if not layer.alpha_composite:
            return base
        if "A" not in layer.sequence.channel_names:
            return base
        if base is None:
            return None  # reader default already RGBA
        if "A" in base:
            return base
        return base + ["A"]

    def _submit_composite(
        self, master_frame: int, layers: list, priority: int,
    ) -> bool:
        """Enqueue a multi-layer over-composite decode.

        Resolves each contributing layer's source path + capture
        the channel selection at submit time so the worker decodes
        against a stable view of the world. Layers whose source
        has a hole at this master frame are skipped (they don't
        contribute) — the layer below shows through the missing
        slot, which matches the user's mental model of compositing.
        """
        plan: list[dict] = []
        for layer in layers:
            self._ensure_index(layer)
            source_frame = layer.source_frame_at(master_frame)
            path = self._path_index[layer.id].get(source_frame)
            if path is None:
                continue  # sparse hole — skip this layer's contribution
            plan.append({
                "layer_id": layer.id,
                "path": path,
                "channels": self._channels_for(layer),
                "ph_w": layer.sequence.width or 512,
                "ph_h": layer.sequence.height or 512,
                # Per-layer convention so the over operator picks the
                # right pre-blend conversion for each contributor.
                "is_straight": bool(layer.alpha_is_straight),
                # Opaque "floor" layers don't compose — when the walker
                # hits one, treat it as alpha=1 even if its source has
                # an A channel.
                "is_opaque_floor": not layer.alpha_composite,
            })
        # User rule: the checker only shows when the bottom contributor
        # has *no* layer beneath it in the stack (regardless of
        # coverage at this master frame). If any layer sits below it
        # in stack order, transparent regions render as solid black
        # instead — the user has explicitly stacked layers, so the
        # checker would feel like a stand-in for content that isn't
        # there. Black is the honest "no contribution here" value.
        # Implementation: force the bottom plan entry's opaque-floor
        # flag when its layer isn't the stack's last one.
        if plan:
            all_ids = [l.id for l in self._stack.layers()]
            bottom_id = plan[-1]["layer_id"]
            if bottom_id in all_ids:
                stack_idx = all_ids.index(bottom_id)
                has_below = stack_idx < len(all_ids) - 1
                if has_below:
                    plan[-1]["is_opaque_floor"] = True
        if not plan:
            # Every layer had a hole at this frame — pre-mark missing.
            # Use the broadest layer dimensions in the stack so the
            # placeholder matches the displayed image size instead of a
            # fixed 512² square (which got letterboxed weirdly when the
            # actual sequence was 1920×1080 / 4K).
            ph_w, ph_h = self._broadest_layer_size()
            with self._lock:
                placeholder = get_missing_placeholder(ph_w, ph_h)
                self._frames[master_frame] = placeholder
                self._missing.add(master_frame)
                self._record_frame_chain(master_frame)
            return False
        return self._pool.submit(
            priority,
            master_frame,
            lambda: self._decode_composited_and_store(master_frame, plan),
        )

    def _decode_composited_and_store(
        self, master_frame: int, plan: list[dict],
    ) -> None:
        """Worker entry: decode each layer's contribution + over-blend
        front-to-back. The first plan entry is the topmost.

        Per-entry convention flags:
        * ``is_straight`` — input RGB is unpremult; multiply by A
          before the blend so all contributors are premult internally.
        * ``is_opaque_floor`` — the layer doesn't compose; treat it as
          fully opaque (force its alpha to 1.0) so layers below are
          masked. Reached when the walker hit a non-composing layer.

        Math (everything in premult):
            accum = top + arr * (1 - accum.a)
        """
        with self._lock:
            epoch = self._epoch
        try:
            top_plan = plan[0]
            top = read_frame(top_plan["path"], channels=top_plan["channels"])
            top = _ensure_rgba(top)
            if top_plan["is_straight"]:
                top = _premultiply(top)
            if top_plan["is_opaque_floor"]:
                top = _force_alpha_one(top)
            # Quick opacity check on the centre row to avoid a full
            # O(WxH) scan when the top is solid; if it's a floor or
            # already opaque we skip the deeper decodes.
            if top.shape[0] > 0 and float(top[top.shape[0] // 2, :, 3].min()) >= 1.0 - 1e-3:
                accum = top
            else:
                accum = top.copy()
                for layer_plan in plan[1:]:
                    arr = read_frame(layer_plan["path"], channels=layer_plan["channels"])
                    arr = _ensure_rgba(arr)
                    if layer_plan["is_straight"]:
                        arr = _premultiply(arr)
                    if layer_plan["is_opaque_floor"]:
                        arr = _force_alpha_one(arr)
                    inv_a = (1.0 - accum[..., 3:4]).astype(accum.dtype)
                    accum = accum + arr * inv_a
                    if float(accum[..., 3].min()) >= 1.0 - 1e-3:
                        break
        except FrameReadError as err:
            log.warning(
                "composite decode failed master=%d: %s", master_frame, err,
            )
            placeholder = get_missing_placeholder(
                plan[0]["ph_w"], plan[0]["ph_h"],
            )
            with self._lock:
                self._decode_errors += 1
                if epoch != self._epoch:
                    return
                if master_frame in self._frames:
                    return
                self._frames[master_frame] = placeholder
                self._missing.add(master_frame)
                self._record_frame_chain(master_frame)
            return

        with self._lock:
            if epoch != self._epoch:
                return
            if master_frame in self._frames:
                return
            self._frames[master_frame] = accum
            self._bytes_used += accum.nbytes
            self._record_frame_chain(master_frame)
            self._evict_if_over_budget()

    def _decode_and_store(
        self,
        master_frame: int,
        path,
        channels: list[str] | None,
        placeholder_w: int,
        placeholder_h: int,
        strip_alpha: bool = False,
    ) -> None:
        """Worker entry point — runs on a decode thread.

        ``placeholder_w / _h`` are captured at submit time from the
        layer that was the topmost-visible-then; they're used only
        if the decode fails so the placeholder matches the expected
        resolution and the GL viewport doesn't have to rescale a
        random size.

        ``strip_alpha`` removes the A channel from the decoded buffer
        before storing — used when the layer is in opaque mode
        (``alpha_composite=False``) so the GL viewport's checker
        compositing has nothing to react to. Without this, an EXR /
        PNG with an alpha channel would still feed the shader an
        ``alpha < 1`` value and surface the checker even though the
        user explicitly disabled transparency on the layer.
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
                self._record_frame_chain(master_frame)
            return

        if strip_alpha and arr.ndim == 3 and arr.shape[2] == 4:
            # Slice the alpha channel out + ensure the result is
            # contiguous so the GL upload's ``glTexSubImage2D`` is
            # happy with the buffer's stride.
            arr = np.ascontiguousarray(arr[..., :3])

        with self._lock:
            if epoch != self._epoch:
                return  # invalidated mid-decode — drop
            if master_frame in self._frames:
                return  # raced — keep existing
            self._frames[master_frame] = arr
            self._bytes_used += arr.nbytes
            self._record_frame_chain(master_frame)
            self._evict_if_over_budget()

    def _evict_if_over_budget(self) -> None:
        """Distance-from-playhead eviction with a behind-the-playhead
        penalty (= we evict frames the user just played first).

        In LOOP mode (``set_loop_range(enabled=True)``), scoring uses
        **ring distance** instead: ``f``'s score is its forward
        wrap-around distance from the playhead within ``[lo, hi]``.
        This keeps the wrap target (``lo``) cheap to retain when the
        playhead approaches ``hi`` — without it, ``lo`` has the
        highest signed-distance score and gets evicted the instant a
        worker finishes decoding it, locking playback at ``hi``
        forever (the loop never visibly fires). Frames outside the
        loop range fall back to signed-distance scoring (effectively
        evicted first since they're not part of the active ring)."""
        if self._bytes_used <= self._budget:
            return
        cur = self._current_frame
        d = self._direction
        penalty = _BEHIND_PLAYHEAD_PENALTY
        loop_lo = self._loop_lo
        loop_hi = self._loop_hi
        loop_on = (
            self._loop_enabled
            and loop_lo is not None
            and loop_hi is not None
        )
        ring_size = (loop_hi - loop_lo + 1) if loop_on else 0

        def score(f: int) -> float:
            if loop_on and loop_lo <= f <= loop_hi:
                # Ring forward distance — frames "ahead" along the
                # loop direction are cheap to keep, frames already
                # passed (which will only be revisited after a full
                # wrap) sit at distance ``ring_size - 1`` and are
                # evicted first. Reverse direction mirrors the wrap.
                if d >= 0:
                    return float((f - cur) % ring_size)
                return float((cur - f) % ring_size)
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
            self._chain_snapshot.pop(f, None)
            self._evictions += 1
