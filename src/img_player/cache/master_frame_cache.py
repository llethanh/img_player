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

import numpy as np

from img_player.cache._common import (
    BEHIND_PLAYHEAD_PENALTY as _BEHIND_PLAYHEAD_PENALTY,
    CacheStats,
    DEFAULT_BUDGET_BYTES as _DEFAULT_BUDGET_BYTES,
    DEFAULT_NUM_WORKERS as _DEFAULT_NUM_WORKERS,
    expected_filename as _expected_filename,
)
from img_player.cache.disk_cache import DiskCache
from img_player.cache.missing_placeholder import get_missing_placeholder
from img_player.cache.source_key import (
    composite_source_key,
    source_key_for_layer_frame,
)
from img_player.cache.worker_pool import WorkerPool
from img_player.io.reader import FrameReadError, read_frame
from img_player.layers import Layer, LayerStack
from img_player.sequence.models import SequenceInfo

log = logging.getLogger(__name__)
# Frames within this many slots BEHIND the playhead are treated as
# "near-rear" by the eviction scoring: small absolute-distance score
# regardless of loop / play direction. Mirrors the controller's
# ``PlayerController.PREFETCH_BEHIND`` so the cache preserves what
# the prefetcher just queued. Without this, LOOP mode (which scores
# by ring forward distance) flags frames at ``cur - 1`` with the
# *maximum* score and evicts them instantly — turning every quick
# scrub-back into a wait-timer poll.
_NEAR_REAR_WINDOW = 8


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
    if arr.shape[2] == 2:
        # Luminance + alpha (some grayscale-with-mask formats, or a
        # 1-channel pass that picked up an A on the way through). Show
        # the lum as monochrome RGB and keep the second channel as
        # the alpha — matches how the GL viewport already presents
        # broadcasted single-channel reads.
        lum = arr[..., 0:1]
        rgb = np.broadcast_to(lum, (arr.shape[0], arr.shape[1], 3))
        return np.ascontiguousarray(np.concatenate([rgb, arr[..., 1:2]], axis=2))
    raise ValueError(f"Unsupported channel count: {arr.shape[2]}")


def _force_alpha_one(arr: np.ndarray) -> np.ndarray:
    """Set every alpha to 1.0 — used for "opaque floor" layers in the
    composite path so the layers below them are masked, regardless of
    the source's actual A channel."""
    out = arr.copy()
    out[..., 3:4] = np.ones_like(out[..., 3:4])
    return out


def _signature_token(
    layer: Layer,
    *,
    sel_label_override: str | None = None,
    offset_override: int | None = None,
) -> str:
    """Build the per-layer signature token used by the multi-version
    cache key.

    The token encodes everything about a layer that, if changed, would
    invalidate the cached composite at this frame:

    * ``id`` — distinguish two layers backed by the same file
    * ``sel_label`` — channel selection (or alt-channel override)
    * ``alpha_composite`` / ``alpha_is_straight`` flags
    * ``offset`` and ``layer_in`` — together determine which source
      frame the layer contributes at a given master frame

    Used by :meth:`_signature_at`, :meth:`_signature_at_with_override`
    and :meth:`_shift_solo_dominant_frames` — all three previously
    open-coded the same f-string and would drift if anyone touched
    the format.

    ``sel_label_override`` swaps the channel label (alt-channel
    prefetch). ``offset_override`` swaps the offset (drag-shift
    re-keying). Pass ``None`` for live state.
    """
    sel = layer.channel_selection
    sel_label = sel_label_override
    if sel_label is None:
        sel_label = sel.active.label if sel is not None else "_"
    ac = "1" if layer.alpha_composite else "0"
    pm = "1" if layer.alpha_is_straight else "0"
    off = int(layer.offset) if offset_override is None else int(offset_override)
    lin = int(layer.layer_in)
    return f"{layer.id}@{sel_label}#{ac}{pm}+{off}+{lin}"


def _layer_source_token(layer: Layer) -> tuple:
    """Identity token for a layer's underlying source.

    Two layers with the same id but different underlying files
    (= post-"Replace source…") produce different tokens. Used by
    :meth:`MasterFrameCache._on_layer_modified` to detect a source
    swap and invalidate the lazily-built ``_path_index`` (which is
    keyed on layer.id alone and would otherwise keep returning the
    OLD source's paths after a swap).

    For image sequences: ``(directory, base_name, padding,
    extension, len(frames))`` — captures both the location and the
    detected pattern so re-scanning the same source produces an
    identical token (= no spurious invalidation).
    For videos: ``("__video__", path)`` — the path uniquely
    identifies the file; the leading sentinel prevents a future
    image-sequence whose base_name happens to be ``"__video__"``
    from colliding.
    """
    md = layer.video_metadata
    if md is not None:
        return ("__video__", str(md.path))
    seq = layer.sequence
    if seq is None:
        return ()
    return (
        str(seq.directory),
        seq.base_name,
        int(seq.padding),
        seq.extension,
        len(seq.frames),
    )


def _premultiply(arr: np.ndarray, *, inplace: bool = False) -> np.ndarray:
    """Multiply RGB by alpha so straight-alpha buffers can feed the
    same over operator that premult buffers do.

    Pass ``inplace=True`` when the caller owns ``arr`` (e.g. it was
    just produced by ``read_frame``) — at 4K that saves an HxWx4
    float copy per contributor (~64 MiB). The default stays
    out-of-place because callers handed read-only views from the
    disk cache would otherwise raise on the in-place ``*=``.
    """
    out = arr if inplace else arr.copy()
    # ``np.multiply(out=...)`` reads each of the RGB channels and
    # multiplies by the alpha column in one pass — measurably
    # cheaper than the explicit slice-multiply on big buffers.
    np.multiply(out[..., :3], out[..., 3:4], out=out[..., :3])
    return out


class MasterFrameCache:
    """RAM cache keyed on master-timeline frames, resolved via a LayerStack."""

    def __init__(
        self,
        stack: LayerStack,
        budget_bytes: int = _DEFAULT_BUDGET_BYTES,
        num_workers: int = _DEFAULT_NUM_WORKERS,
        disk_cache: DiskCache | None = None,
    ) -> None:
        self._stack = stack
        self._budget = budget_bytes
        self._lock = threading.RLock()
        # Optional second cache tier on disk — survives close/reopen.
        # When set, decode workers check it before hitting ``read_frame``
        # and write successful decodes back so the next session can
        # reuse them. ``None`` keeps the legacy RAM-only behaviour.
        self._disk_cache: DiskCache | None = disk_cache
        # Cache keys are ``(master_frame, chain_signature)`` tuples
        # where the signature captures the visible chain at that
        # master frame *plus* each contributor layer's channel
        # selection. Two consequences:
        #
        # 1. Channel switch / visibility toggle / reorder no longer
        #    invalidates anything: the new state has a different
        #    signature, so lookups under it miss and trigger fresh
        #    decodes, while the old entries linger under their
        #    captured signature. Switching back becomes an instant
        #    cache hit.
        # 2. Eviction (``_evict_if_over_budget``) prefers entries
        #    whose signature no longer matches the *current* state at
        #    that frame — those are "alternative views" the user
        #    isn't currently looking at. When budget pressure forces
        #    drops, we drop snapshots before live frames.
        self._frames: dict[tuple[int, str], np.ndarray] = {}
        self._missing: set[tuple[int, str]] = set()
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
        # Snapshot of each layer's source-identity (sequence
        # directory + basename, or video path) so
        # ``_on_layer_modified`` can detect a "Replace source…"
        # swap — same layer.id, different underlying files. The
        # offset / trim / channel paths above keep working when
        # the source is unchanged; only the new source-swap
        # branch reaches for this dict.
        self._last_known_source: dict[str, tuple] = {}
        # Memoised per-master-frame current signature, invalidated on
        # any stack mutation that could shift the chain (layers_changed
        # / visibility_changed / layer_modified). Without this,
        # ``cached_frames()`` / ``missing_frames()`` (called every
        # 200 ms by the cache-bar timer) would recompute the chain
        # signature for every cached entry — O(n_frames × n_layers)
        # of attribute reads. The memo collapses that to one walk per
        # frame per stack mutation.
        self._signature_cache: dict[int, str] = {}
        # Bumped on every invalidation so workers in flight drop
        # their results when the world has moved on (channel change,
        # visibility flip, layer reorder, …).
        self._epoch = 0
        # Per-signature "last touched" timestamp — bumped every time a
        # signature is requested (live ``request`` or
        # ``request_alt_channel``). Kept for compatibility / debugging;
        # eviction uses :attr:`_signature_first_cached` instead because
        # ``last_seen`` is refreshed by every alt-prefetch pass and so
        # collapses to "all sigs touched ~now" under steady-state
        # browsing, making it useless as an age discriminator.
        self._signature_last_seen: dict[str, float] = {}
        # Per-signature "first cached" timestamp — set **exactly once**
        # the moment the first real frame for that signature lands in
        # :attr:`_frames`. Used by :meth:`_evict_if_over_budget` to
        # drop the **oldest cached** alt-channel group first (FIFO over
        # channel cache history) rather than carving a slice off every
        # group at once.
        #
        # Why not reuse ``_signature_last_seen``: that one gets bumped
        # by every alt-prefetch tick, which fires on every pause for
        # every visible layer × every channel group. The result is
        # that all alt signatures end up with near-identical
        # timestamps under steady browsing — the age comparison
        # collapses, the tertiary signature-string sort takes over,
        # and eviction starts deciding "channel order" lexicographically
        # ("emission" before "albedo" before "RGBA" purely because of
        # ASCII ordering). ``_signature_first_cached`` is immutable
        # per sig, so first-in-first-out is preserved even under
        # heavy prefetch traffic.
        #
        # Lifetime: stamped on first frame insert under the sig (real
        # decodes, not missing-frame placeholders). NOT cleared when
        # a sig is fully evicted — a sig that comes back into cache
        # after a full eviction keeps its original "first cached"
        # rank, which is the intuitive behaviour (the channel the
        # user looked at first stays the channel-to-drop-first).
        self._signature_first_cached: dict[str, float] = {}
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
        # Seed ``_last_known_*`` from layers already in the stack at
        # construction time. Without this, layers added pre-cache
        # never populate the snapshot dicts (``layers_changed`` fires
        # before we connect), and the first ``_on_layer_modified``
        # call sees ``prev_state == None`` — which falls through to
        # the trim branch and misses the pure-offset-shift fast path.
        # Idempotent + cheap: no signals fired, no cache mutation.
        for layer in self._stack.layers():
            self._last_known_range[layer.id] = (
                layer.master_start, layer.master_end,
            )
            self._last_known_state[layer.id] = (
                layer.offset, layer.layer_in, layer.layer_out,
                layer.channel_selection,
                layer.alpha_composite, layer.alpha_is_straight,
            )

    # ------------------------------------------------------------------ Lifecycle

    def disk_cache_pending_writes(self) -> int:
        """Pending disk-cache writes (0 if no disk tier). Surface for app.

        Used at shutdown to decide whether to show a "flushing"
        progress widget before blocking on the disk-cache drain.
        """
        if self._disk_cache is None:
            return 0
        try:
            return self._disk_cache.pending_writes()
        except Exception:  # pragma: no cover — defensive
            return 0

    def shutdown(self, disk_progress_callback=None) -> None:  # type: ignore[no-untyped-def]
        """Stop the worker pool **and flush every cached frame**.

        Without the explicit flush the decoded numpy arrays sit in
        ``_frames`` until the Python GC tears down the cache object —
        which on Windows can stall the process exit for a noticeable
        second or two when the cache holds millions of small RGBA
        buffers. Dropping the dict here makes the exit snappy and
        gives the OS the RAM back deterministically. Idempotent: a
        second call after shutdown sees an empty dict and no-ops.

        ``disk_progress_callback`` (optional) is forwarded to
        :meth:`DiskCache.shutdown` — receives the pending-write count
        every ~100 ms during the drain. The app uses it to refresh a
        small "flushing disk cache (N pending)" label so the user
        understands the few-second pause at exit.
        """
        self._pool.shutdown()
        # Flush pending disk-cache writes before clearing the in-RAM
        # arrays they may still reference. ``DiskCache.shutdown`` has
        # a 10 s budget — generous enough for a ~150 frame backlog at
        # ~50 ms / blob, short enough to bail out if something hangs.
        if self._disk_cache is not None:
            try:
                self._disk_cache.shutdown(
                    progress_callback=disk_progress_callback,
                )
            except Exception:  # pragma: no cover — defensive
                log.exception("DiskCache shutdown failed (non-fatal)")
        with self._lock:
            self._frames.clear()
            self._missing.clear()
            self._signature_cache.clear()
            self._signature_last_seen.clear()
            self._signature_first_cached.clear()
            self._path_index.clear()
            self._mtime_index.clear()
            self._last_known_range.clear()
            self._last_known_state.clear()
            self._last_known_source.clear()
            self._bytes_used = 0

    def clear(self) -> None:
        """Drop every cached frame + bump epoch so in-flight decodes
        get discarded at store time."""
        self._pool.clear()
        with self._lock:
            self._frames.clear()
            self._missing.clear()
            self._signature_cache.clear()
            self._signature_last_seen.clear()
            self._signature_first_cached.clear()
            self._bytes_used = 0
            self._epoch += 1

    def invalidate_layer_paths(self, layer_id: str) -> None:
        """Drop the cached path / mtime index for one layer.

        Called from the app's "Replace source on layer" handler:
        ``layer.id`` is preserved across the swap but the underlying
        ``sequence.frames`` points at a different set of files.
        :meth:`_ensure_indexed` skips re-indexing when ``layer.id``
        is already a key in ``_path_index`` — so without this drop,
        every subsequent decode reads from the OLD source's paths
        even though ``layer.sequence`` is the new one.

        Idempotent: a layer never indexed (= cache never accessed
        it) is a no-op.
        """
        with self._lock:
            self._path_index.pop(layer_id, None)
            self._mtime_index.pop(layer_id, None)
            # Drop the per-layer state snapshots too so the next
            # ``_on_layer_modified`` diff doesn't treat the swap as
            # a "small mutation" against the OLD baseline.
            self._last_known_range.pop(layer_id, None)
            self._last_known_state.pop(layer_id, None)
            self._last_known_source.pop(layer_id, None)

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
            # Route 1-frame sequences to ``Layer.from_still`` so a
            # single dropped image (slate, ref) becomes a held still
            # layer rather than a 1-frame sequence the user can't
            # scrub. ``attach`` only fires when the stack is being
            # replaced (the call wipes every existing layer first), so
            # the default-hold heuristic doesn't have other layers to
            # match — fall back to a project-wide default. Caller can
            # bump the hold afterwards via the layer panel.
            layer = Layer.from_image(
                sequence,
                default_still_hold=100,
                offset=sequence.first_frame,
            )
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
        first layer if none is focused). Triggers ``layer_modified``,
        which the cache handles passively now (multi-version key) —
        the existing entries become stale snapshots accessible via
        their captured signature, and lookups under the new selection
        miss + queue fresh decodes. ``channels`` is either ``None``
        (reader default) or a list of names.
        """
        from img_player.sequence.channels import (
            ChannelGroup,
            ChannelSelection,
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

    def pending_decodes(self) -> int:
        """Number of decode tasks queued or in flight. Mirrors
        :meth:`FrameCache.pending_decodes` so :class:`PlayerController`
        can duck-type both cache flavours."""
        return self._pool.pending()

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
        # With multi-version keys we drop EVERY signature variant
        # for the affected master frame — a disk-mtime change
        # invalidates every cached snapshot of that frame, regardless
        # of which channels they captured.
        with self._lock:
            for source_frame in set(old_mtimes) | set(new_mtimes):
                master_frame = target.offset + (source_frame - target.layer_in)
                old_mt = old_mtimes.get(source_frame, 0.0)
                new_mt = new_mtimes.get(source_frame, 0.0)
                # Find every signature variant cached for this master frame.
                variants = [
                    k for k in self._frames if k[0] == master_frame
                ]
                if old_mt == new_mt and source_frame in new_paths:
                    # File unchanged on disk: keep all variants.
                    if any(k not in self._missing for k in variants):
                        kept += 1
                    continue
                # File changed / disappeared: drop all variants.
                for k in variants:
                    arr = self._frames.pop(k, None)
                    if arr is not None and k not in self._missing:
                        self._bytes_used -= arr.nbytes
                        dropped += 1
                    self._missing.discard(k)
            self._epoch += 1
            # Mtime delta means coverage / signature relevance might
            # have shifted (e.g. a new file appeared where a missing
            # placeholder lived); reset the memo.
            self._invalidate_signature_cache()
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
            1 for k in self._missing
            if target.master_start <= k[0] <= target.master_end
        )
        return (kept, dropped, missing)

    # ------------------------------------------------------------------ Public read API

    def get(self, master_frame: int) -> np.ndarray | None:
        """Non-blocking fetch. ``None`` when the frame is not cached.

        Looks up under the current chain × channel signature so a
        stale snapshot (different channels, hidden layer that's now
        back, …) is treated as a miss even though its bytes still
        live in ``_frames`` under another signature key. Updates the
        playhead position so the next eviction round scores frames
        against this center.
        """
        with self._lock:
            self._current_frame = master_frame
            sig = self._signature_at(master_frame)
            arr = self._frames.get((master_frame, sig))
            if arr is not None:
                self._hits += 1
                return arr
            self._misses += 1
            return None

    def contains(self, master_frame: int) -> bool:
        with self._lock:
            sig = self._signature_at(master_frame)
            return (master_frame, sig) in self._frames

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
        """Master frames cached *under the current signature* — i.e.
        what the timeline cache bar should paint as ready for the
        currently-displayed view. Stale snapshots (different channels
        / visibility) live in ``_frames`` too but aren't surfaced
        here: the cache bar reflects the live view, not memory
        accounting.

        Called every ~200 ms by the cache-bar refresh timer, so we
        snapshot all the keys + signatures under **one** lock
        acquisition rather than reacquiring per key. ``_signature_at``
        reads the layer stack (which has its own locking) but the
        master-cache lock guards ``_frames`` only — keeping it held
        for the whole walk is safe because the signature memo is
        managed inside ``_signature_at`` itself.
        """
        with self._lock:
            # Snapshot to a tuple so any concurrent mutation of
            # ``_frames`` after the lock release doesn't matter.
            keys = tuple(self._frames.keys())
            result: set[int] = set()
            for mf, sig in keys:
                if self._signature_at(mf) == sig:
                    result.add(mf)
        return frozenset(result)

    def missing_frames(self) -> frozenset[int]:
        """Master frames whose decode failed (file missing /
        unreadable) under the current signature. Same filter rule
        as :meth:`cached_frames`."""
        with self._lock:
            keys = tuple(self._missing)
            result: set[int] = set()
            for mf, sig in keys:
                if self._signature_at(mf) == sig:
                    result.add(mf)
        return frozenset(result)

    def disk_available_master_frames(self) -> frozenset[int]:
        """Master frames whose **disk** blob is available for the
        current chain × channel state.

        Per-contributor semantics since 1.5.4 — a frame counts as
        "disk available" iff **every** contributing layer at that
        master frame has its individual blob on disk. The composite
        is rebuilt on the fly at decode time from those per-layer
        blobs (cheap over-blend), so the pre-paint indicator is
        truthful: a green dim-orange means the next scrub will hit
        the fast path for that frame.

        Used at session-load to pre-paint the timeline cache bar so
        the user sees the shot is warm on disk before scrubbing.
        Cheap one-shot operation: O(N × L) hash computes + one
        SQLite bulk-IN query (typically ~50 ms for a 1000-frame
        single-layer sequence; ~100 ms for a 3-layer composite).

        Returns an empty set when:
          * Disk cache is disabled / not configured.
          * The stack has no covered range.
          * No frame's source path / mtime is yet indexed.
        """
        if self._disk_cache is None or self._stack is None:
            return frozenset()
        first, last = self._stack.master_range()
        if last <= first:
            return frozenset()
        # ``_ensure_index`` is idempotent + cheap; force-build the
        # path/mtime index for every layer in the stack before walking
        # the range so we don't bail out per-frame.
        for layer in self._stack.layers():
            self._ensure_index(layer)
        per_frame_keys: dict[int, list[str]] = {}
        all_keys: set[str] = set()
        for master_frame in range(first, last + 1):
            keys: list[str] = []
            complete = True
            for layer in self._stack.layers():
                if not layer.visible or not layer.covers(master_frame):
                    continue
                channels = self._channels_for(layer)
                key = self._source_key_for_single_layer(
                    layer, master_frame, channels,
                )
                if key is None:
                    complete = False
                    break
                keys.append(key)
                if not layer.alpha_composite:
                    # Stop at the first opaque-floor layer — the
                    # ``over`` walk wouldn't read further anyway.
                    break
            if complete and keys:
                per_frame_keys[master_frame] = keys
                all_keys.update(keys)
        if not all_keys:
            return frozenset()
        existing = self._disk_cache.contains_keys(all_keys)
        return frozenset(
            mf for mf, keys in per_frame_keys.items()
            if all(k in existing for k in keys)
        )

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
        already cached or no layer covers this master frame.

        Dedups against the ``(master_frame, current_signature)`` key
        — a frame cached under a stale signature (e.g. previous
        channel selection) doesn't block a fresh decode under the
        new state."""
        with self._lock:
            sig = self._signature_at(master_frame)
            # Stamp the LIVE signature so eviction always sees it as
            # "more recent" than any stored alt-channel signature.
            # Keeps tier 0 (alts) evictable before tier 2 (live), even
            # when the user idles on the live channel for a while.
            self._signature_last_seen[sig] = time.monotonic()
            if (master_frame, sig) in self._frames:
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
        # Still-image layers: every master frame in the hold range
        # decodes the SAME file. Decode it once at the layer's
        # ``master_start`` (canonical "anchor" frame) and alias the
        # resulting ndarray into every other slot the request sweeps
        # over — zero redundant OIIO opens, zero extra memory (Python
        # ref-shares the buffer). Eviction may drop aliases
        # independently of the anchor; that's fine, the next request
        # re-aliases on demand.
        if getattr(topmost, "is_still", False):
            anchor = topmost.master_start
            with self._lock:
                anchor_sig = self._signature_at(anchor)
                anchor_arr = self._frames.get((anchor, anchor_sig))
            if anchor_arr is not None:
                if master_frame != anchor:
                    with self._lock:
                        cur_sig = self._signature_at(master_frame)
                        self._frames[(master_frame, cur_sig)] = anchor_arr
                return False
            if master_frame != anchor:
                # Anchor not decoded yet — drive the decode toward
                # ``anchor`` so subsequent frames in the hold range
                # can free-ride. The next request for any frame in
                # range will alias on hit.
                return self.request(anchor, priority)
            # Fallthrough: master_frame == anchor → real decode.
        if not topmost.alpha_composite:
            # Topmost is opaque — single-decode fast path.
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
            # hole there (sparse sequence). Pre-mark missing — using
            # the *shared* (memoised) placeholder rather than the
            # per-frame variant. The filename-baked overlay is meant
            # for "this file failed to decode" cases where the user
            # needs to know which path is bad; for a sparse-sequence
            # gap the absence is intentional / expected and a fresh
            # ~33 MB array per gap (each with its own QImage paint
            # pass) freezes the UI for seconds when load_sequence
            # primes a sparse range. The shared placeholder is built
            # once per (w, h) and aliased across every gap — drop
            # from O(gaps × decode-cost) to O(1).
            with self._lock:
                # Source + master frame numbers are baked into the
                # overlay so the placeholder mirrors the info-band's
                # ``Layer 220 / 1140`` + ``Frame 1101 / 1140``
                # breakdown. Costs one per-call ndarray + QPainter
                # pass (~30-50 ms at HD), but the cache stores the
                # result keyed on ``master_frame`` so subsequent
                # lookups of the same hole hit the stored buffer
                # directly — no rebuild on repeated visits.
                master_first, master_last = self._stack.master_range()
                placeholder = get_missing_placeholder(
                    layer.sequence.width or 512,
                    layer.sequence.height or 512,
                    frame_number=master_frame,
                    frame_max=master_last if master_last else None,
                    source_frame=layer.source_frame_at(master_frame),
                    source_max=layer.layer_out,
                )
                key = (master_frame, self._signature_at(master_frame))
                self._frames[key] = placeholder
                self._missing.add(key)
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
        # Capture the signature *now* so the worker stores its
        # result under the state that motivated the submit. If the
        # user toggles channels between submit and store, the in-
        # flight buffer lands under the original signature (still
        # accessible if they switch back) and a fresh request fires
        # under the new signature. Same idea for ``disk_key``: it
        # must be computed from the captured channel selection so
        # the disk cache writes/reads agree on what's in the blob
        # (the live state at decode time can differ — e.g. the user
        # switched channels while the worker was queued).
        with self._lock:
            sig = self._signature_at(master_frame)
            disk_key = self._source_key_for_single_layer(
                layer, master_frame, channels,
            )
        return self._pool.submit(
            priority,
            (master_frame, sig),
            lambda: self._decode_and_store(
                master_frame, sig, path, channels, ph_w, ph_h,
                strip_alpha=strip_alpha,
                disk_key=disk_key,
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

    # ------------------------------------------------------------------ Alt-channel background prefetch

    def request_alt_channel(
        self,
        layer_id: str,
        master_frame: int,
        alt_channels: tuple[str, ...],
        alt_label: str,
        priority: int,
    ) -> bool:
        """Submit a decode of ``layer_id`` at ``master_frame`` using
        ``alt_channels`` (instead of the layer's live selection).

        Single-layer stacks take the cheap single-decode path; multi-
        layer stacks go through the composite path with an override
        on this layer's channels — both store under the synthetic
        signature ``_signature_at_with_override`` produces, so a
        future channel switch on this layer hits the cache regardless
        of how many other layers it composites with.

        Returns ``False`` when the layer can't contribute (hidden,
        doesn't cover, video, still, sparse hole), when the focused
        layer is masked above (e.g. an opaque-floor topmost), or
        when the entry is already cached / pending.

        ``priority`` should be very large (alt_base + offset) so
        these decodes only run when the worker pool has nothing of
        active priority left to do.
        """
        # Throttle when RAM is already saturated. Without this gate
        # the worker pool keeps churning through alt-channel decodes
        # that are immediately evicted (tier 0) the moment they land
        # — burning CPU + GIL + I/O for nothing AND starving the live
        # signature, which sees its own frames pushed past the budget
        # by the inflight alts. 92 % rather than 100 % so we leave
        # headroom for the live decodes already in flight to finish
        # without immediately tripping eviction either.
        if self._bytes_used >= int(self._budget * 0.92):
            return False
        layer = self._stack.find(layer_id)
        if layer is None or not layer.visible or not layer.covers(master_frame):
            return False
        if getattr(layer, "is_video", False):
            return False
        if getattr(layer, "is_still", False):
            # Stills resolve to one anchor frame regardless of mf —
            # alt prefetch already handled by the still fan-out path
            # the moment the anchor is decoded. Skip the per-frame
            # repeats so we don't burn priority slots on no-ops.
            return False
        sig = self._signature_at_with_override(
            master_frame, layer_id, alt_label,
        )
        # Stamp the signature as "freshly requested" so it survives
        # eviction longer than older alt groups. See
        # :attr:`_signature_last_seen` for the rationale.
        self._signature_last_seen[sig] = time.monotonic()
        key = (master_frame, sig)
        with self._lock:
            if key in self._frames:
                return False

        # Walk the visible+covering chain at this master frame; stop
        # at the first opaque-floor (= what the regular composite
        # path does in ``request``).
        visible = [
            l for l in self._stack.layers()
            if l.visible and l.covers(master_frame)
            and not getattr(l, "is_video", False)
        ]
        chain: list[Layer] = []
        for l in visible:
            chain.append(l)
            if not l.alpha_composite:
                break
        # Focused layer must be in the chain — if it's masked above
        # by an opaque floor, no alt decode of it would change the
        # composite the user sees, so caching is pointless.
        if not any(l.id == layer_id for l in chain):
            return False

        # Single-layer fast path: chain is just the focused layer
        # AND it's an opaque floor (= no compositing needed). One
        # OIIO read straight into the cache.
        if (
            len(chain) == 1
            and chain[0].id == layer_id
            and not chain[0].alpha_composite
        ):
            source_frame = layer.source_frame_at(master_frame)
            self._ensure_index(layer)
            path = self._path_index[layer.id].get(source_frame)
            if path is None:
                return False
            channels = self._channels_for_with_override(layer, alt_channels)
            ph_w = layer.sequence.width or 512
            ph_h = layer.sequence.height or 512
            strip_alpha = not layer.alpha_composite
            # Same submit-time capture as the live decode path —
            # the disk key uses the *override* channels we're about
            # to decode with, not the live state. Without this the
            # alt-channel prefetch would write each blob under the
            # live channel's source key in the disk cache, scrambling
            # cross-session lookups.
            disk_key = self._source_key_for_single_layer(
                layer, master_frame, channels,
            )
            return self._pool.submit(
                priority,
                key,
                self._make_alt_task(
                    lambda: self._decode_and_store(
                        master_frame, sig, path, channels, ph_w, ph_h,
                        strip_alpha=strip_alpha,
                        disk_key=disk_key,
                    ),
                ),
            )

        # Composite path with override. Mirrors ``_submit_composite``'s
        # plan-building loop but with the focused layer's channels
        # swapped to ``alt_channels``.
        plan = self._build_composite_plan_with_override(
            master_frame, chain, layer_id, alt_channels,
        )
        if not plan:
            return False
        # Same opaque-floor-on-bottom rule as the regular composite
        # path (transparency BG only shows when no other visible
        # layer covers below the chain).
        covering_ids = [
            l.id for l in self._stack.layers()
            if l.visible and l.covers(master_frame)
        ]
        bottom_id = plan[-1]["layer_id"]
        if bottom_id in covering_ids:
            stack_idx = covering_ids.index(bottom_id)
            if stack_idx < len(covering_ids) - 1:
                plan[-1]["is_opaque_floor"] = True
        # Per-layer disk keys live inside each ``plan`` entry now
        # (built in ``_build_composite_plan_with_override``), so no
        # composite-level key needs to be captured here.
        return self._pool.submit(
            priority,
            key,
            self._make_alt_task(
                lambda: self._decode_composited_and_store(
                    master_frame, sig, plan,
                ),
            ),
        )

    def _make_alt_task(self, decode_fn):  # type: ignore[no-untyped-def]
        """Wrap an alt-channel decode lambda with a dequeue-time budget
        check. The pool may hold hundreds of alt tasks submitted before
        the cache filled up; without this gate the worker keeps
        decoding them (and the cache evicts them immediately on
        store), burning CPU + GIL + I/O while the user sees the RAM
        ticking up. Re-checking here lets queued-but-not-yet-run alt
        tasks self-cancel once budget is reached.

        Live (non-alt) decodes don't go through this path — they keep
        the existing direct submission and always run, since they're
        what the user is currently looking at.
        """
        threshold = int(self._budget * 0.92)

        def _gated() -> None:
            if self._bytes_used >= threshold:
                return
            decode_fn()

        return _gated

    def _build_composite_plan_with_override(
        self,
        master_frame: int,
        chain: list[Layer],
        override_layer_id: str,
        override_channels: tuple[str, ...],
    ) -> list[dict]:
        """Mirror of the plan-building loop inside
        :meth:`_submit_composite`, but the layer matching
        ``override_layer_id`` reads ``override_channels`` instead of
        its live ``channel_selection``.

        Used by :meth:`request_alt_channel` to pre-cache an
        alt-channel composite at the synthetic signature the user's
        future channel switch would compute.
        """
        plan: list[dict] = []
        for layer in chain:
            self._ensure_index(layer)
            source_frame = layer.source_frame_at(master_frame)
            path = self._path_index[layer.id].get(source_frame)
            if path is None:
                continue
            if layer.id == override_layer_id:
                channels = self._channels_for_with_override(
                    layer, override_channels,
                )
            else:
                channels = self._channels_for(layer)
            plan.append({
                "layer_id": layer.id,
                "path": path,
                "channels": channels,
                "ph_w": layer.sequence.width or 512,
                "ph_h": layer.sequence.height or 512,
                "is_straight": bool(layer.alpha_is_straight),
                "is_opaque_floor": not layer.alpha_composite,
                # Same per-contributor disk key as the regular submit
                # path (see :meth:`_submit_composite`). ``channels``
                # here already has the focused-layer override applied,
                # so the key naturally points at the alt-channel
                # blob the worker will produce.
                "disk_key": self._source_key_for_single_layer(
                    layer, master_frame, channels,
                ),
            })
        return plan

    def alt_channel_groups_for_focused(
        self,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """List the focused layer's channel groups *other than the
        active one*, ordered by ``group_channels`` (RGB first, then
        AOVs in their natural order).

        Used by the channel-menu UI to show per-group cache fill
        for the layer the menu is currently configuring (= the
        focused layer). Alt prefetch itself iterates every visible
        layer (see :meth:`visible_alt_layer_ids`).
        """
        layer = self._focused_alt_layer()
        if layer is None:
            return ()
        return self._alt_groups_for_layer(layer)

    def alt_channel_groups_for_layer_id(
        self, layer_id: str,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Like :meth:`alt_channel_groups_for_focused` but for an
        arbitrary layer in the stack. Used by the controller when
        queueing alt prefetch for every visible layer (top-down).
        Empty when the layer isn't found / has no alt groups."""
        layer = self._stack.find(layer_id)
        if layer is None:
            return ()
        return self._alt_groups_for_layer(layer)

    def _alt_groups_for_layer(
        self, layer: Layer,
    ) -> tuple[tuple[str, tuple[str, ...]], ...]:
        from img_player.sequence.channels import group_channels
        groups = group_channels(layer.sequence.channel_names)
        if len(groups) <= 1:
            return ()
        active_label = (
            layer.channel_selection.active.label
            if layer.channel_selection is not None else None
        )
        out: list[tuple[str, tuple[str, ...]]] = []
        for grp in groups:
            if grp.label == active_label:
                continue
            out.append((grp.label, grp.channels))
        return tuple(out)

    def visible_alt_layer_ids(self) -> tuple[str, ...]:
        """Layer ids the alt-channel prefetch should target, ordered
        top-down (= stack order, which the user reads as priority).

        Filters out hidden / video / still layers because none of
        them benefit from alt-channel pre-decode (hidden = invisible
        to the user; video = not OIIO; still = a single anchor frame
        already covers the whole hold range via fan-out).

        When the result is a single id the original "focused layer
        only" behaviour is preserved (= caching for the layer that
        actually drives the viewport, even if it's at the bottom of
        the stack)."""
        out: list[str] = []
        for layer in self._stack.layers():
            if not layer.visible:
                continue
            if getattr(layer, "is_video", False):
                continue
            if getattr(layer, "is_still", False):
                continue
            out.append(layer.id)
        return tuple(out)

    def layer_master_range(self, layer_id: str) -> tuple[int, int] | None:
        """``(first, last)`` master-frame range covered by the
        layer, or ``None`` when the layer isn't in the stack."""
        layer = self._stack.find(layer_id)
        if layer is None:
            return None
        return (layer.master_start, layer.master_end)

    def focused_layer_master_range(self) -> tuple[int, int] | None:
        """``(first, last)`` master-frame range covered by the
        focused (single-layer) layer, or ``None`` when alt prefetch
        doesn't apply."""
        layer = self._focused_alt_layer()
        if layer is None:
            return None
        return (layer.master_start, layer.master_end)

    def focused_layer_id_for_alt(self) -> str | None:
        """Layer id the alt-channel prefetch should target, or
        ``None`` when alt prefetch doesn't apply (multi-layer stack
        / empty stack). Wrapped here so callers don't reach through
        ``_stack``."""
        layer = self._focused_alt_layer()
        return layer.id if layer is not None else None

    def _focused_alt_layer(self) -> Layer | None:
        """Resolve the focused layer for alt-channel prefetch — the
        layer whose channel groups the menu drives.

        Multi-layer stacks are supported: alt decodes go through the
        composite path with an override on this layer's channels, so
        the cached composite matches the signature a future channel
        switch on this layer would compute.

        Falls back to the first layer when no focus is set (= the
        immediately-after-attach state on a freshly-loaded sequence).
        ``None`` only when the stack is empty.
        """
        focused = self._stack.focused()
        if focused is not None:
            return focused
        layers = self._stack.layers()
        if not layers:
            return None
        return layers[0]

    def alt_channel_progress(self) -> dict[str, tuple[int, int]]:
        """Per-channel-group cache fill ``(cached, total)`` for the
        focused layer. Includes the active group (so the channel
        menu can show its progress too).

        Iterates every channel group × every covered master frame
        and counts how many ``(mf, signature)`` entries are present
        in ``_frames``. O(groups × frames) per call — typically a
        few thousand dict lookups, fine for the menu's
        ``aboutToShow`` poll cadence (every 200 ms while the menu
        is open). Empty when alt prefetch doesn't apply (multi-layer
        stack, no groups, empty stack).
        """
        layer = self._focused_alt_layer()
        if layer is None:
            return {}
        from img_player.sequence.channels import group_channels
        groups = group_channels(layer.sequence.channel_names)
        if not groups:
            return {}
        first, last = layer.master_start, layer.master_end
        total = last - first + 1
        if total <= 0:
            return {}
        # When the focused layer is hidden, every override signature
        # collapses to "same as live" (``_signature_at_with_override``
        # skips invisible layers — the override has no effect on the
        # composite). Every group's lookup would then hit the live
        # composite's entry, painting every bar as fully cached even
        # though ``visible_alt_layer_ids`` already excluded the layer
        # from prefetch. Surface the truth: nothing is cached for a
        # hidden layer's alt channels. Bars render empty.
        if not layer.visible:
            return {grp.label: (0, total) for grp in groups}
        active_label = (
            layer.channel_selection.active.label
            if layer.channel_selection is not None else None
        )
        out: dict[str, tuple[int, int]] = {}
        with self._lock:
            for grp in groups:
                count = 0
                for mf in range(first, last + 1):
                    if grp.label == active_label:
                        sig = self._signature_at(mf)
                    else:
                        sig = self._signature_at_with_override(
                            mf, layer.id, grp.label,
                        )
                    key = (mf, sig)
                    # Exclude missing-frame placeholders: they live in
                    # ``_frames`` (so the GL pipeline can paint the
                    # checker without crashing) but they're tracked in
                    # ``_missing`` and **not** counted in
                    # ``_bytes_used``. Without this filter the per-
                    # channel progress bars would claim "fully cached"
                    # for groups where every alt-prefetch had failed /
                    # produced a placeholder, even though no real RAM
                    # had been spent on them — making the bars
                    # contradict the RAM gauge ("51 GB / 51 GB but
                    # all 23 channels look 100% cached").
                    if key in self._frames and key not in self._missing:
                        count += 1
                out[grp.label] = (count, total)
        return out

    def _signature_at(self, master_frame: int) -> str:
        """Stable string id for the chain × channel-selection tuple
        that would produce the pixels at ``master_frame`` under the
        current stack state.

        Cache entries are keyed by ``(master_frame, signature)`` so
        a channel switch / visibility toggle / reorder leaves the
        old decoded buffer accessible under its captured signature
        — switching back becomes a hit instead of a full re-decode.

        Memoised in ``_signature_cache``; invalidated by every
        stack-mutation handler.
        """
        cached = self._signature_cache.get(master_frame)
        if cached is not None:
            return cached
        # Per-contributor tokens encode everything that would invalidate
        # the composite at this frame: layer id, channel label,
        # alpha-composite / straight flags, offset, layer_in. See
        # :func:`_signature_token` for the full rationale.
        parts: list[str] = []
        for layer in self._stack.layers():
            if not layer.visible or not layer.covers(master_frame):
                continue
            parts.append(_signature_token(layer))
            if not layer.alpha_composite:
                break
        sig = "|".join(parts)
        self._signature_cache[master_frame] = sig
        return sig

    def _invalidate_signature_cache(self) -> None:
        """Clear the per-frame signature memo — call from any path
        that mutates the layer stack (visibility, channels, reorder,
        add / remove, trim)."""
        self._signature_cache.clear()

    def _source_key_for_single_layer(
        self,
        layer: Layer,
        master_frame: int,
        channels: list[str] | None,
    ) -> str | None:
        """Compute the disk-cache key for one layer's contribution at
        ``master_frame`` using **explicit** ``channels``.

        Bypasses ``layer.channel_selection`` (= live state) so the
        key matches what the decode worker will actually decode. This
        is what callers should use at **submit time** to capture a
        stable key alongside the captured ``signature``; using the
        live state at decode time race-conditions with the user
        toggling channels (or with the alt-channel prefetch which
        decodes one channel while the live state is on another).
        """
        if self._disk_cache is None:
            return None
        source_frame = layer.source_frame_at(master_frame)
        path_map = self._path_index.get(layer.id)
        mtime_map = self._mtime_index.get(layer.id)
        if path_map is None or mtime_map is None:
            return None
        path = path_map.get(source_frame)
        mtime = mtime_map.get(source_frame)
        if path is None or mtime is None:
            return None
        return source_key_for_layer_frame(
            source_path=path,
            mtime=float(mtime),
            size=0,
            channels=channels,
            alpha_composite=bool(layer.alpha_composite),
            alpha_is_straight=bool(layer.alpha_is_straight),
        )

    def _signature_at_with_override(
        self,
        master_frame: int,
        override_layer_id: str,
        override_label: str,
    ) -> str:
        """Same as :meth:`_signature_at` but with one layer's channel
        label swapped to ``override_label``.

        Used by the background alt-channel prefetch to compute the
        synthetic signature an alt-channel snapshot would land
        under — the user's future channel switch to that group
        would compute the same key and hit the cache.

        Not memoised: alt prefetch is rare-path; keeping the memo
        keyed only on the live state avoids cross-contamination.
        """
        parts: list[str] = []
        for layer in self._stack.layers():
            if not layer.visible or not layer.covers(master_frame):
                continue
            override = override_label if layer.id == override_layer_id else None
            parts.append(_signature_token(layer, sel_label_override=override))
            if not layer.alpha_composite:
                break
        return "|".join(parts)

    def _channels_for_with_override(
        self, layer: Layer, override_channels: tuple[str, ...],
    ) -> list[str] | None:
        """Mirror of :meth:`_channels_for` but with the layer's
        ``channel_selection.active.channels`` replaced by
        ``override_channels`` (without mutating the layer).

        Used by the alt-channel prefetch to ask OIIO for a different
        channel set than the layer's live selection.
        """
        available = set(layer.sequence.channel_names)
        picked = [c for c in override_channels if c in available]
        base: list[str] | None = picked or None
        if not layer.alpha_composite:
            return base
        if "A" not in available:
            if base is None:
                return None
            stripped = [c for c in base if c != "A"]
            return stripped or None
        if base is None:
            return None
        if len(base) >= 4:
            return base
        if len(base) == 1:
            return base
        if "A" in base:
            return base
        return base + ["A"]

    # ------------------------------------------------------------------ Stack signals → invalidation

    def _on_layers_changed(self) -> None:
        """Stack composition mutated (add / remove / reorder).

        With the multi-version cache key, most "this changed the
        chain at frame F" cases are handled passively: lookups under
        the new signature miss → fresh decode; the old entries
        linger as alternative-view snapshots until eviction reaps
        them. So we no longer drop frames here.

        Two remaining concerns:

        * Layers that have been **removed** from the stack leave
          their ids in stale signatures forever — those entries are
          unreachable via ``_signature_at`` (which only walks live
          layers), so they'd just consume budget until evicted. Drop
          them eagerly.
        * In-flight decodes captured a signature against the *old*
          stack — they're not necessarily wrong (their key still
          encodes the world they decoded), but ``pool.clear()``
          opens the dedup slots so the controller's follow-up
          ``replan_prefetch`` can re-submit under the new active
          signature.
        """
        live_ids = {layer.id for layer in self._stack.layers()}
        self._invalidate_signature_cache()

        with self._lock:
            cached_keys = list(self._frames.keys())

        def sig_refs_dead_layer(sig: str) -> bool:
            if not sig:
                return False
            for token in sig.split("|"):
                lid = token.split("@", 1)[0]
                if lid and lid not in live_ids:
                    return True
            return False

        keys_to_drop = [
            k for k in cached_keys if sig_refs_dead_layer(k[1])
        ]

        self._pool.clear()
        with self._lock:
            for k in keys_to_drop:
                arr = self._frames.pop(k, None)
                if arr is not None and k not in self._missing:
                    self._bytes_used -= arr.nbytes
                self._missing.discard(k)
            self._epoch += 1

        # Drop indexes for layers that no longer exist; rebuild for
        # layers that do.
        for stale_id in list(self._path_index.keys()):
            if stale_id not in live_ids:
                self._path_index.pop(stale_id, None)
                self._mtime_index.pop(stale_id, None)
                self._last_known_range.pop(stale_id, None)
                self._last_known_source.pop(stale_id, None)
        # Source-swap detection on LIVE layers — covers the undo /
        # redo path (LayerStack._restore fires layers_changed, NOT
        # layer_modified). A layer kept its id but its underlying
        # sequence may have flipped between sessions of the undo
        # stack. Drop the lazily-built path_index for any such
        # layer + clear all cached frames so the next decode reads
        # the new files. The catch-all clear is overkill (we don't
        # know WHICH frames touched this layer) but undo of a
        # source swap is rare enough that the simplicity is worth
        # the collateral re-decode.
        any_source_changed = False
        for layer in self._stack.layers():
            prev_token = self._last_known_source.get(layer.id)
            new_token = _layer_source_token(layer)
            if prev_token is not None and prev_token != new_token:
                self._path_index.pop(layer.id, None)
                self._mtime_index.pop(layer.id, None)
                any_source_changed = True
        if any_source_changed:
            with self._lock:
                self._frames.clear()
                self._missing.clear()
                self._signature_last_seen.clear()
                self._signature_first_cached.clear()
                self._bytes_used = 0
                self._epoch += 1
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
            self._last_known_source[layer.id] = _layer_source_token(layer)
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
        # Master-range last frame is the same for every layer in this
        # stack — pull it once outside the inner loop so each
        # placeholder paint pass doesn't re-query.
        _master_first, master_last = self._stack.master_range()
        for layer in self._stack.layers():
            self._ensure_index(layer)
            paths = self._path_index[layer.id]
            ph_w = layer.sequence.width or 512
            ph_h = layer.sequence.height or 512
            for master_frame in range(layer.master_start, layer.master_end + 1):
                with self._lock:
                    sig = self._signature_at(master_frame)
                key = (master_frame, sig)
                if key in self._frames:
                    continue  # already populated (shouldn't happen post-clear)
                source_frame = layer.source_frame_at(master_frame)
                if paths.get(source_frame) is None:
                    # Sparse hole on this layer. Only mark missing if
                    # NO higher layer covers this master frame with a
                    # real path — otherwise the displayed pixel comes
                    # from the layer above and isn't actually missing.
                    topmost = self._stack.topmost_visible_at(master_frame)
                    if topmost is None or topmost.id == layer.id:
                        # Source + master frame coords baked into the
                        # overlay so the placeholder mirrors the
                        # info-band's "Layer X / Frame Y" breakdown.
                        # Per-frame allocation (~30-50 ms / frame at
                        # HD, ~150 ms at 4K) at session-load time —
                        # a sparse sequence with hundreds of gaps can
                        # stall the UI briefly on load. We accept the
                        # cost: the alternative was a shared
                        # placeholder that didn't tell the user which
                        # frame they were stuck on.
                        placeholder = get_missing_placeholder(
                            ph_w, ph_h,
                            frame_number=master_frame,
                            frame_max=master_last if master_last else None,
                            source_frame=source_frame,
                            source_max=layer.layer_out,
                        )
                        with self._lock:
                            self._frames[key] = placeholder
                            self._missing.add(key)

    def _on_visibility_changed(self, layer_id: str) -> None:
        """Toggling a layer's eye changes the chain at every frame it
        covers. With the multi-version cache, the resulting different
        signature naturally turns the existing entries into stale
        snapshots — lookups under the new signature miss and queue
        fresh decodes. We just invalidate the per-frame signature
        memo so subsequent reads recompute against the new visible
        set.

        Also drops the worker pool's pending queue: alt-channel
        prefetch priorities are computed from
        ``visible_alt_layer_ids`` ordering (= top-down through the
        currently-visible layers). Hiding / un-hiding a layer changes
        that ordering, but the previously-queued entries stay in the
        pool with their old priorities + old dedup keys —
        ``replan_prefetch`` afterwards would dedup-reject the
        re-submissions and the new ordering would never take effect
        until the old queue drains. Clearing here lets the next
        ``replan_prefetch`` queue fresh.
        """
        del layer_id  # signature memo doesn't track per-layer
        self._invalidate_signature_cache()
        self._pool.clear()

    def _on_layer_modified(self, layer_id: str) -> None:
        """Trim / offset / channel change on a layer.

        Three paths:

        * **Pure offset shift** (no trim / channel / alpha-flag
          change): re-key cached frames where the moved layer was
          the *sole contributor* at that master frame, mapping
          ``mf → mf + Δ``. Decoded pixels are identical between the
          two offsets — only the master-frame key moves. Works in
          both single-layer and multi-layer stacks: in multi-layer
          we filter by signature equality (a single-token signature
          referencing the moved layer at its OLD offset), so only
          frames whose composite was determined entirely by this
          layer get rekeyed. Frames where higher / lower layers
          also contributed stay under their old keys as stale
          snapshots — they're correctly invalidated passively by
          the multi-version cache key (new signature lookup misses
          → fresh decode).

        * **Channel selection or alpha-flag change**: the multi-
          version cache key handles this passively — the new state
          has a different signature, so lookups miss and queue fresh
          decodes; the old entries become stale snapshots, accessible
          if the user reverts (e.g. switches A → B → A). We just
          invalidate the signature memo and bump the epoch so any
          in-flight decode against the OLD signature still lands
          under its OLD key (= still-useful snapshot).

        * **Trim** (layer_in / layer_out): coverage shrunk on one
          end means frames the layer no longer reaches must surface
          whatever's underneath. Drop the now-uncovered slots from
          ``[prev_start, prev_end] \\ [new_start, new_end]``.
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
        prev_source = self._last_known_source.get(layer_id)
        new_source = _layer_source_token(layer)

        # Stack mutation always invalidates the per-frame signature
        # memo: the next ``_signature_at`` for any frame this layer
        # touches must reflect the new state.
        self._invalidate_signature_cache()

        # Source-identity change (= "Replace source…" forward OR
        # an undo that reverts to a previous sequence): the layer
        # kept its id but ``layer.sequence`` now points at a
        # different set of files. Two consequences:
        #
        # * ``_path_index[layer.id]`` caches ``frame_number -> Path``
        #   built lazily; it short-circuits when the id is already
        #   a key. Without dropping the entry, every subsequent
        #   decode reads from the OLD source's paths.
        # * Cached frame buffers were decoded from the OLD files;
        #   the signature token doesn't encode the source path, so
        #   lookups would still hit them. Clear the whole cache so
        #   the next render produces a fresh decode against the
        #   new sequence.
        #
        # Done BEFORE the existing offset / trim handling so those
        # branches don't try to rekey against the stale state.
        if prev_source is not None and prev_source != new_source:
            self._path_index.pop(layer_id, None)
            self._mtime_index.pop(layer_id, None)
            self._pool.clear()
            self._frames.clear()
            self._missing.clear()
            self._signature_last_seen.clear()
            self._signature_first_cached.clear()
            self._bytes_used = 0
            self._epoch += 1
            self._last_known_range[layer_id] = (new_start, new_end)
            self._last_known_state[layer_id] = new_state
            self._last_known_source[layer_id] = new_source
            return

        # Pure-offset shift fast path — works for any stack size.
        # In multi-layer, ``_shift_solo_dominant_frames`` rekeys only
        # the subset of cached entries where the moved layer was the
        # sole contributor (= signature is a single token referencing
        # this layer at its old offset). Other entries remain under
        # their old keys as snapshots; the multi-version cache key
        # ensures correctness without an explicit invalidate.
        is_offset_only = (
            prev_state is not None
            and prev_state[0] != new_state[0]            # offset differs
            and prev_state[1:] == new_state[1:]          # rest equal
        )
        if is_offset_only:
            shift = new_state[0] - prev_state[0]
            self._shift_solo_dominant_frames(layer, prev_state, shift)
            self._last_known_range[layer_id] = (new_start, new_end)
            self._last_known_state[layer_id] = new_state
            self._last_known_source[layer_id] = new_source
            return

        # Trim detection: layer_in / layer_out changed. The slots
        # the layer no longer covers may need re-decode (different
        # topmost-visible underneath). Channel selection /
        # alpha-flag changes alone don't reach this branch — they're
        # handled passively by the signature key.
        trim_changed = (
            prev_state is None
            or prev_state[1] != new_state[1]   # layer_in
            or prev_state[2] != new_state[2]   # layer_out
        )
        if trim_changed:
            # Frames in the symmetric difference between old and new
            # coverage hold stale pixels.
            old_set = (
                set(range(prev_start, prev_end + 1))
                if prev_start <= prev_end else set()
            )
            new_set = (
                set(range(new_start, new_end + 1))
                if new_start <= new_end else set()
            )
            now_uncovered = sorted(old_set - new_set)
            if now_uncovered:
                self._invalidate_master_range(
                    now_uncovered[0], now_uncovered[-1],
                )

        # Always reopen pool dedup slots so the controller's follow-
        # up ``replan_prefetch`` can submit fresh decodes against
        # the new signature without being rejected by stale pending
        # entries.
        self._pool.clear()
        self._last_known_range[layer_id] = (new_start, new_end)
        self._last_known_state[layer_id] = new_state
        self._last_known_source[layer_id] = new_source

    # ------------------------------------------------------------------ Internals

    def _shift_solo_dominant_frames(
        self,
        layer: Layer,
        prev_state: tuple,
        shift: int,
    ) -> None:
        """Re-key cached entries where ``layer`` was the sole contributor.

        Used by the pure-offset-shift fast path. Walks the cache and
        identifies entries whose stored signature is exactly the
        single-token form ``"{layer.id}@{sel}#{ac}{pm}+{prev_off}+{lin}"``
        — those are entries where the composite at master frame F
        was determined ENTIRELY by this layer at its old offset. For
        each, we map ``F → F + shift`` and verify the live signature
        at the new master frame is also a single token referring to
        this layer (now at its new offset). When both conditions
        hold the rekey is pixel-correct: the decoded source frame
        baked into the buffer (= ``L.layer_in + (F - L.old_master_start)``)
        is exactly what L would produce at ``F + shift`` under its
        new offset (= ``L.layer_in + ((F + shift) - L.new_master_start)``,
        and ``new_master_start = old_master_start + shift``).

        Entries the moved layer didn't dominate (multi-token sigs)
        stay under their old keys. They remain accessible as
        snapshots if the user undoes the shift; otherwise the
        multi-version cache key handles them passively — the live
        lookup at the post-shift signature misses, triggering a
        fresh decode.

        Bumps the epoch so any in-flight decode (which captured the
        OLD master_frame at submit time) gets dropped at store time
        rather than landing under a now-stale key.
        """
        if shift == 0:
            return
        # Reconstruct the signature token L would have produced under
        # its previous state. ``prev_state[1:] == new_state[1:]`` was
        # asserted at the call site, so the channel / alpha / layer_in
        # parts of the token are unchanged — we can read them from
        # the current ``layer`` and only swap the offset.
        prev_off = int(prev_state[0])
        old_solo_token = _signature_token(layer, offset_override=prev_off)
        new_solo_token = _signature_token(layer)
        with self._lock:
            # Snapshot the entries to consider; iterate later to avoid
            # mutating the dict while iterating it.
            to_move: list[tuple[int, np.ndarray, bool]] = []
            for k in list(self._frames.keys()):
                _mf, sig = k
                if sig != old_solo_token:
                    continue
                arr = self._frames.pop(k)
                is_missing = k in self._missing
                self._missing.discard(k)
                to_move.append((k[0], arr, is_missing))
            # Recompute signatures against the post-shift state.
            self._invalidate_signature_cache()
            # Apply the shift. Verify the live sig at the new master
            # frame matches our expected new_solo_token — if a higher
            # / lower layer has crept in at the destination, the
            # composite math differs and rekey would be wrong; in
            # that case we put the entry back under its old key as a
            # snapshot.
            for old_mf, arr, is_missing in to_move:
                new_mf = old_mf + shift
                live_sig = self._signature_at(new_mf)
                if live_sig != new_solo_token:
                    # Other layers now contribute at new_mf — restore
                    # the entry as a stale snapshot under the old
                    # solo-token key (still pixel-correct for that
                    # signature; useful if the user reverts).
                    self._frames[(old_mf, old_solo_token)] = arr
                    if is_missing:
                        self._missing.add((old_mf, old_solo_token))
                    continue
                new_key = (new_mf, live_sig)
                if new_key in self._frames:
                    # A snapshot already occupies the destination —
                    # keep the incumbent and drop the shifted entry.
                    # The frame will be re-decoded by the next
                    # prefetch wave if the live sig disagrees later.
                    if not is_missing:
                        self._bytes_used -= arr.nbytes
                    continue
                self._frames[new_key] = arr
                if is_missing:
                    self._missing.add(new_key)
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
                k for k in self._frames
                if first <= k[0] <= last and k not in self._missing
            ]
            if not to_drop:
                return
            for k in to_drop:
                arr = self._frames.pop(k)
                self._bytes_used -= arr.nbytes
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

        Always filters the returned list against
        ``layer.sequence.channel_names``. Without this, a
        ``channel_selection`` carried over from a previous layer
        (the channel menu's prefs-restore writes the saved selection
        onto whichever layer takes focus, including freshly added
        layers whose source might have a narrower channel set) can
        request channels the source doesn't have — the OIIO reader
        then raises ``requested channels not found`` and the cache
        records a placeholder for every frame in the layer's range.
        """
        available = set(layer.sequence.channel_names)
        sel = layer.channel_selection
        if sel is None:
            base: list[str] | None = None
        else:
            picked = [c for c in sel.active.channels if c in available]
            base = picked or None
        if not layer.alpha_composite:
            return base
        if "A" not in available:
            # Source is RGB-only — strip A from the explicit selection
            # too. Returning a list that includes A would feed the
            # reader a request it can't satisfy.
            if base is None:
                return None
            stripped = [c for c in base if c != "A"]
            return stripped or None
        if base is None:
            return None  # reader default already RGBA
        # A 4-channel group already includes its own alpha as the last
        # entry — that's how ``group_channels`` packs RGBA-shaped layers
        # (bare ``RGBA``, ``albedo.R/G/B/A``, etc.). Don't append the
        # bare ``A`` on top: that would make a 5-channel buffer the
        # cache compositor can't handle (``_ensure_rgba`` raises). Only
        # the 3-channel "RGB without alpha" case needs the implicit
        # bare-A pickup so the over-blend has an opacity to work with.
        if len(base) >= 4:
            return base
        # Single-channel masks / depth passes (Z, normal.X, mattes…)
        # are displayed as monochrome by the reader and the compositor
        # treats them as fully opaque via the 1-channel branch in
        # ``_ensure_rgba``. Appending the bare A would yield a
        # 2-channel buffer the compositor can't handle, and conceptually
        # a one-channel pass has no opacity to blend through anyway.
        if len(base) == 1:
            return base
        if "A" in base:
            return base
        return [*base, "A"]

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
            channels = self._channels_for(layer)
            plan.append({
                "layer_id": layer.id,
                "path": path,
                "channels": channels,
                "ph_w": layer.sequence.width or 512,
                "ph_h": layer.sequence.height or 512,
                # Per-layer convention so the over operator picks the
                # right pre-blend conversion for each contributor.
                "is_straight": bool(layer.alpha_is_straight),
                # Opaque "floor" layers don't compose — when the walker
                # hits one, treat it as alpha=1 even if its source has
                # an A channel.
                "is_opaque_floor": not layer.alpha_composite,
                # Per-contributor disk-cache key. Captured at submit
                # time from the layer's path + mtime + channel
                # selection + alpha flags (= what
                # ``_read_contributor_cached`` will look up at decode
                # time). One key per contributor, *not* one per
                # composite — so e.g. hiding one layer in the stack
                # still hits the cache for the others.
                "disk_key": self._source_key_for_single_layer(
                    layer, master_frame, channels,
                ),
            })
        # User rule: the checker only shows when no other layer would
        # actually contribute pixels beneath the bottom of the chain.
        # "Would contribute" means *visible AND covers this master
        # frame* — hidden layers and trimmed-out layers both fall
        # outside the chain at this frame, so neither should mask the
        # transparency background. Without filtering by coverage, a
        # layer visible globally but trimmed past the current frame
        # still forced the bottom plan entry to opaque-floor, and the
        # transparent regions rendered black instead of the BG.
        if plan:
            covering_ids = [
                l.id for l in self._stack.layers()
                if l.visible and l.covers(master_frame)
            ]
            bottom_id = plan[-1]["layer_id"]
            if bottom_id in covering_ids:
                stack_idx = covering_ids.index(bottom_id)
                has_below_covering = stack_idx < len(covering_ids) - 1
                if has_below_covering:
                    plan[-1]["is_opaque_floor"] = True
        if not plan:
            # Every layer had a hole at this frame — pre-mark missing.
            # Use the broadest layer dimensions in the stack so the
            # placeholder matches the displayed image size instead of a
            # fixed 512² square (which got letterboxed weirdly when the
            # actual sequence was 1920×1080 / 4K).
            ph_w, ph_h = self._broadest_layer_size()
            # No specific covering layer here (the plan was empty =
            # every layer had a hole), so report the topmost layer's
            # source frame — at least it tells the user "this is the
            # frame the layer that *would* have shown a pixel here
            # is missing".
            top = self._stack.topmost_visible_at(master_frame)
            _first, master_last = self._stack.master_range()
            with self._lock:
                placeholder = get_missing_placeholder(
                    ph_w, ph_h,
                    frame_number=master_frame,
                    frame_max=master_last if master_last else None,
                    source_frame=(
                        top.source_frame_at(master_frame)
                        if top is not None else None
                    ),
                    source_max=top.layer_out if top is not None else None,
                )
                key = (master_frame, self._signature_at(master_frame))
                self._frames[key] = placeholder
                self._missing.add(key)
            return False
        # Capture the signature at submit time — the worker will store
        # under the same key even if the stack mutates mid-decode, so
        # the in-flight buffer lands as a "stale" snapshot accessible
        # if the user reverts. Per-contributor disk keys live in each
        # plan entry (built above), no composite-level key needed.
        with self._lock:
            sig = self._signature_at(master_frame)
        return self._pool.submit(
            priority,
            (master_frame, sig),
            lambda: self._decode_composited_and_store(
                master_frame, sig, plan,
            ),
        )

    def _decode_composited_and_store(
        self,
        master_frame: int,
        signature: str,
        plan: list[dict],
        disk_key: str | None = None,  # noqa: ARG002 — kept for back-compat
    ) -> None:
        """Worker entry: decode each layer's contribution + over-blend
        front-to-back. The first plan entry is the topmost.

        Per-entry convention flags:
        * ``is_straight`` — input RGB is unpremult; multiply by A
          before the blend so all contributors are premult internally.
        * ``is_opaque_floor`` — the layer doesn't compose; treat it as
          fully opaque (force its alpha to 1.0) so layers below are
          masked. Reached when the walker hit a non-composing layer.
        * ``disk_key`` (optional) — per-contributor disk-cache key
          computed at submit time. Drives the per-layer cache lookup
          inside the read loop.

        The ``disk_key`` *parameter* is no longer consulted — it used
        to be the composite-level key, but caching was migrated to
        per-contributor in 1.5.4 so a one-layer change in the stack
        doesn't invalidate every cached blob for that frame. Kept on
        the signature for callers that still pass it (pure no-op).

        Math (everything in premult):
            accum = top + arr * (1 - accum.a)
        """
        with self._lock:
            epoch = self._epoch
        key = (master_frame, signature)
        try:
            top_plan = plan[0]
            top = self._read_contributor_cached(top_plan)
            if top_plan["is_opaque_floor"]:
                top = _force_alpha_one(top)
            # Quick opacity check on the centre row to avoid a full
            # O(WxH) scan when the top is solid; if it's a floor or
            # already opaque we skip the deeper decodes.
            if top.shape[0] > 0 and float(top[top.shape[0] // 2, :, 3].min()) >= 1.0 - 1e-3:
                accum = top
            else:
                accum = top.copy()
                # Reusable scratch buffer for the per-layer
                # ``arr * inv_a`` product. Naive ``accum + arr * inv_a``
                # allocates two HxWx4 floats per layer (~128 MiB at 4K
                # for an 8-layer composite). Reusing one scratch buffer
                # cuts that to one alloc total.
                tmp = np.empty_like(accum)
                for layer_plan in plan[1:]:
                    arr = self._read_contributor_cached(layer_plan)
                    if layer_plan["is_opaque_floor"]:
                        arr = _force_alpha_one(arr)
                    inv_a = (1.0 - accum[..., 3:4]).astype(accum.dtype)
                    # ``arr`` may be a read-only view (disk cache); we
                    # write the product to ``tmp`` instead of into
                    # ``arr``. ``accum`` is ours so ``+=`` is safe.
                    np.multiply(arr, inv_a, out=tmp)
                    accum += tmp
                    if float(accum[..., 3].min()) >= 1.0 - 1e-3:
                        break
        except FrameReadError as err:
            log.warning(
                "composite decode failed master=%d: %s", master_frame, err,
            )
            top = self._stack.topmost_visible_at(master_frame)
            _first, master_last = self._stack.master_range()
            placeholder = get_missing_placeholder(
                plan[0]["ph_w"], plan[0]["ph_h"],
                frame_number=master_frame,
                frame_max=master_last if master_last else None,
                source_frame=(
                    top.source_frame_at(master_frame)
                    if top is not None else None
                ),
                source_max=top.layer_out if top is not None else None,
            )
            key = (master_frame, signature)
            with self._lock:
                self._decode_errors += 1
                if epoch != self._epoch:
                    return
                if key in self._frames:
                    return
                self._frames[key] = placeholder
                self._missing.add(key)
            return

        key = (master_frame, signature)
        with self._lock:
            if epoch != self._epoch:
                return
            if key in self._frames:
                return
            self._frames[key] = accum
            self._bytes_used += accum.nbytes
            # Stamp ``first_cached`` for FIFO-by-channel eviction (only
            # on the first frame for this sig; later frames share the
            # original timestamp so the whole channel ages together).
            self._signature_first_cached.setdefault(
                signature, time.monotonic(),
            )
            # Same still-fan-out as ``_decode_and_store`` — covers the
            # case where a user explicitly toggles ``alpha_composite``
            # on a still (uncommon, but supported via the T button).
            still_layer = self._still_layer_anchored_at(master_frame)
            if still_layer is not None:
                for f in range(
                    still_layer.master_start,
                    still_layer.master_end + 1,
                ):
                    if f == master_frame:
                        continue
                    f_sig = self._signature_at(f)
                    f_key = (f, f_sig)
                    if f_key in self._frames:
                        continue
                    self._frames[f_key] = accum
                    self._signature_first_cached.setdefault(
                        f_sig, time.monotonic(),
                    )
            self._evict_if_over_budget()
        # NB: no composite-level disk write any more (since 1.5.4).
        # Each contributor was already persisted in
        # :meth:`_read_contributor_cached` so the per-layer tier
        # reuses across stack-composition changes. Skipping the
        # composite blob saves disk space + a write per multi-layer
        # frame at the cost of ~5-10 ms of over-blend math on the
        # next session's first scrub.

    def _read_contributor_cached(self, entry: dict) -> np.ndarray:
        """Decode one composite contributor with a per-layer disk
        cache wrapper.

        The cached blob is the **post-premultiply, pre-opaque-floor**
        buffer — premult is deterministic for a given (path, channels,
        alpha_is_straight) tuple, so caching it produces a stable
        result. ``_force_alpha_one`` is NOT folded in here because
        ``is_opaque_floor`` depends on *which other layers are visible
        below this one in the current stack* — caching that would
        invalidate the entry on any stack reorder. The composite
        caller applies ``_force_alpha_one`` if needed after this
        returns.
        """
        disk_key = entry.get("disk_key")
        if disk_key and self._disk_cache is not None:
            cached = self._disk_cache.get(disk_key)
            if cached is not None:
                return cached
        arr = read_frame(entry["path"], channels=entry["channels"])
        arr = _ensure_rgba(arr)
        if entry["is_straight"]:
            # ``arr`` was just produced by ``read_frame`` (or by
            # ``_ensure_rgba`` allocating an RGBA layout) — we own
            # it, so premultiply in place rather than copying.
            arr = _premultiply(arr, inplace=True)
        if disk_key and self._disk_cache is not None:
            self._disk_cache.put(disk_key, arr)
        return arr

    def _decode_and_store(
        self,
        master_frame: int,
        signature: str,
        path,
        channels: list[str] | None,
        placeholder_w: int,
        placeholder_h: int,
        strip_alpha: bool = False,
        disk_key: str | None = None,
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

        ``disk_key`` is the cross-session cache key computed at submit
        time (from the captured ``channels`` selection). Critical that
        it doesn't get recomputed here from *live* state — the alt-
        channel prefetch decodes with overridden channels while the
        live state is on another, and a mismatched key reads / writes
        the wrong blob, mixing channels across cache slots.
        """
        with self._lock:
            epoch = self._epoch
        key = (master_frame, signature)
        # Disk-tier lookup — short-circuit the expensive ``read_frame``
        # when the same source was decoded in a previous session (or
        # earlier in this one before RAM eviction). The disk cache
        # stores half-float lz4 blobs; a 4K hit comes back in ~5-8 ms
        # versus ~300 ms for a fresh EXR decode.
        if disk_key is not None and self._disk_cache is not None:
            cached = self._disk_cache.get(disk_key)
            if cached is not None:
                arr = cached
                if strip_alpha and arr.ndim == 3 and arr.shape[2] == 4:
                    arr = np.ascontiguousarray(arr[..., :3])
                with self._lock:
                    if epoch != self._epoch:
                        return
                    if key in self._frames:
                        return
                    self._frames[key] = arr
                    self._bytes_used += arr.nbytes
                    self._signature_first_cached.setdefault(
                        signature, time.monotonic(),
                    )
                    self._evict_if_over_budget()
                return
        try:
            arr = read_frame(path, channels=channels)
        except FrameReadError as err:
            log.warning(
                "decode failed master=%d path=%s: %s",
                master_frame, path, err,
            )
            top = self._stack.topmost_visible_at(master_frame)
            _first, master_last = self._stack.master_range()
            placeholder = get_missing_placeholder(
                placeholder_w, placeholder_h,
                filename=getattr(path, "name", None) or str(path),
                frame_number=master_frame,
                frame_max=master_last if master_last else None,
                source_frame=(
                    top.source_frame_at(master_frame)
                    if top is not None else None
                ),
                source_max=top.layer_out if top is not None else None,
            )
            with self._lock:
                self._decode_errors += 1
                if epoch != self._epoch:
                    return
                if key in self._frames:
                    return
                self._frames[key] = placeholder
                self._missing.add(key)
            return

        if strip_alpha and arr.ndim == 3 and arr.shape[2] == 4:
            # Slice the alpha channel out + ensure the result is
            # contiguous so the GL upload's ``glTexSubImage2D`` is
            # happy with the buffer's stride.
            arr = np.ascontiguousarray(arr[..., :3])

        with self._lock:
            if epoch != self._epoch:
                return  # invalidated mid-decode — drop
            if key in self._frames:
                return  # raced — keep existing
            self._frames[key] = arr
            self._bytes_used += arr.nbytes
            # Stamp ``first_cached`` for FIFO-by-channel eviction (only
            # on the first frame for this sig; later frames share the
            # original timestamp so the whole channel ages together).
            self._signature_first_cached.setdefault(
                signature, time.monotonic(),
            )
            # Still-image fan-out: when the just-decoded frame is the
            # anchor (master_start) of a still layer, alias the same
            # ndarray into every other master frame in the still's
            # hold range — under each frame's *own* signature so
            # they survive the same eviction rules. The prefetch
            # loop already called ``request(N)`` for each of those,
            # but they got dedup-rejected before the anchor was
            # decoded. Without the fan-out, the cache bar would show
            # only one decoded frame for a 100-frame still.
            # Aliases share the buffer (zero extra memory) and are
            # exempt from byte accounting (no double-count).
            still_layer = self._still_layer_anchored_at(master_frame)
            if still_layer is not None:
                for f in range(
                    still_layer.master_start,
                    still_layer.master_end + 1,
                ):
                    if f == master_frame:
                        continue
                    f_sig = self._signature_at(f)
                    f_key = (f, f_sig)
                    if f_key in self._frames:
                        continue
                    self._frames[f_key] = arr  # ref-share, no nbytes bump
                    self._signature_first_cached.setdefault(
                        f_sig, time.monotonic(),
                    )
            self._evict_if_over_budget()
        # Async disk-cache write — survives this session's process
        # exit so a re-open lands on the fast path. Outside the lock
        # because ``put`` only enqueues; the actual write happens on
        # the DiskCache writer thread.
        if disk_key is not None and self._disk_cache is not None:
            self._disk_cache.put(disk_key, arr)

    def _still_layer_anchored_at(self, master_frame: int):  # type: ignore[no-untyped-def]
        """Return the topmost-visible still layer whose ``master_start``
        equals ``master_frame``, else ``None``.

        Used by :meth:`_decode_and_store` to fan a freshly-decoded
        anchor across its entire hold range. We pick the topmost
        rather than any covering still so the alias semantics line up
        with what :meth:`request` already chose to decode.
        """
        for layer in self._stack.layers():
            if not layer.visible:
                continue
            if not getattr(layer, "is_still", False):
                continue
            if layer.master_start == master_frame:
                return layer
        return None

    # ``-inf`` makes beauty (RGB/RGBA) snapshots sort lower than any
    # real first-cached timestamp → evicted LAST within tier 0.
    _AGE_BEAUTY_PIN = float("-inf")

    # Priority threshold for ``drop_above_priority`` after an eviction
    # round. Matches ``PlayerController._ALT_CHANNEL_BASE_PRIORITY`` —
    # any task whose priority is at or above this is an alt-channel
    # decode that's safe to drop in bulk under memory pressure.
    _ALT_TASK_PRIORITY_THRESHOLD = 1_000_000

    def _evict_if_over_budget(self) -> None:
        """Multi-tier eviction prioritising live frames over stale
        snapshots. Tiers, evicted in order:

        1. **Stale-signature** entries — frames whose stored
           signature no longer matches the chain × channel state at
           that master frame. Generated by channel switches /
           visibility toggles / reorders; the user can re-hit them
           by reverting, but they're not the current view.
        2. **Invisible-layer-only** entries — frames whose stored
           signature references at least one layer that's currently
           invisible. Same idea but stricter: the user has actively
           hidden some contributor of the snapshot, so the snapshot
           is unreachable until they unhide.
        3. **Live distance-based** — what the original eviction did:
           score by signed distance from the playhead (with
           behind-penalty + loop-ring awareness), evict farthest
           first.

        Tier ordering means a budget-pressure event drops "alternative
        view" pixels first; only when the alt-pool is empty do we
        start touching live frames. That's exactly what the user
        asked for: keep channel snapshots, surrender them under
        memory pressure.

        Used to be a 180-LOC method with three nested closures; now
        a 25-LOC orchestrator that delegates classification and
        eviction to dedicated helpers below.
        """
        if self._bytes_used <= self._budget:
            return

        visible_ids = self._visible_layer_ids_snapshot()

        # Sort: tier 0 first (smallest -tier), then tier 1, then
        # tier 2. Within tier 0, oldest signature first (beauty
        # pinned at ``-inf``); same-sig frames clump together;
        # within signature, farthest frame first. Within tier 1/2,
        # signature still groups but secondary ordering is distance.
        by_priority = sorted(
            self._frames.keys(),
            key=lambda item: self._eviction_sort_key(item, visible_ids),
            reverse=True,
        )

        self._evict_in_order(by_priority)

        # Cull queued alt-channel tasks. Without this the worker
        # pool keeps decoding stale alts that immediately re-
        # trigger eviction → constant 32 MB alloc/free cycles →
        # Python heap fragmentation → process RSS climbs even
        # though ``_bytes_used`` stays at budget.
        self._pool.drop_above_priority(self._ALT_TASK_PRIORITY_THRESHOLD)

    # --- Eviction helpers ---------------------------------------------------

    def _visible_layer_ids_snapshot(self) -> set[str]:
        """Set of layer ids currently flagged visible. Snapshotted
        once at the start of an eviction round so we don't pay the
        per-key attribute read during sorting."""
        return {layer.id for layer in self._stack.layers() if layer.visible}

    def _eviction_distance_score(self, frame: int) -> float:
        """Score a frame by its signed distance from the playhead,
        with behind-penalty + loop-ring awareness. Lower = closer to
        the playhead = preserved longer.

        Encodes the three core eviction policies:

        * Frames in a small "near-rear window" behind the playhead
          stay cheap so a quick scrub-back doesn't fall off cache.
        * Frames inside an active loop range use modular distance
          (= a frame "behind" the playhead in loop mode is one
          full ring trip ahead, the cheapest of all).
        * Frames truly behind (no loop, outside near-rear) get
          multiplied by :data:`_BEHIND_PLAYHEAD_PENALTY` — we'd
          rather drop them than the frames coming up next.
        """
        cur = self._current_frame
        d = self._direction
        delta_signed = (frame - cur) * d
        if -_NEAR_REAR_WINDOW <= delta_signed < 0:
            return -delta_signed  # 1, 2, 3, ... 8
        loop_lo = self._loop_lo
        loop_hi = self._loop_hi
        if (
            self._loop_enabled
            and loop_lo is not None
            and loop_hi is not None
            and loop_lo <= frame <= loop_hi
        ):
            ring_size = loop_hi - loop_lo + 1
            if d >= 0:
                return float((frame - cur) % ring_size)
            return float((cur - frame) % ring_size)
        if delta_signed < 0:
            return -delta_signed * _BEHIND_PLAYHEAD_PENALTY
        return float(delta_signed)

    @staticmethod
    def _signature_refs_invisible(sig: str, visible_ids: set[str]) -> bool:
        """True when the signature mentions at least one
        currently-hidden layer id. Empty signatures (= no layer
        covered the frame) don't count."""
        if not sig:
            return False
        for token in sig.split("|"):
            # ``layer_id@selection_label#flags`` — peel the id off.
            lid = token.split("@", 1)[0]
            if lid and lid not in visible_ids:
                return True
        return False

    @staticmethod
    def _signature_is_beauty(sig: str) -> bool:
        """True when *any* contributor token in the signature uses
        the beauty-pass label (``RGB`` / ``RGBA``).

        The beauty pass is the visual "anchor" — the user almost
        always wants it preserved across channel switches so they
        can flip back to a familiar view instantly. We bias the
        eviction to keep beauty snapshots last; only when every
        non-beauty alt has been freed AND we're still over budget
        do beauty entries become evictable.

        Heuristic for multi-layer stacks: if *any* layer in the
        chain shows beauty, the whole snapshot is "beauty-
        anchored" and gets the preservation bonus. Imperfect but
        cheap, and the common case is single-layer EXR review
        where this collapses to "exactly the RGB(A) snapshot".
        """
        if not sig:
            return False
        for token in sig.split("|"):
            # token shape: "layer_id@LABEL#flags+off+lin"
            at_idx = token.find("@")
            if at_idx < 0:
                continue
            hash_idx = token.find("#", at_idx + 1)
            if hash_idx < 0:
                continue
            label = token[at_idx + 1:hash_idx]
            if label in ("RGB", "RGBA"):
                return True
        return False

    def _eviction_sort_key(
        self,
        item: tuple[int, str],
        visible_ids: set[str],
    ) -> tuple[int, float, str, float]:
        """Build the 4-tuple sort key for one cache entry.

        Tier 0 = stale-sig (drop first), Tier 1 = sig refs an
        invisible layer, Tier 2 = live frame. Within tier 0,
        beauty snapshots are pinned at ``-inf`` age so they sort
        below every real timestamp → evicted last. Other stale sigs
        sort by negated first-cached timestamp (older → earlier
        evicted). ``sig`` is the third key so same-sig frames clump
        together — without it, every alt-prefetch wave stamps
        similar timestamps and eviction spreads across all groups
        instead of clearing one whole group at a time.
        """
        mf, sig = item
        current_sig = self._signature_at(mf)
        if sig != current_sig:
            tier = 0
            if self._signature_is_beauty(sig):
                age = self._AGE_BEAUTY_PIN
            else:
                # Negate so older (smaller first_cached) sorts
                # higher under reverse=True → evicted first.
                # Sigs without a first-cached stamp default to
                # 0.0 → ``age = 0`` → highest priority for
                # eviction (covers placeholder-only sigs that
                # never had a real frame stored).
                age = -self._signature_first_cached.get(sig, 0.0)
        elif self._signature_refs_invisible(sig, visible_ids):
            tier = 1
            age = 0.0
        else:
            tier = 2
            age = 0.0
        return (-tier, age, sig, self._eviction_distance_score(mf))

    def _evict_in_order(self, ordered_keys: list[tuple[int, str]]) -> None:
        """Walk ``ordered_keys`` (highest-priority-to-evict first)
        and remove entries until ``_bytes_used`` is within budget.

        ``ordered_keys`` is already sorted by the caller; this is
        the pure side-effect loop split out so the sort key + the
        actual eviction can be tested in isolation.
        """
        for k in ordered_keys:
            if self._bytes_used <= self._budget:
                break
            arr = self._frames.pop(k)
            if k not in self._missing:
                self._bytes_used -= arr.nbytes
            self._missing.discard(k)
            self._evictions += 1
