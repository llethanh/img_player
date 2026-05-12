"""Tests for :class:`MasterFrameCache` — invalidation logic + master-frame
resolution. Decoding itself is stubbed so the tests don't touch OIIO.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from img_player.cache.master_frame_cache import MasterFrameCache
from img_player.layers import Layer, LayerStack
from img_player.sequence.models import FrameInfo, SequenceInfo

# ============================================================================
# Helpers
# ============================================================================


def _seq(first: int = 1001, last: int = 1100) -> SequenceInfo:
    frames = tuple(
        FrameInfo(path=Path(f"/fake/{n}.exr"), frame_number=n)
        for n in range(first, last + 1)
    )
    return SequenceInfo(
        base_name="x", extension=".exr", directory=Path("/fake"),
        padding=4, frames=frames, width=64, height=64,
    )


def _layer(first: int = 1001, last: int = 1100, offset: int = 0) -> Layer:
    return Layer.from_sequence(_seq(first, last), offset=offset)


def _fake_decode(path, channels=None, **kw):
    """Stand-in for :func:`io.reader.read_frame` — returns a tiny
    distinguishable buffer so the cache can store it."""
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    return arr


# ============================================================================
# Construction + empty stack
# ============================================================================


class TestEmptyStack:
    def test_no_layers_no_decode(self, qtbot) -> None:
        stack = LayerStack()
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            assert cache.request(50) is False
            assert cache.get(50) is None
            assert cache.cached_frames() == frozenset()
        finally:
            cache.shutdown()


# ============================================================================
# Resolution: master_frame → topmost-visible layer
# ============================================================================


class TestResolution:
    def test_decodes_topmost_visible(self, qtbot) -> None:
        stack = LayerStack()
        bottom = _layer(offset=0)
        top = _layer(offset=50)
        stack.add(bottom)
        stack.add(top)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            with patch(
                "img_player.cache.master_frame_cache.read_frame",
                side_effect=_fake_decode,
            ) as mock:
                cache.request(60)  # both layers cover 60 — top wins
                cache.wait_idle(timeout=2.0)
            # Decode was called with the top layer's source path:
            # offset 50, layer_in 1001 → master 60 = source 1011.
            mock.assert_called_once()
            called_path = mock.call_args.args[0]
            assert called_path == Path("/fake/1011.exr")
        finally:
            cache.shutdown()

    def test_no_request_for_uncovered_master_frame(self, qtbot) -> None:
        stack = LayerStack()
        stack.add(_layer(offset=0))  # 0..99
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            # Master 200 is past every layer — no decode submission.
            assert cache.request(200) is False
        finally:
            cache.shutdown()


# ============================================================================
# Invalidation on stack changes
# ============================================================================


class TestInvalidation:
    def test_visibility_toggle_invalidates_layer_range(self, qtbot) -> None:
        stack = LayerStack()
        layer = _layer(offset=0)
        stack.add(layer)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            with patch(
                "img_player.cache.master_frame_cache.read_frame",
                side_effect=_fake_decode,
            ):
                cache.request(50)
                cache.wait_idle(timeout=2.0)
            assert 50 in cache.cached_frames()
            # Toggling visibility on a layer that covers master 50
            # must wipe it from the cache.
            stack.toggle_visible(layer.id)
            assert 50 not in cache.cached_frames()
        finally:
            cache.shutdown()

    def test_layer_modified_invalidates_range(self, qtbot) -> None:
        stack = LayerStack()
        layer = _layer(offset=0)
        stack.add(layer)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            with patch(
                "img_player.cache.master_frame_cache.read_frame",
                side_effect=_fake_decode,
            ):
                cache.request(25)
                cache.wait_idle(timeout=2.0)
            assert 25 in cache.cached_frames()
            stack.update(layer.id, exposure=2.0)
            # Modified → invalidate the layer's master range.
            assert 25 not in cache.cached_frames()
        finally:
            cache.shutdown()

    def test_layers_changed_keeps_frames_with_unchanged_chain(self, qtbot) -> None:
        """Selective invalidation: adding a layer that doesn't cover
        frame 25 leaves frame 25's chain unchanged → cache keeps it.
        Replaces the legacy "nuclear clear" semantic — see the
        ``_on_layers_changed`` docstring for the rationale."""
        stack = LayerStack()
        layer = _layer(offset=0)
        stack.add(layer)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            with patch(
                "img_player.cache.master_frame_cache.read_frame",
                side_effect=_fake_decode,
            ):
                cache.request(25)
                cache.wait_idle(timeout=2.0)
            assert 25 in cache.cached_frames()
            # New layer at offset=200 covers master 200..299 — frame
            # 25's chain still resolves to the original layer alone.
            stack.add(_layer(offset=200))
            assert 25 in cache.cached_frames(), (
                "frame 25 should survive a stack change that doesn't "
                "alter its contributor chain"
            )
        finally:
            cache.shutdown()

    def test_layers_changed_drops_frames_with_changed_chain(self, qtbot) -> None:
        """Selective invalidation drops a frame when the layer added
        on TOP becomes the new topmost-visible at that frame."""
        stack = LayerStack()
        # Layer A covers master 0..99.
        layer_a = _layer(offset=0)
        stack.add(layer_a)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            with patch(
                "img_player.cache.master_frame_cache.read_frame",
                side_effect=_fake_decode,
            ):
                cache.request(25)
                cache.wait_idle(timeout=2.0)
            assert 25 in cache.cached_frames()
            # Add Layer B on top covering master 0..99 (default add
            # position is top of stack). B is now topmost-visible at
            # frame 25, so the cached pixels (which came from A) are
            # stale → must be dropped.
            stack.add(_layer(offset=0))
            assert 25 not in cache.cached_frames(), (
                "frame 25 should be invalidated when a new layer "
                "becomes the topmost-visible at that frame"
            )
        finally:
            cache.shutdown()


# ============================================================================
# Pure-offset shift fast path (multi-layer rekey)
# ============================================================================


class TestOffsetShift:
    """Pure-offset shift should re-key cached entries instead of
    invalidating + re-decoding. Covers single-layer and the multi-
    layer "solo dominant" extension where the moved layer is the
    only contributor at certain master frames."""

    def test_single_layer_offset_shift_rekeys_cache(self, qtbot) -> None:
        """One layer, one cached frame. Shifting the layer's offset
        by Δ must move the cached entry from F to F+Δ — zero
        re-decodes."""
        stack = LayerStack()
        layer = _layer(offset=0)
        stack.add(layer)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            with patch(
                "img_player.cache.master_frame_cache.read_frame",
                side_effect=_fake_decode,
            ) as mock:
                cache.request(25)
                cache.wait_idle(timeout=2.0)
            assert mock.call_count == 1
            assert 25 in cache.cached_frames()
            # Shift the layer right by 10. Cache should rekey 25 → 35.
            stack.update(layer.id, offset=10)
            cached = cache.cached_frames()
            assert 25 not in cached, "old key should be gone after shift"
            assert 35 in cached, (
                "cached frame should follow the layer's new offset"
            )
        finally:
            cache.shutdown()

    def test_offset_shift_no_op_when_delta_zero(self, qtbot) -> None:
        """Re-emitting layer_modified with the same offset shouldn't
        churn the cache."""
        stack = LayerStack()
        layer = _layer(offset=5)
        stack.add(layer)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            with patch(
                "img_player.cache.master_frame_cache.read_frame",
                side_effect=_fake_decode,
            ):
                cache.request(30)
                cache.wait_idle(timeout=2.0)
            assert 30 in cache.cached_frames()
            # Updating an unrelated no-op field path (offset to its
            # current value) should not invalidate.
            stack.update(layer.id, offset=5)
            assert 30 in cache.cached_frames()
        finally:
            cache.shutdown()

    def test_multilayer_solo_dominant_frames_rekey(self, qtbot) -> None:
        """Two layers that don't overlap: the moved layer is the sole
        contributor at every frame it covers. All its cached entries
        should rekey forward — same outcome as the single-layer case."""
        stack = LayerStack()
        # Layer A: master [0, 99].
        layer_a = _layer(offset=0)
        # Layer B: master [200, 299] — disjoint from A.
        layer_b = _layer(offset=200)
        stack.add(layer_a)
        stack.add(layer_b)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            with patch(
                "img_player.cache.master_frame_cache.read_frame",
                side_effect=_fake_decode,
            ):
                # Cache one frame in A's range, one in B's range.
                cache.request(50)
                cache.request(250)
                cache.wait_idle(timeout=2.0)
            assert 50 in cache.cached_frames()
            assert 250 in cache.cached_frames()
            # Shift A right by 20. A's covered range becomes [20, 119].
            # The cached entry at master 50 (where A was solo) should
            # rekey to master 70.
            stack.update(layer_a.id, offset=20)
            cached = cache.cached_frames()
            assert 50 not in cached, "A's cached frame at 50 should move"
            assert 70 in cached, "A's cached frame should land at 70"
            # B's cached entry at 250 should be untouched — B didn't move
            # and B was solo there.
            assert 250 in cached
        finally:
            cache.shutdown()

    def test_overlapping_layers_skip_rekey_keep_snapshot(self, qtbot) -> None:
        """When the moved layer overlaps with another contributor at a
        cached master frame, the rekey isn't safe (the other layer's
        source frame would shift too in the rekeyed composite). The
        entry must NOT be rekeyed; instead it lingers as a stale
        snapshot under its old signature key."""
        stack = LayerStack()
        # Two transparent layers both covering master [0, 99] →
        # signature at every frame has TWO tokens (no opaque break).
        # No entry will match the "solo dominant" condition.
        bottom = _layer(offset=0)
        top = _layer(offset=0)
        stack.add(bottom)
        stack.add(top)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            with patch(
                "img_player.cache.master_frame_cache.read_frame",
                side_effect=_fake_decode,
            ):
                cache.request(40)
                cache.wait_idle(timeout=2.0)
            # cached_frames() reports under the live signature; both
            # layers contribute so the entry was stored under a 2-token
            # sig. After we shift TOP, the live sig at master 40 has
            # TOP at the new offset → the old 2-token entry is now a
            # stale snapshot (different signature). It should NOT
            # appear in cached_frames() for master 40 (different live
            # sig) AND should NOT have been rekeyed to master 50
            # (rekey is unsafe — bottom's contribution would shift).
            assert 40 in cache.cached_frames()
            stack.update(top.id, offset=10)
            cached = cache.cached_frames()
            assert 40 not in cached, (
                "after TOP shifts, master 40's live signature is "
                "different → no hit"
            )
            assert 50 not in cached, (
                "rekey is unsafe in multi-contributor frames; entry "
                "must stay as snapshot under old key, not migrate"
            )
        finally:
            cache.shutdown()


# ============================================================================
# request_range clamping
# ============================================================================


class TestRequestRange:
    def test_clamps_to_master_range(self, qtbot) -> None:
        stack = LayerStack()
        stack.add(_layer(offset=10))  # 10..109
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            calls = []
            with patch(
                "img_player.cache.master_frame_cache.read_frame",
                side_effect=lambda p, **kw: calls.append(p) or _fake_decode(p, **kw),
            ):
                # Request way outside; only frames in 10..109 should
                # actually queue.
                cache.request_range(-50, 500, direction=1)
                cache.wait_idle(timeout=5.0)
            # We can't easily assert exact count without flaky timing,
            # but every decoded path must be inside the source range.
            for path in calls:
                stem = path.stem
                # source frame numbers are 1001..1100
                assert 1001 <= int(stem) <= 1100
        finally:
            cache.shutdown()

    def test_no_op_on_empty_stack(self, qtbot) -> None:
        cache = MasterFrameCache(LayerStack(), num_workers=1)
        try:
            cache.request_range(0, 100)
            # Just checks we don't crash.
        finally:
            cache.shutdown()


# ============================================================================
# alt_channel_progress — per-channel cache-fill progress
# ============================================================================


class TestAltChannelProgress:
    """The orange progress bars in the channel-picker menu rely on
    :meth:`MasterFrameCache.alt_channel_progress`. Two requirements:

    1. **Real frames count.** A live decoded entry under the right
       signature → the bar advances.
    2. **Missing-frame placeholders DO NOT count.** Failed alt-channel
       decodes store a placeholder under ``(mf, sig)`` and add the
       key to :attr:`_missing` — they live in ``_frames`` so the GL
       pipeline can paint the checker without crashing, but they
       spent **zero** real RAM and don't represent a decoded frame.
       Without the missing-filter, the bar would claim "fully cached"
       for groups where every prefetch failed, contradicting the RAM
       gauge (51 GB / 51 GB but the bars claim every AOV is in RAM).
    """

    def _multi_aov_seq(self, first: int = 1001, last: int = 1010) -> SequenceInfo:
        """Build a sequence with two channel groups: bare ``RGB`` and a
        layered ``albedo.R/G/B`` AOV. ``group_channels`` produces two
        ``ChannelGroup`` entries from this — enough surface to exercise
        the "active vs override signature" split in the progress fn.
        """
        frames = tuple(
            FrameInfo(path=Path(f"/fake/{n}.exr"), frame_number=n)
            for n in range(first, last + 1)
        )
        return SequenceInfo(
            base_name="x", extension=".exr", directory=Path("/fake"),
            padding=4, frames=frames, width=64, height=64,
            channel_names=(
                "R", "G", "B",
                "albedo.R", "albedo.G", "albedo.B",
            ),
        )

    def _layer_with_active(self, sequence: SequenceInfo, active_label: str) -> Layer:
        from img_player.sequence.channels import ChannelSelection, group_channels
        layer = Layer.from_sequence(sequence)
        groups = group_channels(sequence.channel_names)
        active = next(g for g in groups if g.label == active_label)
        layer.channel_selection = ChannelSelection(active=active)
        return layer

    def test_missing_placeholders_excluded_from_progress(self, qtbot) -> None:
        """Inject two entries under the ``albedo`` signature: one real
        decoded buffer, one placeholder. The progress count for
        ``albedo`` must be 1 (not 2)."""
        stack = LayerStack()
        seq = self._multi_aov_seq()
        layer = self._layer_with_active(seq, "RGB")
        stack.add(layer)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            # Build the synthetic signature the alt-prefetch would
            # store under for ``albedo`` at master frames 1001 / 1002.
            sig_albedo = cache._signature_at_with_override(
                1001, layer.id, "albedo",
            )
            real_arr = np.zeros((4, 4, 3), dtype=np.float32)
            placeholder = np.zeros((4, 4, 4), dtype=np.float32)
            with cache._lock:
                # 1001 = real decode
                cache._frames[(1001, sig_albedo)] = real_arr
                # 1002 = decode failed → placeholder + _missing entry
                cache._frames[(1002, sig_albedo)] = placeholder
                cache._missing.add((1002, sig_albedo))

            progress = cache.alt_channel_progress()
            cached, total = progress["albedo"]
            # 10 frames in the sequence (1001..1010 inclusive).
            assert total == 10
            # Only the real entry at 1001 counts. The 1002 placeholder
            # is in ``_missing`` so it's excluded.
            assert cached == 1, (
                f"expected 1 real cached frame, got {cached} — "
                "missing-frame placeholder leaked into progress count"
            )
        finally:
            cache.shutdown()

    def test_real_frames_under_active_signature_count(self, qtbot) -> None:
        """Sanity: the *active* group's bar advances when real frames
        land under the live signature. Belt-and-braces — covers the
        ``grp.label == active_label`` branch that uses
        :meth:`_signature_at` instead of the override path."""
        stack = LayerStack()
        seq = self._multi_aov_seq()
        layer = self._layer_with_active(seq, "RGB")
        stack.add(layer)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            sig_rgb = cache._signature_at(1001)
            real_arr = np.zeros((4, 4, 3), dtype=np.float32)
            with cache._lock:
                cache._frames[(1001, sig_rgb)] = real_arr
                cache._frames[(1005, sig_rgb)] = real_arr

            progress = cache.alt_channel_progress()
            cached, total = progress["RGB"]
            assert (cached, total) == (2, 10)
        finally:
            cache.shutdown()


# ============================================================================
# Eviction ordering — whole-channel-at-a-time + RGBA pin
# ============================================================================


class TestEvictionOrdering:
    """:meth:`_evict_if_over_budget` should drop alt-channel snapshots
    **one whole group at a time** (oldest first), and the beauty pass
    (RGB / RGBA) should be the last thing evicted.

    Rationale: under heavy AOV browsing, the user has 5-20 alt
    snapshots cached + the beauty pass. When budget pressure hits,
    they want a clean "lose volume_Z entirely, keep everything else
    full" rather than "lose 30% of every channel" (which leaves all
    channels half-cached and nothing instantly browseable). And
    they want to swap back to RGBA from any AOV in O(0) decode time,
    so RGBA stays pinned.
    """

    def _multi_aov_seq(self, first: int = 1001, last: int = 1010) -> SequenceInfo:
        frames = tuple(
            FrameInfo(path=Path(f"/fake/{n}.exr"), frame_number=n)
            for n in range(first, last + 1)
        )
        return SequenceInfo(
            base_name="x", extension=".exr", directory=Path("/fake"),
            padding=4, frames=frames, width=64, height=64,
            channel_names=(
                "R", "G", "B", "A",
                "albedo.R", "albedo.G", "albedo.B",
                "emission.R", "emission.G", "emission.B",
            ),
        )

    def _layer_with_active(self, sequence: SequenceInfo, active_label: str) -> Layer:
        from img_player.sequence.channels import ChannelSelection, group_channels
        layer = Layer.from_sequence(sequence)
        groups = group_channels(sequence.channel_names)
        active = next(g for g in groups if g.label == active_label)
        layer.channel_selection = ChannelSelection(active=active)
        return layer

    def test_evicts_oldest_non_beauty_channel_first(self, qtbot) -> None:
        """3 cached channels: RGBA (beauty), albedo (older), emission
        (newer). Currently looking at a 4th hypothetical state so
        all three are stale (tier 0). Force eviction; emission
        should empty before albedo, and RGBA should stay intact."""
        import time as _time
        stack = LayerStack()
        seq = self._multi_aov_seq()
        # Active label = something not in cache; pretend by setting a
        # group that doesn't have RGBA-style entries we'll inject.
        layer = self._layer_with_active(seq, "RGBA")
        stack.add(layer)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            sig_rgba = cache._signature_at_with_override(
                1001, layer.id, "RGBA",
            )
            sig_albedo = cache._signature_at_with_override(
                1001, layer.id, "albedo",
            )
            sig_emission = cache._signature_at_with_override(
                1001, layer.id, "emission",
            )
            arr = np.zeros((100, 100, 3), dtype=np.float32)  # ~120 KB
            nbytes_per = arr.nbytes
            # Inject 3 frames per channel — enough to verify "whole
            # channel clumped" vs "spread across channels".
            with cache._lock:
                for mf in (1001, 1002, 1003):
                    cache._frames[(mf, sig_rgba)] = arr
                    cache._frames[(mf, sig_albedo)] = arr
                    cache._frames[(mf, sig_emission)] = arr
                    cache._bytes_used += 3 * nbytes_per
                # Stamp first_cached: albedo cached FIRST (10s ago),
                # emission cached LATER (1s ago). RGBA cached most
                # recently but beauty-pinned so timing doesn't matter.
                cache._signature_first_cached[sig_albedo] = _time.monotonic() - 10.0
                cache._signature_first_cached[sig_emission] = _time.monotonic() - 1.0
                cache._signature_first_cached[sig_rgba] = _time.monotonic()
                # Switch the layer's live signature to something else
                # so ALL three injected sigs are tier 0 (stale). We
                # bump the offset which is part of the signature.
                layer.offset = 500  # detached from the injected sigs
                cache._invalidate_signature_cache()

                # Budget: enough for just 3 frames. Force eviction
                # to drop 6 frames (3 emission + 3 albedo first).
                cache._budget = 3 * nbytes_per
                cache._evict_if_over_budget()

                remaining_sigs = {sig for _mf, sig in cache._frames.keys()}
            # The 3 RGBA frames must survive (beauty-pinned).
            assert sig_rgba in remaining_sigs, (
                "RGBA snapshots were evicted despite the beauty-pass pin"
            )
            # Emission (newer non-beauty) should be GONE entirely —
            # the oldest-stamped beauty pin means emission is the
            # newest among non-beauty, evicted second by age order,
            # but we only had budget for 3 frames so both non-beauty
            # groups are out completely.
            assert sig_emission not in remaining_sigs
            assert sig_albedo not in remaining_sigs

        finally:
            cache.shutdown()

    def test_partial_eviction_drops_whole_channel_not_a_slice(
        self, qtbot,
    ) -> None:
        """Budget pressure that requires evicting *some* but not
        *all* non-beauty entries should still respect channel
        boundaries — drop the oldest channel entirely, leave the
        newer one intact. The old behaviour mixed: would have
        dropped e.g. 2 albedo frames + 2 emission frames at the
        same time."""
        import time as _time
        stack = LayerStack()
        seq = self._multi_aov_seq()
        layer = self._layer_with_active(seq, "RGBA")
        stack.add(layer)
        cache = MasterFrameCache(stack, num_workers=1)
        try:
            sig_albedo = cache._signature_at_with_override(
                1001, layer.id, "albedo",
            )
            sig_emission = cache._signature_at_with_override(
                1001, layer.id, "emission",
            )
            arr = np.zeros((100, 100, 3), dtype=np.float32)
            nbytes_per = arr.nbytes
            with cache._lock:
                for mf in (1001, 1002, 1003, 1004):
                    cache._frames[(mf, sig_albedo)] = arr
                    cache._frames[(mf, sig_emission)] = arr
                    cache._bytes_used += 2 * nbytes_per
                # albedo cached 10s ago (first), emission 1s ago.
                # First-cached channel leaves first under FIFO.
                cache._signature_first_cached[sig_albedo] = _time.monotonic() - 10.0
                cache._signature_first_cached[sig_emission] = _time.monotonic() - 1.0
                # Detach live state so both sigs are stale (tier 0).
                layer.offset = 500
                cache._invalidate_signature_cache()
                # Budget keeps 4 frames out of 8 — must drop exactly
                # the 4 albedo frames (oldest non-beauty) and leave
                # emission whole.
                cache._budget = 4 * nbytes_per
                cache._evict_if_over_budget()
                emission_left = sum(
                    1 for _mf, sig in cache._frames if sig == sig_emission
                )
                albedo_left = sum(
                    1 for _mf, sig in cache._frames if sig == sig_albedo
                )
            assert albedo_left == 0, (
                f"expected oldest channel (albedo) fully evicted, "
                f"{albedo_left} frames remain — eviction is still slicing "
                "across channels"
            )
            assert emission_left == 4, (
                f"expected newer channel (emission) fully preserved, "
                f"only {emission_left} frames left"
            )
        finally:
            cache.shutdown()
