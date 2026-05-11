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
