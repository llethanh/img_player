"""Tests for :class:`MasterFrameCache` — invalidation logic + master-frame
resolution. Decoding itself is stubbed so the tests don't touch OIIO.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

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
