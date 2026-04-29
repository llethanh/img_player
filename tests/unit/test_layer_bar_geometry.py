"""Pure-function tests for ``BarGeometry`` + ``snap_master_frame``.

The Qt drag interactions live in :class:`LayerBar` and need a real
event loop to exercise; the math underneath is plain Python and can
be pinned here with no Qt at all.
"""

from __future__ import annotations

import pytest

from img_player.ui.layer_bar import BarGeometry, PADDING_X, snap_master_frame


# ============================================================================
# BarGeometry — frame ↔ pixel conversion
# ============================================================================


class TestBarGeometry:
    def test_first_frame_at_left_edge(self) -> None:
        geom = BarGeometry(width=200, master_first=0, master_last=99)
        # "Frame as point": ``master_first`` lands exactly at
        # ``PADDING_X`` (= the bar's drawable left edge), pixel-flush
        # with the timeline's first tick.
        assert abs(geom.frame_to_x(0) - PADDING_X) < 1e-6

    def test_last_frame_at_right_edge(self) -> None:
        geom = BarGeometry(width=200, master_first=0, master_last=99)
        # ``master_last`` lands exactly at the drawable right edge.
        right_edge = 200 - PADDING_X
        assert abs(geom.frame_to_x(99) - right_edge) < 1e-6

    def test_x_to_frame_round_trip(self) -> None:
        geom = BarGeometry(width=200, master_first=0, master_last=99)
        for f in (0, 25, 50, 75, 99):
            x = geom.frame_to_x(f)
            assert geom.x_to_frame(x) == f

    def test_master_length_with_negative_first(self) -> None:
        # Negative offsets are legal (Layer.master_start can be < 0).
        # Length is the *step count* (last - first), not the frame
        # count, so [-10, 10] has length 20.
        geom = BarGeometry(width=200, master_first=-10, master_last=10)
        assert geom.master_length == 20

    def test_zero_width_falls_back_safely(self) -> None:
        # Defensive: widget might be 0 px wide before its first
        # resize. Math must not divide by zero.
        geom = BarGeometry(width=0, master_first=0, master_last=10)
        # Just check no crash.
        geom.frame_to_x(5)
        geom.x_to_frame(0)


# ============================================================================
# snap_master_frame
# ============================================================================


class TestSnap:
    def _geom(self) -> BarGeometry:
        return BarGeometry(width=200, master_first=0, master_last=99)

    def test_no_targets_returns_candidate(self) -> None:
        g = self._geom()
        assert snap_master_frame(50, g, [], snap_distance_frames=5) == 50

    def test_within_snap_distance_uses_target(self) -> None:
        g = self._geom()
        # Target 100, candidate 97 → within 5 frames → snap to 100.
        assert snap_master_frame(97, g, [100], 5) == 100

    def test_beyond_snap_distance_keeps_candidate(self) -> None:
        g = self._geom()
        # Target 100, candidate 80 → 20 frames away → keep 80.
        assert snap_master_frame(80, g, [100], 5) == 80

    def test_picks_closest_target(self) -> None:
        g = self._geom()
        # Two targets within range — closer one wins.
        assert snap_master_frame(50, g, [48, 52, 100], 5) == 48
        assert snap_master_frame(51, g, [48, 52, 100], 5) == 52

    def test_exact_distance_includes_boundary(self) -> None:
        g = self._geom()
        # Distance == snap_distance is still a snap.
        assert snap_master_frame(95, g, [100], 5) == 100
