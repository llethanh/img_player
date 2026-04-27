"""Pure-math tests for :mod:`img_player.annotate.overlay`.

The QWidget itself can't be unit-tested without a Qt event loop —
verified manually via the GUI test plan in the PR. But the three
module-level helpers (``widget_to_image``, ``image_to_widget``,
``nearest_stroke_index``) are pure functions with concrete contracts
worth pinning here.
"""

from __future__ import annotations

import math

import pytest

from img_player.annotate.overlay import (
    image_to_widget,
    nearest_stroke_index,
    widget_to_image,
)
from img_player.annotate.stroke import Stroke


# ============================================================================
# widget_to_image / image_to_widget — coordinate transform
# ============================================================================


class TestRoundTrip:
    """Whatever the transform parameters, applying ``widget_to_image``
    then ``image_to_widget`` (or vice versa) must return the input
    point modulo float rounding. This is the headline contract — if
    it breaks, every stroke is rendered at the wrong place."""

    @pytest.mark.parametrize(
        ("widget_xy", "factor", "pan", "img_size", "win_size"),
        [
            ((512.0, 384.0), 1.0, (0.0, 0.0), (1024, 768), (1024, 768)),
            ((100.0, 200.0), 0.5, (50.0, -25.0), (4096, 2160), (1280, 720)),
            ((0.0, 0.0), 2.0, (10.0, 10.0), (1920, 1080), (960, 540)),
            ((1280.0, 720.0), 0.314, (-7.5, 12.3), (3840, 2160), (1280, 720)),
        ],
    )
    def test_widget_to_image_to_widget(
        self,
        widget_xy: tuple[float, float],
        factor: float,
        pan: tuple[float, float],
        img_size: tuple[int, int],
        win_size: tuple[int, int],
    ) -> None:
        image_xy = widget_to_image(
            widget_xy=widget_xy,
            widget_size=win_size,
            img_size=img_size,
            factor=factor,
            pan=pan,
        )
        back = image_to_widget(
            image_xy=image_xy,
            widget_size=win_size,
            img_size=img_size,
            factor=factor,
            pan=pan,
        )
        assert math.isclose(back[0], widget_xy[0], abs_tol=1e-9)
        assert math.isclose(back[1], widget_xy[1], abs_tol=1e-9)


class TestKnownPoints:
    """Anchor cases: hand-computable so a regression that breaks the
    formula on a specific axis trips the test loudly."""

    def test_centred_at_actual_size_no_pan(self) -> None:
        """At factor=1, no pan, widget centre maps to image centre."""
        result = widget_to_image(
            widget_xy=(512.0, 384.0),
            widget_size=(1024, 768),
            img_size=(1024, 768),
            factor=1.0,
            pan=(0.0, 0.0),
        )
        assert math.isclose(result[0], 512.0)
        assert math.isclose(result[1], 384.0)

    def test_pan_shifts_centre(self) -> None:
        """Pan (50, 0) puts the image 50px right of widget centre →
        widget centre is now 50 image-pixels LEFT of the image centre."""
        result = widget_to_image(
            widget_xy=(512.0, 384.0),
            widget_size=(1024, 768),
            img_size=(1024, 768),
            factor=1.0,
            pan=(50.0, 0.0),
        )
        assert math.isclose(result[0], 512.0 - 50.0)  # = 462.0
        assert math.isclose(result[1], 384.0)

    def test_zoom_2x_halves_image_offset(self) -> None:
        """At zoom 2, a widget pixel 100 right of centre is only 50
        image-pixels right of the image centre (image is bigger on
        screen, so the same widget distance covers fewer image px)."""
        result = widget_to_image(
            widget_xy=(612.0, 384.0),  # 100px right of centre
            widget_size=(1024, 768),
            img_size=(1024, 768),
            factor=2.0,
            pan=(0.0, 0.0),
        )
        # 100 widget-px / factor 2 = 50 image-px; plus img_w/2 = 512.
        assert math.isclose(result[0], 562.0)


# ============================================================================
# nearest_stroke_index — eraser hit-testing
# ============================================================================


def _line_stroke(p0: tuple[float, float], p1: tuple[float, float]) -> Stroke:
    return Stroke(points=(p0, p1), color="#FF0000", size=5.0)


class TestNearestStroke:
    def test_empty_strokes_returns_none(self) -> None:
        assert nearest_stroke_index((0.0, 0.0), [], hit_radius=8.0) is None

    def test_cursor_outside_radius_returns_none(self) -> None:
        s = _line_stroke((0.0, 0.0), (100.0, 0.0))
        # 100 px above the segment — far outside the 8 px radius.
        assert nearest_stroke_index((50.0, 100.0), [s], hit_radius=8.0) is None

    def test_cursor_on_segment_hits(self) -> None:
        s = _line_stroke((0.0, 0.0), (100.0, 0.0))
        # 5 px above the segment — within 8 px radius.
        assert nearest_stroke_index((50.0, 5.0), [s], hit_radius=8.0) == 0

    def test_picks_closest_among_multiple(self) -> None:
        """Two strokes within reach — return the one closer to the
        cursor, not the first or last."""
        far = _line_stroke((0.0, 7.0), (100.0, 7.0))   # 7 px above
        near = _line_stroke((0.0, 1.0), (100.0, 1.0))  # 1 px above
        # Cursor on the X axis (y=0) — both are within 8 px radius
        # but `near` is much closer.
        idx = nearest_stroke_index((50.0, 0.0), [far, near], hit_radius=8.0)
        assert idx == 1

    def test_one_point_stroke_uses_point_distance(self) -> None:
        """A stroke with a single point is rendered as a dot — the
        eraser hit-test must compute distance to that point, not
        treat it as a zero-length segment that could behave oddly."""
        dot = Stroke(points=((50.0, 50.0),), color="#FF0000", size=5.0)
        assert nearest_stroke_index((52.0, 52.0), [dot], hit_radius=8.0) == 0
        assert nearest_stroke_index((100.0, 100.0), [dot], hit_radius=8.0) is None

    def test_polyline_min_distance_across_segments(self) -> None:
        """A multi-segment polyline: the cursor is far from the first
        segment but close to the last. Hit-test must scan all
        segments, not just the first."""
        zigzag = Stroke(
            points=(
                (0.0, 0.0),
                (10.0, 0.0),
                (10.0, 100.0),
                (50.0, 100.0),  # last segment near the cursor
            ),
            color="#FF0000",
            size=5.0,
        )
        # Cursor at (40, 102) — close to the last segment.
        assert nearest_stroke_index((40.0, 102.0), [zigzag], hit_radius=8.0) == 0

    def test_hit_radius_boundary(self) -> None:
        """A cursor exactly at hit_radius should still hit (≤, not <)."""
        s = _line_stroke((0.0, 0.0), (100.0, 0.0))
        # Exactly 8 px above.
        assert nearest_stroke_index((50.0, 8.0), [s], hit_radius=8.0) == 0
        # Just above.
        assert (
            nearest_stroke_index((50.0, 8.001), [s], hit_radius=8.0) is None
        )
