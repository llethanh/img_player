"""Tests for :class:`img_player.annotate.stroke.Stroke`.

The stroke is the atomic unit. Tests cover construction validation
(must have at least one point, positive size, valid hex color),
immutability (frozen dataclass), and JSON round-trip.
"""

from __future__ import annotations

import pytest

from img_player.annotate.stroke import Stroke


class TestConstruction:
    def test_minimal_stroke_is_one_point(self) -> None:
        """A click without movement is still a valid stroke (rendered
        as a dot). The lower bound is 1 point, not 2."""
        s = Stroke(points=((10.0, 20.0),), color="#FF0000", size=5.0)
        assert s.points == ((10.0, 20.0),)
        assert s.color == "#FF0000"
        assert s.size == 5.0

    def test_zero_points_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one point"):
            Stroke(points=(), color="#FF0000", size=5.0)

    def test_zero_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="size must be positive"):
            Stroke(points=((0.0, 0.0),), color="#FF0000", size=0.0)

    def test_negative_size_rejected(self) -> None:
        with pytest.raises(ValueError, match="size must be positive"):
            Stroke(points=((0.0, 0.0),), color="#FF0000", size=-1.0)

    @pytest.mark.parametrize(
        "color",
        # No '#' prefix; or wrong-length hex (4 digits, 5 digits); or
        # CSS-style names; or bogus filler.
        ["FF0000", "#FFFF", "#FFFFF", "red", "rgb(255,0,0)", "#TOOLONG12345"],
    )
    def test_invalid_color_format_rejected(self, color: str) -> None:
        with pytest.raises(ValueError, match="hex string"):
            Stroke(points=((0.0, 0.0),), color=color, size=5.0)

    @pytest.mark.parametrize(
        "color",
        ["#F00", "#FF0000", "#FF000080"],  # 3, 6, and 8 hex digits
    )
    def test_valid_hex_formats_accepted(self, color: str) -> None:
        Stroke(points=((0.0, 0.0),), color=color, size=5.0)


class TestImmutability:
    def test_dataclass_is_frozen(self) -> None:
        """The undo stack relies on Stroke being immutable — a frozen
        dataclass refuses attribute assignment, which is what we want."""
        s = Stroke(points=((0.0, 0.0),), color="#FF0000", size=5.0)
        with pytest.raises((AttributeError, TypeError)):
            s.size = 10.0  # type: ignore[misc]

    def test_points_tuple_is_immutable(self) -> None:
        """Points are stored as a tuple, which is itself immutable —
        no caller can sneak in a list-mutation that re-orders the
        polyline."""
        s = Stroke(points=((1.0, 2.0), (3.0, 4.0)), color="#FF0000", size=5.0)
        assert isinstance(s.points, tuple)
        # Tuples don't support item assignment.
        with pytest.raises(TypeError):
            s.points[0] = (99.0, 99.0)  # type: ignore[index]

    def test_strokes_are_hashable(self) -> None:
        """Frozen dataclasses with hashable fields are hashable —
        useful for set deduplication / dict keys in future features."""
        s1 = Stroke(points=((0.0, 0.0),), color="#FF0000", size=5.0)
        s2 = Stroke(points=((0.0, 0.0),), color="#FF0000", size=5.0)
        assert hash(s1) == hash(s2)
        assert {s1, s2} == {s1}


class TestJsonRoundTrip:
    def test_to_dict_and_back_preserves_value(self) -> None:
        """The on-disk format must round-trip without loss."""
        original = Stroke(
            points=((1024.5, 532.0), (1028.1, 535.7), (1031.0, 540.2)),
            color="#E84A4A",
            size=5.0,
        )
        out = Stroke.from_dict(original.to_dict())
        assert out == original

    def test_to_dict_uses_lists_for_points(self) -> None:
        """JSON has no tuple type — points must serialise as lists so
        downstream JSON tooling doesn't see a Python-only repr."""
        s = Stroke(points=((1.0, 2.0),), color="#FF0000", size=5.0)
        d = s.to_dict()
        assert d["points"] == [[1.0, 2.0]]
        assert isinstance(d["points"][0], list)

    def test_from_dict_coerces_int_points_to_float(self) -> None:
        """The from_dict path tolerates int coords (e.g. produced by
        a hand-written sidecar) and casts them to float."""
        d = {"points": [[10, 20]], "color": "#FF0000", "size": 5}
        s = Stroke.from_dict(d)
        assert s.points == ((10.0, 20.0),)
        assert isinstance(s.points[0][0], float)
        assert s.size == 5.0
        assert isinstance(s.size, float)

    def test_from_dict_missing_field_raises(self) -> None:
        """from_dict is the strict layer — the persistence module
        catches the exception and skips the bad stroke."""
        with pytest.raises(KeyError):
            Stroke.from_dict({"points": [[0, 0]], "size": 5.0})  # missing color
