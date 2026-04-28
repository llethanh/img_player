"""Tests for the pure :func:`alpha_at` helper in
``img_player.annotate.ephemeral``.

Level 1 of the ephemeral feature's testing strategy (spec
§8.1) — no Qt, no QObject, no widget. Just the math.
"""

from __future__ import annotations

import pytest

from img_player.annotate.ephemeral import alpha_at


class TestAlphaAt:
    def test_birth_is_full_opacity(self) -> None:
        """At ``now == birth`` the stroke is brand new — alpha 1.0."""
        assert alpha_at(birth_ts=0.0, now_ts=0.0, duration_s=5.0) == 1.0

    def test_end_of_life_is_transparent(self) -> None:
        """At ``now == birth + duration`` the stroke just expired."""
        assert alpha_at(birth_ts=0.0, now_ts=5.0, duration_s=5.0) == 0.0

    def test_midlife_is_half(self) -> None:
        """Linear curve: at half-duration we should read 0.5 exactly."""
        assert alpha_at(birth_ts=0.0, now_ts=2.5, duration_s=5.0) == 0.5

    def test_quarter_life(self) -> None:
        """Sanity check at 25 % age — 0.75 alpha."""
        assert alpha_at(birth_ts=0.0, now_ts=1.25, duration_s=5.0) == 0.75

    def test_beyond_duration_clamps_to_zero(self) -> None:
        """Past the death point we never go negative."""
        assert alpha_at(birth_ts=0.0, now_ts=100.0, duration_s=5.0) == 0.0

    def test_negative_age_clamps_to_one(self) -> None:
        """Clock jitter — ``now < birth`` shouldn't make the stroke
        spookily faint. We treat age as 0 → alpha 1.0."""
        # E.g. birth at monotonic = 100, but a suspend-resume reads
        # now = 90 on the next tick. Defensive clamp.
        assert alpha_at(birth_ts=100.0, now_ts=90.0, duration_s=5.0) == 1.0

    def test_zero_duration_is_instant_death(self) -> None:
        """If the user somehow set duration to 0, every stroke dies
        on the next tick — no division-by-zero crash."""
        assert alpha_at(birth_ts=0.0, now_ts=1.0, duration_s=0.0) == 0.0

    def test_negative_duration_is_instant_death(self) -> None:
        """Defensive: a negative duration is invalid input but we
        don't crash. Same outcome as zero."""
        assert alpha_at(birth_ts=0.0, now_ts=1.0, duration_s=-3.0) == 0.0

    @pytest.mark.parametrize(
        ("now_ts", "expected"),
        [
            (0.0, 1.0),
            (1.0, 0.9),
            (2.0, 0.8),
            (5.0, 0.5),
            (9.0, 0.1),
            (10.0, 0.0),
            (10.5, 0.0),
        ],
    )
    def test_linear_curve_10s_duration(
        self, now_ts: float, expected: float
    ) -> None:
        """Spot-check the linear ramp at a 10 s duration."""
        result = alpha_at(birth_ts=0.0, now_ts=now_ts, duration_s=10.0)
        assert result == pytest.approx(expected)

    def test_birth_offset_is_respected(self) -> None:
        """Time can start anywhere — only the *delta* matters."""
        # birth at t=1000, now at t=1003, 5s duration → 2/5 elapsed
        assert alpha_at(
            birth_ts=1000.0, now_ts=1003.0, duration_s=5.0
        ) == pytest.approx(0.4)

    def test_returns_float(self) -> None:
        """Type sanity — the overlay multiplies this into a QColor's
        alpha channel and we'd rather not surprise it with an int."""
        result = alpha_at(birth_ts=0.0, now_ts=2.5, duration_s=5.0)
        assert isinstance(result, float)
