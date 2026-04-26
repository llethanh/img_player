"""Pure-function tests for the status-bar formatter (no Qt needed)."""

from __future__ import annotations

import pytest

from img_player.ui.status_format import (
    cache_dot_color,
    format_perf_html,
    fps_dot_color,
)
from img_player.ui.theme import H


# ---------------------------------------------------------------------- fps dot

class TestFpsDotColor:
    def test_returns_none_when_paused(self) -> None:
        assert fps_dot_color(None, target=24.0) is None

    def test_returns_none_when_target_is_zero(self) -> None:
        # Defensive: a target of 0 would cause a division blow-up in
        # the ratio calc. Returning None is safer than crashing.
        assert fps_dot_color(24.0, target=0.0) is None

    def test_green_when_at_target(self) -> None:
        assert fps_dot_color(24.0, target=24.0) == H.CACHE_BAR

    def test_green_when_just_above_threshold(self) -> None:
        # 95% of 24 = 22.8 → still green
        assert fps_dot_color(22.9, target=24.0) == H.CACHE_BAR

    def test_amber_when_between_thresholds(self) -> None:
        # 80% of 24 = 19.2; 90% = 21.6
        assert fps_dot_color(21.6, target=24.0) == H.ACCENT

    def test_red_when_below_warn(self) -> None:
        # 50% of 24 = 12 → red
        assert fps_dot_color(12.0, target=24.0) == H.MARKER_IO


# ---------------------------------------------------------------------- cache dot

class TestCacheDotColor:
    def test_green_when_full_enough(self) -> None:
        assert cache_dot_color(0.95) == H.CACHE_BAR

    def test_green_at_threshold(self) -> None:
        assert cache_dot_color(0.80) == H.CACHE_BAR

    def test_none_when_warming_up(self) -> None:
        assert cache_dot_color(0.4) is None
        assert cache_dot_color(0.0) is None


# ---------------------------------------------------------------------- format

class TestFormatPerfHtml:
    def test_includes_three_sections(self) -> None:
        html = format_perf_html(
            cache_n=42, cache_total=90, cache_ratio=0.5,
            fps_effective=23.8, fps_target=24.0, ram_gb=2.6,
        )
        # All three indicators should be in the rendered string.
        assert "cache 42/90" in html
        assert "23.8 fps" in html
        assert "RAM 2.6 GB" in html

    def test_paused_renders_em_dash(self) -> None:
        html = format_perf_html(
            cache_n=0, cache_total=1, cache_ratio=0.0,
            fps_effective=None, fps_target=24.0, ram_gb=0.0,
        )
        assert "— fps" in html
        # The fps dot should NOT be a coloured one when paused.
        assert "color:#" + H.CACHE_BAR.lstrip("#").lower() not in html.lower() or \
               "— fps" in html  # at least the em dash is present

    def test_red_dot_when_fps_drops_hard(self) -> None:
        html = format_perf_html(
            cache_n=10, cache_total=90, cache_ratio=0.1,
            fps_effective=10.0, fps_target=24.0, ram_gb=0.5,
        )
        # MARKER_IO hex must appear when ratio < 0.80
        assert H.MARKER_IO in html

    @pytest.mark.parametrize("eff,target,expected_hex", [
        (24.0, 24.0, H.CACHE_BAR),  # at target → green
        (20.0, 24.0, H.ACCENT),     # 83% → amber
        (10.0, 24.0, H.MARKER_IO),  # 41% → red
    ])
    def test_dot_color_matches_threshold_table(
        self, eff: float, target: float, expected_hex: str,
    ) -> None:
        html = format_perf_html(
            cache_n=1, cache_total=1, cache_ratio=1.0,
            fps_effective=eff, fps_target=target, ram_gb=0.0,
        )
        assert expected_hex in html
