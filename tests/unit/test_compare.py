"""Tests for the two-layer compare overlay.

Covers :class:`CompareState` (round-trip, defaults, clamping),
:mod:`img_player.compare.compose` (the four blend modes + size
matching), and a small smoke test of :class:`CompareDecoder`'s
caching.

Pure data + numpy; no Qt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from img_player.compare import (
    MODE_HORIZONTAL,
    MODE_OPACITY,
    MODE_VERTICAL,
    CompareState,
)
from img_player.compare.compose import compose
from img_player.compare.decode import CompareDecoder


# ============================================================================
# CompareState
# ============================================================================


class TestCompareState:
    def test_defaults(self) -> None:
        s = CompareState()
        assert s.enabled is False
        assert s.layer_a_id is None
        assert s.layer_b_id is None
        assert s.mode == MODE_VERTICAL  # default after swap-mode retired
        assert s.seam == 0.5
        assert s.swap_showing_b is False

    def test_is_active_requires_enabled_and_both_layers(self) -> None:
        s = CompareState(enabled=True)
        assert not s.is_active()  # both ids missing
        s = CompareState(enabled=True, layer_a_id="a")
        assert not s.is_active()  # B missing
        s = CompareState(enabled=True, layer_a_id="a", layer_b_id="b")
        assert s.is_active()
        s = CompareState(enabled=False, layer_a_id="a", layer_b_id="b")
        assert not s.is_active()  # disabled

    def test_seam_clamping(self) -> None:
        s = CompareState()
        assert s.with_seam(-0.5).seam == 0.0
        assert s.with_seam(2.0).seam == 1.0
        assert s.with_seam(0.42).seam == 0.42

    def test_round_trip(self) -> None:
        original = CompareState(
            enabled=True,
            layer_a_id="aaa",
            layer_b_id="bbb",
            mode=MODE_VERTICAL,
            seam=0.33,
            swap_showing_b=True,
        )
        roundtripped = CompareState.from_dict(original.to_dict())
        assert roundtripped == original

    def test_from_dict_recovers_from_garbage(self) -> None:
        # Unknown mode → falls back to default; non-numeric seam → 0.5;
        # seam out of range → clamped.
        s = CompareState.from_dict({"mode": "weird", "seam": "nope"})
        assert s.mode == MODE_VERTICAL
        assert s.seam == 0.5
        s2 = CompareState.from_dict({"seam": 2.5})
        assert s2.seam == 1.0

    def test_legacy_swap_mode_migrates_to_solo_b(self) -> None:
        # Pre-1.2.1 sessions saved ``mode == "swap"``. Loading them
        # should preserve the visual ("show full B") via the new
        # always-on ``swap_showing_b`` toggle.
        s = CompareState.from_dict({"mode": "swap"})
        assert s.mode == MODE_VERTICAL
        assert s.swap_showing_b is True

    def test_from_dict_skipped_keys_use_defaults(self) -> None:
        s = CompareState.from_dict({})
        assert s == CompareState()


# ============================================================================
# compose — modes
# ============================================================================


def _block(value: int, h: int = 4, w: int = 4, c: int = 3) -> np.ndarray:
    """Solid-coloured ndarray, same value in every channel/pixel."""
    return np.full((h, w, c), value, dtype=np.uint8)


class TestComposeSoloB:
    """``swap_showing_b`` is an override that returns full B
    regardless of the picked blend mode."""

    def test_returns_b_when_override_on_opacity(self) -> None:
        a, b = _block(10), _block(200)
        out = compose(a, b, mode=MODE_OPACITY, seam=0.0, swap_showing_b=True)
        # Without the override, seam=0 would be pure A (= 10). The
        # override forces full B (= 200).
        assert np.array_equal(out, b)

    def test_returns_b_when_override_on_vertical_wipe(self) -> None:
        a, b = _block(10, w=8), _block(200, w=8)
        out = compose(a, b, mode=MODE_VERTICAL, seam=0.5, swap_showing_b=True)
        assert np.array_equal(out, b)

    def test_normal_compose_when_override_off(self) -> None:
        a, b = _block(10), _block(200)
        out = compose(a, b, mode=MODE_OPACITY, seam=0.0, swap_showing_b=False)
        assert np.array_equal(out, a)


class TestComposeOpacity:
    def test_pure_a_at_seam_zero(self) -> None:
        a, b = _block(0), _block(200)
        out = compose(a, b, mode=MODE_OPACITY, seam=0.0)
        assert np.array_equal(out, a)

    def test_pure_b_at_seam_one(self) -> None:
        a, b = _block(0), _block(200)
        out = compose(a, b, mode=MODE_OPACITY, seam=1.0)
        assert np.array_equal(out, b)

    def test_midpoint_average(self) -> None:
        a, b = _block(0), _block(200)
        out = compose(a, b, mode=MODE_OPACITY, seam=0.5)
        # Linear blend: 0 * 0.5 + 200 * 0.5 = 100.
        assert int(out[0, 0, 0]) == 100


class TestComposeVerticalWipe:
    def test_left_half_a_right_half_b(self) -> None:
        a = _block(10, h=4, w=8)
        b = _block(200, h=4, w=8)
        out = compose(
            a, b, mode=MODE_VERTICAL, seam=0.5, draw_seam_line=False,
        )
        assert (out[:, :4] == 10).all()
        assert (out[:, 4:] == 200).all()

    def test_seam_zero_all_b(self) -> None:
        a, b = _block(10, w=8), _block(200, w=8)
        out = compose(a, b, mode=MODE_VERTICAL, seam=0.0, draw_seam_line=False)
        assert (out == 200).all()

    def test_seam_one_all_a(self) -> None:
        a, b = _block(10, w=8), _block(200, w=8)
        out = compose(a, b, mode=MODE_VERTICAL, seam=1.0, draw_seam_line=False)
        assert (out == 10).all()

    def test_seam_line_blended_orange(self) -> None:
        a, b = _block(10, w=8), _block(200, w=8)
        out = compose(a, b, mode=MODE_VERTICAL, seam=0.5)
        # The seam is alpha-blended accent-orange (R≈232, G≈144,
        # B≈28) over the underlying B-side gray (= 200). With ~55%
        # alpha:
        #   R = 200*0.45 + 232*0.55 ≈ 217
        #   G = 200*0.45 + 144*0.55 ≈ 169
        #   B = 200*0.45 +  28*0.55 ≈  105
        # The R channel stays warm and brightest, B stays dimmest —
        # we just check the relative ordering rather than exact
        # rounded values so the test stays robust to small alpha
        # tweaks.
        seam_pixel = out[:, 4]
        assert (seam_pixel[..., 0] > seam_pixel[..., 1]).all()  # R > G
        assert (seam_pixel[..., 1] > seam_pixel[..., 2]).all()  # G > B


class TestComposeHorizontalWipe:
    def test_top_a_bottom_b(self) -> None:
        a = _block(10, h=8, w=4)
        b = _block(200, h=8, w=4)
        out = compose(
            a, b, mode=MODE_HORIZONTAL, seam=0.5, draw_seam_line=False,
        )
        assert (out[:4] == 10).all()
        assert (out[4:] == 200).all()


class TestComposeSizeMatching:
    def test_b_resized_to_a(self) -> None:
        a = _block(10, h=8, w=8)
        b = _block(200, h=4, w=4)  # half size
        out = compose(a, b, mode=MODE_OPACITY, seam=0.5)
        # Result is a's size — the resize made B match.
        assert out.shape == (8, 8, 3)

    def test_channel_padding(self) -> None:
        # RGB vs RGBA → both come out with same channel count.
        a = _block(10, c=3)
        b = _block(200, c=4)
        out = compose(a, b, mode=MODE_OPACITY, seam=0.5)
        assert out.shape[2] == 4


class TestComposeRejects:
    def test_2d_arrays_raise(self) -> None:
        a = np.zeros((4, 4), dtype=np.uint8)
        b = np.zeros((4, 4), dtype=np.uint8)
        with pytest.raises(ValueError, match="HxWxC"):
            compose(a, b, mode=MODE_OPACITY, seam=0.5)

    def test_unknown_mode_raises(self) -> None:
        a, b = _block(0), _block(0)
        with pytest.raises(ValueError, match="Unknown compare mode"):
            compose(a, b, mode="bogus", seam=0.5)


# ============================================================================
# CompareDecoder — caching behaviour
# ============================================================================


class _FakeLayer:
    """Minimal Layer-shaped object for the decoder's image path."""

    def __init__(
        self, layer_id: str, source_frame: int, path: Path,
    ) -> None:
        self.id = layer_id
        self._source_frame = source_frame
        # Mimic ``layer.sequence.frames`` enough to satisfy the
        # decoder's path lookup.
        from types import SimpleNamespace

        self.sequence = SimpleNamespace(
            frames=[SimpleNamespace(frame_number=source_frame, path=path)],
        )
        self.is_video = False
        self.video_metadata = None
        self.channel_selection = None

    def covers(self, _master_frame: int) -> bool:
        return True

    def source_frame_at(self, _master_frame: int) -> int:
        return self._source_frame


class TestCompareDecoderCaching:
    def test_invalidate_clears_cache(self, tmp_path: Path) -> None:
        decoder = CompareDecoder(video_sources=None)
        # Inject a fake last-decode entry directly so we don't need
        # an actual OIIO read to hit the cache path.
        from img_player.compare.decode import _LastDecode

        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        decoder._last["fake"] = _LastDecode(
            layer_id="fake", source_frame=0, arr=arr,
        )
        assert "fake" in decoder._last
        decoder.invalidate()
        assert "fake" not in decoder._last

    def test_invalidate_specific_id(self) -> None:
        decoder = CompareDecoder(video_sources=None)
        from img_player.compare.decode import _LastDecode

        arr = np.zeros((2, 2, 3), dtype=np.uint8)
        decoder._last["a"] = _LastDecode("a", 0, arr)
        decoder._last["b"] = _LastDecode("b", 0, arr)
        decoder.invalidate(layer_id="a")
        assert "a" not in decoder._last
        assert "b" in decoder._last

    def test_decode_returns_none_for_out_of_range_layer(
        self, tmp_path: Path,
    ) -> None:
        layer = _FakeLayer("x", source_frame=0, path=tmp_path / "missing.png")

        # Override covers() to simulate an out-of-range master frame.
        layer.covers = lambda _f: False  # type: ignore[assignment]
        decoder = CompareDecoder(video_sources=None)
        assert decoder.decode(layer, master_frame=100) is None
