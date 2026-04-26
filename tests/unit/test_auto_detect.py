"""Pure-function tests for the source-colorspace auto-detector.

We pin every level of the cascade:
* explicit tags win over OIIO's tag, which wins over chromaticities,
  which wins over the extension fallback;
* unknown tags / mismatched chromaticities don't crash;
* canonical names are matched fuzzily against the OCIO config so
  "ACES - ACEScg" picks up our "ACEScg" claim.
"""

from __future__ import annotations

import pytest

from img_player.color.auto_detect import (
    DetectionResult,
    detect_source_colorspace,
)


# A representative OCIO config — the names mirror what the ACES studio
# config exposes (close enough to the built-in ocio://default).
OCIO_CONFIG = [
    "Raw",
    "ACES - ACES2065-1",
    "ACES - ACEScg",
    "ACES - ACEScct",
    "Utility - Linear - sRGB",
    "Utility - Linear - Rec.709",
    "Utility - Linear - Rec.2020",
    "Output - sRGB",
    "Output - Rec.709",
    "Cineon",
    "scene_linear",
]


# ---------------------------------------------------------------------- Cascade levels

class TestExplicitColorspaceTag:
    def test_colorspace_name_attribute(self) -> None:
        meta = {"colorSpaceName": "ACES - ACEScg"}
        result = detect_source_colorspace(meta, ".exr", OCIO_CONFIG)
        assert result.colorspace == "ACES - ACEScg"
        assert "colorSpaceName" in result.reason

    def test_nuke_colorspace_uk_spelling(self) -> None:
        meta = {"nuke/input/colourspace": "linear"}
        result = detect_source_colorspace(meta, ".exr", OCIO_CONFIG)
        # "linear" appears as a substring in "Utility - Linear - sRGB" etc.
        assert result.colorspace is not None
        assert "linear" in result.colorspace.lower()

    def test_unknown_explicit_tag_returns_none_with_reason(self) -> None:
        meta = {"colorSpaceName": "Nonexistent CS Studio Internal"}
        result = detect_source_colorspace(meta, ".exr", OCIO_CONFIG)
        assert result.colorspace is None
        assert "Nonexistent" in result.reason


class TestOiioColorspace:
    def test_srgb_via_oiio_tag(self) -> None:
        meta = {"oiio:ColorSpace": "sRGB"}
        result = detect_source_colorspace(meta, ".png", OCIO_CONFIG)
        assert result.colorspace is not None
        assert "sRGB" in result.colorspace
        assert "sRGB" in result.reason

    def test_acescg_via_oiio_tag(self) -> None:
        meta = {"oiio:ColorSpace": "ACEScg"}
        result = detect_source_colorspace(meta, ".exr", OCIO_CONFIG)
        assert result.colorspace == "ACES - ACEScg"

    def test_alias_linear_picks_scene_linear(self) -> None:
        meta = {"oiio:ColorSpace": "Linear"}
        result = detect_source_colorspace(meta, ".exr", OCIO_CONFIG)
        assert result.colorspace is not None
        assert "linear" in result.colorspace.lower()


class TestChromaticitiesGamut:
    def test_acescg_primaries_match_ap1(self) -> None:
        # Exact ACES AP1 primaries.
        chroma = (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)
        result = detect_source_colorspace(
            {"chromaticities": chroma}, ".exr", OCIO_CONFIG
        )
        assert result.colorspace == "ACES - ACEScg"
        assert "chromaticities" in result.reason

    def test_rec709_primaries_match(self) -> None:
        chroma = (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.329)
        result = detect_source_colorspace(
            {"chromaticities": chroma}, ".exr", OCIO_CONFIG
        )
        assert result.colorspace is not None
        assert "Rec.709" in result.colorspace

    def test_chromaticities_within_tolerance_still_match(self) -> None:
        # +0.003 on each channel — under our 0.005 tolerance.
        chroma = (0.713 + 0.003, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)
        result = detect_source_colorspace(
            {"chromaticities": chroma}, ".exr", OCIO_CONFIG
        )
        assert result.colorspace == "ACES - ACEScg"

    def test_chromaticities_outside_tolerance_do_not_match(self) -> None:
        # +0.05 on red.x — clearly out of any standard gamut.
        chroma = (0.85, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)
        result = detect_source_colorspace(
            {"chromaticities": chroma}, ".exr", OCIO_CONFIG
        )
        # Should fall through to extension fallback.
        assert result.colorspace == "scene_linear"
        # And the reason names the fallback (so the user knows we
        # didn't actually match the gamut).
        assert "EXR" in result.reason or "scene_linear" in result.reason

    def test_chromaticities_wrong_length_ignored(self) -> None:
        result = detect_source_colorspace(
            {"chromaticities": (0.7, 0.3)}, ".exr", OCIO_CONFIG
        )
        # No crash, falls to extension.
        assert result.colorspace == "scene_linear"


class TestExtensionFallback:
    def test_exr_no_metadata_uses_scene_linear(self) -> None:
        result = detect_source_colorspace({}, ".exr", OCIO_CONFIG)
        assert result.colorspace == "scene_linear"

    def test_png_no_metadata_uses_srgb_family(self) -> None:
        result = detect_source_colorspace({}, ".png", OCIO_CONFIG)
        assert result.colorspace is not None
        assert "sRGB" in result.colorspace or "srgb" in result.colorspace.lower()

    def test_dpx_no_metadata_uses_cineon(self) -> None:
        result = detect_source_colorspace({}, ".dpx", OCIO_CONFIG)
        assert result.colorspace == "Cineon"

    def test_unknown_extension_returns_none(self) -> None:
        result = detect_source_colorspace({}, ".raw", OCIO_CONFIG)
        assert result.colorspace is None
        assert "no signal" in result.reason


class TestCascadePriority:
    """Higher-priority signals win over lower-priority ones."""

    def test_explicit_tag_beats_extension(self) -> None:
        # PNG would extension-fallback to sRGB — but the file says ACEScg.
        meta = {"colorSpaceName": "ACES - ACEScg"}
        result = detect_source_colorspace(meta, ".png", OCIO_CONFIG)
        assert result.colorspace == "ACES - ACEScg"

    def test_oiio_tag_beats_chromaticities(self) -> None:
        # Inconsistent file: oiio:ColorSpace says ACEScg but chromaticities
        # encode Rec.709. The explicit tag wins because it's a higher
        # cascade level.
        meta = {
            "oiio:ColorSpace": "ACEScg",
            "chromaticities": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.329),
        }
        result = detect_source_colorspace(meta, ".exr", OCIO_CONFIG)
        assert result.colorspace == "ACES - ACEScg"

    def test_chromaticities_beat_extension(self) -> None:
        chroma = (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)
        result = detect_source_colorspace(
            {"chromaticities": chroma}, ".exr", OCIO_CONFIG
        )
        # ACEScg wins over scene_linear extension fallback.
        assert result.colorspace == "ACES - ACEScg"


class TestEmptyMetadata:
    def test_empty_metadata_no_extension_returns_none(self) -> None:
        result = detect_source_colorspace({}, ".weird_format", OCIO_CONFIG)
        assert result.colorspace is None

    def test_empty_metadata_empty_config(self) -> None:
        # Defensive — caller should never hand an empty config but the
        # function should at least return cleanly.
        result = detect_source_colorspace({}, ".exr", [])
        assert result.colorspace is None


class TestDetectionResult:
    def test_is_a_frozen_dataclass(self) -> None:
        # Belt-and-suspenders: results are immutable and equality-comparable.
        a = DetectionResult("ACES - ACEScg", "test")
        b = DetectionResult("ACES - ACEScg", "test")
        assert a == b
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            a.colorspace = "other"  # type: ignore[misc]
