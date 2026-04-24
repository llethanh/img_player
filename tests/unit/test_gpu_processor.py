"""Tests for color/gpu_processor.py — ShaderBundle generation."""

from __future__ import annotations

import numpy as np
import pytest

from img_player.color.gpu_processor import (
    _OCIO_PLACEHOLDER,
    ShaderBundle,
    build_shader_bundle,
)
from img_player.color.ocio_manager import OCIOManager


@pytest.fixture(scope="module")
def manager() -> OCIOManager:
    return OCIOManager()


def test_bundle_has_compile_ready_sources(manager: OCIOManager) -> None:
    cs = manager.role("scene_linear")
    assert cs is not None
    bundle = build_shader_bundle(
        manager,
        source_colorspace=cs,
        display=manager.default_display(),
        view=manager.default_view(),
    )
    assert isinstance(bundle, ShaderBundle)
    assert bundle.vertex_source.startswith("#version")
    assert bundle.fragment_source.startswith("#version")
    assert _OCIO_PLACEHOLDER not in bundle.fragment_source
    assert "OCIOMain" in bundle.fragment_source
    assert "uImage" in bundle.fragment_source
    assert "uExposure" in bundle.fragment_source
    assert "uGamma" in bundle.fragment_source


def test_bundle_exposes_colorspace_metadata(manager: OCIOManager) -> None:
    cs = manager.role("scene_linear")
    display = manager.default_display()
    view = manager.default_view(display)
    bundle = build_shader_bundle(manager, source_colorspace=cs, display=display, view=view)
    assert bundle.source_colorspace == cs
    assert bundle.display == display
    assert bundle.view == view


def test_lut_textures_are_float32_arrays(manager: OCIOManager) -> None:
    # ACES display transforms are rich enough to include 1D LUTs.
    bundle = build_shader_bundle(
        manager,
        source_colorspace="ACEScg",
        display=manager.default_display(),
        view=manager.default_view(),
    )
    for tex in bundle.textures_1d:
        assert tex.values.dtype == np.float32
        assert tex.channel in ("RED", "RGB")
        if tex.channel == "RED":
            assert tex.values.ndim == 1
        else:
            assert tex.values.ndim == 2
            assert tex.values.shape[1] == 3
    for tex3 in bundle.textures_3d:
        assert tex3.values.shape == (tex3.edge, tex3.edge, tex3.edge, 3)


def test_identity_transform_still_builds(manager: OCIOManager) -> None:
    # source == display/view combination may not be common but should at least
    # produce a valid ShaderBundle (it'll be shorter).
    cs = manager.role("scene_linear")
    assert cs is not None
    bundle = build_shader_bundle(
        manager,
        source_colorspace=cs,
        display=manager.default_display(),
        view=manager.default_view(),
    )
    assert len(bundle.fragment_source) > 0
