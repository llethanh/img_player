"""Tests for color/ocio_manager.py."""

from __future__ import annotations

import pytest

from img_player.color.ocio_manager import OCIOManager


@pytest.fixture(scope="module")
def manager() -> OCIOManager:
    return OCIOManager()


def test_builtin_fallback_loads(manager: OCIOManager) -> None:
    assert manager.source.origin in ("builtin", "env")
    # The built-in default config always exposes at least a handful of colorspaces.
    assert len(manager.list_colorspaces()) >= 5


def test_default_display_and_view(manager: OCIOManager) -> None:
    display = manager.default_display()
    assert display
    view = manager.default_view(display)
    assert view
    assert view in manager.list_views(display)


def test_scene_linear_role_exists(manager: OCIOManager) -> None:
    # Any sensible config has the scene_linear role.
    assert manager.role("scene_linear") is not None


def test_explicit_config_is_preserved() -> None:
    mgr1 = OCIOManager()
    mgr2 = OCIOManager(config=mgr1.config)
    assert mgr2.source.origin == "explicit"
    assert mgr2.list_colorspaces() == mgr1.list_colorspaces()


def test_get_processor_round_trip(manager: OCIOManager) -> None:
    # Identity transform should produce a valid processor.
    cs = manager.role("scene_linear")
    assert cs is not None
    proc = manager.get_processor(cs, cs)
    assert proc is not None


def test_get_display_view_processor(manager: OCIOManager) -> None:
    cs = manager.role("scene_linear")
    assert cs is not None
    display = manager.default_display()
    view = manager.default_view(display)
    proc = manager.get_display_view_processor(cs, display, view)
    assert proc is not None
