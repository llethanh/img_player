"""Smoke tests: package imports and version exposes correctly."""

from __future__ import annotations

import img_player


def test_package_imports() -> None:
    assert hasattr(img_player, "__version__")


def test_version_is_semver_like() -> None:
    parts = img_player.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
