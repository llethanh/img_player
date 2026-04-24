"""Tests for io/formats.py — the list of OIIO-supported extensions."""

from __future__ import annotations

from pathlib import Path

import pytest

from img_player.io.formats import is_supported, supported_extensions


def test_supported_extensions_is_nonempty() -> None:
    exts = supported_extensions()
    assert len(exts) > 0


@pytest.mark.parametrize("ext", [".exr", ".png", ".jpg", ".tif", ".tiff"])
def test_common_extensions_are_supported(ext: str) -> None:
    assert ext in supported_extensions()


def test_is_supported_path_and_str() -> None:
    assert is_supported(Path("foo.png"))
    assert is_supported("foo.EXR")  # case-insensitive
    assert not is_supported(Path("foo.xyz"))


def test_supported_extensions_are_lowercase() -> None:
    exts = supported_extensions()
    for ext in exts:
        assert ext == ext.lower()
        assert ext.startswith(".")
