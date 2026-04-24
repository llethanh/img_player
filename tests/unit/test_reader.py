"""Tests for io/reader.py — OIIO-backed frame decoding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from img_player.io.reader import FrameReadError, read_frame, read_header


def test_read_png_returns_float32_hwc(png_path: Path) -> None:
    arr = read_frame(png_path)
    assert arr.dtype == np.float32
    assert arr.ndim == 3
    assert arr.shape[2] in (3, 4)
    assert 0.0 <= arr.min() and arr.max() <= 1.0


def test_read_exr_returns_rgba(exr_path: Path) -> None:
    arr = read_frame(exr_path)
    assert arr.shape[2] == 4
    assert arr.dtype == np.float32


def test_read_multichannel_exr_full(exr_multichannel_path: Path) -> None:
    arr = read_frame(exr_multichannel_path)
    assert arr.shape[2] == 6  # R, G, B, A, Z, AO


def test_read_multichannel_exr_subset(exr_multichannel_path: Path) -> None:
    arr = read_frame(exr_multichannel_path, channels=["Z", "AO"])
    assert arr.shape[2] == 2
    # Z was set to 10.0, AO to 0.75 in the fixture
    assert np.allclose(arr[:, :, 0], 10.0)
    assert np.allclose(arr[:, :, 1], 0.75)


def test_read_multichannel_missing_channel_raises(exr_multichannel_path: Path) -> None:
    with pytest.raises(FrameReadError, match="not found"):
        read_frame(exr_multichannel_path, channels=["R", "ZZZ"])


def test_read_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FrameReadError, match="not found"):
        read_frame(tmp_path / "nope.exr")


def test_read_corrupt_file_raises(corrupt_file: Path) -> None:
    with pytest.raises(FrameReadError):
        read_frame(corrupt_file)


def test_read_header_returns_spec(exr_path: Path) -> None:
    spec = read_header(exr_path)
    assert spec.width > 0
    assert spec.height > 0
    assert spec.nchannels == 4
