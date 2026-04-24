"""Shared pytest fixtures: generate tiny synthetic images and sequences on disk."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import OpenImageIO as oiio
import pytest


def _write_image(
    path: Path,
    pixels: np.ndarray,
    channel_names: Iterable[str] | None = None,
) -> None:
    """Write a numpy array (float32 HxWxC) to disk via OIIO."""
    assert pixels.dtype == np.float32
    assert pixels.ndim == 3
    height, width, nchannels = pixels.shape

    spec = oiio.ImageSpec(width, height, nchannels, oiio.FLOAT)
    if channel_names is not None:
        spec.channelnames = tuple(channel_names)

    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"OIIO cannot create output for {path}: {oiio.geterror()}")
    if not out.open(str(path), spec):
        raise RuntimeError(f"OIIO cannot open {path} for writing: {out.geterror()}")
    try:
        if not out.write_image(pixels):
            raise RuntimeError(f"OIIO write_image failed for {path}: {out.geterror()}")
    finally:
        out.close()


def _make_rgba(height: int = 32, width: int = 32) -> np.ndarray:
    """Generate a simple RGBA gradient."""
    arr = np.zeros((height, width, 4), dtype=np.float32)
    arr[:, :, 0] = np.linspace(0.0, 1.0, width, dtype=np.float32)  # R across
    arr[:, :, 1] = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]  # G down
    arr[:, :, 2] = 0.5
    arr[:, :, 3] = 1.0
    return arr


def _make_multichannel(height: int = 32, width: int = 32) -> np.ndarray:
    """Generate an RGBA + Z + AO (6 channels) image."""
    arr = np.zeros((height, width, 6), dtype=np.float32)
    arr[:, :, :4] = _make_rgba(height, width)
    arr[:, :, 4] = 10.0  # constant Z depth
    arr[:, :, 5] = 0.75  # constant AO
    return arr


@pytest.fixture(scope="session")
def png_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("images") / "sample.png"
    _write_image(path, _make_rgba(), channel_names=["R", "G", "B", "A"])
    return path


@pytest.fixture(scope="session")
def exr_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("images") / "sample.exr"
    _write_image(path, _make_rgba(), channel_names=["R", "G", "B", "A"])
    return path


@pytest.fixture(scope="session")
def exr_multichannel_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("images") / "multi.exr"
    _write_image(path, _make_multichannel(), channel_names=["R", "G", "B", "A", "Z", "AO"])
    return path


@pytest.fixture(scope="session")
def sequence_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 10-frame contiguous PNG sequence: render.0001.png … render.0010.png."""
    directory = tmp_path_factory.mktemp("seq")
    pixels = _make_rgba(16, 16)
    for frame in range(1, 11):
        _write_image(directory / f"render.{frame:04d}.png", pixels, ["R", "G", "B", "A"])
    return directory


@pytest.fixture(scope="session")
def sequence_with_gaps_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A sparse PNG sequence at frames 1, 3, 5, 8."""
    directory = tmp_path_factory.mktemp("seq_gaps")
    pixels = _make_rgba(16, 16)
    for frame in (1, 3, 5, 8):
        _write_image(directory / f"shot.{frame:04d}.png", pixels, ["R", "G", "B", "A"])
    return directory


@pytest.fixture(scope="session")
def mixed_sequences_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two distinct sequences in the same directory + a non-sequence file."""
    directory = tmp_path_factory.mktemp("mixed")
    pixels = _make_rgba(16, 16)
    # large sequence: 5 frames
    for frame in range(1, 6):
        _write_image(directory / f"big.{frame:04d}.png", pixels, ["R", "G", "B", "A"])
    # smaller sequence: 2 frames
    for frame in (10, 11):
        _write_image(directory / f"small.{frame:04d}.exr", pixels, ["R", "G", "B", "A"])
    # a standalone file (not part of a sequence)
    _write_image(directory / "poster.png", pixels, ["R", "G", "B", "A"])
    return directory


@pytest.fixture(scope="session")
def corrupt_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A file with a valid-looking extension but invalid contents."""
    path = tmp_path_factory.mktemp("bad") / "broken.exr"
    path.write_bytes(b"not an exr file")
    return path
