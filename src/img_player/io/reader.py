"""Single-frame image decoding on top of OpenImageIO."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import OpenImageIO as oiio


class FrameReadError(RuntimeError):
    """Raised when a frame cannot be decoded (missing file, bad format, ...)."""


def read_frame(path: Path | str, channels: Sequence[str] | None = None) -> np.ndarray:
    """Decode a frame to a float32 HxWxC numpy array.

    Values are returned in the file's native color space — no OCIO transform
    is applied here. That's the job of the render layer.

    Parameters
    ----------
    path:
        Filesystem path to the image file.
    channels:
        Optional list of channel names to keep. When ``None`` (default), we
        pick a minimal sensible subset for *display* — the named R/G/B/A
        channels when they exist, otherwise the first up to four channels.
        This is critical for multichannel EXRs (AOVs, depth, normals,
        cryptomattes): reading only the beauty can be 3-4x smaller in RAM
        and faster to decode than pulling every layer.

    Raises
    ------
    FrameReadError
        When the file is missing, unreadable, or doesn't contain the
        requested channels.
    """
    path = Path(path)
    if not path.exists():
        raise FrameReadError(f"File not found: {path}")

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FrameReadError(f"Failed to open {path}: {oiio.geterror()}")

    try:
        spec = inp.spec()
        available = list(spec.channelnames)
        selected = _resolve_channels(available, channels, path)
        indices = [available.index(c) for c in selected]

        # Fast path: contiguous channel range → ask OIIO to decode only that
        # slice, skipping the AOV layers entirely.
        if _is_contiguous(indices):
            chbegin = indices[0]
            chend = indices[-1] + 1
            pixels = inp.read_image(chbegin, chend, oiio.FLOAT)
        else:
            # Rare: non-contiguous selection. Read everything and subset.
            pixels = inp.read_image(oiio.FLOAT)
            pixels = pixels[..., indices]

        if pixels is None:
            raise FrameReadError(f"{path}: OIIO read returned None ({inp.geterror()})")

        arr = np.asarray(pixels, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr
    finally:
        inp.close()


def read_header(path: Path | str) -> oiio.ImageSpec:
    """Read only the file header (no pixel data). Cheap."""
    path = Path(path)
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FrameReadError(f"Failed to open header for {path}: {oiio.geterror()}")
    try:
        return inp.spec()
    finally:
        inp.close()


def _resolve_channels(
    available: list[str], requested: Sequence[str] | None, path: Path
) -> list[str]:
    if requested is None:
        preferred = [c for c in ("R", "G", "B", "A") if c in available]
        if preferred:
            return preferred
        if not available:
            raise FrameReadError(f"{path}: no channels in spec")
        return available[: min(4, len(available))]

    missing = [c for c in requested if c not in available]
    if missing:
        raise FrameReadError(
            f"{path}: requested channels not found: {missing}. Available: {available}"
        )
    return list(requested)


def _is_contiguous(indices: list[int]) -> bool:
    if not indices:
        return False
    return indices == list(range(indices[0], indices[-1] + 1))
