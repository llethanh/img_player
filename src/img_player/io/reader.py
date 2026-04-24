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
        Optional list of channel names to keep (EXR multichannel). When None,
        the main subimage's channels are returned as-is.

    Raises
    ------
    FrameReadError
        When the file is missing, unreadable, or doesn't contain the
        requested channels.
    """
    path = Path(path)
    if not path.exists():
        raise FrameReadError(f"File not found: {path}")

    buf = oiio.ImageBuf(str(path))
    if buf.has_error:
        raise FrameReadError(f"Failed to open {path}: {buf.geterror()}")

    if channels is not None:
        available = set(buf.spec().channelnames)
        missing = [c for c in channels if c not in available]
        if missing:
            raise FrameReadError(
                f"{path}: requested channels not found: {missing}. "
                f"Available: {buf.spec().channelnames}"
            )
        buf = oiio.ImageBufAlgo.channels(buf, tuple(channels))
        if buf.has_error:
            raise FrameReadError(f"{path}: channel selection failed: {buf.geterror()}")

    pixels = buf.get_pixels(oiio.FLOAT)
    if pixels is None:
        raise FrameReadError(f"{path}: pixel read returned None ({buf.geterror()})")

    arr = np.asarray(pixels, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return arr


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
