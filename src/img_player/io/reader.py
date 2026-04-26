"""Single-frame image decoding on top of OpenImageIO."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import OpenImageIO as oiio

log = logging.getLogger(__name__)


class FrameReadError(RuntimeError):
    """Raised when a frame cannot be decoded (missing file, bad format, ...)."""


def configure_oiio(threads: int | None = None) -> int:
    """Set the OpenImageIO global thread pool size.

    OIIO uses a *single* shared thread pool across the whole process. Our
    own decode worker pool runs ``N`` Python workers that each call into
    OIIO in parallel — OIIO's pool is what lets each individual decode
    parallelise (e.g. across EXR scanlines or channels).

    Sizing rule of thumb:
    * ``None`` (default) → use ``os.cpu_count()``. OIIO is free to use
      every logical core, but it shares them across all in-flight decodes
      so we don't actually create N×workers threads.
    * Pass an explicit number to constrain it (useful when sharing the
      machine with other heavy work).

    Returns the value that was actually applied so the caller can log it.
    """
    desired = threads if threads is not None else (os.cpu_count() or 1)
    desired = max(1, int(desired))
    try:
        oiio.attribute("threads", desired)
    except Exception:
        log.exception("OIIO: failed to set 'threads' attribute to %d", desired)
        return -1

    # Best-effort: also nudge the EXR plugin specifically. Older OIIO
    # builds may not expose this attribute; ignore failures.
    try:
        oiio.attribute("exr_threads", desired)
    except Exception:
        pass

    log.info("OIIO threads attribute set to %d (was reading defaults)", desired)
    return desired


def read_frame(
    path: Path | str,
    channels: Sequence[str] | None = None,
    *,
    as_half: bool = True,
) -> np.ndarray:
    """Decode a frame to a float HxWxC numpy array.

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
    as_half:
        When True (default) decodes to ``float16`` — half the RAM and half
        the PCIe upload bandwidth, which matters for realtime 4K playback.
        Precision is still fine for display (EXR's native format anyway).
        Set to False to force float32 output.

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

    oiio_type = oiio.HALF if as_half else oiio.FLOAT
    numpy_dtype = np.float16 if as_half else np.float32

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
            pixels = inp.read_image(chbegin, chend, oiio_type)
        else:
            # Rare: non-contiguous selection. Read everything and subset.
            pixels = inp.read_image(oiio_type)
            pixels = pixels[..., indices]

        if pixels is None:
            raise FrameReadError(f"{path}: OIIO read returned None ({inp.geterror()})")

        arr = np.asarray(pixels, dtype=numpy_dtype)
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


# Attributes we care about for colorspace auto-detection. Listed in
# priority order: explicit colorspace tags first (most reliable), then
# chromaticities (gamut signature), finally OIIO's own normalisation.
_COLOR_METADATA_ATTRS: tuple[str, ...] = (
    # Set by OCIO-aware renderers / pipelines.
    "colorSpaceName",
    # Nuke writes this when it bakes a colorspace into an EXR.
    "nuke/input/colorspace",
    "nuke/input/colourspace",  # British spelling shows up too
    # ICC profile bakers tend to set these.
    "ICCProfile:cprt",
    "ICCProfile:desc",
    # OIIO's own classification of the file's encoding.
    "oiio:ColorSpace",
    # Standard EXR header attribute — 8 floats describing R/G/B/W
    # primaries. Lets us match against ACES AP0 / AP1, Rec.709,
    # Rec.2020, P3 etc. by gamut signature.
    "chromaticities",
)


def read_color_metadata(path: Path | str) -> dict[str, object]:
    """Return whatever colour-related attributes the file's header
    advertises. Cheap — opens the file but doesn't decode pixels.

    The returned dict's keys are the OIIO attribute names from
    :data:`_COLOR_METADATA_ATTRS`. Missing attributes are simply
    absent (not present as ``None``). The values come straight from
    OIIO so they may be strings, tuples of floats, or other types
    depending on the attribute.
    """
    path = Path(path)
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        # We don't raise — auto-detection is best-effort, so a
        # failure to read metadata just means "no metadata, fall
        # back to extension".
        log.debug("read_color_metadata: cannot open %s (%s)", path, oiio.geterror())
        return {}
    try:
        spec = inp.spec()
        meta: dict[str, object] = {}
        for attr in _COLOR_METADATA_ATTRS:
            value = spec.getattribute(attr)
            if value is not None:
                meta[attr] = value
        return meta
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
