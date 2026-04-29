"""Checkerboard "Missing frame" placeholder used by the cache.

When a frame's source file is missing or unreadable the cache stores
this placeholder so playback can continue. The pattern is deliberately
ugly: 32×32 checker squares of two near-grey shades + a centred
"MISSING FRAME" label. Anyone looking at the playback knows
immediately that something is wrong with the source data, not with
the player.

The buffer is generated lazily once per (width, height) pair and
cached at module level. Generation is cheap (~5 ms for a 1920×1080
checker via numpy) but doing it on every cache miss would still cost.
"""

from __future__ import annotations

from threading import Lock

import numpy as np

from img_player.cache.missing_frame import generate_missing_frame_rgba_float

_cache: dict[tuple[int, int], np.ndarray] = {}
_cache_lock = Lock()


def get_missing_placeholder(width: int, height: int) -> np.ndarray:
    """Return a HxWx4 float32 RGBA placeholder. Memoised by (w, h)."""
    width = max(2, int(width))
    height = max(2, int(height))
    key = (width, height)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        arr = _build(width, height)
        _cache[key] = arr
        return arr


def _build(width: int, height: int) -> np.ndarray:
    """Generate the 'Missing Frame' placeholder as float32 RGBA.

    Delegates to :mod:`missing_frame` which produces a richer visual
    (greyscale damier + chromatic aberration + 4-corner registration
    crosshairs + central boxed "MISSING FRAME" label + vignette). We
    convert its QPixmap output to the float32 RGBA array shape the GL
    viewport / multi-layer compositor consumes directly.
    """
    return generate_missing_frame_rgba_float(width, height)


def reset_cache() -> None:
    """Test helper: drop the memoised placeholders so a fresh run
    rebuilds them. Not used in production."""
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------- Empty placeholder

# Used by File → New: solid dark-grey buffer matching the app's
# BG_DEEP. We deliberately don't reuse the missing-frame
# checkerboard here — "no sequence loaded" is a different state
# from "this frame's source is missing" and the user shouldn't
# have to think "what's wrong, did I delete something?" when they
# clicked New on purpose.
_EMPTY_GREY = (20, 20, 22)  # = theme.H.BG_DEEP

_empty_cache: dict[tuple[int, int], np.ndarray] = {}
_empty_lock = Lock()


def get_empty_placeholder(width: int, height: int) -> np.ndarray:
    """Return a HxWx4 float32 RGBA solid-dark-grey buffer."""
    width = max(1, int(width))
    height = max(1, int(height))
    key = (width, height)
    with _empty_lock:
        cached = _empty_cache.get(key)
        if cached is not None:
            return cached
        rgba8 = np.empty((height, width, 4), dtype=np.uint8)
        rgba8[..., 0] = _EMPTY_GREY[0]
        rgba8[..., 1] = _EMPTY_GREY[1]
        rgba8[..., 2] = _EMPTY_GREY[2]
        rgba8[..., 3] = 255
        arr = (rgba8.astype(np.float32) / 255.0)
        _empty_cache[key] = arr
        return arr
