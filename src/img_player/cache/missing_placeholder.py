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


def get_missing_placeholder(
    width: int, height: int, filename: str | None = None,
) -> np.ndarray:
    """Return a HxWx4 float32 RGBA placeholder.

    The base placeholder (no filename) is memoised by (w, h) — every
    "missing slot" in the cache aliases the same shared buffer, so
    hundreds of holes cost a single ndarray.

    When ``filename`` is provided the result is built per call (no
    memoisation): the filename is baked into the overlay so the user
    can see *which* file is missing directly on the placeholder. Each
    such call allocates a fresh ndarray (~33 MB at 1920×1080), so
    callers pay one buffer per missing frame. For typical sparse
    sequences this is negligible; for pathological cases (thousands
    of missing frames) the memory cost is real — call sites that
    don't have a meaningful filename should pass ``None`` to keep
    the shared-buffer behaviour.
    """
    width = max(2, int(width))
    height = max(2, int(height))
    if filename:
        # Per-frame variant — never cached. Stripping to basename keeps
        # the overlay readable; callers that already passed a basename
        # are no-ops here.
        return generate_missing_frame_rgba_float(width, height, filename)
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


