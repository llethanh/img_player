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
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter

# Shade pair tuned for "obviously missing" without being painful.
# Both lighter than the toolbar accents so the user's eye doesn't
# mistake it for a UI element overlaid on top of the image.
_GREY_DARK = (102, 102, 102)
_GREY_LIGHT = (153, 153, 153)
_CHECKER_PX = 32

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
    """Generate a checkerboard + 'MISSING FRAME' label as float32 RGBA."""
    # Step 1 — checker via integer division on row/col indices.
    rows = np.arange(height, dtype=np.int32) // _CHECKER_PX
    cols = np.arange(width, dtype=np.int32) // _CHECKER_PX
    parity = (rows[:, None] + cols[None, :]) & 1  # H×W bool

    rgb = np.empty((height, width, 3), dtype=np.uint8)
    rgb[parity == 0] = _GREY_DARK
    rgb[parity == 1] = _GREY_LIGHT
    rgba8 = np.concatenate(
        [rgb, np.full((height, width, 1), 255, dtype=np.uint8)], axis=-1,
    )
    rgba8 = np.ascontiguousarray(rgba8)

    # Step 2 — paint "MISSING FRAME" text over the checker via QImage.
    # We draw on the uint8 buffer in place, then convert.
    qimg = QImage(
        rgba8.data, width, height, width * 4, QImage.Format.Format_RGBA8888,
    )
    painter = QPainter(qimg)
    try:
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        # Font size scales with image height so the label stays
        # readable from the lowest 256 px tests up to 4K.
        font = QFont("Sans Serif")
        font.setPointSizeF(max(12.0, min(64.0, height / 14.0)))
        font.setBold(True)
        painter.setFont(font)
        text = "MISSING FRAME"
        # Drop-shadow for legibility on whichever shade lands behind.
        painter.setPen(QColor(0, 0, 0, 160))
        painter.drawText(
            qimg.rect().translated(QPoint(2, 2)),
            Qt.AlignmentFlag.AlignCenter,
            text,
        )
        painter.setPen(QColor(255, 80, 80))  # red — same logic as the timeline
        painter.drawText(
            qimg.rect(), Qt.AlignmentFlag.AlignCenter, text,
        )
    finally:
        painter.end()
    # Step 3 — float32 in [0, 1], the dtype the GL viewport expects.
    return (rgba8.astype(np.float32) / 255.0)


def reset_cache() -> None:
    """Test helper: drop the memoised placeholders so a fresh run
    rebuilds them. Not used in production."""
    with _cache_lock:
        _cache.clear()
