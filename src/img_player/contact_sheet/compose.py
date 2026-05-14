"""Numpy + QPainter tile-grid compositor for contact sheet mode.

Pure functions — no app / state references. Easier to unit-test
without spinning up the cache or the layer stack:

* :func:`auto_grid_dimensions` — pick ``(cols, rows)`` that keeps
  the output composite aspect close to the source image aspect.
* :func:`render_contact_sheet` — given N decoded arrays (one per
  layer), arrange them in a ``cols × rows`` grid, optionally label
  each tile with the layer's display name.

The composite output mirrors what the rest of the pipeline expects:
HxWx{3,4} float32, RGBA when any input had an alpha channel,
otherwise RGB. The GL viewport handles both.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen

log = logging.getLogger(__name__)


# Label band height as a fraction of the tile height. ~6 % gives a
# readable line on a 1080p tile (= 65 px) without eating too much of
# the image. Bigger on small tiles is wrong (text becomes huge), so
# we also clamp to an absolute minimum of 14 px and maximum of 60 px
# below.
_LABEL_HEIGHT_FRACTION = 0.06
_LABEL_MIN_PX = 14
_LABEL_MAX_PX = 60
_LABEL_BG_RGBA = (0.0, 0.0, 0.0, 0.55)  # semi-transparent black
_LABEL_FG_RGB = (1.0, 1.0, 1.0)


def auto_grid_dimensions(n: int, image_aspect: float) -> tuple[int, int]:
    """Pick ``(cols, rows)`` so the composite output's aspect ratio
    stays close to ``image_aspect`` (= source frame W/H).

    Math: with all tiles sharing the source aspect, the output
    aspect is ``(cols / rows) × image_aspect``. For the composite to
    preserve the source aspect we'd want ``cols / rows == 1`` —
    i.e. a square grid. The classic ``cols = ceil(sqrt(n))`` does
    exactly that, with the bottom row partially empty when ``n``
    isn't a perfect square.

    ``image_aspect`` is currently unused but kept on the signature
    because a future change (e.g. fill the empty trailing cells
    with extra rows of a different aspect) would want it. Passing
    it is also forward-compat with a tighter optimisation that
    actually does ``cols / rows = source_aspect / output_aspect``
    for non-square outputs.
    """
    del image_aspect  # see docstring
    n = max(1, n)
    cols = max(1, int(math.ceil(math.sqrt(n))))
    rows = max(1, int(math.ceil(n / cols)))
    return (cols, rows)


def render_contact_sheet(
    tiles: Sequence[np.ndarray | None],
    *,
    names: Sequence[str],
    cols: int,
    rows: int,
    target_w: int,
    target_h: int,
    show_labels: bool = False,
) -> np.ndarray:
    """Compose ``tiles`` into a ``cols × rows`` grid.

    ``tiles[i]`` is either:
    * an HxWx{3,4} float ndarray — the layer's decoded frame, or
    * ``None`` — the layer fell off its range (decode failed or
      the contact-sheet playhead is past the layer's last frame).
      The slot is filled with solid black + a "—" marker.

    ``names[i]`` is the layer's display name; rendered as a label
    strip at the bottom of the tile when ``show_labels`` is True.

    ``target_w`` / ``target_h`` are the composite output size in
    pixels. Each tile is resized (nearest-neighbour, cheap) to fit
    ``(target_w // cols, target_h // rows)``; the remaining 1-2 px
    when the sizes don't divide evenly are absorbed by the rightmost
    column / bottom row so the output is exactly target-sized.

    The output dtype matches the first non-None tile's dtype (the
    GL viewport then handles either uint8 or float).
    """
    if cols <= 0 or rows <= 0:
        raise ValueError(f"cols / rows must be positive, got {cols}x{rows}")
    if target_w <= 0 or target_h <= 0:
        raise ValueError(f"target size must be positive, got {target_w}x{target_h}")

    tile_w = target_w // cols
    tile_h = target_h // rows
    if tile_w <= 0 or tile_h <= 0:
        # Pathological case: target smaller than the grid — give every
        # tile at least 1px and let the composite truncate.
        tile_w = max(1, tile_w)
        tile_h = max(1, tile_h)

    # Pick output channel count + dtype from the first real tile.
    sample = next((t for t in tiles if t is not None), None)
    if sample is None:
        # No tiles — empty grid. Return black float32 RGB at target
        # size so the GL viewport still has something to upload.
        return np.zeros((target_h, target_w, 3), dtype=np.float32)
    n_channels = sample.shape[2] if sample.ndim == 3 else 3
    if n_channels not in (3, 4):
        n_channels = 4 if n_channels >= 4 else 3
    out_dtype = sample.dtype

    out = np.zeros((target_h, target_w, n_channels), dtype=out_dtype)
    # The label strip eats the bottom of each tile when enabled.
    # Compute once outside the loop so every tile gets the same band
    # geometry.
    label_h = 0
    if show_labels:
        label_h = max(_LABEL_MIN_PX, int(tile_h * _LABEL_HEIGHT_FRACTION))
        label_h = min(label_h, _LABEL_MAX_PX, tile_h - 1)
        label_h = max(0, label_h)

    for idx in range(cols * rows):
        col = idx % cols
        row = idx // cols
        # Rightmost / bottom cells absorb the modulo remainder so the
        # composite exactly fills target_w × target_h.
        x0 = col * tile_w
        x1 = target_w if col == cols - 1 else x0 + tile_w
        y0 = row * tile_h
        y1 = target_h if row == rows - 1 else y0 + tile_h
        cell_w = x1 - x0
        cell_h = y1 - y0

        if idx >= len(tiles):
            # No more layers — leave the cell black.
            continue
        tile = tiles[idx]
        name = names[idx] if idx < len(names) else ""
        image_h = cell_h - label_h
        if image_h <= 0:
            image_h = cell_h  # label band wouldn't fit, drop it
            actual_label_h = 0
        else:
            actual_label_h = label_h
        if tile is None:
            # Layer fell off its range — paint a placeholder dash.
            _fill_unavailable(
                out[y0:y0 + image_h, x0:x1],
                n_channels=n_channels,
                dtype=out_dtype,
            )
        else:
            resized = _resize_nearest(tile, cell_w, image_h, n_channels)
            out[y0:y0 + image_h, x0:x1] = resized

        if show_labels and actual_label_h > 0 and name:
            _paint_label(
                out[y0 + image_h:y1, x0:x1],
                name=name,
                n_channels=n_channels,
                dtype=out_dtype,
            )

    return out


# ----------------------------------------------------------------- internals


def _resize_nearest(
    arr: np.ndarray, w: int, h: int, n_channels: int,
) -> np.ndarray:
    """Nearest-neighbour resize via numpy fancy-index.

    Faster than calling out to cv2 / PIL for our scale (~1 ms on a
    1080p → 540p downsample) and avoids a heavy dependency on the
    composite path. The compose runs on every frame change in
    contact-sheet mode, so this matters.

    Also normalises the input channel count to ``n_channels`` —
    padding RGB → RGBA with full alpha or trimming RGBA → RGB.
    """
    src_h, src_w = arr.shape[:2]
    # Generate index arrays once, broadcast.
    ys = (np.arange(h) * src_h // h).astype(np.intp)
    xs = (np.arange(w) * src_w // w).astype(np.intp)
    if arr.ndim == 2:
        # Grayscale — broadcast to 3 channels then re-channel below.
        sampled = arr[ys[:, None], xs[None, :]]
        sampled = np.stack([sampled, sampled, sampled], axis=2)
    else:
        sampled = arr[ys[:, None], xs[None, :], :]

    src_channels = sampled.shape[2]
    if src_channels == n_channels:
        return sampled
    if src_channels < n_channels:
        # Pad with full alpha (1.0 for float, 255 for uint8).
        pad = np.full(
            (h, w, n_channels - src_channels),
            _opaque_for(arr.dtype),
            dtype=arr.dtype,
        )
        return np.concatenate([sampled, pad], axis=2)
    return sampled[..., :n_channels]


def _opaque_for(dtype: np.dtype) -> object:
    """Pick the "fully opaque" alpha value for ``dtype``.

    Mirrors :func:`compare.compose._opaque_for` — kept private here
    so contact_sheet doesn't reach across modules for a trivial
    helper.
    """
    if np.issubdtype(dtype, np.integer):
        return np.iinfo(dtype).max
    return 1.0


def _fill_unavailable(
    region: np.ndarray, *, n_channels: int, dtype: np.dtype,
) -> None:
    """Paint a dashed "—" marker into an empty / out-of-range tile.

    Visually distinct from black-on-load so the user sees "this
    layer doesn't reach this contact-sheet frame" rather than
    "decode is still pending". Cheap diagonal stripes pattern
    (every 16 px) — no font rendering, works at any tile size.
    """
    h, w = region.shape[:2]
    # Base: dark grey instead of pitch black so the stripes contrast.
    base = 0.08 if not np.issubdtype(dtype, np.integer) else 20
    stripe = 0.16 if not np.issubdtype(dtype, np.integer) else 40
    region[:, :, :3] = base
    if n_channels == 4:
        region[:, :, 3] = _opaque_for(dtype)
    # Diagonal stripes every 16 px.
    # Two pixel-wide line every 32-pixel band.
    yy, xx = np.indices((h, w))
    mask = ((xx + yy) % 32) < 2
    region[mask, :3] = stripe


def _paint_label(
    band: np.ndarray, *, name: str, n_channels: int, dtype: np.dtype,
) -> None:
    """Render ``name`` as a white-on-translucent-black strip into
    ``band`` (HxWxC, the last few rows of a tile).

    Uses QPainter for text — it gives us hinted antialiased glyphs
    "for free" and matches the typography of the rest of the UI.
    The painted QImage is converted to numpy and blended into the
    band; we don't blow away whatever was there (the underlying
    image bleeds through the semi-transparent black).
    """
    h, w = band.shape[:2]
    if h <= 0 or w <= 0:
        return
    img = QImage(w, h, QImage.Format.Format_ARGB32_Premultiplied)
    # Semi-transparent black background — uses Premultiplied so the
    # alpha math matches the numpy blend below.
    bg = QColor(
        0, 0, 0,
        int(_LABEL_BG_RGBA[3] * 255),
    )
    img.fill(bg)
    painter = QPainter(img)
    try:
        # Font sized to ~60 % of the band height — leaves margin so
        # descenders aren't clipped on tight bands. Capped at the
        # band height -2 px just in case.
        px = max(8, int(h * 0.60))
        px = min(px, h - 2) if h > 10 else px
        font = QFont()
        font.setPixelSize(px)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        # Left-padded 6 px so the text doesn't kiss the tile edge.
        painter.drawText(
            6, 0, w - 12, h,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            name,
        )
    finally:
        painter.end()

    # Convert QImage → numpy. Format_ARGB32_Premultiplied is BGRA
    # in memory on little-endian — we re-channel to RGBA on read.
    ptr = img.constBits()
    # bytesPerLine may include trailing padding on some Qt builds —
    # use it to slice instead of assuming row stride.
    bytes_per_line = img.bytesPerLine()
    raw = np.frombuffer(ptr, dtype=np.uint8, count=bytes_per_line * h)
    bgra = raw.reshape(h, bytes_per_line)[:, : w * 4].reshape(h, w, 4)
    # BGRA → RGBA reorder so colour math below works on the right
    # channel positions.
    rgba = bgra[..., [2, 1, 0, 3]]

    # Premultiplied alpha → straight alpha for the blend. We were
    # told the format is premultiplied so divide RGB by alpha when
    # alpha > 0; on alpha==0 the source contributes nothing anyway.
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    # Avoid divide-by-zero: where alpha is 0 the colour will be
    # multiplied out by alpha in the blend, so the temporary
    # divisor doesn't matter (clamp to 1 to keep numpy quiet).
    safe_alpha = np.where(alpha > 0, alpha, 1.0)
    src_rgb_f = rgba[..., :3].astype(np.float32) / 255.0 / safe_alpha

    # Convert band to float for the blend, then cast back.
    is_uint = np.issubdtype(dtype, np.integer)
    if is_uint:
        scale = float(np.iinfo(dtype).max)
        band_f = band[..., :3].astype(np.float32) / scale
    else:
        scale = 1.0
        band_f = band[..., :3].astype(np.float32, copy=False)

    out_rgb = band_f * (1.0 - alpha) + src_rgb_f * alpha
    band[..., :3] = (out_rgb * scale).astype(dtype) if is_uint else out_rgb.astype(dtype)
    if n_channels == 4:
        band[..., 3] = _opaque_for(dtype)
