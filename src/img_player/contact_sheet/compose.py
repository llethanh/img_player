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


# Label pill colours. The label is rendered as a rounded pill behind
# the layer-name text, semi-transparent black so the image bleeds
# through (= the user can still see the corner content) but the
# text reads cleanly on any background.
_LABEL_BG_RGBA = (0.0, 0.0, 0.0, 0.55)  # semi-transparent black


def auto_grid_dimensions(
    n: int,
    image_aspect: float = 1.0,
    canvas_aspect: float | None = None,
) -> tuple[int, int]:
    """Pick ``(cols, rows)`` for ``n`` tiles given the tile aspect
    and (optionally) the canvas aspect we're rendering into.

    Two strategies:

    * **No canvas hint** (``canvas_aspect is None``) — fall back to
      the classic ``cols = ceil(sqrt(n))`` square-ish grid. Used as
      a defensive default for code paths that don't have a viewport
      size yet (boot, headless tests).

    * **With a canvas hint** — :func:`smart_grid_dimensions`. Picks
      ``(cols, rows)`` to maximise the per-tile usable area inside
      the canvas, accounting for both the per-tile aspect (= tiles
      are letterboxed inside their cells when the cell aspect
      differs from the image aspect) and the canvas aspect (= the
      whole composite gets letterboxed inside the GL viewport if
      its aspect mismatches).
    """
    if canvas_aspect is None:
        n = max(1, n)
        cols = max(1, int(math.ceil(math.sqrt(n))))
        rows = max(1, int(math.ceil(n / cols)))
        return (cols, rows)
    return smart_grid_dimensions(n, image_aspect, canvas_aspect)


def smart_grid_dimensions(
    n: int,
    image_aspect: float,
    canvas_aspect: float,
) -> tuple[int, int]:
    """Pick the grid that maximises composite efficiency.

    Score per ``(c, r)`` candidate with ``c × r ≥ n``:

    * **Cell efficiency** — fraction of cells actually filled,
      ``n / (c × r)``. A 3×3 layout for 7 tiles wastes 2/9 of the
      canvas; a 4×2 lays out the same 7 tiles wasting 1/8. Higher
      is better.
    * **Composite aspect match** — how close ``cols / rows ×
      image_aspect`` (the composite's natural aspect with all
      tiles at source aspect) is to ``canvas_aspect`` (the GL
      viewport's). Mismatch makes the GL viewport letterbox the
      whole composite on top of the per-tile letterboxing, wasting
      pixels. Computed as ``min(a, b) / max(a, b) ∈ (0, 1]`` so
      ties are symmetric.

    The two factors are multiplied — the best grid keeps both
    cells full and the composite aspect close to the viewport.
    """
    n = max(1, n)
    image_aspect = max(image_aspect, 0.01)
    canvas_aspect = max(canvas_aspect, 0.01)
    # The candidate space is small (n options) so we materialise
    # the per-candidate stats and pick with a deterministic
    # multi-key sort instead of an in-loop best-tracker. Reads
    # cleaner and makes the tie-breaking hierarchy explicit.
    candidates: list[tuple[int, int, float, float, int]] = []
    for cols in range(1, n + 1):
        rows = int(math.ceil(n / cols))
        composite_aspect = (cols / rows) * image_aspect
        a, b = composite_aspect, canvas_aspect
        ar_eff = (min(a, b) / max(a, b)) if max(a, b) > 0 else 0.0
        cell_eff = n / (cols * rows)
        landscape_bias = 1 if cols >= rows else 0
        candidates.append((cols, rows, ar_eff * cell_eff, cell_eff, landscape_bias))

    # Sort key (descending priority):
    # 1. Combined score (ar_eff × cell_eff) — primary efficiency.
    # 2. cell_eff — among ties, prefer the layout with fewer empty
    #    cells (= 3×3 over 4×3 for 9 tiles). Psychological / UX win:
    #    a complete grid feels "right", holes feel like a bug.
    # 3. landscape_bias — prefer cols ≥ rows on ties. Photographic
    #    contact sheets are wider-than-tall by convention; landscape
    #    monitors render that layout better.
    # The negatives flip the sort to descending without ``reverse=True``
    # (so the secondary keys can stay ascending where appropriate).
    candidates.sort(key=lambda c: (-c[2], -c[3], -c[4]))
    best_cols, best_rows, _score, _cell, _bias = candidates[0]
    return (best_cols, best_rows)


def render_contact_sheet(
    tiles: Sequence[np.ndarray | None],
    *,
    names: Sequence[str],
    cols: int,
    rows: int,
    target_w: int,
    target_h: int,
    show_labels: bool = False,
    scrub_indicator: tuple[int, float] | None = None,
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
    # Labels live ON the tile (top-left overlay) not as a separate
    # strip — cells are fully used by the image, no vertical real
    # estate eaten by a band. See :func:`_paint_label_overlay`.

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

        if tile is None:
            # Layer fell off its range — paint a placeholder dash.
            _fill_unavailable(
                out[y0:y0 + cell_h, x0:x1],
                n_channels=n_channels,
                dtype=out_dtype,
            )
        else:
            # Stretch the tile to fill the cell exactly. Tiles touch
            # edge-to-edge — no per-cell letterbox, no bars between
            # neighbours. Layers with mismatched aspect ratios get
            # mildly distorted; the trade-off the user chose when
            # asking for "images collées".
            resized = _resize_nearest_raw(tile, cell_w, cell_h, n_channels)
            out[y0:y0 + cell_h, x0:x1] = resized

        # Label overlay: rendered ON the tile near the top-left
        # corner rather than as a strip below the image, so the
        # image stays the dominant visual + the label doesn't eat
        # vertical space. Falls back to a no-op when ``name`` is
        # empty (= layer has no display name, defensive).
        if show_labels and name:
            _paint_label_overlay(
                out[y0:y0 + cell_h, x0:x1],
                name=name,
                n_channels=n_channels,
                dtype=out_dtype,
            )

        # Scrub-progress indicator: a translucent orange bar at the
        # bottom of the actively-scrubbed tile, width proportional
        # to the layer's current frame within its trim range. Only
        # the dragged tile shows it; clears the next render when the
        # gesture ends (caller passes ``None``).
        if scrub_indicator is not None and scrub_indicator[0] == idx:
            _paint_scrub_progress_bar(
                out[y0:y0 + cell_h, x0:x1],
                pct=scrub_indicator[1],
                n_channels=n_channels,
                dtype=out_dtype,
            )

    return out


# ----------------------------------------------------------------- internals


def _resize_nearest_raw(
    arr: np.ndarray, w: int, h: int, n_channels: int,
) -> np.ndarray:
    """Nearest-neighbour resize via numpy fancy-index, no aspect
    preservation — exact ``(h, w)`` output regardless of input
    shape. Used as the low-level engine of
    :func:`_letterbox_into_region` after the caller has computed
    the aspect-preserving ``(h, w)`` target.

    Faster than calling out to cv2 / PIL for our scale (~1 ms on a
    1080p → 540p downsample) and avoids a heavy dependency on the
    composite path. Also normalises the input channel count to
    ``n_channels`` — padding RGB → RGBA with full alpha or
    trimming RGBA → RGB.
    """
    src_h, src_w = arr.shape[:2]
    ys = (np.arange(h) * src_h // h).astype(np.intp)
    xs = (np.arange(w) * src_w // w).astype(np.intp)
    if arr.ndim == 2:
        sampled = arr[ys[:, None], xs[None, :]]
        sampled = np.stack([sampled, sampled, sampled], axis=2)
    else:
        sampled = arr[ys[:, None], xs[None, :], :]

    src_channels = sampled.shape[2]
    if src_channels == n_channels:
        return sampled
    if src_channels < n_channels:
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


def _paint_label_overlay(
    tile_region: np.ndarray, *, name: str, n_channels: int, dtype: np.dtype,
) -> None:
    """Stamp ``name`` near the top-left of ``tile_region`` (HxWxC).

    Two layers:

    * A small **semi-transparent black pill** behind the text — just
      a few px of padding around the glyphs so the text stays
      readable on any background (sky, white, dark grey, …).
    * The **white bold text** itself, antialiased via QPainter.

    Both are alpha-blended into the underlying tile pixels —
    we don't overwrite — so the image bleeds through the pill's
    transparency. Position is the top-left corner with a small
    inset so the label doesn't kiss the tile edge.

    No-op when the tile is too small (< ~40 px on either side) to
    render any legible label.
    """
    h, w = tile_region.shape[:2]
    if h < 40 or w < 40:
        return

    # Geometry. Inset from the tile edge so the label sits clearly
    # inside the image rather than touching the border. Text size
    # scales with tile height — ~3.5 % is a sweet spot that stays
    # readable on a 540 px tile (= ~19 px) without crowding a
    # 2160 px tile (= ~75 px).
    inset = max(8, int(min(h, w) * 0.02))
    px = max(14, min(int(h * 0.035), 64))
    pad_x = max(4, px // 3)
    pad_y = max(2, px // 5)

    # Measure the text first so the pill background is sized to fit.
    # We paint into a probe QImage just to get the text bounding
    # box — Qt's ``QFontMetrics`` would also work but stays cheap
    # in either direction.
    font = QFont()
    font.setPixelSize(px)
    font.setBold(True)
    from PySide6.QtGui import QFontMetrics  # noqa: PLC0415 — only here

    metrics = QFontMetrics(font)
    text_w = metrics.horizontalAdvance(name)
    text_h = metrics.height()
    # Clamp the label width so it doesn't push past the tile bounds
    # on very long layer names. We let the text get elided with an
    # ellipsis instead.
    max_text_w = w - 2 * inset - 2 * pad_x
    if text_w > max_text_w:
        elided = metrics.elidedText(
            name, Qt.TextElideMode.ElideMiddle, max_text_w,
        )
    else:
        elided = name
    text_w = min(text_w, max_text_w)
    if text_w <= 0:
        return

    box_w = text_w + 2 * pad_x
    box_h = text_h + 2 * pad_y
    if box_w <= 0 or box_h <= 0:
        return

    # Allocate an RGBA buffer the size of the label box and render
    # pill + text into it. The buffer is then alpha-blended onto
    # the tile at ``(inset, inset)``.
    img = QImage(box_w, box_h, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(0)  # fully transparent
    painter = QPainter(img)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        # Rounded pill background — 6 px radius works at any text
        # size we render here (~14..64 px). The colour matches the
        # info-band convention: dark + semi-transparent.
        from PySide6.QtGui import QBrush  # noqa: PLC0415
        bg = QColor(0, 0, 0, int(_LABEL_BG_RGBA[3] * 255))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(bg))
        radius = max(4, box_h // 4)
        painter.drawRoundedRect(0, 0, box_w, box_h, radius, radius)
        # Text on top — white, bold, centred vertically inside the pill.
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.setFont(font)
        painter.drawText(
            pad_x, 0, text_w, box_h,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
            elided,
        )
    finally:
        painter.end()

    # Convert QImage → numpy RGBA (premultiplied → straight alpha
    # math for the blend). Same dance as the missing-frame helper.
    ptr = img.constBits()
    bytes_per_line = img.bytesPerLine()
    raw = np.frombuffer(ptr, dtype=np.uint8, count=bytes_per_line * box_h)
    bgra = raw.reshape(box_h, bytes_per_line)[:, : box_w * 4].reshape(
        box_h, box_w, 4,
    )
    rgba = bgra[..., [2, 1, 0, 3]]

    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    safe_alpha = np.where(alpha > 0, alpha, 1.0)
    src_rgb_f = rgba[..., :3].astype(np.float32) / 255.0 / safe_alpha

    # Carve out the destination slice on the tile. Clamp to bounds
    # so a future change to inset / pad doesn't reach past the tile
    # edges.
    y0 = inset
    x0 = inset
    y1 = min(h, y0 + box_h)
    x1 = min(w, x0 + box_w)
    if y1 <= y0 or x1 <= x0:
        return
    src = src_rgb_f[: y1 - y0, : x1 - x0]
    alpha_clip = alpha[: y1 - y0, : x1 - x0]
    dst = tile_region[y0:y1, x0:x1, :3]

    is_uint = np.issubdtype(dtype, np.integer)
    if is_uint:
        scale = float(np.iinfo(dtype).max)
        dst_f = dst.astype(np.float32) / scale
    else:
        scale = 1.0
        dst_f = dst.astype(np.float32, copy=False)
    out_rgb = dst_f * (1.0 - alpha_clip) + src * alpha_clip
    tile_region[y0:y1, x0:x1, :3] = (
        (out_rgb * scale).astype(dtype) if is_uint else out_rgb.astype(dtype)
    )
    # Force full opacity on the alpha channel under the label so the
    # GL viewport's premultiplied path doesn't make the label
    # appear ghostly when the tile itself has reduced alpha.
    if n_channels == 4:
        tile_region[y0:y1, x0:x1, 3] = _opaque_for(dtype)


def _paint_scrub_progress_bar(
    tile_region: np.ndarray,
    *,
    pct: float,
    n_channels: int,
    dtype: np.dtype,
) -> None:
    """Stamp a translucent orange progress bar at the bottom of a tile.

    The bar's width is ``pct × tile_w`` (clamped to ``[0, 1]``) so the
    user reads the scrubbed layer's position within its trim range at
    a glance — left edge = first frame, right edge = last frame.

    Only the actively scrubbed tile gets the overlay; the caller
    omits ``scrub_indicator`` (or passes a different tile index) on
    every other render so the bar disappears the moment the gesture
    ends. Painted directly into the numpy buffer via an alpha-blend
    (no QPainter — a flat rect doesn't need it), so the cost per
    paint stays under a microsecond for typical tile sizes.

    No-op for tiny tiles (< 16 px on either side) so the bar doesn't
    overwrite the entire cell on stamp-sized previews.
    """
    h, w = tile_region.shape[:2]
    if h < 16 or w < 16:
        return
    p = max(0.0, min(1.0, float(pct)))
    fill_w = int(round(p * w))
    if fill_w <= 0:
        return
    # Bar geometry: ~3% of tile height, clamped to [3, 12] px so it
    # stays visible on a 200 px review tile and doesn't dominate on
    # a 2160 px hero tile.
    bar_h = max(3, min(int(h * 0.03), 12))
    y_start = h - bar_h

    # Orange in dtype-appropriate scale. Matches the app's accent
    # colour family (FF8C00 → ~1.0 / 0.55 / 0.0). Premultiplied alpha
    # blend so the underlying pixels show through at the bar edges.
    if np.issubdtype(dtype, np.integer):
        max_v = float(np.iinfo(dtype).max)
        rgb = np.array(
            [max_v, max_v * 0.55, 0.0], dtype=np.float32,
        )
        scale = max_v
    else:
        rgb = np.array([1.0, 0.55, 0.0], dtype=np.float32)
        scale = 1.0
    alpha = 0.7  # translucent — image still bleeds through

    region = tile_region[y_start:h, :fill_w, :3]
    dst_f = region.astype(np.float32, copy=False)
    if scale != 1.0:
        dst_f = dst_f / scale
        src = rgb / scale
    else:
        src = rgb
    out_rgb = dst_f * (1.0 - alpha) + src * alpha
    if np.issubdtype(dtype, np.integer):
        tile_region[y_start:h, :fill_w, :3] = (
            out_rgb * scale
        ).astype(dtype)
    else:
        tile_region[y_start:h, :fill_w, :3] = out_rgb.astype(dtype)
    # Bar pixels = fully opaque so the GL upload doesn't ghost it.
    if n_channels == 4:
        tile_region[y_start:h, :fill_w, 3] = _opaque_for(dtype)
