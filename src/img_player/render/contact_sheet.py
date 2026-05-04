"""CPU-side contact-sheet compositor.

Builds a single ``np.ndarray`` containing all currently-selected channel
groups laid out in a grid. The result is fed to the existing
:class:`~img_player.render.gl_viewport.GLViewport` via its standard
``set_frame`` path — no GL pipeline changes required.

Why CPU and not GPU multi-quad? The viewport's shader pipeline is one
texture + OCIO display transform applied as a fullscreen quad; making
it draw N quads would touch every part of that file (shader, uniforms,
upload, draw call, hit-testing). Compositing in NumPy keeps the GL
side untouched and is fast enough — the cost is dominated by the
single OCIO upload, not the slicing.

If we later need per-tile labels, hover, or click-to-isolate (= Slice 3
of the contact-sheet feature), we'll replace this module with a real
multi-texture path. Until then, this is the MVP.

The function is *pure* (no Qt, no GL) so it lives outside the UI tree
and can be unit-tested with plain numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from img_player.sequence.channels import ChannelSelection, auto_grid

# Token strings used by the channel menu's layout combo. Kept in sync
# with ``ui/channel_menu.LAYOUT_MODES`` (we don't import that module
# here to keep ``contact_sheet`` Qt-free / pure-NumPy testable).
LAYOUT_AUTO = "Auto"
LAYOUT_1xN = "1×N"
LAYOUT_Nx1 = "N×1"
LAYOUT_2x2 = "2×2"
LAYOUT_3x3 = "3×3"
LAYOUT_4x4 = "4×4"

# Visual gap (in source pixels) between adjacent tiles in the composite.
# Painted with the background colour so the eye separates tiles cleanly
# without distracting borders. Kept small relative to typical tile
# sizes (a few px on a 1080 grid).
GAP_PX = 8

# Background fill for the composite (gap colour + areas where a tile
# is smaller than its slot). Matches the GL viewport clear colour
# (~``BG_DEEP`` from the studio-dark palette) so the seam is invisible
# when the composite is drawn on top of the cleared viewport.
BG_FILL = (0.0235, 0.0235, 0.0235)  # ≈ #060606

# Hard cap on how many tiles we'll composite. Beyond this, a contact
# sheet is unreadable AND the decode cost (one channel per tile) makes
# playback drop. The UI clamps the user's selection to this number;
# the compositor enforces it as a defensive guard.
MAX_TILES = 16


@dataclass(frozen=True)
class TileSpec:
    """One tile in the composite: where it lives + which channels feed it."""

    label: str
    channels: tuple[int, ...]   # column indices into the union buffer
    row: int
    col: int


@dataclass(frozen=True)
class TileRect:
    """Pixel rectangle of a tile *inside the composite image*.

    Coordinates are in composite-image space (origin top-left of the
    composite array). Used by the click-to-isolate hit-test and by
    the label baker — both need to know where each tile lives.
    """

    label: str
    x: int
    y: int
    w: int
    h: int

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


@dataclass(frozen=True)
class CompositeGeometry:
    """Layout of the contact-sheet composite produced by :func:`compose`.

    All sizes are in *composite-image pixels* (not viewport widget
    pixels — the GL viewport's transform handles fit/zoom on top).
    Caller must combine with ``GLViewport.current_transform`` to map
    a click in widget coords to image coords before hit-testing.
    """

    rows: int
    cols: int
    composite_w: int
    composite_h: int
    tiles: tuple[TileRect, ...]

    @property
    def is_contact_sheet(self) -> bool:
        return len(self.tiles) > 1 or (self.rows * self.cols) > 1


def tile_at(geometry: CompositeGeometry, image_x: int, image_y: int) -> str | None:
    """Return the label of the tile under ``(image_x, image_y)`` in
    composite-image coords. ``None`` outside any tile (gap or beyond
    the composite bounds)."""
    for rect in geometry.tiles:
        if rect.contains(image_x, image_y):
            return rect.label
    return None


def _broadcast_to_rgb(tile: np.ndarray) -> np.ndarray:
    """Make ``tile`` an HxWx3 array regardless of how many channels it had.

    1ch → broadcast to RGB (monochrome). 2ch → take the first as luma,
    pad to RGB. 3ch → pass through. 4+ → drop alpha and any extra.
    Always returns a contiguous array because the GL upload path
    chokes on zero-strided views.
    """
    if tile.ndim == 2:
        tile = tile[:, :, np.newaxis]
    n = tile.shape[2]
    if n == 1:
        tile = np.broadcast_to(tile, (tile.shape[0], tile.shape[1], 3))
    elif n == 2:
        # Treat as luma+alpha; show luma in all three channels.
        luma = tile[:, :, 0:1]
        tile = np.broadcast_to(luma, (tile.shape[0], tile.shape[1], 3))
    elif n >= 4:
        tile = tile[:, :, :3]
    return np.ascontiguousarray(tile)


def _downsample(tile: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    """Reduce ``tile`` to fit inside ``(max_w, max_h)`` while preserving aspect.

    Uses simple stride sampling — fast, no scipy/PIL dependency, good
    enough for a review-grade contact sheet. The factor is integer so
    aliasing stays predictable; sub-integer scaling would need a real
    resampler. Tiles smaller than the slot are returned unchanged.
    """
    h, w = tile.shape[:2]
    if w <= max_w and h <= max_h:
        return tile
    # Integer downsample factor — round UP so the result fits the slot.
    factor = max(1, max(int(np.ceil(w / max_w)), int(np.ceil(h / max_h))))
    return tile[::factor, ::factor]


def resolve_grid(
    mode: str,
    n: int,
    viewport_aspect: float,
    tile_aspect: float,
) -> tuple[int, int, int]:
    """Return ``(rows, cols, displayed_n)`` for a layout mode + tile count.

    * ``"Auto"`` → :func:`auto_grid` decides ``(rows, cols)`` from the
      viewport aspect; every requested tile is displayed.
    * ``"1×N"`` / ``"N×1"`` → all tiles stacked horizontally / vertically.
    * ``"2×2"`` / ``"3×3"`` / ``"4×4"`` → fixed grid; tiles beyond the
      grid's capacity (4 / 9 / 16) are dropped. The user opted into a
      specific grid shape, so silently overflowing into auto would
      contradict the choice.

    Anything unrecognised falls back to ``"Auto"``.
    """
    n = max(0, n)
    if mode == LAYOUT_1xN:
        return 1, max(1, n), n
    if mode == LAYOUT_Nx1:
        return max(1, n), 1, n
    if mode == LAYOUT_2x2:
        return 2, 2, min(n, 4)
    if mode == LAYOUT_3x3:
        return 3, 3, min(n, 9)
    if mode == LAYOUT_4x4:
        return 4, 4, min(n, 16)
    # "Auto" or unrecognised → pick the best-fit shape for the viewport.
    rows, cols = auto_grid(n, viewport_aspect, tile_aspect)
    return rows, cols, n


def plan_layout(
    selection: ChannelSelection,
    viewport_w: int,
    viewport_h: int,
    tile_w: int,
    tile_h: int,
    layout_mode: str = LAYOUT_AUTO,
) -> tuple[int, int, tuple[TileSpec, ...]]:
    """Return ``(rows, cols, specs)`` for the current selection.

    Pure: no array data needed, just sizes. Useful for tests and for
    the UI to know "how big will the composite be?" before paying for
    a decode.
    """
    layout = selection.tile_layout()[:MAX_TILES]
    vp_aspect = viewport_w / max(1, viewport_h)
    tile_aspect = tile_w / max(1, tile_h)
    rows, cols, displayed = resolve_grid(layout_mode, len(layout), vp_aspect, tile_aspect)
    layout = layout[:displayed]
    specs = tuple(
        TileSpec(label=label, channels=channels, row=i // cols, col=i % cols)
        for i, (label, channels) in enumerate(layout)
    )
    return rows, cols, specs


def compose(
    union_buffer: np.ndarray,
    selection: ChannelSelection,
    viewport_w: int,
    viewport_h: int,
    layout_mode: str = LAYOUT_AUTO,
) -> tuple[np.ndarray, CompositeGeometry]:
    """Build the contact-sheet image for one frame.

    Parameters
    ----------
    union_buffer:
        Output of ``read_frame(path, channels=selection.union_channels())``
        — an ``HxWxC`` float array where the channel axis matches the
        union order.
    selection:
        The current :class:`ChannelSelection`. ``selection.is_contact_sheet``
        controls whether we composite at all (single-tile mode just
        returns the active group as RGB, fast path).
    viewport_w / viewport_h:
        Pixel size of the destination (the GL viewport widget). Used
        to pick the grid shape and to decide how aggressively to
        downsample tiles.
    layout_mode:
        One of the :data:`LAYOUT_*` tokens. ``"Auto"`` picks the
        best-fit shape for the viewport; ``"2×2"``/``"3×3"``/``"4×4"``
        force a fixed grid and drop overflow tiles; ``"1×N"``/``"N×1"``
        stack horizontally / vertically.

    Returns
    -------
    ``HxWx3`` ``float32`` (or whatever dtype the input was) array
    ready to be passed to ``GLViewport.set_frame``.
    """
    h, w = union_buffer.shape[:2]
    buffer_channels = union_buffer.shape[2]
    layout = selection.tile_layout()[:MAX_TILES]

    # Drop tiles whose channel indices exceed the buffer width.
    # ``MasterFrameCache`` has a documented case where a frame was
    # decoded with the reader's default channel set (RGBA = 4) before
    # the user toggled extra tile groups: the buffer is stale relative
    # to the new selection and a re-decode is queued, but the next
    # paint hits the old buffer. Without this filter, indexing into
    # the stale buffer with ``selection.union_channels()``-derived
    # indices raises ``IndexError`` (= the user-reported channel-menu
    # multi-tile crash). Letting the offending tiles vanish for one
    # paint is the right trade — the next decode lands and they
    # reappear with no further input.
    layout = [
        (label, channels)
        for label, channels in layout
        if all(c < buffer_channels for c in channels)
    ]

    # Resolve the grid first — fixed modes (e.g. "2×2") may truncate
    # the displayed tiles count, which we must apply *before* the
    # single-tile fast path below so a 2×2 with one tile checked
    # still composites a 2×2 grid (with three empty slots).
    rows, cols, displayed = resolve_grid(
        layout_mode,
        len(layout),
        viewport_w / max(1, viewport_h),
        w / max(1, h),
    )
    layout = layout[:displayed]
    n = len(layout)

    # Single-tile fast path: ONLY when the layout shape itself is 1×1
    # (auto-mode with one tile, or N×1 / 1×N degenerated to 1). When
    # the user has explicitly picked a fixed grid like 2×2 we always
    # composite so they see the grid even with a single tile.
    if n <= 1 and rows == 1 and cols == 1:
        if n == 0:
            arr = _broadcast_to_rgb(union_buffer)
            return arr, CompositeGeometry(
                rows=1, cols=1,
                composite_w=arr.shape[1], composite_h=arr.shape[0],
                tiles=(),
            )
        label, channels = layout[0]
        tile = union_buffer[:, :, list(channels)]
        arr = _broadcast_to_rgb(tile)
        return arr, CompositeGeometry(
            rows=1, cols=1,
            composite_w=arr.shape[1], composite_h=arr.shape[0],
            tiles=(TileRect(label=label, x=0, y=0, w=arr.shape[1], h=arr.shape[0]),),
        )

    # Slot size = how much room each tile gets. We aim slightly larger
    # than the viewport so OpenGL's MAG_FILTER does the final fit
    # without us having to upscale on CPU; downsampling LARGE source
    # tiles to the slot is the only step that has a real cost.
    slot_w = max(1, viewport_w // cols)
    slot_h = max(1, viewport_h // rows)

    # Pre-compute downsampled tiles so we know exact composite size.
    downsampled: list[np.ndarray] = []
    for _, channels in layout:
        tile = union_buffer[:, :, list(channels)]
        tile = _broadcast_to_rgb(tile)
        tile = _downsample(tile, slot_w, slot_h)
        downsampled.append(tile)

    # Tile size after downsampling. We standardise on the LARGEST tile
    # in each axis so columns/rows align even when the source channels
    # have different aspect ratios (rare, but happens with cropped
    # AOVs). Smaller tiles get centered in their slot.
    cell_w = max(t.shape[1] for t in downsampled)
    cell_h = max(t.shape[0] for t in downsampled)

    composite_w = cols * cell_w + (cols + 1) * GAP_PX
    composite_h = rows * cell_h + (rows + 1) * GAP_PX
    composite = np.empty((composite_h, composite_w, 3), dtype=union_buffer.dtype)
    composite[...] = np.array(BG_FILL, dtype=union_buffer.dtype)

    tile_rects: list[TileRect] = []
    for i, tile in enumerate(downsampled):
        r = i // cols
        c = i % cols
        # Centre the tile in its slot when it doesn't fill it (smaller
        # source aspect than slot, or odd-pixel rounding).
        slot_x = GAP_PX + c * (cell_w + GAP_PX)
        slot_y = GAP_PX + r * (cell_h + GAP_PX)
        th, tw = tile.shape[:2]
        ox = (cell_w - tw) // 2
        oy = (cell_h - th) // 2
        composite[
            slot_y + oy : slot_y + oy + th,
            slot_x + ox : slot_x + ox + tw,
            :,
        ] = tile
        # Store the *slot* rect (not the tile rect) for hit-testing —
        # clicks anywhere in the slot, including the letterbox bands,
        # should isolate the tile. More forgiving target than the
        # actual painted pixels.
        label = layout[i][0]
        tile_rects.append(TileRect(
            label=label, x=slot_x, y=slot_y, w=cell_w, h=cell_h,
        ))

    geometry = CompositeGeometry(
        rows=rows, cols=cols,
        composite_w=composite_w, composite_h=composite_h,
        tiles=tuple(tile_rects),
    )
    return composite, geometry


# ---------------------------------------------------------------- Label baking
#
# Tiles get a small "chip" at the top-left with their channel-group
# name, baked onto the composite *before* upload. The alternative —
# overlaying labels via a transparent QWidget child of the GL viewport
# — would conflict with the existing AnnotationOverlay (mouse-event
# pass-through is brittle when two siblings stack). Baking keeps the
# whole GL pipeline single-texture and lets OpenGL filter the labels
# along with the image when the user zooms.
#
# Qt is imported lazily so the rest of this module stays Qt-free and
# unit-testable with plain numpy.

# Chip styling — kept here as constants so a future tweak doesn't
# scatter magic values.
_LABEL_CHIP_PAD_X = 6   # horizontal padding inside the chip
_LABEL_CHIP_PAD_Y = 2
_LABEL_CHIP_OFFSET_X = 6   # distance from tile's top-left corner
_LABEL_CHIP_OFFSET_Y = 6
_LABEL_CHIP_BG_RGBA = (0, 0, 0, 165)   # ≈ 65 % opacity — readable over bright tiles
_LABEL_CHIP_FG = "#F5F5F5"
_LABEL_FONT_PT = 10


def bake_labels(
    composite: np.ndarray,
    geometry: CompositeGeometry,
) -> np.ndarray:
    """Paint each tile's label as a small chip at its top-left corner.

    Returns a NEW array (same shape + dtype as ``composite``) — the
    input is not modified, so callers can keep the un-labelled
    version for export. Drawing happens in 8-bit space (QPainter)
    against a transparent ARGB layer the same size as the composite,
    then alpha-blended into the float composite.

    No-op when there are zero or one tiles (single-channel display
    doesn't need a label — the transport bar already shows it).
    """
    # Lazy import: keeps this module Qt-free at import time so tests
    # that only need ``compose`` / ``tile_at`` don't pull Qt.
    from PySide6.QtCore import QRect, Qt
    from PySide6.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter

    if len(geometry.tiles) <= 1:
        return composite

    h, w = composite.shape[:2]
    layer = QImage(w, h, QImage.Format.Format_ARGB32)
    layer.fill(0)
    painter = QPainter(layer)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

    font = QFont("Inter", _LABEL_FONT_PT)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    painter.setFont(font)
    metrics = QFontMetrics(font)

    chip_bg = QColor(*_LABEL_CHIP_BG_RGBA)
    chip_fg = QColor(_LABEL_CHIP_FG)

    for rect in geometry.tiles:
        text_w = metrics.horizontalAdvance(rect.label)
        text_h = metrics.height()
        chip = QRect(
            rect.x + _LABEL_CHIP_OFFSET_X,
            rect.y + _LABEL_CHIP_OFFSET_Y,
            text_w + 2 * _LABEL_CHIP_PAD_X,
            text_h + 2 * _LABEL_CHIP_PAD_Y,
        )
        painter.fillRect(chip, chip_bg)
        painter.setPen(chip_fg)
        painter.drawText(
            chip,
            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignHCenter),
            rect.label,
        )
    painter.end()

    # QImage Format_ARGB32 is (B, G, R, A) byte order on little-endian.
    # Convert to a numpy view, swap to (R, G, B, A), normalise to float.
    ptr = layer.constBits()
    overlay_bgra = np.frombuffer(memoryview(ptr), dtype=np.uint8).reshape(h, w, 4)
    overlay_rgba = overlay_bgra[..., [2, 1, 0, 3]].astype(np.float32) / 255.0
    alpha = overlay_rgba[..., 3:4]

    out = composite.astype(np.float32, copy=True)
    out[..., :3] = out[..., :3] * (1.0 - alpha) + overlay_rgba[..., :3] * alpha
    return out.astype(composite.dtype)
