"""Pure-numpy A/B compose for the compare overlay.

Three blend modes (mutually exclusive):

* :data:`MODE_VERTICAL` — left half is A, right half is B; seam at
  ``seam`` of the width.
* :data:`MODE_HORIZONTAL` — top half is A, bottom half is B; seam at
  ``seam`` of the height.
* :data:`MODE_OPACITY`  — linear blend ``A * (1 - seam) + B * seam``.

Plus a separate ``swap_showing_b`` override: when ``True``, returns
full B regardless of the active mode (= the "preview B in isolation"
gesture used to A/B-spot subtle differences).

Inputs may have different sizes — the second argument is resized to
match the first via nearest-neighbour (no extra deps; bilinear quality
isn't worth pulling OpenCV for a review overlay).

Pure numpy. No Qt. Lives outside the UI tree so the compose decisions
stay testable with synthetic ndarrays.
"""

from __future__ import annotations

import numpy as np

from img_player.compare.state import (
    MODE_HORIZONTAL,
    MODE_OPACITY,
    MODE_VERTICAL,
)

# Seam line painted on top of the wipe composite. Accent orange
# (matches ``H.ACCENT`` in the theme), alpha-blended at ~50 % so the
# stroke reads as a thin tint rather than a hard 1-px wall over the
# image — the user can still see the pixels under the seam. Baked
# into the buffer because the BracketsOverlay pipeline has no easy
# hook for "draw a single line at this fraction of the viewport".
_SEAM_COLOR_RGB: tuple[int, int, int] = (232, 144, 28)  # H.ACCENT
_SEAM_ALPHA: float = 0.55
_SEAM_THICKNESS_PX: int = 1


def compose(
    a: np.ndarray,
    b: np.ndarray,
    *,
    mode: str,
    seam: float,
    swap_showing_b: bool = False,
    draw_seam_line: bool = True,
) -> np.ndarray:
    """Combine two layer buffers according to ``mode``.

    Returns an ndarray sized like ``a`` (the second buffer is resized
    to match if it differs). The seam line is painted in white on
    wipe modes when ``draw_seam_line`` is True; opacity / swap modes
    don't get a seam (it would have nowhere meaningful to land).
    """
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError(f"compose expects HxWxC arrays, got {a.shape}, {b.shape}")
    # Match channel count: pad A or B to RGBA if mixed.
    a, b = _match_channels(a, b)
    # Match resolution: B → A's size when they differ. We take A's
    # size as the canonical output so the GL viewport's existing
    # zoom/pan keeps working off the layer-A geometry.
    if b.shape[:2] != a.shape[:2]:
        b = _nn_resize(b, target_h=a.shape[0], target_w=a.shape[1])

    # ``swap_showing_b`` is a global override: when set, return full
    # B regardless of the picked blend mode. Equivalent to a "solo
    # B" preview button — flip on, see only B; flip off, see the
    # blend back. Faster than swapping dropdowns for spot-checking.
    if swap_showing_b:
        return b.copy()
    if mode == MODE_OPACITY:
        t = max(0.0, min(1.0, float(seam)))
        # Cast float32 explicitly so an uint8 input doesn't get
        # truncated by integer arithmetic.
        return (a.astype(np.float32) * (1.0 - t)
                + b.astype(np.float32) * t).astype(a.dtype)
    if mode == MODE_VERTICAL:
        # ``np.empty_like`` + two half-writes is ~50 % less memory
        # traffic than the more obvious ``out = a.copy(); out[..] = b[..]``
        # (a full ``a.copy`` then a half overwrite). Matters during
        # interactive seam-drag, where every mouse-move triggers
        # one of these per frame.
        h, w = a.shape[:2]
        split = int(round(w * max(0.0, min(1.0, float(seam)))))
        out = np.empty_like(a)
        if split > 0:
            out[:, :split] = a[:, :split]
        if split < w:
            out[:, split:] = b[:, split:]
        if draw_seam_line and 0 < split < w:
            _paint_vertical_seam(out, split)
        return out
    if mode == MODE_HORIZONTAL:
        h, w = a.shape[:2]
        split = int(round(h * max(0.0, min(1.0, float(seam)))))
        out = np.empty_like(a)
        if split > 0:
            out[:split] = a[:split]
        if split < h:
            out[split:] = b[split:]
        if draw_seam_line and 0 < split < h:
            _paint_horizontal_seam(out, split)
        return out
    raise ValueError(f"Unknown compare mode: {mode!r}")


# ----------------------------------------------------------------- helpers


def _match_channels(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pad whichever input has fewer channels to RGBA / RGB common shape.

    GL renderer accepts both RGB and RGBA; we keep whatever is the
    *higher* channel count of the two so we never lose information.
    """
    ca, cb = a.shape[2], b.shape[2]
    if ca == cb:
        return a, b
    target = max(ca, cb)
    return _to_channel_count(a, target), _to_channel_count(b, target)


def _to_channel_count(arr: np.ndarray, target: int) -> np.ndarray:
    """Pad ``arr`` to ``target`` channels (3 or 4) by appending an
    opaque alpha or replicating G as B for grayscale-ish inputs."""
    c = arr.shape[2]
    if c == target:
        return arr
    if c == 1 and target >= 3:
        rgb = np.repeat(arr, 3, axis=2)
        if target == 4:
            alpha = np.full(
                (arr.shape[0], arr.shape[1], 1), _opaque_for(arr.dtype),
                dtype=arr.dtype,
            )
            return np.concatenate([rgb, alpha], axis=2)
        return rgb
    if c == 3 and target == 4:
        alpha = np.full(
            (arr.shape[0], arr.shape[1], 1), _opaque_for(arr.dtype),
            dtype=arr.dtype,
        )
        return np.concatenate([arr, alpha], axis=2)
    if c > target:
        # Caller asked for fewer channels — drop trailing ones.
        # Doesn't happen on the live path (max() above) but defensive
        # for tests that pass exotic shapes.
        return arr[..., :target]
    raise ValueError(f"Cannot reshape {c}-channel array to {target} channels")


def _opaque_for(dtype: np.dtype) -> object:
    """Pick the "fully opaque" value for ``dtype`` — 1.0 for float
    buffers, 255 for uint8 (the two we ever see in the compose path)."""
    if np.issubdtype(dtype, np.floating):
        return 1.0
    if dtype == np.uint8:
        return 255
    if dtype == np.uint16:
        return 65535
    return 1


def _nn_resize(arr: np.ndarray, *, target_h: int, target_w: int) -> np.ndarray:
    """Nearest-neighbour resize. Crude but dependency-free; the
    compare mode is for review at full resolution most of the time
    so a precise resampler isn't critical here.
    """
    src_h, src_w = arr.shape[:2]
    if src_h == target_h and src_w == target_w:
        return arr
    # Index arrays so we can resize once for both axes — much faster
    # than a Python loop and fine for review-time refresh rates.
    row_idx = (np.arange(target_h) * (src_h / target_h)).astype(np.int64)
    col_idx = (np.arange(target_w) * (src_w / target_w)).astype(np.int64)
    row_idx = np.clip(row_idx, 0, src_h - 1)
    col_idx = np.clip(col_idx, 0, src_w - 1)
    return arr[row_idx[:, None], col_idx[None, :]]


def _paint_vertical_seam(out: np.ndarray, x: int) -> None:
    """Alpha-blend the accent-orange seam line onto column ``x`` of
    ``out`` (in-place). The seam is ``_SEAM_THICKNESS_PX`` wide and
    blended at ``_SEAM_ALPHA`` so the underlying pixels stay
    partially visible — reads as a thin tint, not a hard divider."""
    x0 = max(0, x - _SEAM_THICKNESS_PX // 2)
    x1 = min(out.shape[1], x0 + _SEAM_THICKNESS_PX)
    if x0 >= x1:
        return
    _blend_seam_strip(out, np.s_[:, x0:x1])


def _paint_horizontal_seam(out: np.ndarray, y: int) -> None:
    """Alpha-blend the accent-orange seam line onto row ``y`` of
    ``out`` (in-place). See :func:`_paint_vertical_seam`."""
    y0 = max(0, y - _SEAM_THICKNESS_PX // 2)
    y1 = min(out.shape[0], y0 + _SEAM_THICKNESS_PX)
    if y0 >= y1:
        return
    _blend_seam_strip(out, np.s_[y0:y1, :])


def _blend_seam_strip(out: np.ndarray, slc: tuple) -> None:
    """In-place alpha blend of the seam colour over ``out[slc]``.

    Float output: linear blend in the buffer's native scale.
    Integer output (uint8 / uint16): blend in float32 then cast back
    so the rounded result stays inside the dtype's range.
    """
    color = _seam_value(out.dtype, channels=out.shape[2])
    strip = out[slc]
    if np.issubdtype(out.dtype, np.floating):
        out[slc] = strip * (1.0 - _SEAM_ALPHA) + color * _SEAM_ALPHA
    else:
        blended = (
            strip.astype(np.float32) * (1.0 - _SEAM_ALPHA)
            + color.astype(np.float32) * _SEAM_ALPHA
        )
        out[slc] = blended.astype(out.dtype)


def _seam_value(dtype: np.dtype, channels: int) -> np.ndarray:
    """Build the per-channel seam colour vector for ``dtype``."""
    r, g, b = _SEAM_COLOR_RGB
    if np.issubdtype(dtype, np.floating):
        rgb = (r / 255.0, g / 255.0, b / 255.0)
        opaque: float = 1.0
    else:
        rgb = (r, g, b)
        opaque = _opaque_for(dtype)  # type: ignore[assignment]
    if channels == 4:
        return np.array([*rgb, opaque], dtype=dtype)
    if channels == 3:
        return np.array(rgb, dtype=dtype)
    if channels == 1:
        return np.array([rgb[0]], dtype=dtype)
    raise ValueError(f"Unsupported seam channel count: {channels}")
