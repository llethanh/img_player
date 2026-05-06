"""Bridges between the :class:`CompareBand` UI and the app singleton.

Free functions (matching the rest of the ``*_handler`` modules) that
mutate ``app._compare_state`` and trigger a redisplay. Kept outside
``app.py`` so the imperative bookkeeping doesn't bloat the bootstrap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from img_player.compare.compose import compose
from img_player.compare.state import (
    MODE_HORIZONTAL,
    MODE_OPACITY,
    MODE_VERTICAL,
    CompareState,
)
from img_player.ui.compare_band import _LayerOption

if TYPE_CHECKING:
    from img_player.app import ImgPlayerApp

log = logging.getLogger(__name__)


# ============================================================================
# Layer enumeration for the dropdowns
# ============================================================================


def available_layer_options(app: ImgPlayerApp) -> list[_LayerOption]:
    """Snapshot the layer stack into the dropdown's data model.

    Top-of-stack first (matches the layer panel ordering). Empty
    stacks return ``[]`` — the band hides naturally when the user
    never had two layers to compare against.
    """
    return [
        _LayerOption(layer_id=layer.id, name=layer.name or "(unnamed)")
        for layer in app._layer_stack.layers()
    ]


def refresh_band_layers(app: ImgPlayerApp) -> None:
    """Repopulate both dropdowns from the live LayerStack.

    Called from :meth:`_refresh_after_stack_change` so a layer
    add/remove is reflected immediately in the band. If the
    currently-selected A or B was deleted, fall back to ``None`` —
    the band's compose path then draws the remaining layer at full
    coverage.
    """
    band = app._window.compare_band
    options = available_layer_options(app)
    state: CompareState = app._compare_state
    valid_ids = {opt.layer_id for opt in options}
    a_id = state.layer_a_id if state.layer_a_id in valid_ids else None
    b_id = state.layer_b_id if state.layer_b_id in valid_ids else None
    if a_id != state.layer_a_id or b_id != state.layer_b_id:
        # Patch state to drop stale ids so the next decode pass
        # doesn't try to read a vanished layer.
        state.layer_a_id = a_id
        state.layer_b_id = b_id
    band.set_available_layers(options, a_id=a_id, b_id=b_id)


# ============================================================================
# Toggle
# ============================================================================


def toggle_compare(app: ImgPlayerApp) -> None:
    """Flip ``CompareState.enabled``, with smart auto-pick of A/B.

    Entering compare mode with no layers selected auto-fills A and B
    from the top two layers — saves the user one click in the
    typical "I want to A/B these two" case.
    """
    state = app._compare_state
    # Block the toggle when there aren't two layers to compare. The
    # button is already gated, but the W shortcut isn't — guard here.
    if not state.enabled and len(list(app._layer_stack.layers())) < 2:
        app._window.set_status(
            "Compare needs at least 2 layers — load another sequence first.",
        )
        return
    state.enabled = not state.enabled
    band = app._window.compare_band
    # Sync the transport button so its checked state always
    # reflects ``state.enabled`` regardless of how the toggle was
    # triggered (button click, W shortcut, ✕ on the band).
    app._window.transport.set_compare_checked(state.enabled)
    if state.enabled:
        # Auto-pick: top two layers if A/B aren't set yet.
        layers = list(app._layer_stack.layers())
        if state.layer_a_id is None and layers:
            state.layer_a_id = layers[0].id
        if state.layer_b_id is None and len(layers) >= 2:
            state.layer_b_id = layers[1].id
        # If we still don't have both, fall back to A only — the
        # decoder will return one buffer and the compose helper's
        # fallback path will display it as-is.
        refresh_band_layers(app)
        band.set_mode(state.mode)
        band.set_seam(state.seam)
        band.set_swap_showing_b(state.swap_showing_b)
        app._window.set_compare_band_visible(True)
        band.raise_()
    else:
        app._window.set_compare_band_visible(False)
        # Drop the per-layer decoder caches so a re-entry doesn't
        # paint stale pixels (rare, but cheap).
        app._compare_decoder.invalidate()
    # Either way: trigger a redisplay so the viewport reflects the
    # new mode (compare overlay or back to the normal composite).
    app._redisplay_current()


# ============================================================================
# Field setters — wired straight from the band's signals
# ============================================================================


def set_layer_a(app: ImgPlayerApp, layer_id: str) -> None:
    state = app._compare_state
    if state.layer_a_id == layer_id:
        return
    state.layer_a_id = layer_id
    app._compare_decoder.invalidate(layer_id)
    app._redisplay_current()


def set_layer_b(app: ImgPlayerApp, layer_id: str) -> None:
    state = app._compare_state
    if state.layer_b_id == layer_id:
        return
    state.layer_b_id = layer_id
    app._compare_decoder.invalidate(layer_id)
    app._redisplay_current()


def set_mode(app: ImgPlayerApp, mode: str) -> None:
    state = app._compare_state
    if state.mode == mode:
        return
    state.mode = mode
    app._window.compare_band.set_mode(mode)
    app._redisplay_current()


def set_seam(app: ImgPlayerApp, seam: float) -> None:
    state = app._compare_state
    state.seam = max(0.0, min(1.0, float(seam)))
    # Don't re-feed the slider — the band already shows the new
    # value (the signal came from there). Drag-from-viewer routes
    # through ``set_seam_from_viewport`` which DOES re-feed the
    # slider so the two stay in sync.
    app._redisplay_current()


def set_seam_from_viewport(app: ImgPlayerApp, seam: float) -> None:
    """Drag-on-image → mirror the new seam to the slider too."""
    set_seam(app, seam)
    app._window.compare_band.set_seam(app._compare_state.seam)


def toggle_swap(app: ImgPlayerApp) -> None:
    """Flip the always-visible Solo-B override.

    Works regardless of the active blend mode: when the override is
    True, the compose path returns full B (overriding the wipe /
    opacity); when False, the picked mode applies normally.
    """
    state = app._compare_state
    state.swap_showing_b = not state.swap_showing_b
    app._window.compare_band.set_swap_showing_b(state.swap_showing_b)
    app._redisplay_current()


def swap_layers(app: ImgPlayerApp) -> None:
    """Permute A and B in the dropdowns + state."""
    state = app._compare_state
    state.layer_a_id, state.layer_b_id = state.layer_b_id, state.layer_a_id
    refresh_band_layers(app)
    app._redisplay_current()


def nudge_seam(app: ImgPlayerApp, delta: float) -> None:
    """Bump the seam by ±``delta`` (e.g. 0.01 for 1%). Wraps the
    band sync so the slider follows."""
    set_seam_from_viewport(app, app._compare_state.seam + delta)


# ============================================================================
# Render hook — called by app._on_frame_changed
# ============================================================================


def render_compare(app: ImgPlayerApp, master_frame: int) -> bool:
    """Try to render the compare overlay for ``master_frame``.

    Returns ``True`` when the overlay handled the upload (caller
    should skip the normal display path); ``False`` when compare
    isn't active or both layer buffers couldn't be obtained, in
    which case the caller falls through to the standard composite.
    """
    state = app._compare_state
    if not state.is_active():
        return False
    layer_a = app._layer_stack.find(state.layer_a_id)
    layer_b = app._layer_stack.find(state.layer_b_id)
    if layer_a is None or layer_b is None:
        # Stale ids — refresh the band so the dropdowns drop them
        # and try again on the next frame change.
        refresh_band_layers(app)
        return False
    arr_a = app._compare_decoder.decode(layer_a, master_frame)
    arr_b = app._compare_decoder.decode(layer_b, master_frame)
    # Out-of-range fallbacks: if only one is available, draw it
    # alone; if neither, give up and let the normal path try.
    if arr_a is None and arr_b is None:
        return False
    if arr_a is None:
        composed = arr_b
    elif arr_b is None:
        composed = arr_a
    else:
        try:
            composed = compose(
                arr_a, arr_b,
                mode=state.mode,
                seam=state.seam,
                swap_showing_b=state.swap_showing_b,
            )
        except Exception:
            log.exception("[compare] compose failed at master frame %d", master_frame)
            return False
    if composed.ndim != 3:
        return False
    if composed.shape[2] > 4:
        composed = composed[:, :, :4]
    composed = _ensure_contiguous(composed)
    app._window.viewer.gl.set_frame(composed)
    return True


def _ensure_contiguous(arr: np.ndarray) -> np.ndarray:
    """GL upload requires contiguous strides — the wipe path slices
    + assigns view-style buffers that may be non-contiguous. Cheap
    no-op when already contiguous."""
    if arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr)


# Re-exports for the typing-only convenience of app.py / main_window.
__all__ = [
    "MODE_HORIZONTAL",
    "MODE_OPACITY",
    "MODE_VERTICAL",
    "available_layer_options",
    "nudge_seam",
    "refresh_band_layers",
    "render_compare",
    "set_layer_a",
    "set_layer_b",
    "set_mode",
    "set_seam",
    "set_seam_from_viewport",
    "swap_layers",
    "toggle_compare",
    "toggle_swap",
]
