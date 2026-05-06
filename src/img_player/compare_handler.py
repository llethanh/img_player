"""Bridges between the :class:`CompareBand` UI and the app singleton.

Free functions (matching the rest of the ``*_handler`` modules) that
mutate ``app._compare_state`` and trigger a redisplay. Kept outside
``app.py`` so the imperative bookkeeping doesn't bloat the bootstrap.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QCursor

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


def set_seam_from_pointer(
    app: ImgPlayerApp, x: float, y: float, widget_w: int, widget_h: int,
) -> None:
    """Map a viewport mouse position to a seam value.

    Called by the compare-mode mouse filter (installed in
    ``app._wire_compare``) on left-click + drag. The mapping depends
    on the active blend mode:

    * **Vertical** wipe — horizontal drag (X / widget_w) controls the
      left↔right seam.
    * **Horizontal** wipe — vertical drag (Y / widget_h) controls the
      top↕bottom seam.
    * **Opacity** blend — horizontal drag (X / widget_w). Both axes
      were considered, but the band's slider runs left-to-right and
      sticking to the same axis keeps muscle memory consistent.
    """
    state = app._compare_state
    if state.mode == MODE_VERTICAL:
        seam = x / max(1.0, float(widget_w))
    elif state.mode == MODE_HORIZONTAL:
        seam = y / max(1.0, float(widget_h))
    elif state.mode == MODE_OPACITY:
        seam = x / max(1.0, float(widget_w))
    else:
        return
    set_seam_from_viewport(app, seam)


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


# ============================================================================
# Viewport mouse filter — left-click drag updates the seam
# ============================================================================


class _ViewportSeamFilter(QObject):
    """Intercept right-click drag on the GL viewport while compare
    mode is active and route the cursor to the seam.

    Sits as a Qt event filter on ``viewer.gl`` (installed in
    ``app._wire_compare``). Only acts when ``CompareState.is_active``
    — outside compare mode it falls through so the GL viewport's
    own gestures keep working. Right-click was picked over
    left-click so the standard left-drag gesture (= drag-scrub the
    timeline) stays available even while comparing.

    Behaviour per blend mode:

    * **Vertical / Horizontal wipe** — *absolute*: pressing snaps
      the seam to the cursor (so the wipe line jumps to where the
      user grabbed), and subsequent moves keep the seam attached to
      the cursor. Feels like "grab the seam line and drag it".
    * **Opacity blend** — *relative + amplified*: pressing captures
      an anchor (current seam + cursor coords). Each move adds the
      cursor delta (× ``_OPACITY_GAIN``) to the captured seam.
      Multiplier > 1 means less mouse distance is needed to sweep
      0 → 1 — a short gesture covers the full opacity range, which
      feels nicer than dragging across the whole viewport.

    Returning ``True`` from ``eventFilter`` suppresses the event so
    the GL viewport doesn't ALSO try to drag-scrub the timeline at
    the same gesture.
    """

    # Opacity drag amplification. ``3.0`` means a third of the
    # viewport width drags the opacity from 0 to 1, which is a
    # comfortable wrist gesture without being twitchy.
    _OPACITY_GAIN: float = 3.0

    def __init__(self, app: ImgPlayerApp) -> None:
        super().__init__(app._window)
        self._app = app
        self._dragging = False
        # Press-time anchors — the seam moves relative to these so a
        # short drag near the seam line nudges it slightly rather
        # than teleporting it to the absolute cursor position.
        self._press_x: float = 0.0
        self._press_y: float = 0.0
        self._press_seam: float = 0.5

    def eventFilter(self, watched: object, event: QEvent) -> bool:  # type: ignore[override]
        state = self._app._compare_state
        if not state.is_active():
            return False
        et = event.type()
        if (
            et == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.RightButton
        ):
            pos = event.position()
            self._dragging = True
            self._press_x = pos.x()
            self._press_y = pos.y()
            self._press_seam = state.seam
            # Wipe modes: snap the seam to the cursor at press time
            # so the line jumps under the click. Opacity stays
            # delta-based — there's no visible seam line, so an
            # absolute snap would feel disorienting.
            if state.mode in (MODE_VERTICAL, MODE_HORIZONTAL):
                self._apply_absolute(watched, event)
            return True
        if et == QEvent.Type.MouseMove and self._dragging:
            if state.mode in (MODE_VERTICAL, MODE_HORIZONTAL):
                self._apply_absolute(watched, event)
            else:
                self._apply_delta(watched, event)
            # Force a synchronous repaint so the seam line catches up
            # with the cursor on the same event tick. ``set_frame``
            # only calls ``update`` which schedules an async paint —
            # at fast mouse-drag rates Qt can stack 2-3 mouse-move
            # events between paints, producing visible lag between
            # cursor and seam. ``repaint`` flushes the pending GL
            # upload before returning.
            self._app._window.viewer.gl.repaint()
            return True
        if (
            et == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.RightButton
            and self._dragging
        ):
            self._dragging = False
            return True
        return False

    def _apply_absolute(self, watched: object, event: QEvent) -> None:
        """Wipe modes: seam follows the cursor's absolute position
        in *image space*.

        We use ``QCursor.pos()`` (= the latest OS cursor position) +
        ``mapFromGlobal`` instead of ``event.position()`` because Qt
        can queue several mouse-move events between paints during a
        fast drag — using the embedded position would have us paint
        to a stale spot. Querying the cursor at handler-time gives
        the most recent position regardless of event backlog.

        The cursor's widget coordinates are then converted to
        image-space via the GL viewport's current zoom + pan
        transform, so the seam stays under the cursor regardless of
        how the image is fit / zoomed / panned.
        """
        from img_player.annotate.overlay import widget_to_image

        gl = self._app._window.viewer.gl
        # Latest cursor position in this widget's local coords.
        local = watched.mapFromGlobal(QCursor.pos())
        cur_x = float(local.x())
        cur_y = float(local.y())
        img_w, img_h = gl.image_size()
        if img_w <= 0 or img_h <= 0:
            # No image loaded yet — fall back to widget-space mapping
            # so the press still produces *something* consistent.
            try:
                w = int(watched.width())
                h = int(watched.height())
            except AttributeError:
                return
            set_seam_from_pointer(self._app, cur_x, cur_y, w, h)
            return
        factor, pan_x, pan_y = gl.current_transform()
        if factor <= 0:
            return
        # ``event`` is unused in this code path — see ``QCursor.pos``
        # rationale in the docstring above.
        del event
        ix, iy = widget_to_image(
            widget_xy=(cur_x, cur_y),
            widget_size=(gl.width(), gl.height()),
            img_size=(img_w, img_h),
            factor=factor,
            pan=(pan_x, pan_y),
        )
        # Map image-space cursor to a [0..1] seam value. Clamping is
        # done by ``set_seam_from_viewport`` downstream, so a cursor
        # outside the image bounds (= the user dragged off-image)
        # still pegs the seam at 0 or 1 cleanly.
        state = self._app._compare_state
        if state.mode == MODE_HORIZONTAL:
            seam = iy / float(img_h)
        else:
            seam = ix / float(img_w)
        set_seam_from_viewport(self._app, seam)

    def _apply_delta(self, watched: object, event: QEvent) -> None:
        """Opacity mode: seam moves relative to press, amplified by
        ``_OPACITY_GAIN`` so a short drag covers the full 0 → 1 range.
        ``QCursor.pos`` over the queued event position for the same
        reason as ``_apply_absolute`` (avoid stale events during
        fast drags).
        """
        del event  # unused — see ``QCursor.pos`` rationale above.
        try:
            w = float(watched.width())
        except AttributeError:
            return
        local = watched.mapFromGlobal(QCursor.pos())
        # X axis on opacity (consistent with the band's left-to-right
        # slider). The Y component is not folded in — adding it would
        # make the gesture jittery for a user who's just shaking
        # their hand horizontally.
        denom = max(1.0, w)
        delta_frac = (float(local.x()) - self._press_x) / denom
        set_seam_from_viewport(
            self._app,
            self._press_seam + delta_frac * self._OPACITY_GAIN,
        )


# Re-exports for the typing-only convenience of app.py / main_window.
__all__ = [
    "MODE_HORIZONTAL",
    "MODE_OPACITY",
    "MODE_VERTICAL",
    "available_layer_options",
    "set_seam_from_pointer",
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
