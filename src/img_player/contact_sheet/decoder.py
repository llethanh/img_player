"""Per-layer decode for contact sheet mode.

Mirrors :class:`compare.decode.CompareDecoder` but with two
contact-sheet-specific tweaks:

* **Offset is ignored.** The user-facing semantic is "show every
  layer as if it started at the same frame" — so the source frame
  we look up is ``layer.layer_in + contact_frame`` rather than
  ``layer.layer_in + (master_frame - layer.master_start)``. If the
  contact-sheet playhead is past a layer's last frame, the decoder
  returns the layer's last successfully-decoded buffer (clamp) so
  the tile freezes on its tail instead of going black mid-scrub.

* **All visible layers, not just two.** Returns a list of
  ``(layer, ndarray | None)`` pairs in stack order (top-down)
  ready for the compositor to grid-place.

The decoder keeps a per-layer 1-slot cache (same shape as
``CompareDecoder``) — cheap insurance against re-decoding the same
file when the contact-sheet playhead idles on a frame.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from img_player.io.reader import FrameReadError, read_frame
from img_player.layers import Layer

log = logging.getLogger(__name__)


@dataclass
class _LastDecode:
    """One-slot cache per layer id."""

    layer_id: str
    source_frame: int
    arr: np.ndarray


class ContactSheetDecoder:
    """Decodes every visible layer at the same "from-start" frame.

    ``video_sources`` is the app's :class:`VideoSourceManager`; we
    reuse it so contact-sheet video tiles share PyAV containers
    with the live viewport (memory + open-handle savings on shots
    that mix sequence + video layers).
    """

    def __init__(self, video_sources) -> None:  # type: ignore[no-untyped-def]
        self._video_sources = video_sources
        self._last: dict[str, _LastDecode] = {}

    # ------------------------------------------------------------------ Public

    def decode_all(
        self,
        layers: list[Layer],
        contact_offset: int,
    ) -> list[tuple[Layer, np.ndarray | None]]:
        """Return ``(layer, decoded_arr_or_None)`` for every layer.

        ``contact_offset`` is the **0-based playback offset** — the
        number of frames since playback started. At offset 0 every
        tile shows its layer's first source frame; at offset N each
        tile shows its layer's (N+1)-th frame, clamped to the
        layer's trim length so shorter layers freeze on their last
        frame rather than reading out of range. The caller (=
        ``ImgPlayerApp._render_contact_sheet``) converts the
        master-frame number into this offset by subtracting the
        navigable-range start.

        Layers that failed to decode at all return ``None`` so the
        compositor can paint the unavailable-stripes placeholder.
        """
        out: list[tuple[Layer, np.ndarray | None]] = []
        for layer in layers:
            arr = self.decode_one(layer, contact_offset)
            out.append((layer, arr))
        return out

    def decode_one(
        self, layer: Layer, contact_offset: int,
    ) -> np.ndarray | None:
        """Decode ``layer`` at ``contact_offset``-from-start.

        Returns the layer's last cached buffer when the requested
        frame is past the layer's trim range — the "freeze on tail"
        behaviour the user expects when layers have different
        lengths.
        """
        if layer.is_video:
            return self._decode_video(layer, contact_offset)
        return self._decode_image(layer, contact_offset)

    def invalidate(self, layer_id: str | None = None) -> None:
        """Drop cached buffers. ``None`` = wipe everything (used at
        mode entry / exit so we don't leak stale state)."""
        if layer_id is None:
            self._last.clear()
        else:
            self._last.pop(layer_id, None)

    # ------------------------------------------------------------------ Internals

    def _decode_image(
        self, layer: Layer, contact_offset: int,
    ) -> np.ndarray | None:
        """Image-sequence layer: pick the source frame at the requested
        contact-sheet offset, clamping to the layer's trim range so
        layers shorter than the contact sheet's longest layer freeze
        on their final frame instead of decoding garbage."""
        trim_length = layer.trim_length
        if trim_length <= 0:
            return self._last_arr_or_none(layer.id)
        # Stills hold the same source frame across their whole range.
        if layer.is_still:
            source_frame = layer.layer_in
        else:
            clamped = max(0, min(contact_offset, trim_length - 1))
            source_frame = layer.layer_in + clamped

        last = self._last.get(layer.id)
        if last is not None and last.source_frame == source_frame:
            return last.arr

        path = None
        for fi in layer.sequence.frames:
            if fi.frame_number == source_frame:
                path = fi.path
                break
        if path is None:
            # Source frame doesn't have a matching file — fall back to
            # the previous decode if we have one, else give up.
            return self._last_arr_or_none(layer.id)

        sel = layer.channel_selection
        channels = list(sel.active.channels) if sel is not None else None
        try:
            arr = read_frame(path, channels=channels, as_half=False)
        except FrameReadError as err:
            log.warning("[contact_sheet] decode failed for %s: %s", path, err)
            return self._last_arr_or_none(layer.id)
        self._last[layer.id] = _LastDecode(
            layer_id=layer.id, source_frame=source_frame, arr=arr,
        )
        return arr

    def _decode_video(
        self, layer: Layer, contact_offset: int,
    ) -> np.ndarray | None:
        if layer.video_metadata is None:
            return None
        meta = layer.video_metadata
        if meta.fps is None or meta.fps <= 0:
            return None
        # Clamp to the layer's trim length so video tiles freeze on
        # their last frame past the end — same semantic as images.
        clamped = max(0, min(contact_offset, layer.trim_length - 1))
        t_seconds = clamped / float(meta.fps)
        try:
            return self._video_sources.decode_at(layer.id, meta.path, t_seconds)
        except Exception:
            log.exception("[contact_sheet] video decode failed for %s", layer.id)
            return self._last_arr_or_none(layer.id)

    def _last_arr_or_none(self, layer_id: str) -> np.ndarray | None:
        """Return the most recently cached array for ``layer_id``, or
        ``None`` if we never successfully decoded this layer."""
        last = self._last.get(layer_id)
        return last.arr if last is not None else None
