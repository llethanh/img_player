"""Video / audio handlers extracted from app.py.

Free functions taking the :class:`ImgPlayerApp` as first arg —
mirrors the pattern in :mod:`channel_handler` and :mod:`scan_handler`.
The thin methods on :class:`ImgPlayerApp` delegate here so the
existing signal connections (which bind to ``self._on_*``,
``self._refresh_*``, etc.) keep working unchanged.

Two responsibilities collapsed into one module because video and
audio are tightly coupled in this codebase:

* Video decode = ``VideoSourceManager`` per-layer worker threads
  (bypass the OIIO master-frame cache).
* Audio playback = sounddevice ``OutputStream`` driven by an
  :class:`AudioSource` opened lazily for the topmost-visible-with-
  audio video layer.

Both paths share the same source of truth — the ``LayerStack`` and
the layer's ``video_metadata`` — so keeping them in one module
avoids cross-module coupling for the active-layer policy.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QMessageBox

if TYPE_CHECKING:
    from img_player.app import ImgPlayerApp
    from img_player.layers.models import Layer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------- Video

def close_orphan_video_sources(app: ImgPlayerApp) -> None:
    """Close video decoders whose layer is no longer in the stack.

    Hooked into ``layers_changed`` — without it, removing a video
    layer leaks a PyAV container (file handle on Windows). Cheap:
    the manager dict is small (one entry per video layer) and
    ``close`` is a no-op when the layer id is unknown.
    """
    if not app._video_sources._sources:
        return
    live_ids = {layer.id for layer in app._layer_stack}
    for layer_id in list(app._video_sources._sources.keys()):
        if layer_id not in live_ids:
            app._video_sources.close(layer_id)


def decode_video_layer(
    app: ImgPlayerApp, layer: Layer, master_frame: int,
):  # type: ignore[no-untyped-def]
    """Pull pixels for ``master_frame`` from a video-backed layer.

    Maps master_frame → video-source time using the layer's offset
    and the video's native fps (taken from the probe), then delegates
    to :class:`VideoSourceManager` which handles the seek-then-decode-
    forward strategy and its single-frame cache.

    Returns ``(H, W, 4)`` float32 RGBA in [0, 1], ready for the GL
    viewport's ``set_frame``. Returns ``None`` when the master frame
    falls outside the layer's range (caller falls back to the gap
    placeholder logic via the cache path).
    """
    if not layer.is_video or layer.video_metadata is None:
        return None
    if not layer.covers(master_frame):
        return None
    meta = layer.video_metadata
    if meta.fps is None or meta.fps <= 0:
        return None
    # ``master_frame - master_start`` is the source frame index in
    # the video's own 0..N-1 range. Convert to seconds at the
    # video's native rate so a session FPS that differs from the
    # video FPS still hits the right time on the source clock —
    # the v1 simplification (session FPS == video FPS) means this
    # falls out as ``frame / fps``, but the math stays correct
    # once we mix sources later.
    source_frame_idx = master_frame - layer.master_start
    t_seconds = source_frame_idx / float(meta.fps)
    return app._video_sources.decode_at(layer.id, meta.path, t_seconds)


def open_video_path(app: ImgPlayerApp, path: Path) -> None:
    """Replace the current state with a single video file.

    Probes the container, builds a ``Layer.from_video``, wipes the
    previous LayerStack contents, and seeds the controller with the
    synthetic SequenceInfo so transport / timeline / frame navigation
    work the same way as for image sequences. The actual decode
    happens lazily in ``_on_frame_changed`` via :class:`VideoSourceManager`.
    """
    from img_player.layers.models import Layer
    from img_player.media import probe_video
    try:
        metadata = probe_video(path)
    except Exception as exc:
        log.exception("video probe failed for %s", path)
        QMessageBox.critical(
            app._window, "Cannot open video", f"{path.name}\n\n{exc}",
        )
        app._window.set_status("Ready.")
        return
    if not metadata.has_video:
        QMessageBox.warning(
            app._window, "Cannot open video",
            f"{path.name} has no video stream.",
        )
        app._window.set_status("Ready.")
        return

    layer = Layer.from_video(metadata)
    # Force-exit any active review mode (compare, contact-sheet)
    # before swapping in the video layer. Same rationale as the
    # image-sequence path in ``scan_handler.apply_scan_result`` —
    # the mode state references layer ids from the OLD stack and
    # would point at nothing meaningful after the swap.
    app._exit_review_modes()
    # Detach the cache first — without this the previous sequence's
    # prefetch keeps walking image paths in the background while we
    # swap in a video layer the cache can't decode.
    app._cache.detach()
    # Wipe the previous stack in one batched undo step, then add the
    # video layer. We do NOT call ``controller.load_sequence`` because
    # that calls ``cache.attach(sequence)`` which would rebuild an
    # image-sequence Layer wrapping the synthetic video sequence —
    # clobbering Layer.from_video and pointing the cache's path index
    # at the .mp4 container. Instead, push the controller's sequence +
    # navigable range directly, then re-emit a synthetic frame_changed
    # so the viewport renders the first video frame.
    with app._layer_stack.batch():
        for existing in tuple(app._layer_stack):
            app._layer_stack.remove(existing.id)
        app._layer_stack.add(layer, position=0)
        app._layer_stack.set_focus(layer.id)
    app._controller._sequence = layer.sequence  # type: ignore[attr-defined]
    app._controller._state = replace(  # type: ignore[attr-defined]
        app._controller._state,  # type: ignore[attr-defined]
        current_frame=layer.master_start,
        is_playing=False,
        in_frame=None,
        out_frame=None,
        dropped_frames=0,
    )
    app._controller.set_navigable_range(
        layer.master_start, layer.master_end,
    )
    if metadata.fps is not None:
        app._controller.set_fps(float(metadata.fps))
    # Broadcast the state we just installed by hand so transport,
    # timeline, layer panel etc. all rebind to the new clip.
    app._controller.state_changed.emit(app._controller._state)  # type: ignore[attr-defined]
    app._controller.frame_changed.emit(layer.master_start)
    app._window.update_sequence_info(layer.sequence)
    # Mirror the image-sequence load path (``scan_handler``): run the
    # source-colorspace auto-detect against the new layer. The video
    # branch of ``_guess_source_colorspace`` reads the container's
    # color tags (color_primaries / color_trc) from the layer's
    # ``video_metadata`` — without this a loaded mp4 / mov / mkv
    # would stay on the default colorspace even when the container
    # explicitly declares Rec.709, HDR PQ, HLG, Rec.2020 etc.
    app._guess_source_colorspace(layer.sequence, layer=layer)
    if metadata.fps is not None:
        app._window.set_status(
            f"Loaded video {path.name} "
            f"({metadata.width}×{metadata.height}, "
            f"{float(metadata.fps):.3f} fps, "
            f"{metadata.frame_count} frames)"
        )
    else:
        app._window.set_status(f"Loaded video {path.name}")


def add_video_layer(app: ImgPlayerApp, path: Path) -> bool:
    """Probe ``path`` and append a :class:`Layer.from_video` at the
    top of the stack. Returns ``True`` on success, ``False`` when
    the probe failed (unsupported / corrupt / no video stream).

    Used both by ``_on_add_layer_requested`` (drop on layer panel,
    no replace) and by ``_open_path`` when a multi-source drop
    contains a mix of video and image sequences.
    """
    from img_player.layers.models import Layer
    from img_player.media import probe_video
    try:
        metadata = probe_video(path)
    except Exception as exc:
        log.exception("video probe failed for %s", path)
        app._window.set_status(f"Cannot add video {path.name}: {exc}")
        return False
    if not metadata.has_video:
        app._window.set_status(
            f"Cannot add {path.name}: no video stream."
        )
        return False
    layer = Layer.from_video(metadata)
    app._layer_stack.add(layer, position=0)
    # Same rationale as ``open_video_path``: feed the container's
    # color tags into the source-colorspace auto-detect so adding a
    # video layer matches the behaviour of adding an image sequence.
    app._guess_source_colorspace(layer.sequence, layer=layer)
    return True


# ---------------------------------------------------------------------- Audio

def pick_active_audio_layer(app: ImgPlayerApp) -> Layer | None:
    """Choose which video layer's audio should play right now.

    Policy (option 1c — "solo / mute per layer with topmost-fallback"),
    coverage-aware:

    * **Coverage matters.** A layer that doesn't cover the current
      playhead can't be active — the audio would play in advance
      of the video reaching that clip. Without this guard, the
      feeder reads continuously while the playhead is in a void
      (offset > 0, between two clips, etc.), and by the time the
      playhead enters the layer the audio is already N seconds
      ahead of it.
    * Solo wins. If any video layer has ``audio_solo=True`` AND
      covers the playhead, that one plays even if another video
      layer is on top.
    * Otherwise the topmost-visible video layer with audio that
      covers the playhead plays.
    * Layers with ``audio_mute=True`` never play.
    * Layers without an audio stream (``has_audio=False``) never
      play, regardless of solo / mute.

    Returns ``None`` when no layer qualifies.
    """
    if app._layer_stack is None:
        return None
    cur = app._controller.state.current_frame
    # Solo first.
    for layer in app._layer_stack.layers():
        if (
            layer.is_video
            and layer.audio_solo
            and not layer.audio_mute
            and layer.video_metadata
            and layer.video_metadata.has_audio
            and layer.covers(cur)
        ):
            return layer
    # No covering solo — pick the topmost visible video layer that
    # covers + has audio.
    for layer in app._layer_stack.layers():
        if (
            layer.visible
            and layer.is_video
            and not layer.audio_mute
            and layer.video_metadata
            and layer.video_metadata.has_audio
            and layer.covers(cur)
        ):
            return layer
    return None


def current_layer_time(
    app: ImgPlayerApp, layer: Layer,
) -> float | None:
    """Master playhead → seconds on ``layer``'s native timebase.

    Mirrors the math in :func:`decode_video_layer`. Returns ``None``
    when the layer doesn't cover the playhead.
    """
    if not layer.is_video or layer.video_metadata is None:
        return None
    meta = layer.video_metadata
    if meta.fps is None or meta.fps <= 0:
        return None
    cur = app._controller.state.current_frame
    if not layer.covers(cur):
        return None
    source_frame_idx = cur - layer.master_start
    return source_frame_idx / float(meta.fps)


def refresh_active_audio(app: ImgPlayerApp) -> None:
    """Sync the AudioOutput's source + gain with the layer stack.

    Idempotent: if the active layer is unchanged we just update the
    gain (cheap). When the active layer changes we open a new
    AudioSource and ``set_source`` it on the output (which closes
    the previous one). When no layer qualifies we set source to None
    — the output goes silent.

    Called on ``frame_changed`` (every tick) and on ``layers_changed``
    / ``visibility_changed``. **Does not reseek** on the same-layer
    branch — reseeking on every tick would flush the AudioOutput
    ring buffer and stutter audibly. Reseek-on-layer-state-change is
    handled by :func:`reseek_active_audio_for_layer_change`.
    """
    from img_player.media import AudioSource
    layer = pick_active_audio_layer(app)
    if layer is None:
        if app._active_audio_layer_id is not None:
            app._audio_output.set_source(None, None)
            app._active_audio_layer_id = None
        return
    if app._active_audio_layer_id == layer.id:
        # Same layer — apply (potentially new) gain only.
        app._audio_output.set_gain(float(layer.audio_gain))
        return
    # Open a new AudioSource for this layer. Failures (corrupt audio
    # stream, sample format we can't resample) downgrade to silence
    # rather than crashing.
    try:
        assert layer.video_metadata is not None
        source = AudioSource(layer.video_metadata.path)
    except Exception:
        log.exception("[audio] failed to open source for layer %s", layer.id)
        app._audio_output.set_source(None, None)
        app._active_audio_layer_id = None
        return
    # Position the new source at the current playhead so the user
    # hears the right offset, not the start of the file.
    try:
        t = current_layer_time(app, layer)
        if t is not None:
            source.seek(t)
    except Exception:
        log.exception("[audio] initial seek failed")
    app._audio_output.set_source(layer.id, source)
    app._audio_output.set_gain(float(layer.audio_gain))
    app._active_audio_layer_id = layer.id


def reseek_active_audio_for_layer_change(app: ImgPlayerApp) -> None:
    """Reseek the audio source after a per-layer mutation that
    shifted the source-time ↔ master-time mapping (offset / trim
    drag, etc.). Refreshes the active layer first so coverage
    transitions caused by the same edit are also handled.

    Called only on ``layer_modified``, never on ``frame_changed``.
    """
    refresh_active_audio(app)
    layer = pick_active_audio_layer(app)
    if layer is None:
        return
    t = current_layer_time(app, layer)
    if t is not None:
        app._audio_output.seek(t)
