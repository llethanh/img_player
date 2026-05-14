"""File → Save Frame As… — single-frame full-image export.

Single entry point :func:`open_save_frame_dialog`. Builds the dialog,
reads the *source* pixels of the current frame at the user-picked
resolution (NOT a screenshot of the GL viewport — so zoom / pan
don't crop the saved image), and writes the file to disk via Qt's
standard image writer (PNG / JPEG / TIFF / BMP / WebP).

This now mirrors the Export pipeline's render tail (OCIO display
transform, OIIO Lanczos resize, optional annotation bake, optional
compare-overlay bake) for a single frame — the only differences vs.
the multi-frame Export are:

* The frame range is exactly one frame (the live ``current_frame``).
* Output dtype is always uint8 (we route through Qt's image writer).
* The user can opt out of the compare overlay even when it's live.

When contact-sheet mode is active the per-tile grid is composed
directly at the user's chosen resolution — same render-tile-then-
compose path as the live viewer's contact sheet, but at a clean
output size instead of the viewport size.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtGui import QImage, QImageWriter

from img_player.export.renderer import (
    CompareRenderContext,
    FrameRenderer,
    RenderContext,
)
from img_player.export.settings import ExportSettings, MissingFramePolicy
from img_player.preferences import _qbool
from img_player.ui.save_frame_dialog import SaveFrameDialog, SaveFrameSettings

if TYPE_CHECKING:
    from img_player.app import ImgPlayerApp

log = logging.getLogger(__name__)


# Qt's QImageWriter accepts these (case-insensitive) as the format
# hint. Mapping kept small and explicit so we don't depend on the
# user's Qt build happening to support whatever extension they typed
# — and so adding a format here is the single source of truth.
_QT_FORMAT_FOR_EXT: dict[str, str] = {
    "png": "png",
    "jpg": "jpg",
    "jpeg": "jpg",
    "tif": "tiff",
    "tiff": "tiff",
    "bmp": "bmp",
    "webp": "webp",
}


def open_save_frame_dialog(app: ImgPlayerApp) -> None:
    """File → Save Frame As… — full-image render of one frame.

    Refuses to run when no sequence is loaded (nothing to render).
    Otherwise builds the dialog with sensible defaults sourced from
    the last-used settings + the current sequence's directory, runs
    the render on accept, and surfaces the result via the status
    bar.
    """
    seq = app._controller.sequence
    if seq is None:
        app._window.set_status("Save Frame: no sequence loaded.")
        return

    state = app._controller.state
    master_frame = state.current_frame

    # Suggested filename: ``{seq_basename}_{master_frame:04d}``. Strips
    # trailing separators on the basename so ``render.`` produces
    # ``render_0042`` rather than ``render._0042``.
    base = (seq.base_name or "frame").rstrip("._- ") or "frame"
    suggested_filename = f"{base}_{master_frame:04d}"

    # Last-used dir wins over the sequence directory when the user
    # has previously saved frames elsewhere — typical workflow is to
    # batch-save snapshots into a "review_screens" folder, not
    # alongside the source plates.
    last = app._prefs.save_frame_settings
    suggested_dir = Path(str(last.get("output_dir") or seq.directory))
    last_format = str(last.get("format") or "png").lower()
    last_with_annotations = _qbool(last.get("with_annotations"), True)
    last_bake_compare = _qbool(last.get("bake_compare"), True)
    last_width = _opt_pos_int(last.get("width"))
    last_height = _opt_pos_int(last.get("height"))

    # Whether the live A/B compare overlay is actually active right
    # now. Drives the visibility of the "Bake compare overlay" row in
    # the dialog: without an active wipe there's nothing to bake, so
    # the toggle would be confusing.
    compare_state = getattr(app, "_compare_state", None)
    compare_active = bool(
        compare_state is not None and compare_state.is_active(),
    )

    # Resolve the source size the dialog should pre-fill on the
    # "Source" preset. Falls back to a sane 1920×1080 if the sequence
    # didn't expose dims (rare — but happens for malformed headers).
    source_w = int(seq.width or 1920)
    source_h = int(seq.height or 1080)

    dialog = SaveFrameDialog(
        suggested_filename=suggested_filename,
        suggested_dir=suggested_dir,
        source_width=source_w,
        source_height=source_h,
        last_format=last_format,
        last_with_annotations=last_with_annotations,
        last_bake_compare=last_bake_compare,
        last_width=last_width,
        last_height=last_height,
        compare_active=compare_active,
        parent=app._window,
    )
    if dialog.exec() != dialog.DialogCode.Accepted:
        return
    settings = dialog.settings()

    # Render + write. The renderer reads the source frame directly
    # (full image, not a viewer screenshot) and applies the OCIO /
    # resize / annotation / compare-bake tail — identical pipeline
    # the multi-frame Export uses, just for one frame.
    try:
        arr = _render_save_frame_array(app, master_frame, settings)
    except Exception:
        log.exception("[save-frame] render failed")
        app._window.set_status("Save Frame failed: render error (see log).")
        return
    if arr is None:
        app._window.set_status("Save Frame failed: nothing to render.")
        return

    image = _ndarray_to_qimage(arr)
    if not _write_image(image, settings):
        app._window.set_status(
            f"Save Frame failed: could not write {settings.path.name}",
        )
        return

    # Persist the user's choices for next time.
    try:
        app._prefs.save_frame_settings = {
            "output_dir": str(settings.path.parent),
            "format": settings.fmt,
            "with_annotations": settings.with_annotations,
            "bake_compare": settings.bake_compare,
            # ``None`` round-trips through QSettings as the literal
            # ``"None"`` string; the ``_opt_pos_int`` helper unwraps
            # both forms when we read it back.
            "width": "None" if settings.width is None else settings.width,
            "height": "None" if settings.height is None else settings.height,
        }
    except Exception:
        log.exception("[save-frame] failed to persist last-used settings")

    app._window.set_status(f"Saved frame to {settings.path}")


# ============================================================================
# Render — pick the right path (regular / compare / contact-sheet)
# ============================================================================


def _render_save_frame_array(
    app: ImgPlayerApp, master_frame: int, settings: SaveFrameSettings,
) -> np.ndarray | None:
    """Build the final uint8 RGB(A) ndarray to write.

    Routes through one of two paths:

    * **Contact-sheet active** → re-decode every visible layer at the
      current contact-sheet offset and compose them into a grid at
      the picked resolution, applying OCIO + clamping to uint8. This
      is the same tile-and-compose path the live viewer uses, just
      sized for the export rather than the viewport.
    * **Single-sequence (or compare overlay)** → reuse the Export
      pipeline's :class:`FrameRenderer` for one frame so the colour /
      resize / annotation / compare-bake math matches the multi-frame
      Export exactly. Avoids drifting two near-identical render tails.
    """
    out_w, out_h = _resolve_output_size(app, settings)

    # Contact-sheet branch: the viewport shows a grid composite, not a
    # single source frame. Compose at the picked size instead of
    # routing through FrameRenderer (which only knows about a single
    # sequence).
    cs_state = getattr(app, "_contact_sheet_state", None)
    if cs_state is not None and cs_state.is_active():
        return _render_contact_sheet_array(
            app, master_frame, settings, out_w, out_h,
        )

    return _render_single_frame_array(
        app, master_frame, settings, out_w, out_h,
    )


def _resolve_output_size(
    app: ImgPlayerApp, settings: SaveFrameSettings,
) -> tuple[int, int]:
    """User-picked dims, falling back to source dims for the
    Source preset (where width / height are both ``None``)."""
    if settings.width is not None and settings.height is not None:
        return int(settings.width), int(settings.height)
    seq = app._controller.sequence
    src_w = int(seq.width) if seq and seq.width else 1920
    src_h = int(seq.height) if seq and seq.height else 1080
    return src_w, src_h


# ----------------------------------------------------------------------------
# Single-frame path — reuses FrameRenderer
# ----------------------------------------------------------------------------


def _render_single_frame_array(
    app: ImgPlayerApp,
    master_frame: int,
    settings: SaveFrameSettings,
    out_w: int,
    out_h: int,
) -> np.ndarray | None:
    """Render one full-image frame via the export :class:`FrameRenderer`.

    Resolves the active sequence's source-frame number for the live
    ``master_frame``, builds a minimal :class:`ExportSettings` +
    :class:`RenderContext`, and asks the renderer for a uint8 RGBA
    buffer at ``(out_w, out_h)``. The renderer's OCIO + resize +
    annotation bake + compare-overlay bake are all driven by those
    contexts — same path the multi-frame Export uses.
    """
    seq = app._controller.sequence
    if seq is None:
        return None

    # Active layer = the visible source the user is currently looking
    # at. Picking the topmost-visible-at-master layer means the saved
    # frame matches what's on screen even in stacked layouts (e.g. a
    # cleanup pass on top of a plate).
    layer = app._layer_stack.topmost_visible_at(master_frame)

    # Compare overlay snapshot — only relevant when compare is live
    # AND the user kept the "Bake compare overlay" checkbox on. The
    # renderer's compare path takes a master frame, decodes both
    # layers, and blends them per the captured snapshot.
    compare_ctx: CompareRenderContext | None = None
    compare_state = getattr(app, "_compare_state", None)
    compare_live = bool(
        compare_state is not None and compare_state.is_active(),
    )
    if (
        settings.bake_compare
        and compare_live
        and app._layer_stack is not None
    ):
        layer_a = app._layer_stack.find(compare_state.layer_a_id)
        layer_b = app._layer_stack.find(compare_state.layer_b_id)
        if layer_a is not None and layer_b is not None:
            compare_ctx = CompareRenderContext(
                layer_a=layer_a,
                layer_b=layer_b,
                mode=compare_state.mode,
                seam=compare_state.seam,
                swap_showing_b=compare_state.swap_showing_b,
            )

    # The renderer iterates by source-frame number for the single-
    # sequence path; for the compare path the same parameter is used
    # as the master frame internally. Resolve master → source through
    # the active layer when we have one; fall back to master_frame
    # itself if the stack is empty (renderer will substitute /
    # placeholder for missing frames per the policy below).
    if compare_ctx is not None:
        frame_arg = master_frame  # renderer interprets as master
    elif layer is not None and layer.covers(master_frame):
        frame_arg = layer.source_frame_at(master_frame)
    else:
        # No live layer at the playhead — last-resort: try the
        # sequence's clamped first/last frame so we still produce
        # *something* useful. Renderer's PLACEHOLDER policy will fill
        # the gap if even that misses.
        frame_arg = max(seq.first_frame, min(master_frame, seq.last_frame))

    # Build a minimal ExportSettings. We always force format_key="png"
    # so the renderer's _to_writer_dtype returns uint8 — Qt's
    # QImageWriter pipeline expects 8-bit input regardless of the
    # user's chosen file format (PNG / JPG / TIFF / BMP / WebP).
    export_settings = ExportSettings(
        output_dir=settings.path.parent,
        in_frame=frame_arg,
        out_frame=frame_arg,
        format_key="png",  # forces uint8 output dtype
        width=int(out_w),
        height=int(out_h),
        apply_display_transform=True,
        bake_annotations=settings.with_annotations,
        bake_compare=settings.bake_compare,
        # Use BLACK on missing so a hole in the sequence at the user's
        # current playhead saves a black frame instead of aborting.
        # ABORT would raise FileNotFoundError up to our caller; the
        # user just wants a save attempt, not a stack trace.
        missing_frame_policy=MissingFramePolicy.BLACK,
    )

    # OCIO processor — built once here rather than via ExportEngine's
    # constructor so we avoid spinning up the writer / engine just to
    # render one frame. Mirrors ExportEngine._build_cpu_processor.
    ocio_proc = _build_save_frame_ocio_processor(app)

    ctx = RenderContext(
        sequence=seq,
        annotation_store=app._annotation_store if settings.with_annotations else None,
        ocio_cpu_processor=ocio_proc,
        channel_selection=app._channel_selection,
        compare=compare_ctx,
    )
    renderer = FrameRenderer(ctx, export_settings)
    return renderer.render(frame_arg, (int(out_w), int(out_h)))


def _build_save_frame_ocio_processor(app: ImgPlayerApp):  # noqa: ANN202
    """CPU OCIO processor for the renderer, or ``None`` on failure.

    Mirrors :meth:`ExportEngine._build_cpu_processor` but kept local
    so the save-frame path doesn't carry an ExportEngine instance
    just for the colour transform. Returns ``None`` if any step
    fails — the renderer treats that as "skip the colour step".
    """
    manager = app._ocio
    if manager is None:
        return None
    try:
        src = (
            app._prefs.source_colorspace
            or manager.role("scene_linear")
            or "Linear Rec.709 (sRGB)"
        )
        disp = app._prefs.display or manager.default_display()
        view = app._prefs.view or manager.default_view(disp)
        proc = manager.get_display_view_processor(src, disp, view)
        return proc.getDefaultCPUProcessor()
    except Exception:
        log.exception("[save-frame] failed to build OCIO processor; baking raw")
        return None


# ----------------------------------------------------------------------------
# Contact-sheet path — re-compose the grid at the picked size
# ----------------------------------------------------------------------------


def _render_contact_sheet_array(
    app: ImgPlayerApp,
    master_frame: int,
    settings: SaveFrameSettings,
    out_w: int,
    out_h: int,
) -> np.ndarray | None:
    """Compose a contact-sheet grid at ``(out_w, out_h)``.

    Replicates :meth:`ImgPlayerApp._render_contact_sheet`'s per-tile
    decode loop but routes the result through the export-style OCIO
    transform + clamp-to-uint8 so the saved file looks like what the
    user sees in the viewport (the viewport itself applies OCIO on
    the GPU; we have to bake it on CPU for a file write).

    ``settings.with_annotations`` is honoured at the composite level:
    annotations are sequence-space strokes, so we bake them after the
    grid is composed and OCIO-transformed.
    """
    from img_player.contact_sheet import render_contact_sheet  # noqa: PLC0415

    cs_state = app._contact_sheet_state
    layers = [
        layer for layer in app._layer_stack.layers()
        if layer.visible
    ]
    if not layers:
        log.info("[save-frame] contact sheet: no visible layers")
        return None

    # Same master → 0-based offset conversion the live render uses.
    # Subtracting the navigable-range start keeps every layer at its
    # source-aligned tile when the master playhead is mid-shot.
    anchor = app._controller._effective_in_frame()  # noqa: SLF001
    global_offset = max(0, master_frame - anchor)

    per_offsets = cs_state.per_layer_offsets
    tiles: list[np.ndarray | None] = []
    names: list[str] = []
    effective_source_frames: list[int] = []
    for layer in layers:
        layer_offset = global_offset + per_offsets.get(layer.id, 0)
        arr = app._contact_sheet_decoder.decode_one(layer, layer_offset)
        tiles.append(arr)
        # Mirror the live decoder's clamp so the per-tile label
        # matches the pixels actually on screen.
        if layer.is_still or layer.trim_length <= 0:
            effective_source_frames.append(int(layer.layer_in))
        else:
            clamped = max(0, min(layer_offset, layer.trim_length - 1))
            effective_source_frames.append(int(layer.layer_in + clamped))
    if all(arr is None for arr in tiles):
        log.warning("[save-frame] contact sheet: every layer's decode returned None")
        return None

    # Resolve grid using the OUTPUT aspect (so the saved composite
    # matches the picked W×H precisely, not the viewport's aspect).
    first_arr = next(arr for arr in tiles if arr is not None)
    src_h, src_w = first_arr.shape[:2]
    image_aspect = src_w / src_h if src_h > 0 else 16 / 9
    canvas_aspect = out_w / out_h if out_h > 0 else image_aspect
    cols, rows = cs_state.effective_grid(
        len(layers), image_aspect, canvas_aspect=canvas_aspect,
    )

    # Reuse the app's label helper so the per-tile label format is
    # identical to the viewport (e.g. ``####`` placeholder
    # substitution → current source frame number).
    from img_player.app import _format_tile_label  # noqa: PLC0415
    label_strs = [
        _format_tile_label(layer.name, frame)
        for layer, frame in zip(layers, effective_source_frames)
    ]

    composite = render_contact_sheet(
        tiles,
        names=label_strs,
        cols=cols,
        rows=rows,
        target_w=out_w,
        target_h=out_h,
        show_labels=cs_state.show_labels,
    )

    # OCIO + clamp to uint8. Tiles come from ``read_frame`` so they're
    # in the source colorspace (linear-ish for EXR, sRGB-ish for PNG).
    # The viewport applies the display transform on the GPU; we have
    # to do it on the CPU for a file write.
    arr = _ensure_rgba_uint8(composite, app)
    return arr


def _ensure_rgba_uint8(arr: np.ndarray, app: ImgPlayerApp) -> np.ndarray:
    """Apply OCIO display transform + clamp/scale to uint8 RGBA.

    Mirrors the tail of :meth:`FrameRenderer._to_writer_dtype` for
    uint8, with the OCIO step inserted so contact-sheet exports look
    like the GL viewport. Channels-only inputs get padded to RGBA
    (alpha=1) so QImage can read the buffer.
    """
    # Pad to RGBA in float32.
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] == 3:
        alpha = np.ones((*arr.shape[:2], 1), dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=2)
    arr = np.ascontiguousarray(arr.astype(np.float32, copy=False))

    proc = _build_save_frame_ocio_processor(app)
    if proc is not None:
        rgb = np.ascontiguousarray(arr[..., :3], dtype=np.float32)
        try:
            proc.applyRGB(rgb)
            arr[..., :3] = rgb
        except Exception:
            log.exception("[save-frame] OCIO apply failed; falling back to raw")

    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


# ============================================================================
# QImage + write
# ============================================================================


def _ndarray_to_qimage(arr: np.ndarray) -> QImage:
    """Wrap a contiguous uint8 RGB(A) ndarray in a QImage.

    The returned QImage references the ndarray's memory directly —
    so the caller MUST keep the array alive for the writer's
    lifetime. We solve that by calling ``copy()`` so the QImage owns
    its own buffer (the array is dropped after this function
    returns and we don't want a dangling pointer).
    """
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0 + 0.5).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8)
        return qimg.copy()
    h, w, c = arr.shape
    if c == 3:
        qimg = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
    elif c == 4:
        qimg = QImage(arr.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    else:
        # Defensive: 2-channel (e.g. luma+alpha) and >4-channel
        # ndarrays shouldn't reach here. Fall back to RGB by slicing
        # the first three channels so the write doesn't fail mid-
        # pipeline.
        rgb = np.ascontiguousarray(arr[..., :3])
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return qimg.copy()
    # ``.copy()`` so the QImage owns its memory (the source ndarray
    # is GC'd as soon as the caller drops its reference).
    return qimg.copy()


def _write_image(image: QImage, settings: SaveFrameSettings) -> bool:
    """Persist ``image`` to ``settings.path`` using Qt's image writer.

    Picks the format hint from the chosen extension rather than
    relying on Qt's filename sniff — this way an unusual stem (e.g.
    a filename containing dots) still saves in the user-picked
    encoding. Returns ``True`` on success.
    """
    fmt_hint = _QT_FORMAT_FOR_EXT.get(settings.fmt.lower())
    settings.path.parent.mkdir(parents=True, exist_ok=True)
    if fmt_hint is None:
        # Defensive: shouldn't happen because the dialog only offers
        # known extensions, but a future format addition could miss
        # the mapping update — fall back to extension sniff so the
        # user gets *something* useful.
        ok = image.save(str(settings.path))
    else:
        # ``QImageWriter`` is the explicit, format-hint-friendly
        # write path. The ``QImage.save`` overload that accepts a
        # format string is finicky in PySide6 (silently rejects
        # uppercase / unknown encodings), so we route through the
        # writer for predictable behaviour across platforms.
        writer = QImageWriter(str(settings.path), fmt_hint.encode("ascii"))
        ok = writer.write(image)
        if not ok:
            log.warning(
                "[save-frame] QImageWriter failed for %s: %s",
                settings.path, writer.errorString(),
            )
    if not ok:
        log.warning("[save-frame] QImage.save returned False for %s", settings.path)
    return bool(ok)


# ============================================================================
# Misc helpers
# ============================================================================


def _opt_pos_int(value: object) -> int | None:
    """Parse a QSettings value as either a positive int or ``None``.

    QSettings round-trips ``None`` as the literal string ``"None"``
    on POSIX .conf files (and on Windows when stored via the
    ``save_frame_settings`` setter above which preserves the
    sentinel). Both forms decode to ``None`` here; non-positive or
    unparseable values also become ``None`` so the dialog falls back
    to the Source preset rather than getting a zero-sized export.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in ("none", ""):
        return None
    try:
        iv = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return iv if iv > 0 else None
