"""File → Save Frame As… — quick WYSIWYG snapshot of the viewer.

Single entry point :func:`open_save_frame_dialog`. Builds the dialog,
captures the viewer area with the user's chosen toggles applied, and
writes the file to disk via Qt's standard image writer (PNG / JPEG /
TIFF / BMP / WebP).

This is intentionally simpler than the full Export pipeline:
no OCIO re-render, no resolution tweaking, no annotation baking by
the FrameRenderer — just a screenshot of what the user is looking
at, minus the overlay widgets they want hidden.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtGui import QImage, QImageWriter

from img_player.preferences import _qbool
from img_player.ui.save_frame_dialog import SaveFrameDialog, SaveFrameSettings

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget

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
    """File → Save Frame As… — snapshot the viewer to a single image.

    Refuses to run when no sequence is loaded (nothing to capture).
    Otherwise builds the dialog with sensible defaults sourced from
    the last-used settings + the current sequence's directory, runs
    the capture on accept, and surfaces the result via the status
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

    # Whether the live A/B compare overlay is actually active right
    # now. Drives the visibility of the "Bake compare overlay" row in
    # the dialog: without an active wipe there's nothing to bake, so
    # the toggle would be confusing.
    compare_state = getattr(app, "_compare_state", None)
    compare_active = bool(
        compare_state is not None and compare_state.is_active(),
    )

    dialog = SaveFrameDialog(
        suggested_filename=suggested_filename,
        suggested_dir=suggested_dir,
        last_format=last_format,
        last_with_annotations=last_with_annotations,
        last_bake_compare=last_bake_compare,
        compare_active=compare_active,
        parent=app._window,
    )
    if dialog.exec() != dialog.DialogCode.Accepted:
        return
    settings = dialog.settings()

    # Capture + write. When the user wants the snapshot WITHOUT the
    # compare overlay even though it's currently on screen, we
    # temporarily flip the compare state off, force a re-render so
    # the GL widget paints the underlying composite, capture, then
    # restore — see :func:`_capture_with_compare_off`.
    try:
        if compare_active and not settings.bake_compare:
            image = _capture_with_compare_off(
                app, master_frame, settings.with_annotations,
            )
        else:
            image = capture_viewer(
                app._window.viewer,
                annotation_overlay=app._annotation_overlay,
                with_annotations=settings.with_annotations,
            )
    except Exception:
        log.exception("[save-frame] capture failed")
        app._window.set_status("Save Frame failed: capture error (see log).")
        return

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
        }
    except Exception:
        log.exception("[save-frame] failed to persist last-used settings")

    app._window.set_status(f"Saved frame to {settings.path}")


def _capture_with_compare_off(
    app: ImgPlayerApp, master_frame: int, with_annotations: bool,
) -> QImage:
    """Snapshot the viewer with the live A/B compare overlay
    temporarily disabled, then restore it.

    Useful when the reviewer wants a clean plate at the current
    frame even though they've got a wipe on screen. We flip
    ``compare_state.enabled`` off, force the frame-changed handler
    to re-render via the regular cache path so the GL widget shows
    the underlying composite, capture, then restore the compare
    state and re-render once more so the user sees their wipe back.

    The QApplication.processEvents() pumps in between are necessary
    so Qt actually paints the new GL contents before ``viewer.grab``
    samples the framebuffer — without them, the grab would catch the
    *old* frame (the compare composite that was last uploaded).
    """
    from PySide6.QtWidgets import QApplication

    compare_state = app._compare_state
    was_enabled = compare_state.enabled
    compare_state.enabled = False
    try:
        # Force a redraw via the regular path. Internally this
        # routes through the master cache, which decodes / hits and
        # uploads the result to the GL widget. Process pending paint
        # events so the GL framebuffer reflects the new frame
        # before we grab it.
        app._on_frame_changed(master_frame)
        QApplication.processEvents()
        return capture_viewer(
            app._window.viewer,
            annotation_overlay=app._annotation_overlay,
            with_annotations=with_annotations,
        )
    finally:
        # Restore the wipe so the live viewer matches what the
        # user expected to keep seeing after the dialog closed.
        compare_state.enabled = was_enabled
        app._on_frame_changed(master_frame)
        QApplication.processEvents()


def capture_viewer(
    viewer: QWidget,
    *,
    annotation_overlay: QWidget | None,
    with_annotations: bool,
) -> QImage:
    """Grab the viewer's current pixels as a QImage.

    Always excludes the on-screen UI overlay (bottom HUD info band,
    decorative brackets) — that chrome has no business in a saved
    snapshot. The annotation overlay is gated by ``with_annotations``
    because annotations are reviewer-authored content the user may
    legitimately want either in or out.

    Visibility is toggled around the grab and restored on the way
    out (even on exception), so a failing capture can't leave the
    live UI in a half-hidden state. Widgets that were already
    hidden by the user stay hidden afterwards.

    Channel labels are baked into the GL composite (contact-sheet
    pass) and aren't a separate widget — they ride along with the
    image regardless.
    """
    # Widgets we'll temporarily hide during the grab. Each entry pairs
    # the widget with the previous visibility we'll restore.
    to_toggle: list[tuple[QWidget, bool]] = []

    # UI chrome — always hidden during capture.
    info_band = getattr(viewer, "_info_band", None)
    brackets = getattr(viewer, "_overlay", None)
    if info_band is not None and info_band.isVisible():
        to_toggle.append((info_band, True))
        info_band.setVisible(False)
    if brackets is not None and brackets.isVisible():
        to_toggle.append((brackets, True))
        brackets.setVisible(False)
    # Annotations — user-controlled.
    if not with_annotations and annotation_overlay is not None:
        if annotation_overlay.isVisible():
            to_toggle.append((annotation_overlay, True))
            annotation_overlay.setVisible(False)

    try:
        # ``QWidget.grab()`` triggers a synchronous paint of the
        # widget tree into a QPixmap. Children render at their
        # current visibility state, which is exactly what we want.
        pixmap = viewer.grab()
        return pixmap.toImage()
    finally:
        # Restore visibility no matter what — even if grab() raised.
        for widget, was_visible in to_toggle:
            widget.setVisible(was_visible)


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


