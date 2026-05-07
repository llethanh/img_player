"""Export-dialog wiring extracted from :mod:`img_player.app`.

Single entry point — :func:`open_export_dialog` — that runs the full
flow: build the dialog, snapshot the user's last-used settings,
construct the engine + worker + progress dialog, and wire the
cancel + status-bar callbacks.

Lives outside ``app.py`` so the (heavy) export imports (PyAV, OIIO
writers, dialog widgets) only get loaded the first time the user
hits Export — keeping the cold-start path slim. The heavy submodule
imports stay deferred inside this function for the same reason.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from img_player.app import ImgPlayerApp

log = logging.getLogger(__name__)


def open_export_dialog(app: ImgPlayerApp) -> None:
    """File → Export… (or 💾 transport button) — open the dialog,
    kick off the worker on accept."""
    from img_player.export import ExportSettings
    from img_player.export.dialog import ExportDialog
    from img_player.export.engine import ExportEngine
    from img_player.export.progress_dialog import ExportProgressDialog
    from img_player.export.renderer import CompareRenderContext
    from img_player.export.worker import ExportWorker

    seq = app._controller.sequence
    if seq is None:
        app._window.set_status("Export: no sequence loaded.")
        return

    # Range defaults: respect the player's current in/out points
    # if the user has set them, otherwise the full sequence.
    state = app._controller.state
    in_f = state.in_frame if state.in_frame is not None else seq.first_frame
    out_f = state.out_frame if state.out_frame is not None else seq.last_frame

    # Snapshot the live compare overlay (if any) so the dialog can
    # decide whether to surface the bake-compare option, and the
    # engine can reproduce the exact A/B blend the user has on screen.
    compare_state = getattr(app, "_compare_state", None)
    compare_active = bool(
        compare_state is not None and compare_state.is_active(),
    )

    # Restore last-used settings (per-key fall back to the
    # ExportSettings dataclass defaults).
    try:
        initial = ExportSettings.from_prefs_dict(
            app._prefs.export_settings, in_frame=in_f, out_frame=out_f,
        )
    except Exception:
        initial = None

    dialog = ExportDialog(
        in_frame=in_f,
        out_frame=out_f,
        source_in_frame=seq.first_frame,
        source_out_frame=seq.last_frame,
        source_width=seq.width or 1920,
        source_height=seq.height or 1080,
        source_fps=state.fps,
        compare_active=compare_active,
        initial_settings=initial,
        parent=app._window,
    )
    if dialog.exec() != dialog.DialogCode.Accepted:
        return
    settings = dialog.get_settings()
    # Persist for next time.
    try:
        app._prefs.export_settings = settings.to_prefs_dict()
    except Exception:
        log.exception("[export] failed to persist last-used settings")

    # Compare overlay snapshot for the engine. Resolved here so the
    # renderer doesn't reach back into the app singleton — it just
    # holds direct refs to the picked layers + the captured blend
    # state, which can't drift mid-export. Set to ``None`` if the
    # user didn't tick bake_compare or compare went stale between
    # dialog open and accept.
    compare_ctx = None
    if settings.bake_compare and compare_active and app._layer_stack is not None:
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
        else:
            log.warning(
                "[export] bake_compare requested but layer A or B "
                "not found in the stack; falling back to single sequence",
            )

    # Build the engine + worker + progress dialog and wire them.
    engine = ExportEngine(
        settings=settings,
        sequence=seq,
        annotation_store=app._annotation_store,
        ocio_manager=app._ocio,
        source_colorspace=app._prefs.source_colorspace,
        display=app._prefs.display,
        view=app._prefs.view,
        sidecar_source=app._annotations_path,
        # Snapshot the live channel state so the export reproduces
        # whatever AOV / channel group is currently on screen. The
        # renderer reads this once at construction — subsequent live
        # changes during the export don't affect the run.
        channel_selection=app._channel_selection,
        compare=compare_ctx,
    )
    worker = ExportWorker(engine, parent=app._window)
    progress = ExportProgressDialog(
        total_frames=settings.total_frames,
        output_path=settings.output_dir,
        parent=app._window,
    )

    worker.progress.connect(progress.update_progress)
    worker.finished_ok.connect(progress.on_finished)
    worker.failed.connect(progress.on_failed)
    worker.canceled.connect(progress.on_canceled)
    # Cancel routing: the dialog's Cancel button asks the worker
    # to stop. The progress dialog itself stays open until the
    # worker reaches its end-state and emits canceled.
    progress.cancel_button.clicked.connect(worker.cancel)
    # Cancel cleanup: the engine no longer auto-deletes partial
    # output (so we can ask the user). For image sequences we prompt
    # keep / discard; for video the partial container is always
    # corrupt so we discard silently.
    worker.canceled.connect(
        lambda _p, n: _handle_cancel_cleanup(app, engine, settings, n),
    )
    # Final status message on success / failure / cancel.
    worker.finished_ok.connect(app._on_export_finished)
    worker.failed.connect(lambda msg: app._window.set_status(f"Export failed: {msg}"))
    worker.canceled.connect(
        lambda _p, n: app._window.set_status(f"Export canceled after {n} frames")
    )
    # Auto-cleanup once the dialog finishes.
    progress.finished.connect(worker.deleteLater)
    progress.finished.connect(progress.deleteLater)

    worker.start()
    app._window.set_status(
        f"Exporting {settings.total_frames} frames → {settings.output_dir}"
    )
    progress.exec()


def _handle_cancel_cleanup(
    app: ImgPlayerApp,
    engine,  # type: ignore[no-untyped-def]
    settings,  # type: ignore[no-untyped-def]
    frames_written: int,
) -> None:
    """After a cancel, decide whether to keep or delete the partial output.

    Image-sequence exports: prompt the user (a half-finished folder
    of frames is sometimes useful — early-look on a long render).
    Video exports: silently discard, mid-encode containers are
    typically unreadable so there's nothing to keep.

    The engine already closed the writer cleanly when it noticed the
    cancel flag; this function only decides whether to invoke
    :meth:`engine.discard_partial_output` to delete the files.
    """
    from PySide6.QtWidgets import QMessageBox

    if frames_written <= 0:
        # Nothing on disk worth prompting about — silently discard
        # any zero-byte stub the writer may have opened.
        engine.discard_partial_output()
        return

    if not settings.is_image_sequence:
        # Video container — partial encode is corrupt anyway.
        engine.discard_partial_output()
        app._window.set_status(
            f"Export canceled after {frames_written} frames (partial video discarded)",
        )
        return

    # Image sequence — let the user decide.
    box = QMessageBox(app._window)
    box.setWindowTitle("Export canceled")
    box.setIcon(QMessageBox.Icon.Question)
    box.setText(
        f"Export canceled after {frames_written} frames.\n\n"
        f"Garder les {frames_written} frame(s) déjà écrite(s) "
        f"dans {settings.output_dir} ?",
    )
    keep_btn = box.addButton("Garder", QMessageBox.ButtonRole.AcceptRole)
    discard_btn = box.addButton(
        "Supprimer", QMessageBox.ButtonRole.DestructiveRole,
    )
    box.setDefaultButton(keep_btn)
    box.exec()
    if box.clickedButton() is discard_btn:
        engine.discard_partial_output()
        app._window.set_status(
            f"Export canceled — {frames_written} partial frames deleted",
        )
    else:
        app._window.set_status(
            f"Export canceled — {frames_written} partial frames kept",
        )
