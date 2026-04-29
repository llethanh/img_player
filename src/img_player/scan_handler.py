"""Sequence-scan plumbing: off-thread scanner + result handler.

Extracted from :mod:`img_player.app` so the bootstrap class doesn't
have to carry the scan loop, the ``_ScanRunner`` QObject, and the
result dispatcher all inline. The module exposes:

* :class:`ScanRunner` — QObject that runs ``scan`` / ``scan_all`` on
  a worker thread and emits the result on its ``done`` signal.
* :func:`open_path` — kicks off the scan and tracks generations so a
  newer drag-and-drop can supersede an in-flight one.
* :func:`apply_scan_result` — dispatches the scanner's emission:
  shows the picker for multi-sequence folders, surfaces errors,
  hands the chosen :class:`SequenceInfo` to the controller and
  reloads sidecar annotations + comments.

The module stays small and Qt-light: it doesn't own any state beyond
``ScanRunner``'s built-in thread reference. The owning
:class:`ImgPlayerApp` keeps ``_scan_generation`` / ``_scan_runner``
on itself so multiple drops in flight stay correctly serialised.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QMessageBox

from img_player.comment import load_comments
from img_player.annotate import load_annotations
from img_player.annotate.persistence import sidecar_path
from img_player.sequence.models import SequenceInfo
from img_player.sequence.scanner import SequenceNotFoundError, scan, scan_all

if TYPE_CHECKING:
    from img_player.app import ImgPlayerApp

log = logging.getLogger(__name__)


class ScanRunner(QObject):  # type: ignore[misc]
    """Runs the appropriate scanner in a worker thread and emits the result.

    The ``done`` signal carries:

    * :class:`SequenceInfo` — single-file path, sequence resolved.
    * ``list[SequenceInfo]`` — directory path. Possibly one item
      (single sequence in folder), possibly many (multi-sequence
      folder; the app then prompts the user).
    * :class:`Exception` — failed scan.

    Qt delivers the signal on the main thread automatically since
    emit happens from the worker thread.
    """

    done = Signal(object)

    def run_async(self, path: Path) -> None:
        def worker() -> None:
            try:
                if path.is_dir():
                    # Directory drop / open: enumerate every sequence
                    # so the controller can prompt when there are
                    # multiple. ``probe=False`` for the same lazy-FS
                    # latency reasons as the file path below.
                    sequences = scan_all(path, probe=False)
                    if not sequences:
                        raise SequenceNotFoundError(
                            f"No sequence found in {path}"
                        )
                    self.done.emit(sequences)
                else:
                    seq = scan(path, probe=False)
                    self.done.emit(seq)
            except Exception as err:
                self.done.emit(err)

        threading.Thread(target=worker, name="scan-worker", daemon=True).start()


def open_path(app: ImgPlayerApp, path: Path) -> None:
    """Scan ``path`` off the main thread so the UI stays responsive."""
    app._window.set_status(f"Scanning {path}…")

    app._scan_generation += 1
    gen = app._scan_generation

    runner = ScanRunner()
    app._scan_runner = runner  # keep a reference so the QObject stays alive

    def on_done(result: object) -> None:
        if gen != app._scan_generation:
            # Superseded by a newer drop — ignore this result.
            return
        apply_scan_result(app, path, result)

    runner.done.connect(on_done)
    # probe=False: don't open any image file just to read metadata.
    # On lazy filesystems (Google Drive Stream) a single header read
    # can trigger a full file download (tens of seconds).
    runner.run_async(path)


def apply_scan_result(app: ImgPlayerApp, path: Path, result: object) -> None:
    """Process the scanner's emission: error, single seq, or list."""
    if isinstance(result, Exception):
        if isinstance(result, SequenceNotFoundError):
            QMessageBox.warning(app._window, "Cannot open", str(result))
        else:
            log.exception("scan failed for %s: %s", path, result)
            QMessageBox.critical(app._window, "Scan failed", str(result))
        app._window.set_status("Ready.")
        return
    # Directory scan returns a list. One sequence → load it
    # directly. Two or more → prompt the user via the picker
    # dialog so they don't get the "largest first" silent fallback.
    if isinstance(result, list):
        sequences: list[SequenceInfo] = result
        if len(sequences) == 1:
            seq = sequences[0]
        else:
            from img_player.ui.sequence_picker import SequencePickerDialog
            picked = SequencePickerDialog.pick(sequences, parent=app._window)
            if picked is None:
                app._window.set_status("Open canceled.")
                return
            seq = picked
    else:
        seq = result  # type: ignore[assignment]
    log.info("loaded sequence: %s (%d frames)", seq.display_pattern(), seq.frame_count)

    # The scanner runs with probe=False to keep the open() snappy on
    # Drive Stream / network paths — that means the SequenceInfo
    # arrives without channel names, width or height. Read the
    # header of the first frame now (one cheap OIIO call, no pixel
    # decode) so the channel selector can be populated and the
    # auto-detector has the resolution it needs.
    seq = app._enrich_with_header(seq)

    app._window.update_sequence_info(seq)
    # Restore the saved channel-menu state (active radio + tile
    # checkboxes + layout). Best-effort: labels not present in
    # this sequence's group list are silently skipped, so the
    # user never sees a stale state cause a crash. Same call
    # also re-emits ``channel_selection_changed`` which drives
    # the cache via :meth:`set_channel_selection`.
    app._window.transport.restore_channel_state(
        app._prefs.channel_active_label,
        app._prefs.channel_tile_labels,
        app._prefs.channel_layout_mode,
        app._prefs.channel_labels_visible,
    )
    app._guess_source_colorspace(seq)
    # ``controller.load_sequence(seq)`` calls ``cache.attach(seq)``,
    # which on the MasterFrameCache path replaces the LayerStack's
    # contents with a single Layer at offset = first_frame. The
    # FrameCache path is a no-op for the stack (the cache holds its
    # own _paths_by_frame map). Either way, no manual stack mutation
    # is required here.
    app._controller.load_sequence(seq)
    app._window.set_status(
        f"Loaded {seq.display_pattern()} ({seq.frame_count} frames) — decoding first frame…"
    )
    # Remember this path for next launch and for the Recent menu.
    app._prefs.last_path = path
    app._prefs.push_recent(path)

    # Drop any live ephemeral strokes from the previous sequence
    # (v0.4.1) — they're image-space anchored, not frame-bound,
    # so they would otherwise float on the new sequence's first
    # frame until they fade. Cheap and idempotent.
    app._ephemeral_manager.clear_all()

    # Load any persisted annotations for this sequence. The
    # sidecar lives next to the frame files; basename routes to
    # the right sub-payload when several sequences share a dir.
    # Strip trailing separators ('.', '_') from the base_name so
    # 'render.' and 'render' both map to the same JSON key — a
    # cosmetic detail, but it would be confusing if a previously-
    # saved sequence's notes silently disappeared after a tool
    # change in the scanner.
    app._annotations_path = sidecar_path(seq.directory)
    app._annotations_basename = seq.base_name.rstrip("._-") or seq.base_name
    loaded = load_annotations(
        app._annotations_path,
        basename=app._annotations_basename,
    )
    if loaded is not None:
        # Replace the in-memory store contents (reuse the live
        # store object so its signal subscribers stay wired).
        app._annotation_store.load_from_dict(loaded.to_dict()["frames"])
        log.info(
            "[annotations] loaded %d annotated frames from %s",
            len(app._annotation_store.annotated_frames()),
            app._annotations_path,
        )
    else:
        # Fresh start — clear any leftover state from a previous
        # sequence in this session.
        app._annotation_store.load_from_dict({})

    # Comments share the same sidecar — reload them too. Same
    # error-tolerant pattern: a missing or unreadable file
    # yields None, and we just clear the in-memory store.
    loaded_comments = load_comments(
        app._annotations_path,
        basename=app._annotations_basename,
    )
    if loaded_comments is not None:
        app._comment_store.load_from_dict(loaded_comments.to_dict())
        log.info(
            "[comment] loaded %d commented frames from %s",
            len(app._comment_store.commented_frames()),
            app._annotations_path,
        )
    else:
        app._comment_store.load_from_dict({})
