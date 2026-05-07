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

from img_player.annotate import load_annotations
from img_player.annotate.persistence import sidecar_path
from img_player.comment import load_comments
from img_player.sequence.models import SequenceInfo
from img_player.sequence.scanner import (
    FolderGroup,
    SequenceNotFoundError,
    scan,
    scan_all,
    scan_paths,
)

if TYPE_CHECKING:
    from img_player.app import ImgPlayerApp
    from img_player.layers.stack import LayerStack

log = logging.getLogger(__name__)


# Default hold (in master frames) for a still loaded into an EMPTY
# session. ~4 seconds at 24 fps — long enough to be scrubbable but
# short enough that a single ref doesn't hijack the timebar. The user
# can resize via the timeline drag handle or the layer panel spinbox.
_EMPTY_STACK_STILL_HOLD = 100


def _default_still_hold(stack: LayerStack) -> int:
    """Pick a sensible hold duration for a still being added to ``stack``.

    Logic:

    * Stack has at least one non-still layer (sequence or video) →
      match the longest one's ``trim_length`` so a slate dropped into
      a 200-frame shot covers the whole shot by default.
    * Stack is empty or contains only other stills → fall back to
      :data:`_EMPTY_STACK_STILL_HOLD` (~4 s @ 24 fps). The user can
      adjust afterwards.
    """
    candidates = [
        layer.trim_length
        for layer in stack.layers()
        if not layer.is_still
    ]
    if not candidates:
        return _EMPTY_STACK_STILL_HOLD
    return max(candidates)


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

    def run_paths_async(self, paths: list[Path]) -> None:
        """Scan a heterogeneous batch (folders + loose files) on a worker.

        Emits ``list[FolderGroup]`` on success, or an :class:`Exception`
        when the scan blew up unexpectedly. Empty results (no
        detectable sequence anywhere in the drop) still emit a list
        so the caller can surface a friendly status message rather
        than an error popup.
        """
        def worker() -> None:
            try:
                groups = scan_paths(paths, probe=False)
                self.done.emit(groups)
            except Exception as err:
                self.done.emit(err)

        threading.Thread(
            target=worker, name="scan-paths-worker", daemon=True,
        ).start()

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


def add_layer(app: ImgPlayerApp, path: Path) -> None:
    """Scan ``path`` and add the result as a new top-of-stack layer.

    Bypasses the controller's load_sequence — the existing
    sequence's master range stays bound to the controller's
    ``_sequence``. The added layer is positioned at
    ``offset = sequence.first_frame`` so its source frames align
    with master frame numbers (= same convention as the original
    sequence's wrapping). The user can later drag the layer to
    shift it once the offset-drag UI lands.

    Single-file paths and multi-sequence folders both supported:
    multi-seq dirs prompt the picker, single-seq / file paths load
    directly. Errors surface via the status bar.
    """
    app._window.set_status(f"Scanning {path} for add-layer…")
    runner = ScanRunner()
    # Same lifecycle dance as ``open_path``: keep a strong reference
    # so the QObject survives until the worker emits.
    app._scan_runner = runner

    def on_done(result: object) -> None:
        from img_player.layers import Layer
        if isinstance(result, Exception):
            log.warning("[add-layer] scan failed for %s: %s", path, result)
            app._window.set_status(f"Add layer failed: {result}")
            return
        extras: list[SequenceInfo] = []
        if isinstance(result, list):
            sequences: list[SequenceInfo] = result
            if not sequences:
                app._window.set_status("Add layer: no sequence found.")
                return
            if len(sequences) == 1:
                seq = sequences[0]
            else:
                from img_player.ui.sequence_picker import SequencePickerDialog
                group = FolderGroup(folder=path, sequences=tuple(sequences))
                picked = SequencePickerDialog.pick_grouped(
                    [group], parent=app._window,
                )
                if not picked:
                    app._window.set_status("Add layer canceled.")
                    return
                seq = picked[0]
                extras = list(picked[1:])
        else:
            seq = result  # type: ignore[assignment]
        # If the interface is empty (no sequence attached to the
        # controller yet), an "add layer" can't actually start
        # playing — controller._sequence stays None so play()
        # early-returns and the viewport never updates. Route to
        # the regular open path so load_sequence(), channel
        # restore, annotations sidecar and friends all run.
        if app._controller.sequence is None:
            log.info("[add-layer] empty interface → routing to open_path")
            # Forward the already-picked ``seq`` (single SequenceInfo)
            # — passing the raw multi-seq ``result`` list here would
            # have apply_scan_result run the picker a SECOND time,
            # which is the bug the user reported as "the dialog
            # reopens after I click Load Selected".
            apply_scan_result(app, path, seq)
            # Re-add any extra sequences the user also ticked in the
            # picker as additional top-of-stack layers (apply_scan_result
            # only handles the primary sequence on this branch).
            for extra in extras:
                extra2 = app._enrich_with_header(extra)
                extra_layer = Layer.from_image(
                    extra2,
                    default_still_hold=_default_still_hold(app._layer_stack),
                    offset=extra2.first_frame,
                )
                app._layer_stack.add(extra_layer)
            return
        # Header probe so the new layer carries width / height /
        # channel info — same enrichment the main load path uses.
        seq = app._enrich_with_header(seq)
        layer = Layer.from_image(
            seq,
            default_still_hold=_default_still_hold(app._layer_stack),
            offset=seq.first_frame,
        )
        app._layer_stack.add(layer)  # auto-positions at top (= focus shifts)
        for extra in extras:
            extra2 = app._enrich_with_header(extra)
            extra_layer = Layer.from_image(
                extra2,
                default_still_hold=_default_still_hold(app._layer_stack),
                offset=extra2.first_frame,
            )
            app._layer_stack.add(extra_layer)
        total = 1 + len(extras)
        if total == 1:
            app._window.set_status(
                f"Added layer: {seq.display_pattern()} ({seq.frame_count} frames)"
            )
        else:
            app._window.set_status(f"Added {total} layers from {path.name}.")

    runner.done.connect(on_done)
    runner.run_async(path)


def _resolve_groups_picker(
    app: ImgPlayerApp, groups: list[FolderGroup],
) -> list[SequenceInfo]:
    """Show the grouped picker and return the user's checked sequences.

    Returns ``[]`` on cancel or when no sequences were detected at
    all (in which case the picker isn't even shown — we just surface
    a status message).
    """
    total = sum(len(g.sequences) for g in groups)
    if total == 0:
        app._window.set_status("Drop: no sequence found.")
        return []
    from img_player.ui.sequence_picker import SequencePickerDialog
    picked = SequencePickerDialog.pick_grouped(groups, parent=app._window)
    return list(picked)


def open_paths(app: ImgPlayerApp, paths: list[Path]) -> None:
    """Multi-source replace flow: scan ``paths``, prompt picker, load.

    The first picked sequence becomes the active sequence (replaces
    the controller's binding via the standard ``apply_scan_result``
    pipeline — channels, annotations sidecar, recent path, etc.).
    Each subsequent picked sequence is appended as a top layer in
    pick order so their on-screen layer-panel order mirrors what
    the user saw in the picker.
    """
    app._window.set_status(
        f"Scanning {len(paths)} sources…"
    )
    app._scan_generation += 1
    gen = app._scan_generation
    runner = ScanRunner()
    app._scan_runner = runner

    def on_done(result: object) -> None:
        if gen != app._scan_generation:
            return
        if isinstance(result, Exception):
            log.exception("[multi-open] scan failed: %s", result)
            QMessageBox.critical(
                app._window, "Scan failed", str(result),
            )
            app._window.set_status("Ready.")
            return
        groups: list[FolderGroup] = result  # type: ignore[assignment]
        picked = _resolve_groups_picker(app, groups)
        if not picked:
            app._window.set_status("Open canceled.")
            return
        # Use the first picked seq's directory as the "primary" path
        # for the recent / sidecar bookkeeping in apply_scan_result.
        primary_path = picked[0].directory
        apply_scan_result(app, primary_path, picked[0])
        from img_player.layers import Layer
        for seq in picked[1:]:
            seq2 = app._enrich_with_header(seq)
            layer = Layer.from_image(
                seq2,
                default_still_hold=_default_still_hold(app._layer_stack),
                offset=seq2.first_frame,
            )
            app._layer_stack.add(layer)
        if len(picked) > 1:
            app._window.set_status(
                f"Loaded {len(picked)} sequences from drop."
            )

    runner.done.connect(on_done)
    runner.run_paths_async(paths)


def add_layers(app: ImgPlayerApp, paths: list[Path]) -> None:
    """Multi-source add-layer flow: scan ``paths``, prompt picker, append.

    Every checked sequence becomes a new top-of-stack layer (in pick
    order). If the player has no active sequence yet, the first
    picked sequence routes through the standard open path so the
    controller binds properly; the rest are appended as layers.
    """
    app._window.set_status(
        f"Scanning {len(paths)} sources for add-layer…"
    )
    runner = ScanRunner()
    app._scan_runner = runner

    def on_done(result: object) -> None:
        if isinstance(result, Exception):
            log.warning("[multi-add-layer] scan failed: %s", result)
            app._window.set_status(f"Add layer failed: {result}")
            return
        groups: list[FolderGroup] = result  # type: ignore[assignment]
        picked = _resolve_groups_picker(app, groups)
        if not picked:
            app._window.set_status("Add layer canceled.")
            return
        from img_player.layers import Layer
        start_index = 0
        if app._controller.sequence is None:
            # Empty player — bootstrap the controller with the first
            # pick the same way ``open_path`` does, so playback is
            # actually functional. The remaining picks become layers.
            apply_scan_result(app, picked[0].directory, picked[0])
            start_index = 1
        for seq in picked[start_index:]:
            seq2 = app._enrich_with_header(seq)
            layer = Layer.from_image(
                seq2,
                default_still_hold=_default_still_hold(app._layer_stack),
                offset=seq2.first_frame,
            )
            app._layer_stack.add(layer)
        app._window.set_status(
            f"Added {len(picked) - start_index} layer"
            f"{'s' if len(picked) - start_index != 1 else ''}."
        )

    runner.done.connect(on_done)
    runner.run_paths_async(paths)


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
    # directly. Two or more → prompt with the grouped picker so the
    # user can also pick MULTIPLE sequences from this single folder
    # (each extra one becomes a top-of-stack layer, same model as a
    # multi-folder drop). Avoids the "largest first" silent fallback
    # AND lines up the single-folder UX with the multi-source one.
    extras: list[SequenceInfo] = []
    if isinstance(result, list):
        sequences: list[SequenceInfo] = result
        if len(sequences) == 1:
            seq = sequences[0]
        else:
            from img_player.ui.sequence_picker import SequencePickerDialog
            group = FolderGroup(folder=path, sequences=tuple(sequences))
            picked = SequencePickerDialog.pick_grouped(
                [group], parent=app._window,
            )
            if not picked:
                app._window.set_status("Open canceled.")
                return
            seq = picked[0]
            # Extra ticks become layers, mirroring the multi-folder
            # path's behaviour.
            extras = list(picked[1:])
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
    app._guess_source_colorspace(seq)
    # ``controller.load_sequence(seq)`` calls ``cache.attach(seq)``,
    # which on the MasterFrameCache path replaces the LayerStack's
    # contents with a single Layer at offset = first_frame. The
    # FrameCache path is a no-op for the stack (the cache holds its
    # own _paths_by_frame map). Either way, no manual stack mutation
    # is required here.
    #
    # Order matters: ``load_sequence`` runs BEFORE
    # ``restore_channel_state`` so the layer exists (and is
    # focused) by the time the menu's ``selection_changed`` fires
    # — otherwise ``set_channel_selection`` early-returns when no
    # layer is focused and the layer's per-layer
    # ``channel_selection`` field stays ``None``, causing the
    # cache to decode the reader's RGB(A) default rather than
    # what the menu shows.
    app._controller.load_sequence(seq)
    # If the multi-pick gave us extra sequences (single-folder picker
    # with several boxes ticked), append each as a top-of-stack layer
    # in pick order. Same enrichment + offset model as the multi-
    # source ``open_paths`` path.
    if extras:
        from img_player.layers import Layer
        for extra in extras:
            extra2 = app._enrich_with_header(extra)
            layer = Layer.from_image(
                extra2,
                default_still_hold=_default_still_hold(app._layer_stack),
                offset=extra2.first_frame,
            )
            app._layer_stack.add(layer)
    # Each new sequence opens on its first channel group (RGB by
    # default for beauty plates) — no cross-sequence carry-over.
    # ``set_available_channels`` (called by the transport's focus
    # sync) already picks the first group of the freshly-loaded
    # sequence, so there's nothing to restore here.
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
