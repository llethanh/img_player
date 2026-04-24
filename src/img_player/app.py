"""Qt application bootstrap: builds the main window, cache, controller and wires them."""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from img_player.cache.frame_cache import FrameCache
from img_player.color.gpu_processor import build_shader_bundle
from img_player.color.ocio_manager import OCIOManager
from img_player.player.controller import PlayerController
from img_player.player.state import PlaybackState
from img_player.sequence.models import SequenceInfo
from img_player.sequence.scanner import SequenceNotFoundError, scan
from img_player.ui.main_window import MainWindow

log = logging.getLogger(__name__)

# Reasonable defaults for an HD VFX perso workstation — tune later via settings.
DEFAULT_CACHE_BUDGET_BYTES = 8 * 1024**3  # 8 GB
DEFAULT_NUM_WORKERS = 6


class _ScanRunner(QObject):  # type: ignore[misc]
    """Runs ``scan(path, probe=False)`` in a worker thread and emits the result.

    The ``done`` signal carries either a :class:`SequenceInfo` or an
    :class:`Exception`. Qt delivers it on the main thread automatically
    since emit happens from the worker thread.
    """

    done = Signal(object)

    def run_async(self, path: Path) -> None:
        def worker() -> None:
            try:
                seq = scan(path, probe=False)
                self.done.emit(seq)
            except Exception as err:
                self.done.emit(err)

        threading.Thread(target=worker, name="scan-worker", daemon=True).start()


class ImgPlayerApp:
    """Owns every long-lived object (cache, controller, window) and their wiring."""

    def __init__(
        self,
        argv: list[str],
        *,
        cache_budget_bytes: int = DEFAULT_CACHE_BUDGET_BYTES,
        num_workers: int = DEFAULT_NUM_WORKERS,
    ) -> None:
        self._qapp = QApplication.instance() or QApplication(argv)

        self._ocio = OCIOManager()
        self._cache = FrameCache(
            budget_bytes=cache_budget_bytes,
            num_workers=num_workers,
        )
        self._controller = PlayerController(self._cache)
        self._window = MainWindow(self._ocio)

        # A light-touch status timer so we can surface cache hit/miss info.
        self._status_timer = QTimer(self._window)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._refresh_status)

        # Polling timer that retries displaying the current frame while the
        # cache is still decoding it (used when not playing — the controller's
        # own QTimer handles drop-and-move-on while playing).
        self._wait_timer = QTimer(self._window)
        self._wait_timer.setInterval(50)
        self._wait_timer.timeout.connect(self._try_display_current_frame)

        # Refresh the timeline's cache-fill bar a few times per second.
        self._cache_bar_timer = QTimer(self._window)
        self._cache_bar_timer.setInterval(200)
        self._cache_bar_timer.timeout.connect(self._refresh_cache_bar)

        # Debounce timeline scrubs: rapid slider dragging would otherwise
        # clear + re-enqueue dozens of prefetch requests per second. We
        # coalesce into one real seek every ~20 ms. The display is updated
        # immediately (from cache only, no prefetch thrash).
        self._pending_seek: int | None = None
        self._scrub_debounce = QTimer(self._window)
        self._scrub_debounce.setSingleShot(True)
        self._scrub_debounce.setInterval(20)
        self._scrub_debounce.timeout.connect(self._apply_pending_seek)

        # Track active scan requests so a newer drag&drop supersedes an older
        # one still running in a background thread.
        self._scan_generation = 0
        self._scan_runner: _ScanRunner | None = None

        # Last frame we actually pushed to the viewport — used to avoid
        # redundant uploads when play falls back to the same nearest frame.
        self._last_displayed: int | None = None

        self._wire()

        # Push the initial color params so the GL shader is ready before any
        # frame arrives.
        self._window.color_panel.emit_current()

    # ------------------------------------------------------------------ Lifecycle

    def run(self, initial_path: Path | None = None) -> int:
        self._window.show()
        self._status_timer.start()
        self._cache_bar_timer.start()
        # Defer the initial load: show the window first so the event loop
        # can paint it, *then* kick off the (potentially slow) scan +
        # prefetch. With no deferral, the window stays invisible during a
        # slow first scan (e.g. Google Drive Stream lazy downloads).
        if initial_path is not None:
            QTimer.singleShot(0, lambda: self._open_path(initial_path))
        exit_code = int(self._qapp.exec())
        self._shutdown()
        return exit_code

    def _shutdown(self) -> None:
        self._status_timer.stop()
        self._wait_timer.stop()
        self._cache_bar_timer.stop()
        self._scrub_debounce.stop()
        self._controller.shutdown()
        self._cache.shutdown()

    # ------------------------------------------------------------------ Wiring

    def _wire(self) -> None:
        # Controller -> UI
        self._controller.frame_changed.connect(self._on_frame_changed)
        self._controller.state_changed.connect(self._on_state_changed)

        # MainWindow -> Controller
        self._window.play_toggled.connect(self._on_play_toggled)
        self._window.stop_clicked.connect(self._controller.stop)
        self._window.step_clicked.connect(self._controller.step)
        self._window.jump_to_ends.connect(self._on_jump_to_ends)
        self._window.frame_requested.connect(self._on_scrub_requested)
        self._window.open_requested.connect(self._open_path)
        self._window.exposure_step.connect(self._window.color_panel.bump_exposure)
        self._window.fps_changed.connect(self._controller.set_fps)

        # ColorPanel -> GL viewport
        self._window.color_panel.color_params_changed.connect(self._on_color_params)

    # ------------------------------------------------------------------ Handlers

    def _on_frame_changed(self, frame: int) -> None:
        self._window.timeline.set_current_frame(frame)
        arr = self._cache.get(frame)
        if arr is not None:
            self._display_array(arr)
            self._last_displayed = frame
            self._wait_timer.stop()
            return

        # Cache miss. What we do depends on whether we're playing.
        if self._controller.state.is_playing:
            # Don't freeze the view — show the nearest already-decoded
            # frame behind the playhead so the user sees continuous
            # (slower but moving) progress while the prefetcher catches up.
            fallback = self._nearest_cached_fallback(frame)
            if fallback is not None and fallback != self._last_displayed:
                fallback_arr = self._cache.get(fallback)
                if fallback_arr is not None:
                    self._display_array(fallback_arr)
                    self._last_displayed = fallback
        else:
            # Parked on this frame (user scrubbed / stopped) — poll until
            # it lands so the display snaps to the exact requested frame.
            if not self._wait_timer.isActive():
                self._wait_timer.start()

    def _nearest_cached_fallback(self, frame: int) -> int | None:
        """Pick the closest cached frame behind (for forward play) or ahead
        (for reverse play) of `frame`. Returns None if the cache is empty."""
        cached = self._cache.cached_frames()
        if not cached:
            return None
        direction = self._controller.state.direction
        if direction >= 0:
            candidates = [f for f in cached if f <= frame]
            return max(candidates) if candidates else min(cached)
        candidates = [f for f in cached if f >= frame]
        return min(candidates) if candidates else max(cached)

    def _try_display_current_frame(self) -> None:
        frame = self._controller.state.current_frame
        arr = self._cache.get(frame)
        if arr is not None:
            self._display_array(arr)
            self._last_displayed = frame
            self._wait_timer.stop()
            # Populate channel panel lazily from the first decoded frame
            # when probe was skipped at scan time.
            seq = self._controller.sequence
            if seq is not None and not seq.channel_names and arr.shape[2] > 0:
                fallback = ("R", "G", "B", "A")[: arr.shape[2]]
                self._window.channel_panel.set_channels(fallback)

    def _display_array(self, arr) -> None:  # type: ignore[no-untyped-def]
        if arr.shape[2] > 4:
            arr = arr[:, :, :4]  # viewport only handles RGB/RGBA today
        self._window.viewer.gl.set_frame(arr)

    def _on_state_changed(self, state: PlaybackState) -> None:
        self._window.transport.update_from_state(state)

    def _on_play_toggled(self) -> None:
        if self._controller.state.is_playing:
            self._controller.pause()
        else:
            self._controller.play()

    def _on_scrub_requested(self, frame: int) -> None:
        """Timeline scrub: update the display immediately from the cache, but
        defer the full seek (which re-does prefetch planning) to coalesce
        rapid slider events."""
        # Immediate visual feedback: show whatever's closest in cache.
        self._show_best_available(frame)
        self._window.timeline.set_current_frame(frame)
        # Defer the expensive part.
        self._pending_seek = frame
        self._scrub_debounce.start()

    def _apply_pending_seek(self) -> None:
        if self._pending_seek is None:
            return
        frame = self._pending_seek
        self._pending_seek = None
        self._controller.seek(frame)

    def _show_best_available(self, frame: int) -> None:
        arr = self._cache.get(frame)
        if arr is None:
            fallback = self._nearest_cached_fallback(frame)
            if fallback is None:
                return
            arr = self._cache.get(fallback)
            if arr is None:
                return
            self._last_displayed = fallback
        else:
            self._last_displayed = frame
        self._display_array(arr)

    def _on_jump_to_ends(self, direction: int) -> None:
        seq = self._controller.sequence
        if seq is None:
            return
        target = seq.first_frame if direction < 0 else seq.last_frame
        self._controller.seek(target)

    def _on_color_params(
        self, src: str, display: str, view: str, exposure: float, gamma: float
    ) -> None:
        try:
            bundle = build_shader_bundle(
                self._ocio, source_colorspace=src, display=display, view=view
            )
        except Exception:
            log.exception("failed to build color shader (%s -> %s/%s)", src, display, view)
            return
        self._window.viewer.gl.set_color_params(bundle=bundle, exposure=exposure, gamma=gamma)

    def _open_path(self, path: Path) -> None:
        """Scan `path` off the main thread so the UI stays responsive."""
        self._window.set_status(f"Scanning {path}…")

        self._scan_generation += 1
        gen = self._scan_generation

        runner = _ScanRunner()
        self._scan_runner = runner  # keep a reference so the QObject stays alive

        def on_done(result: object) -> None:
            if gen != self._scan_generation:
                # Superseded by a newer drop — ignore this result.
                return
            self._apply_scan_result(path, result)

        runner.done.connect(on_done)
        # probe=False: don't open any image file just to read metadata.
        # On lazy filesystems (Google Drive Stream) a single header read
        # can trigger a full file download (tens of seconds).
        runner.run_async(path)

    def _apply_scan_result(self, path: Path, result: object) -> None:
        if isinstance(result, Exception):
            if isinstance(result, SequenceNotFoundError):
                QMessageBox.warning(self._window, "Cannot open", str(result))
            else:
                log.exception("scan failed for %s: %s", path, result)
                QMessageBox.critical(self._window, "Scan failed", str(result))
            self._window.set_status("Ready.")
            return
        seq: SequenceInfo = result  # type: ignore[assignment]
        log.info("loaded sequence: %s (%d frames)", seq.display_pattern(), seq.frame_count)
        self._window.update_sequence_info(seq)
        self._guess_source_colorspace(seq)
        self._controller.load_sequence(seq)
        self._window.set_status(
            f"Loaded {seq.display_pattern()} ({seq.frame_count} frames) — decoding first frame…"
        )

    def _guess_source_colorspace(self, seq: SequenceInfo) -> None:
        """Heuristic: EXR -> scene_linear, PNG/JPG/TGA -> sRGB display encoded.

        The user can always override via the Color panel.
        """
        ext = seq.extension.lower()
        cs: str | None = None
        if ext in (".exr",):
            cs = self._ocio.role("scene_linear")
        elif ext in (".png", ".jpg", ".jpeg", ".tga", ".bmp"):
            cs = _first_existing(
                self._ocio,
                ["sRGB Encoded Rec.709 (sRGB)", "sRGB", "Gamma 2.2 Encoded Rec.709"],
            )
        elif ext in (".dpx", ".cin"):
            cs = _first_existing(self._ocio, ["Cineon", "Log Film"])
        if cs:
            self._window.color_panel.set_source_colorspace(cs)

    def _refresh_cache_bar(self) -> None:
        if self._controller.sequence is None:
            return
        self._window.timeline.set_cached_frames(self._cache.cached_frames())

    def _refresh_status(self) -> None:
        if self._controller.sequence is None:
            return
        stats = self._cache.stats()
        state = self._controller.state
        self._window.set_status(
            f"frame {state.current_frame}  |  "
            f"{'play' if state.is_playing else 'pause'}  |  "
            f"cache {stats.frames_cached} frames, "
            f"{stats.bytes_used / 1024**2:.0f}/{stats.bytes_budget / 1024**2:.0f} MB  |  "
            f"hits {stats.hits} / misses {stats.misses} / dropped {state.dropped_frames}"
        )


def _first_existing(manager: OCIOManager, candidates: list[str]) -> str | None:
    available = set(manager.list_colorspaces())
    for name in candidates:
        if name in available:
            return name
    return None


def run_gui(
    argv: list[str] | None = None,
    initial_path: Path | None = None,
    *,
    cache_budget_bytes: int = DEFAULT_CACHE_BUDGET_BYTES,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> int:
    """Public entry point used by ``python -m img_player``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = ImgPlayerApp(
        argv or sys.argv,
        cache_budget_bytes=cache_budget_bytes,
        num_workers=num_workers,
    )
    return app.run(initial_path=initial_path)
