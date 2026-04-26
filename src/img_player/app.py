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
from img_player.io.reader import configure_oiio
from img_player.player.controller import PlayerController
from img_player.player.state import PlaybackState
from img_player.preferences import Preferences
from img_player.sequence.models import SequenceInfo
from img_player.sequence.scanner import SequenceNotFoundError, scan
from img_player.ui.main_window import MainWindow

log = logging.getLogger(__name__)

# Reasonable defaults for an HD VFX perso workstation — tune later via settings.
DEFAULT_CACHE_BUDGET_BYTES = 8 * 1024**3  # 8 GB
DEFAULT_NUM_WORKERS = 6
# Why 1 instead of os.cpu_count():
# On APUs with shared CPU/GPU memory (tested on AMD Radeon 780M with 16
# logical cores), letting OIIO spawn os.cpu_count() threads *per decode*
# saturates DRAM bandwidth — which also slows down the glTexSubImage2D
# memcpy on the Qt main thread. Empirically threads=1 with 6 workers
# gives +47% playback fps vs threads=16. See perf/BASELINE.md.
DEFAULT_OIIO_THREADS: int | None = 1


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
        oiio_threads: int | None = DEFAULT_OIIO_THREADS,
    ) -> None:
        self._qapp = QApplication.instance() or QApplication(argv)
        self._qapp.setOrganizationName("img_player")
        self._qapp.setApplicationName("img_player")

        from img_player.ui.theme import build_stylesheet
        self._qapp.setStyleSheet(build_stylesheet())

        # Configure OIIO's global thread pool *before* we spin up the cache —
        # any in-flight decode would otherwise see the default value.
        configure_oiio(oiio_threads)

        self._prefs = Preferences()

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

        # Restore user preferences (colorspace, FPS, window geometry, recent
        # files) from previous sessions.
        self._apply_preferences()

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

    def _apply_preferences(self) -> None:
        _apply_preferences_to_window(self)

    def _shutdown(self) -> None:
        # Persist window geometry so it reopens at the same size/position.
        self._prefs.window_geometry = bytes(self._window.saveGeometry())
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
        self._window.step_clicked.connect(self._controller.step)
        self._window.jump_to_ends.connect(self._on_jump_to_ends)
        self._window.frame_requested.connect(self._on_scrub_requested)
        self._window.open_requested.connect(self._open_path)
        self._window.exposure_step.connect(self._window.color_panel.bump_exposure)
        self._window.fps_changed.connect(self._on_fps_changed)
        self._window.direction_play_requested.connect(self._on_direction_play)
        self._window.mark_in_requested.connect(self._on_mark_in)
        self._window.mark_out_requested.connect(self._on_mark_out)
        self._window.clear_in_out_requested.connect(lambda: self._controller.set_in_out(None, None))
        self._window.loop_mode_requested.connect(self._controller.set_loop_mode)

        # Recent-files menu uses callbacks into preferences.
        self._window.install_recent_provider(
            provider=self._prefs.recent_paths,
            clear_callback=self._prefs.clear_recent,
        )

        # ColorPanel -> GL viewport
        self._window.color_panel.color_params_changed.connect(self._on_color_params)

    # ------------------------------------------------------------------ Handlers

    def _on_frame_changed(self, frame: int) -> None:
        self._window.timeline.set_current_frame(frame)
        # The viewport needs to know the current frame so the next
        # drag-scrub can use it as a base reference.
        self._window.viewer.gl.set_current_frame(frame)
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
        # Timeline needs in/out markers and the fps for its timecode labels.
        self._window.timeline.set_in_out(state.in_frame, state.out_frame)
        self._window.timeline.set_fps(state.fps)

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
        # Persist the colorspace triple so the next launch starts with the
        # same look. Exposure / gamma are per-image adjustments and aren't
        # saved.
        self._prefs.source_colorspace = src
        self._prefs.display = display
        self._prefs.view = view

    def _on_fps_changed(self, fps: float) -> None:
        self._controller.set_fps(fps)
        self._prefs.fps = fps

    def _on_direction_play(self, direction: int) -> None:
        # Logic lives on the controller — start / flip / pause based
        # on the requested direction vs current state. See
        # :meth:`PlayerController.play_direction` for the rules.
        self._controller.play_direction(direction)

    def _on_mark_in(self) -> None:
        cur = self._controller.state.current_frame
        self._controller.set_in_out(cur, self._controller.state.out_frame)
        self._window.set_status(f"In point set to frame {cur}")

    def _on_mark_out(self) -> None:
        cur = self._controller.state.current_frame
        self._controller.set_in_out(self._controller.state.in_frame, cur)
        self._window.set_status(f"Out point set to frame {cur}")

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
        # Remember this path for next launch and for the Recent menu.
        self._prefs.last_path = path
        self._prefs.push_recent(path)

    def _guess_source_colorspace(self, seq: SequenceInfo) -> None:
        """Auto-detect the source colorspace from the first frame's
        metadata, with extension as a fallback.

        See :mod:`img_player.color.auto_detect` for the cascade. The
        user can always override via the Color panel.
        """
        from img_player.color.auto_detect import detect_source_colorspace
        from img_player.io.reader import read_color_metadata

        # Read the metadata of the first frame only — colour metadata
        # is invariant across the sequence, and reading one header is
        # cheap (no pixel decode).
        first_path = seq.frames[0].path if seq.frames else None
        metadata: dict[str, object] = {}
        if first_path is not None:
            try:
                metadata = read_color_metadata(first_path)
            except Exception:
                log.exception("failed to read color metadata from %s", first_path)

        result = detect_source_colorspace(
            metadata=metadata,
            extension=seq.extension,
            available_colorspaces=self._ocio.list_colorspaces(),
            scene_linear_role=self._ocio.role("scene_linear"),
        )
        if result.colorspace is not None:
            self._window.color_panel.set_source_colorspace(result.colorspace)
            self._window.set_status(
                f"Source colorspace: {result.colorspace} ({result.reason})"
            )
            log.info(
                "auto-detect: source colorspace = %s (%s)",
                result.colorspace, result.reason,
            )
        else:
            log.info("auto-detect: no source colorspace match (%s)", result.reason)
            self._window.set_status(
                f"Source colorspace: not detected — {result.reason}. "
                f"Pick one in the Color panel."
            )

    def _refresh_cache_bar(self) -> None:
        if self._controller.sequence is None:
            return
        self._window.timeline.set_cached_frames(self._cache.cached_frames())

    def _refresh_status(self) -> None:
        """Update the right-hand perf indicators every 500 ms.

        The left-hand contextual message is owned by other handlers
        (open / mark_in / mark_out / etc.) — we don't touch it here so
        their messages aren't overwritten by the timer.
        """
        seq = self._controller.sequence
        if seq is None:
            return
        from img_player.ui.status_format import format_perf_html

        stats = self._cache.stats()
        state = self._controller.state
        eff = self._controller.effective_fps()
        cache_total = max(1, seq.frame_count)
        cache_ratio = stats.bytes_used / max(1, stats.bytes_budget)
        ram_gb = stats.bytes_used / 1024**3

        self._window.status_right.setText(
            format_perf_html(
                cache_n=stats.frames_cached,
                cache_total=cache_total,
                cache_ratio=cache_ratio,
                fps_effective=eff,
                fps_target=state.fps,
                ram_gb=ram_gb,
            )
        )




# ---------------------------------------------------------------------- Preferences glue


def _apply_preferences_to_window(app: ImgPlayerApp) -> None:
    """Separated so we can call it after the window exists but before show()."""
    prefs = app._prefs
    geom = prefs.window_geometry
    if geom is not None:
        app._window.restoreGeometry(geom)

    # Color defaults — only apply if they still exist in the current OCIO config.
    cs_list = set(app._ocio.list_colorspaces())
    displays = set(app._ocio.list_displays())
    if prefs.source_colorspace and prefs.source_colorspace in cs_list:
        app._window.color_panel.set_source_colorspace(prefs.source_colorspace)
    if prefs.display and prefs.display in displays:
        # Selecting a display also repopulates the view combo, so set view after.
        app._window.color_panel._display_combo.setCurrentText(prefs.display)
    if prefs.view and prefs.display and prefs.view in set(app._ocio.list_views(prefs.display)):
        app._window.color_panel._view_combo.setCurrentText(prefs.view)

    # FPS — push through the controller so transport + timeline pick up
    # the value via state_changed (keeps the FPS combo / timeline TC in sync).
    app._controller.set_fps(prefs.fps)


def run_gui(
    argv: list[str] | None = None,
    initial_path: Path | None = None,
    *,
    cache_budget_bytes: int = DEFAULT_CACHE_BUDGET_BYTES,
    num_workers: int = DEFAULT_NUM_WORKERS,
    oiio_threads: int | None = DEFAULT_OIIO_THREADS,
) -> int:
    """Public entry point used by ``python -m img_player``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = ImgPlayerApp(
        argv or sys.argv,
        cache_budget_bytes=cache_budget_bytes,
        num_workers=num_workers,
        oiio_threads=oiio_threads,
    )
    return app.run(initial_path=initial_path)
