"""Qt application bootstrap: builds the main window, cache, controller and wires them."""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from img_player.annotate import (
    AnnotationOverlay,
    AnnotationStore,
    AnnotationToolbar,
    ToolbarMode,
    ToolKind,
    load_annotations,
    save_annotations,
)
from img_player.annotate.persistence import sidecar_path
from img_player.cache.frame_cache import FrameCache
from img_player.color.gpu_processor import build_shader_bundle
from img_player.color.ocio_manager import OCIOManager
from img_player.io.reader import configure_oiio
from img_player.perf import (
    HardwareProfile,
    PerformanceTune,
    RuntimeMonitor,
    RuntimeState,
    apply_cli_overrides,
    apply_profile_to_tune,
    apply_runtime_constraints,
    build_profile,
    compute_tune,
    detect_hardware,
    load_profile,
    log_applied_tune,
    log_runtime_state,
    log_tune_resolution,
    save_profile,
)
from img_player.player.controller import PlayerController
from img_player.player.state import PlaybackState
from img_player.preferences import Preferences
from img_player.sequence.models import SequenceInfo
from img_player.sequence.scanner import SequenceNotFoundError, scan
from img_player.ui.main_window import MainWindow

log = logging.getLogger(__name__)

# Hard-coded fallback tier of the precedence chain.
#
#     CLI flag (explicit)  >  auto-tune  >  these DEFAULT_* constants
#
# Since slice 2 of the hardware-adaptive perf work, ``__main__.py``
# always runs ``perf.compute_tune()`` and these constants are no
# longer the boot path's source of truth — they're only used if a
# caller instantiates ``ImgPlayerApp`` / ``run_gui`` / ``run_benchmark``
# *programmatically* without passing values. They also match the
# values ``compute_tune()`` returns under ``gpu_kind="unknown"``,
# which is the conservative fallback the auto-tune emits at boot
# (before the GL context exists). Keeping them in sync is a
# non-regression invariant.
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
        cli_args: argparse.Namespace | None = None,
    ) -> None:
        # ``cli_args`` is the Namespace from ``__main__.py``'s parser.
        # Slice 4 needs it so the late-bind tune (after the GL context
        # exists and we know the real GPU) can re-apply CLI overrides
        # at the same precedence as the boot-time pipeline. ``None``
        # means "no overrides ever" — programmatic callers that never
        # touch the CLI fall through to plain auto-tune at late-bind.
        self._cli_args = cli_args
        # Track the OIIO thread count we last asked for so the late-
        # bind can detect a change and re-call ``configure_oiio``.
        self._oiio_threads_active: int | None = oiio_threads

        # Calibration tracking (slice 6). Both fields are populated
        # by the late-bind handler once the GL context reveals the
        # real GPU. At shutdown we persist them to ~/.cache/img_player/
        # profile.json so the next boot can reuse the same tune
        # without re-running compute_tune. Stay None until late-bind.
        #
        # Note: ``_desired_tune`` is the *pre-runtime-constraint* tune
        # (compute_tune + profile + CLI overrides). The runtime
        # safety clamp is intentionally NOT persisted — re-evaluating
        # it from the next boot's actual RAM headroom is the whole
        # point of slice 3. Persisting the clamped value would lock
        # the user into a tight-RAM tune even after they freed the
        # memory.
        self._desired_hw: HardwareProfile | None = None
        self._desired_tune: PerformanceTune | None = None

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

        # Slice 5: 1 Hz watchdog that auto-corrects mid-playback. Hooks
        # the controller's state_changed signal itself, so we just
        # construct it and connect its three user-facing signals to
        # the status bar. Parented to the window so its QTimer is
        # cleaned up at shutdown.
        self._runtime_monitor = RuntimeMonitor(
            self._controller, self._cache, parent=self._window
        )

        # Annotations (slices 2-3 of the feature, see
        # docs/specs/2026-04-27-annotations-design.md). The store is
        # the source of truth for strokes; the overlay is a
        # transparent QWidget child of the GL viewport that captures
        # pen input and paints existing strokes on top of the image;
        # the toolbar is a composite widget with pen / eraser / palette
        # / size / undo / redo + a pin button to bascule float ⇄ dock.
        self._annotation_store = AnnotationStore(parent=self._window)
        self._annotation_overlay = AnnotationOverlay(
            self._window.viewer.gl,
            self._annotation_store,
            parent=self._window,
        )

        # Toolbar — load mode + position from prefs.
        toolbar_mode = (
            ToolbarMode.DOCK
            if self._prefs.annotation_toolbar_mode == "dock"
            else ToolbarMode.FLOAT
        )
        self._annotation_toolbar = AnnotationToolbar(
            self._window.viewer.gl,
            self._window.annotation_dock,
            initial_mode=toolbar_mode,
            initial_floating_pos=self._prefs.annotation_toolbar_pos,
            parent=self._window,
        )
        # Toolbar starts hidden by default. The user opens it with D
        # or via the Annotations transport button (slice 4).
        self._annotation_toolbar.setVisible(self._prefs.annotation_toolbar_visible)
        if self._prefs.annotation_toolbar_visible:
            # If we want it visible AND we are in dock mode, reveal
            # the dock too. In float mode the toolbar's parent is
            # already the viewport so just show()ing is enough.
            if toolbar_mode == ToolbarMode.DOCK:
                self._window.annotation_dock.show()
        # Sync the transport's ✏ toggle button with the persisted
        # visibility so the visual matches reality at boot.
        self._window.transport.set_annotation_toggle_active(
            self._prefs.annotation_toolbar_visible
        )

        # Path of the sidecar for the currently-open sequence — set
        # by ``_open_path`` when a sequence is loaded, used by
        # ``_shutdown`` to save. ``None`` when no sequence is open.
        self._annotations_path: Path | None = None
        self._annotations_basename: str | None = None

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
        # Save annotations to the sidecar JSON if a sequence is open.
        # Done first so a later Qt teardown exception doesn't lose
        # them. ``save_annotations`` is best-effort and never raises
        # — read-only Drive Stream sessions log a warning and the
        # annotations are lost gracefully.
        if (
            self._annotations_path is not None
            and self._annotations_basename is not None
        ):
            save_annotations(
                self._annotations_path,
                self._annotation_store,
                basename=self._annotations_basename,
            )

        # Slice 6: persist the calibration profile if late-bind ran.
        # We do this before window/timer teardown so a Qt teardown
        # exception doesn't lose the profile write. If late-bind never
        # ran (initializeGL never fired — e.g. crash at startup), we
        # have nothing to save and that's fine.
        skip = bool(self._cli_args is not None and self._cli_args.skip_calibration)
        if not skip and self._desired_hw is not None and self._desired_tune is not None:
            try:
                save_profile(build_profile(self._desired_hw, self._desired_tune))
            except Exception as err:  # pragma: no cover — best effort
                log.warning("[calibration] save failed at shutdown: %s", err)

        # Persist window geometry so it reopens at the same size /
        # position; persist the dock-layout state so the side panels
        # come back collapsed / floating / wherever the user left them.
        self._prefs.window_geometry = bytes(self._window.saveGeometry())
        self._prefs.window_state = bytes(self._window.saveState())
        self._status_timer.stop()
        self._wait_timer.stop()
        self._cache_bar_timer.stop()
        self._scrub_debounce.stop()
        self._controller.shutdown()
        self._cache.shutdown()

    # ------------------------------------------------------------------ Late-bind perf tune

    def _on_gpu_renderer_detected(self, renderer: str) -> None:
        """Re-run the auto-tune now that the real GPU is known.

        Boot-time tune (in ``__main__._resolve_tune``) was forced to
        ``gpu_kind="unknown"`` because the GL context didn't exist yet,
        so it picked the conservative defaults (``oiio_threads=1``,
        ``use_pbo=False``). Now that ``initializeGL`` has fired we know
        the real ``GL_RENDERER`` and can switch to the dGPU-tuned
        values where applicable.

        What we ARE allowed to change at this point:

        * ``oiio_threads`` — calling ``configure_oiio`` again is safe;
          the next decode picks up the new value.
        * ``use_pbo`` — the viewport's PBO ring is allocated lazily on
          its first upload, so flipping the switch is just a setter.

        What we are NOT allowed to change here (per spec §4 caveat):

        * ``cache_gb`` — the FrameCache is already running and holds
          decoded frames. Reseating it would drop them and re-trigger
          a warmup. We document this but live with it; the boot-time
          ``unknown`` tune always picked a *smaller* cache than what a
          dGPU classification would (only the ceiling differs), so
          we're not "missing out" much.
        * ``num_workers`` — same reason: the worker pool is alive.
        """
        hw = detect_hardware(gpu_renderer=renderer)
        auto = compute_tune(hw)
        # Slice 6: apply the persisted profile to the late-bind tune
        # too, otherwise oiio_threads and use_pbo from the profile
        # would be silently overwritten by the freshly-computed
        # heuristics. The profile is the source of truth for this
        # machine — load it once, apply it everywhere a tune is
        # resolved.
        skip_cal = bool(
            self._cli_args is not None
            and (self._cli_args.skip_calibration or self._cli_args.recalibrate)
        )
        post_profile = (
            apply_profile_to_tune(auto, load_profile(), hw)
            if not skip_cal
            else auto
        )
        if self._cli_args is not None:
            after_cli = apply_cli_overrides(
                post_profile,
                cache_gb=self._cli_args.cache_gb,
                num_workers=self._cli_args.workers,
                oiio_threads=self._cli_args.oiio_threads,
                no_pbo=self._cli_args.no_pbo,
                force_pbo=self._cli_args.force_pbo,
            )
        else:
            after_cli = post_profile
        log.info("[hw-tune] late-bind (post-GL): re-running auto-tune with %s", renderer)
        log_tune_resolution(hw, auto, after_cli)

        state = RuntimeState.snapshot()
        final = apply_runtime_constraints(after_cli, state)
        log_runtime_state(state, after_cli, final)
        log_applied_tune(final)

        # Apply the runtime-mutable bits.
        if final.oiio_threads != self._oiio_threads_active:
            log.info(
                "[hw-tune] late-bind: oiio_threads %s → %d",
                self._oiio_threads_active,
                final.oiio_threads,
            )
            configure_oiio(final.oiio_threads)
            self._oiio_threads_active = final.oiio_threads
        self._window.viewer.gl.set_pbo_enabled(final.use_pbo)

        if final.cache_gb * 1024**3 > self._cache._budget * 1.05:
            # Diagnostic note only — the cache is already alive, see
            # docstring. We log a single line so a power user can spot
            # that the dGPU tune would have wanted a bigger cache and
            # restart the app to pick it up.
            log.info(
                "[hw-tune] late-bind: dGPU profile would have requested %.1f GB cache "
                "(currently %.1f GB). Restart the app to apply.",
                final.cache_gb,
                self._cache._budget / 1024**3,
            )

        # Slice 6: remember what we want to persist as the calibration
        # profile at shutdown.
        #
        # We save ``after_cli`` (compute_tune → profile → CLI), NOT
        # ``final`` (which is ``after_cli`` minus the runtime
        # memory-pressure clamp). The clamp is a per-boot safety —
        # re-running it on the next launch from the next launch's
        # actual ``available_ram_gb`` is the right behaviour. If we
        # persisted the clamped value, a single tight-RAM session
        # (Notion + browser + Drive open at boot) would lock the user
        # into a tiny cache for all future launches even after they
        # freed memory. The boot pipeline already re-applies the
        # clamp every time, so the safety is preserved.
        self._desired_hw = hw
        self._desired_tune = after_cli

    # ------------------------------------------------------------------ Wiring

    def _wire(self) -> None:
        # Runtime monitor (slice 5) → status bar. The monitor only
        # emits French user-facing strings; we route them straight to
        # set_status so the user sees plain language ("la machine ne
        # suit pas le rythme", "fermez d'autres applications…")
        # whenever the watchdog catches a degraded condition. The
        # status-bar UI stays untouched — same set_status path as
        # every other transient message in the app.
        self._runtime_monitor.playback_struggle.connect(self._window.set_status)
        self._runtime_monitor.memory_pressure.connect(self._window.set_status)
        self._runtime_monitor.frame_pacing_drop.connect(self._window.set_status)

        # GL viewport -> late-bind tune. The viewport emits this signal
        # exactly once per session, on its first ``initializeGL``,
        # carrying the real ``glGetString(GL_RENDERER)``. We then re-run
        # the auto-tune with the actual GPU classification and push the
        # results that can be applied at runtime (use_pbo on the viewport,
        # OIIO thread pool size). The cache budget and worker count are
        # NOT re-applied — the cache is already alive and reseating it
        # would lose its content; spec §4 documents this caveat.
        self._window.viewer.gl.gpu_renderer_detected.connect(
            self._on_gpu_renderer_detected
        )

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
        self._window.channels_requested.connect(self._on_channels_requested)
        self._window.channel_mask_changed.connect(self._on_channel_mask_changed)
        # Zoom from the combo box → propagate to the GL viewport.
        # The wheel-zoom path (viewport → combo) is wired inside
        # MainWindow so app.py doesn't have to care.
        self._window.zoom_requested.connect(self._on_zoom_requested)

        # Recent-files menu uses callbacks into preferences.
        self._window.install_recent_provider(
            provider=self._prefs.recent_paths,
            clear_callback=self._prefs.clear_recent,
        )

        # ColorPanel -> GL viewport
        self._window.color_panel.color_params_changed.connect(self._on_color_params)

        # Annotations: toolbar wiring + keyboard shortcuts.
        #
        # Toolbar -> overlay / store: the toolbar is the UI source of
        # truth for which tool / color / size is active; we forward
        # those to the overlay when they change. Undo / redo dispatch
        # against the current frame's stack.
        self._annotation_toolbar.tool_changed.connect(
            self._annotation_overlay.set_tool
        )
        self._annotation_toolbar.color_changed.connect(
            self._annotation_overlay.set_color
        )
        self._annotation_toolbar.size_changed.connect(
            self._annotation_overlay.set_size
        )
        self._annotation_toolbar.undo_requested.connect(self._undo_annotation)
        self._annotation_toolbar.redo_requested.connect(self._redo_annotation)
        # Persist mode + float position to prefs so the toolbar
        # comes back where the user left it next session.
        self._annotation_toolbar.mode_changed.connect(self._on_toolbar_mode_changed)
        self._annotation_toolbar.floating_pos_changed.connect(
            self._on_toolbar_floating_pos_changed
        )

        # Store -> timeline + transport (slice 4): when the set of
        # annotated frames changes, the timeline repaints its markers
        # and the transport's prev/next-annotation buttons re-enable
        # themselves accordingly.
        self._annotation_store.annotated_frames_changed.connect(
            self._on_annotated_frames_changed
        )

        # Transport annotation buttons -> store / toolbar.
        self._window.transport.annotation_toggle_clicked.connect(
            self._toggle_toolbar_visible
        )
        self._window.transport.annotation_prev_clicked.connect(
            self._on_annotation_prev
        )
        self._window.transport.annotation_next_clicked.connect(
            self._on_annotation_next
        )

        # Initialise the overlay from the toolbar's defaults so a user
        # who jumps straight to drawing has the expected color/size.
        self._annotation_overlay.set_color(self._annotation_toolbar.current_color())
        self._annotation_overlay.set_size(self._annotation_toolbar.current_size())

        # Keyboard shortcuts: parented to the window so they fire
        # from anywhere in the app, not just when the GL viewport has
        # focus. Existing convention (main_window.py): blocked while a
        # QLineEdit owns focus, so typing a frame number can't toggle
        # the pen.
        from PySide6.QtGui import QKeySequence, QShortcut

        QShortcut(
            QKeySequence("D"),
            self._window,
            activated=self._toggle_toolbar_visible,
        )
        QShortcut(
            QKeySequence("P"),
            self._window,
            activated=self._toggle_pen_mode,
        )
        QShortcut(
            QKeySequence("E"),
            self._window,
            activated=self._toggle_eraser_mode,
        )
        QShortcut(
            QKeySequence("A"),
            self._window,
            activated=self._toggle_show_annotations_during_play,
        )
        QShortcut(
            QKeySequence.StandardKey.Undo,
            self._window,
            activated=self._undo_annotation,
        )
        QShortcut(
            QKeySequence.StandardKey.Redo,
            self._window,
            activated=self._redo_annotation,
        )
        # Prev / next annotated frame.
        QShortcut(
            QKeySequence("["),
            self._window,
            activated=self._on_annotation_prev,
        )
        QShortcut(
            QKeySequence("]"),
            self._window,
            activated=self._on_annotation_next,
        )

    # ------------------------------------------------------------------ Handlers

    def _on_frame_changed(self, frame: int) -> None:
        self._window.timeline.set_current_frame(frame)
        # The viewport needs to know the current frame so the next
        # drag-scrub can use it as a base reference.
        self._window.viewer.gl.set_current_frame(frame)
        # The annotation overlay paints strokes for the current
        # frame — keep it in sync.
        self._annotation_overlay.set_current_frame(frame)
        # Prev/next-annotation transport buttons depend on the
        # playhead position vs the annotated set — re-evaluate.
        self._refresh_annotation_nav_buttons()
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
        # Tell the annotation overlay whether to render: hidden during
        # play unless the show-during-playback toggle is on.
        self._annotation_overlay.set_is_playing(state.is_playing)

    def _on_play_toggled(self) -> None:
        if self._controller.state.is_playing:
            self._controller.pause()
        else:
            self._controller.play()

    # -------------------------- Annotation shortcuts --------------------------
    # These are slice 2 of the annotations feature. Slice 3 will add a
    # toolbar UI that mirrors the same state and provides palette /
    # size pickers; the keyboard shortcuts below stay working.

    def _toggle_toolbar_visible(self) -> None:
        """``D`` — show or hide the annotation toolbar.

        In float mode the toolbar overlays the viewport; in dock mode
        it lives in a right-side dock. Either way, hiding it returns
        the viewport to a clean review state and disables the active
        tool (so the mouse passes through to drag-scrub etc.).
        """
        was_visible = self._annotation_toolbar.isVisible()
        if was_visible:
            # Hide and disable any active tool.
            self._annotation_toolbar.hide()
            if self._annotation_toolbar.mode() == ToolbarMode.DOCK:
                self._window.annotation_dock.hide()
            self._annotation_toolbar.set_current_tool(ToolKind.NONE)
            self._window.set_status("Annotations : toolbar masquée")
        else:
            self._annotation_toolbar.show()
            if self._annotation_toolbar.mode() == ToolbarMode.DOCK:
                self._window.annotation_dock.show()
            self._window.set_status("Annotations : toolbar visible (D pour masquer)")
        self._prefs.annotation_toolbar_visible = not was_visible
        # Reflect the new state on the transport's ✏ button so the
        # checkable visual matches reality whether the user toggled
        # via D, the toolbar's hide-on-pen-off, or the button itself.
        self._window.transport.set_annotation_toggle_active(not was_visible)

    def _on_annotation_prev(self) -> None:
        """``[`` or transport prev button — seek to the highest
        annotated frame strictly less than the current frame. No-op
        if none (button is disabled in that case but we double-check
        to keep the keyboard path robust)."""
        cur = self._controller.state.current_frame
        candidates = [
            f for f in self._annotation_store.annotated_frames() if f < cur
        ]
        if candidates:
            self._controller.seek(max(candidates))

    def _on_annotation_next(self) -> None:
        """``]`` or transport next button — seek to the lowest
        annotated frame strictly greater than the current frame."""
        cur = self._controller.state.current_frame
        candidates = [
            f for f in self._annotation_store.annotated_frames() if f > cur
        ]
        if candidates:
            self._controller.seek(min(candidates))

    def _on_annotated_frames_changed(self) -> None:
        """Pushes the new annotated-frames set to the timeline (markers
        repaint) and re-evaluates whether the transport's prev/next
        buttons are reachable from the current playhead."""
        annotated = self._annotation_store.annotated_frames()
        self._window.timeline.set_annotated_frames(annotated)
        self._refresh_annotation_nav_buttons()

    def _refresh_annotation_nav_buttons(self) -> None:
        """Enable / disable the transport prev/next buttons based on
        the current playhead position relative to the annotated set.
        Called from ``_on_annotated_frames_changed`` (set changed) and
        ``_on_frame_changed`` (playhead moved)."""
        annotated = self._annotation_store.annotated_frames()
        cur = self._controller.state.current_frame
        prev_avail = any(f < cur for f in annotated)
        next_avail = any(f > cur for f in annotated)
        self._window.transport.set_annotation_nav_enabled(prev_avail, next_avail)

    def _toggle_pen_mode(self) -> None:
        """``P`` — toggle the pen tool through the toolbar.

        Going through the toolbar (instead of straight to the overlay)
        keeps its UI checkboxes in sync — clicking the pen icon and
        pressing P are now interchangeable.
        """
        new_tool = (
            ToolKind.NONE
            if self._annotation_toolbar.current_tool() == ToolKind.PEN
            else ToolKind.PEN
        )
        # If the toolbar is hidden, show it so the user gets visual
        # feedback (the shortcut would otherwise just emit signals).
        if not self._annotation_toolbar.isVisible():
            self._toggle_toolbar_visible()
        self._annotation_toolbar.set_current_tool(new_tool)
        self._window.set_status(
            "Annotation : pen on" if new_tool == ToolKind.PEN else "Annotation : pen off"
        )

    def _toggle_eraser_mode(self) -> None:
        """``E`` — toggle the eraser tool through the toolbar."""
        new_tool = (
            ToolKind.NONE
            if self._annotation_toolbar.current_tool() == ToolKind.ERASER
            else ToolKind.ERASER
        )
        if not self._annotation_toolbar.isVisible():
            self._toggle_toolbar_visible()
        self._annotation_toolbar.set_current_tool(new_tool)
        self._window.set_status(
            "Annotation : eraser on"
            if new_tool == ToolKind.ERASER
            else "Annotation : eraser off"
        )

    def _on_toolbar_mode_changed(self, mode: ToolbarMode) -> None:
        """Persist the new mode to prefs."""
        self._prefs.annotation_toolbar_mode = mode.value

    def _on_toolbar_floating_pos_changed(self, x: int, y: int) -> None:
        """Persist the float-mode position after the user drags it."""
        self._prefs.annotation_toolbar_pos = (x, y)

    def _toggle_show_annotations_during_play(self) -> None:
        """``A`` — flip the store's flag and ask the overlay to repaint."""
        new = not self._annotation_store.show_during_playback
        self._annotation_store.show_during_playback = new
        # The overlay's paintEvent reads the flag — trigger a repaint.
        self._annotation_overlay.update()
        self._window.set_status(
            f"Annotations pendant lecture : {'visibles' if new else 'masquées'}"
        )

    def _undo_annotation(self) -> None:
        """``Ctrl+Z`` — undo on the current frame's stack only."""
        frame = self._controller.state.current_frame
        if not self._annotation_store.undo(frame):
            self._window.set_status("Annotation : rien à annuler sur cette frame")

    def _redo_annotation(self) -> None:
        """``Ctrl+Y`` — redo on the current frame's stack only."""
        frame = self._controller.state.current_frame
        if not self._annotation_store.redo(frame):
            self._window.set_status("Annotation : rien à rétablir sur cette frame")

    def _on_channel_mask_changed(self, mask: tuple) -> None:
        """Forward the four RGBA visibility booleans to the GL viewport.

        The viewer multiplies each component by 0 or 1 in the
        fragment shader — no texture re-upload, no cache eviction.
        Toggling is essentially free at runtime.
        """
        # Defensive: accept any 4-element sequence and coerce to floats.
        try:
            r, g, b, a = (1.0 if bool(x) else 0.0 for x in mask)
        except (TypeError, ValueError):
            return
        self._window.viewer.gl.set_color_params(channel_mask=(r, g, b, a))

    def _on_zoom_requested(self, factor: object) -> None:
        """Forward a zoom request from the combo to the GL viewport.

        ``factor`` is either ``None`` (= fit-to-window) or a float.
        """
        try:
            zoom: float | None = None if factor is None else float(factor)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        self._window.viewer.gl.set_zoom(zoom)

    def _on_channels_requested(self, channels: object) -> None:
        """Switch which channels the cache decodes for subsequent frames.

        ``channels`` is either ``None`` (RGB composite default) or a
        list like ``["Z"]`` or ``["albedo.R", "albedo.G", "albedo.B"]``.
        We push it to the cache, which clears currently-cached frames
        (they were decoded with the previous selection) and re-prefetches
        around the playhead.

        EXR multichannel decode is intrinsically slow (~700 ms – 2 s per
        4K frame on this hardware), so the user sees the *previous*
        channel for a couple of seconds while the worker pool
        re-decodes. We surface that with an explicit status message —
        better than silence.
        """
        # mypy gets a Signal(object) here, normalise the type at the boundary.
        cs: list[str] | None = list(channels) if isinstance(channels, list) else None
        self._cache.set_channels(cs)
        # The cache just got wiped (frames decoded with the previous
        # selection are now invalid). The timeline's cache bar polls
        # at ~200 ms, so without an eager reset the user briefly sees
        # the *previous* channel's cache runs lingering after the
        # switch — confusing, especially mid-prefetch. Mirror the same
        # eager reset we do on sequence load.
        self._window.timeline.set_cached_frames(frozenset())

        if cs is None:
            label = "RGB (composite)"
        elif len(cs) == 1:
            label = cs[0]
        elif "." in cs[0]:
            # Layer composite (e.g. ["albedo.R", "albedo.G", "albedo.B"])
            # — pull the layer prefix for a tidy "albedo (composite)"
            # message rather than a mile-long channel list.
            label = f"{cs[0].rsplit('.', 1)[0]} (composite)"
        else:
            label = " / ".join(cs)

        self._window.set_status(f"Loading channel: {label} — decoding…")

        # Re-trigger the prefetch so the new channel set fills the
        # cache around the playhead immediately. Easiest path: drive
        # through the existing seek-to-current-frame route.
        if self._controller.sequence is not None:
            self._controller.seek(self._controller.state.current_frame)

    def _on_scrub_requested(self, frame: int) -> None:
        """Timeline scrub: update the display immediately from the cache, but
        defer the full seek (which re-does prefetch planning) to coalesce
        rapid slider events."""
        # Immediate visual feedback: show whatever's closest in cache.
        self._show_best_available(frame)
        self._window.timeline.set_current_frame(frame)
        # Push the requested frame straight into the readout. Without
        # this, the number lags by one debounce window (~20 ms) plus
        # whatever decode time the seek racks up — the cursor jumps
        # under the mouse but the digits limp behind. The eventual
        # state_changed (after the debounced seek) re-asserts the
        # final value; in the common case it matches what we set here.
        self._window.transport.set_frame_immediate(frame)
        # Same reasoning for the annotation overlay: without this push,
        # the overlay would keep painting the strokes of the frame we
        # *left* until the debounced seek lands. The user sees an
        # annotation "stick" to the cursor during the drag — exactly
        # the bug reported. Pushing the frame here removes the lag
        # without altering the seek path.
        self._annotation_overlay.set_current_frame(frame)
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

        # The scanner runs with probe=False to keep the open() snappy on
        # Drive Stream / network paths — that means the SequenceInfo
        # arrives without channel names, width or height. Read the
        # header of the first frame now (one cheap OIIO call, no pixel
        # decode) so the channel selector can be populated and the
        # auto-detector has the resolution it needs.
        seq = self._enrich_with_header(seq)

        self._window.update_sequence_info(seq)
        self._guess_source_colorspace(seq)
        self._controller.load_sequence(seq)
        self._window.set_status(
            f"Loaded {seq.display_pattern()} ({seq.frame_count} frames) — decoding first frame…"
        )
        # Remember this path for next launch and for the Recent menu.
        self._prefs.last_path = path
        self._prefs.push_recent(path)

        # Load any persisted annotations for this sequence. The
        # sidecar lives next to the frame files; basename routes to
        # the right sub-payload when several sequences share a dir.
        # Strip trailing separators ('.', '_') from the base_name so
        # 'render.' and 'render' both map to the same JSON key — a
        # cosmetic detail, but it would be confusing if a previously-
        # saved sequence's notes silently disappeared after a tool
        # change in the scanner.
        self._annotations_path = sidecar_path(seq.directory)
        self._annotations_basename = seq.base_name.rstrip("._-") or seq.base_name
        loaded = load_annotations(
            self._annotations_path,
            basename=self._annotations_basename,
        )
        if loaded is not None:
            # Replace the in-memory store contents (reuse the live
            # store object so its signal subscribers stay wired).
            self._annotation_store.load_from_dict(loaded.to_dict()["frames"])
            log.info(
                "[annotations] loaded %d annotated frames from %s",
                len(self._annotation_store.annotated_frames()),
                self._annotations_path,
            )
        else:
            # Fresh start — clear any leftover state from a previous
            # sequence in this session.
            self._annotation_store.load_from_dict({})

    def _enrich_with_header(self, seq: SequenceInfo) -> SequenceInfo:
        """Fill in channel_names / width / height from the first frame's
        header if the scanner skipped them. Returns the same seq when
        already populated."""
        if seq.channel_names and seq.width and seq.height:
            return seq
        if not seq.frames:
            return seq
        try:
            from dataclasses import replace
            from img_player.io.reader import read_header

            spec = read_header(seq.frames[0].path)
            channels = tuple(spec.channelnames or ())
            log.info(
                "header probe: %d channels (%s), %dx%d",
                len(channels), ", ".join(channels[:8]) + ("…" if len(channels) > 8 else ""),
                spec.width, spec.height,
            )
            return replace(
                seq,
                channel_names=channels or seq.channel_names,
                width=spec.width or seq.width,
                height=spec.height or seq.height,
            )
        except Exception:
            log.exception("could not read header from %s", seq.frames[0].path)
            return seq

    def _guess_source_colorspace(self, seq: SequenceInfo) -> None:
        """Auto-detect the source colorspace + the right view for it.

        See :mod:`img_player.color.auto_detect` for the cascades. The
        user can always override via the Color panel.
        """
        from img_player.color.auto_detect import (
            detect_source_colorspace,
            detect_view,
        )
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

        source_result = detect_source_colorspace(
            metadata=metadata,
            extension=seq.extension,
            available_colorspaces=self._ocio.list_colorspaces(),
            scene_linear_role=self._ocio.role("scene_linear"),
        )
        if source_result.colorspace is not None:
            self._window.color_panel.set_source_colorspace(source_result.colorspace)
            log.info(
                "auto-detect: source colorspace = %s (%s)",
                source_result.colorspace, source_result.reason,
            )
        else:
            log.info("auto-detect: no source colorspace match (%s)", source_result.reason)

        # Now pick the view appropriate for that source. A scene-
        # referred input wants tone mapping (ACES SDR / Filmic); a
        # display-referred input wants Raw / Un-tone-mapped to avoid
        # doubling up the EOTF.
        view_msg = ""
        if source_result.colorspace is not None:
            current_display = self._window.color_panel._display_combo.currentText()
            if current_display:
                available_views = self._ocio.list_views(current_display)
                view_result = detect_view(
                    source_colorspace=source_result.colorspace,
                    available_views=available_views,
                    default_view=self._ocio.default_view(current_display),
                )
                if view_result.colorspace is not None:
                    self._window.color_panel._view_combo.setCurrentText(view_result.colorspace)
                    view_msg = f" → view: {view_result.colorspace} ({view_result.reason})"
                    log.info(
                        "auto-detect: view = %s (%s)",
                        view_result.colorspace, view_result.reason,
                    )

        # Surface the combined choice in the status bar (left side).
        if source_result.colorspace is not None:
            self._window.set_status(
                f"Source: {source_result.colorspace} ({source_result.reason}){view_msg}"
            )
        else:
            self._window.set_status(
                f"Source colorspace: not detected — {source_result.reason}. "
                f"Pick one in the Color panel."
            )
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
    state = prefs.window_state
    if state is not None:
        # restoreState reapplies dock visibility / position / floating
        # from the previous session.
        app._window.restoreState(state)

    # Color defaults — only apply if they still exist in the current OCIO config.
    cs_list = set(app._ocio.list_colorspaces())
    displays = set(app._ocio.list_displays())
    if prefs.source_colorspace and prefs.source_colorspace in cs_list:
        app._window.color_panel.set_source_colorspace(prefs.source_colorspace)

    # Display: prefer a stored preference; otherwise auto-detect from
    # the current screen's color profile.
    display_name = prefs.display if prefs.display in displays else None
    if display_name is None:
        display_name = _autodetect_display(app)

    if display_name and display_name in displays:
        # Selecting a display also repopulates the view combo, so set view after.
        app._window.color_panel._display_combo.setCurrentText(display_name)

    if prefs.view and display_name and prefs.view in set(app._ocio.list_views(display_name)):
        app._window.color_panel._view_combo.setCurrentText(prefs.view)

    # FPS — push through the controller so transport + timeline pick up
    # the value via state_changed (keeps the FPS combo / timeline TC in sync).
    app._controller.set_fps(prefs.fps)


def _autodetect_display(app: ImgPlayerApp) -> str | None:
    """Inspect Qt's view of the primary screen and pick the matching
    OCIO display.

    Returns the display name that was applied, or ``None`` if even
    the safe sRGB fallback wasn't in the config.
    """
    from img_player.color.auto_detect import detect_display

    hint = _qt_screen_colorspace_hint(app._qapp)
    result = detect_display(hint, app._ocio.list_displays())
    if result.colorspace is not None:
        log.info("auto-detect: display = %s (%s)", result.colorspace, result.reason)
        # Surface to the user; will be replaced by the source-colorspace
        # message once they load a sequence.
        app._window.set_status(f"Display: {result.colorspace} ({result.reason})")
        return result.colorspace
    log.info("auto-detect: no display match (%s)", result.reason)
    return None


def _qt_screen_colorspace_hint(qapp: QApplication) -> str | None:
    """Translate Qt's QColorSpace introspection into the lowercase
    canonical name our :func:`auto_detect.detect_display` expects.

    Returns ``None`` when the screen has a custom ICC profile that
    Qt couldn't classify into a named colorspace; the detector then
    falls back to sRGB.
    """
    from PySide6.QtGui import QColorSpace

    screen = qapp.primaryScreen() if qapp is not None else None
    if screen is None:
        return None
    qcs = screen.colorSpace()
    if not qcs.isValid():
        return None

    # Qt 6 enum → canonical lowercase string. Anything not listed
    # (e.g. Undefined, or a custom-named ICC) returns None, and
    # detect_display() handles that with the sRGB fallback.
    mapping = {
        QColorSpace.NamedColorSpace.SRgb: "srgb",
        QColorSpace.NamedColorSpace.SRgbLinear: "srgblinear",
        QColorSpace.NamedColorSpace.AdobeRgb: "adobergb",
        QColorSpace.NamedColorSpace.DisplayP3: "displayp3",
        QColorSpace.NamedColorSpace.ProPhotoRgb: "prophotorgb",
        QColorSpace.NamedColorSpace.Bt2020: "bt2020",
        QColorSpace.NamedColorSpace.Bt2100Pq: "bt2100pq",
        QColorSpace.NamedColorSpace.Bt2100Hlg: "bt2100hlg",
    }
    return mapping.get(qcs.namedColorSpace())


def run_gui(
    argv: list[str] | None = None,
    initial_path: Path | None = None,
    *,
    cache_budget_bytes: int = DEFAULT_CACHE_BUDGET_BYTES,
    num_workers: int = DEFAULT_NUM_WORKERS,
    oiio_threads: int | None = DEFAULT_OIIO_THREADS,
    cli_args: argparse.Namespace | None = None,
) -> int:
    """Public entry point used by ``python -m img_player``.

    ``cli_args`` propagates the parsed argparse Namespace down to
    ``ImgPlayerApp`` so the late-bind perf tune (slice 4) can re-apply
    user overrides at the same precedence as the boot pipeline. Older
    callers that don't pass it fall through to plain auto-tune, which
    is also fine.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = ImgPlayerApp(
        argv or sys.argv,
        cache_budget_bytes=cache_budget_bytes,
        num_workers=num_workers,
        oiio_threads=oiio_threads,
        cli_args=cli_args,
    )
    return app.run(initial_path=initial_path)
