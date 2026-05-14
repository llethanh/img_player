"""Qt application bootstrap: builds the main window, cache, controller and wires them."""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from img_player.annotate import (
    AnnotationOverlay,
    AnnotationStore,
    AnnotationToolbar,
    EphemeralStrokeManager,
    ToolbarMode,
    ToolKind,
    save_annotations,
)
# Handlers are hoisted to module top (they were lazy-imported at 17
# call sites, several of which fire per-frame on ``_on_frame_changed``).
# All ``*_handler`` modules TYPE_CHECKING-import ImgPlayerApp so the
# back-edge isn't a real cycle.
from img_player.channel_handler import (
    on_channel_selection_changed,
    set_channel_selection,
)
from img_player.color.gpu_processor import build_shader_bundle
from img_player.color.ocio_manager import OCIOManager
from img_player.comment import CommentStore, save_comments
from img_player.compare_handler import (
    render_compare as _render_compare,
)
from img_player.io.reader import configure_oiio
from img_player.media_handler import (
    close_orphan_video_sources as _close_orphan_video_sources,
    current_layer_time as _current_layer_time,
    decode_video_layer as _decode_video_layer,
    pick_active_audio_layer as _pick_active_audio_layer,
    refresh_active_audio as _refresh_active_audio,
    reseek_active_audio_for_layer_change as _reseek_active_audio_for_layer_change,
)
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
from img_player.sequence.channels import ChannelSelection
from img_player.sequence.models import SequenceInfo
from img_player.sequence.scanner import enrich_with_header
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
        boot_tune: PerformanceTune | None = None,
    ) -> None:
        # Construction is split into phases so each block stays small
        # enough to scan. Phase order is load-bearing — Qt objects
        # need a live QApplication, the window needs the OCIO/cache
        # backbones, the annotation overlays need the window's GL
        # viewport. Don't reorder without checking the deps.
        self._init_plain_state(cli_args, oiio_threads, boot_tune)
        self._build_qt_runtime(argv, oiio_threads)
        self._build_models(cache_budget_bytes, num_workers)
        self._build_window_and_overlays()
        self._build_timers()
        self._init_channel_state()

        self._wire()

        # Restore user preferences (colorspace, FPS, window geometry, recent
        # files) from previous sessions.
        self._apply_preferences()

        # Push the initial color params so the GL shader is ready before any
        # frame arrives.
        self._window.color_panel.emit_current()

    # -- __init__ phases -------------------------------------------------

    def _init_plain_state(
        self,
        cli_args: argparse.Namespace | None,
        oiio_threads: int | None,
        boot_tune: PerformanceTune | None = None,
    ) -> None:
        """Pre-Qt attribute setup. No Qt objects are constructed here."""
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
        # Pre-runtime-constraint tune resolved at boot. Used by
        # ``_retune_for_current_ram`` as the ceiling for the per-session
        # cache budget recompute: closing other apps between Flick
        # launch and a project open lets the next project enjoy a
        # bigger cache without restart, but never beyond what
        # ``compute_tune`` would have asked for if the boot had had
        # the same RAM available. Falls back to ``_desired_tune`` once
        # the late-bind GPU re-tune fires (= more accurate ceiling
        # because the real GPU is known at that point).
        self._boot_tune: PerformanceTune | None = boot_tune

        # Path of the sidecar for the currently-open sequence — set
        # by ``_open_path`` when a sequence is loaded, used by
        # ``_shutdown`` to save. ``None`` when no sequence is open.
        self._annotations_path: Path | None = None
        self._annotations_basename: str | None = None

        # Track active scan requests so a newer drag&drop supersedes an older
        # one still running in a background thread. The runner type is
        # ``scan_handler.ScanRunner`` but typed as ``object`` here to
        # avoid pulling Qt into ``_init_plain_state``.
        self._scan_generation = 0
        self._scan_runner: object | None = None

        # Last frame we actually pushed to the viewport — used to avoid
        # redundant uploads when play falls back to the same nearest frame.
        self._last_displayed: int | None = None

        # Pending seek payload for the scrub debouncer (real timer
        # built later in ``_build_timers``).
        self._pending_seek: int | None = None

    def _build_qt_runtime(self, argv: list[str], oiio_threads: int | None) -> None:
        """QApplication + global stylesheet + OIIO thread pool."""
        # Set the Windows AppUserModelID *before* the QApplication is
        # created. Without this Windows attributes the taskbar entry
        # to the .exe path or — worse, on a freshly-cloned dev env —
        # to the Python interpreter, which means the taskbar shows a
        # generic placeholder instead of the ``flick.ico`` artwork.
        # Setting it explicitly lets Windows group all FlickPlayer
        # instances under the same icon (= the one PyInstaller
        # embedded in the .exe and Qt loads via ``setWindowIcon``
        # below).
        import sys as _sys
        if _sys.platform == "win32":
            try:
                import ctypes as _ctypes
                _ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "flickplayer.app",
                )
            except Exception:  # pragma: no cover — defensive
                log.exception("failed to set AppUserModelID")
        self._qapp = QApplication.instance() or QApplication(argv)
        self._qapp.setOrganizationName("img_player")
        self._qapp.setApplicationName("img_player")

        # App-level icon — picked up by Qt for the title bar, the
        # taskbar grouping, and any tray icon listeners. Same artwork
        # as the ``flick.ico`` baked into the .exe by PyInstaller.
        # Resolved by Path arithmetic from the package — same pattern
        # ``cache.missing_frame`` uses for its bundled font.
        from pathlib import Path as _Path

        from PySide6.QtGui import QIcon
        ico_path = (
            _Path(__file__).resolve().parent
            / "assets" / "icons" / "flick.ico"
        )
        if ico_path.is_file():
            self._qapp.setWindowIcon(QIcon(str(ico_path)))

        from img_player.ui.theme import build_stylesheet
        self._qapp.setStyleSheet(build_stylesheet())

        # Register the bundled "Big Shoulders Display" TTF (if shipped
        # in ``assets/fonts/``) so the missing-frame placeholder picks
        # it up. No-op when the file isn't present — the module falls
        # back to Arial silently.
        from img_player.cache.missing_frame import ensure_font_loaded
        ensure_font_loaded()

        # Configure OIIO's global thread pool *before* we spin up the cache —
        # any in-flight decode would otherwise see the default value.
        configure_oiio(oiio_threads)

    def _build_models(self, cache_budget_bytes: int, num_workers: int) -> None:
        """Non-UI domain objects: prefs, cache, controller, stores."""
        self._prefs = Preferences()
        self._ocio = OCIOManager()
        # Multi-layer foundation (v1.0 phase 2c). The LayerStack is
        # the source of truth for "what's loaded"; the cache reads
        # from it for path / channel resolution at decode time. Order
        # matters: stack first, then cache (which subscribes to stack
        # signals), then controller (which uses the cache).
        from img_player.cache.disk_cache import DiskCache, default_cache_dir
        from img_player.cache.master_frame_cache import MasterFrameCache
        from img_player.layers import LayerStack
        from img_player.media import VideoSourceManager
        self._layer_stack = LayerStack()
        # On-disk frame cache — survives close/reopen so the next
        # session re-opens warm. Path + budget driven by preferences;
        # disabling switches the tier off entirely (the cache falls
        # back to its legacy RAM-only behaviour).
        disk_cache: DiskCache | None = None
        if self._prefs.disk_cache_enabled:
            try:
                cache_dir = self._prefs.disk_cache_path or default_cache_dir()
                budget_bytes = self._prefs.disk_cache_budget_gb * (1024 ** 3)
                disk_cache = DiskCache(
                    cache_dir,
                    budget_bytes=budget_bytes,
                    compress=self._prefs.disk_cache_compression,
                )
            except Exception:  # pragma: no cover — defensive
                log.exception(
                    "DiskCache init failed; falling back to RAM-only cache",
                )
                disk_cache = None
        self._disk_cache = disk_cache
        self._cache = MasterFrameCache(
            self._layer_stack,
            budget_bytes=cache_budget_bytes,
            num_workers=num_workers,
            disk_cache=disk_cache,
        )
        # Video decoders. Image-sequence layers go through ``self._cache``;
        # video layers (mp4 / mov / …) bypass the cache and pull pixels
        # directly from this manager — long-GOP video has fundamentally
        # different access patterns (sequential cheap, random expensive)
        # that don't fit the cache's per-frame independent model.
        self._video_sources = VideoSourceManager()
        # Persistent audio output (sounddevice + feeder thread). Stays
        # open from boot through shutdown — option (b) of the design:
        # no play-time latency at the cost of holding the device.
        # Initially silent (no source); ``_refresh_active_audio`` swaps
        # the active AudioSource on layer-stack / playhead changes.
        from img_player.media import AudioOutput
        self._audio_output = AudioOutput()
        self._audio_output.open()
        # Tracks which layer id is currently feeding the AudioOutput,
        # so ``_refresh_active_audio`` doesn't re-open the source on
        # every frame_changed when the active layer is unchanged.
        self._active_audio_layer_id: str | None = None
        self._controller = PlayerController(self._cache)
        # Comment store — owned by the app, passed to MainWindow so
        # the Comments tab can read / write directly. Cohérent with
        # how the AnnotationStore is owned: app-level for lifecycle,
        # widgets just hold a reference. Constructed before the
        # window because MainWindow takes it as a constructor arg.
        self._comment_store = CommentStore()

    def _build_window_and_overlays(self) -> None:
        """Main window + annotation store / overlay / toolbar.

        Window must come before overlays (overlays parent themselves
        on the GL viewport). RuntimeMonitor parents on the window for
        QTimer cleanup at shutdown.
        """
        self._window = MainWindow(
            self._ocio, self._comment_store, layer_stack=self._layer_stack,
        )

        # Slice 5: 1 Hz watchdog that auto-corrects mid-playback. Hooks
        # the controller's state_changed signal itself, so we just
        # construct it and connect its three user-facing signals to
        # the status bar.
        self._runtime_monitor = RuntimeMonitor(
            self._controller, self._cache, parent=self._window,
        )

        # E3 — auto-reload on disk changes. Wraps a QFileSystemWatcher
        # on the layer source directories with a 200 ms debounce; on
        # change fires the existing smart-reload pipeline so a
        # re-rendered EXR shows up without the artist having to
        # press Ctrl+R. Wired below in ``_wire_layer_stack`` so the
        # watch list tracks the live layer stack.
        from img_player.sequence.source_watcher import SourceWatcher

        self._source_watcher = SourceWatcher(parent=self._window)
        self._source_watcher.sources_changed.connect(
            self._on_source_watcher_fired,
        )

        # Annotations: store + overlay + toolbar. The store is the
        # source of truth for strokes; the overlay is a transparent
        # QWidget child of the GL viewport that captures pen input
        # and paints existing strokes; the toolbar is a composite
        # widget with pen / eraser / palette / size / undo / redo + a
        # pin button to bascule float ⇄ dock.
        self._annotation_store = AnnotationStore(parent=self._window)
        # Ephemeral mode (v0.4.1) — companion store for live, fading,
        # never-saved strokes. Owned at app-level so its QTimer
        # survives a sequence reload (we just call clear_all() to
        # drop ghosts from the previous sequence).
        self._ephemeral_manager = EphemeralStrokeManager(parent=self._window)
        # Seed the duration from the persisted preset preference.
        from img_player.annotate.toolbar import EPHEMERAL_PRESETS_S
        try:
            preset_idx = self._prefs.ephemeral_duration_preset
            self._ephemeral_manager.set_duration(EPHEMERAL_PRESETS_S[preset_idx])
        except (IndexError, ValueError):
            pass  # manager already has its 5s default
        self._annotation_overlay = AnnotationOverlay(
            self._window.viewer.gl,
            self._annotation_store,
            parent=self._window,
        )
        # Inject the manager into the overlay so ephemeral routes work
        # at the very first mouseRelease — before any signal-wiring
        # below has a chance to fire.
        self._annotation_overlay.set_ephemeral_manager(self._ephemeral_manager)

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
            initial_ephemeral_preset=self._prefs.ephemeral_duration_preset,
            initial_stabilizer_level=self._prefs.pen_stabilizer_level,
            parent=self._window,
        )
        # Toolbar starts hidden by default. The user opens it with D
        # or via the Annotations transport button.
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
            self._prefs.annotation_toolbar_visible,
        )

    def _build_timers(self) -> None:
        """All QTimer instances. Started in :meth:`run`."""
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
        self._scrub_debounce = QTimer(self._window)
        self._scrub_debounce.setSingleShot(True)
        self._scrub_debounce.setInterval(20)
        self._scrub_debounce.timeout.connect(self._apply_pending_seek)

    def _init_channel_state(self) -> None:
        """Channel-selection + compare-mode bookkeeping.

        ``_channel_selection`` is the app-level fallback for the
        active channel group, used until the first layer focuses
        (the per-layer ``Layer.channel_selection`` then takes over).
        """
        self._channel_selection: ChannelSelection | None = None
        # Compare-mode (two-layer A/B overlay) state + decoder. The
        # band itself lives on the viewer; this app-side state is the
        # single source of truth that the band, keyboard shortcuts
        # and session save all sync against.
        from img_player.compare import CompareState
        from img_player.compare.decode import CompareDecoder

        self._compare_state: CompareState = CompareState()
        self._compare_decoder = CompareDecoder(self._video_sources)

        # Contact-sheet mode (multi-layer grid view, layers aligned to
        # the same "frame 0" regardless of timeline offset). Same
        # shape as compare-mode: state held here, edited from the
        # View menu / a small settings band on the viewer.
        from img_player.contact_sheet import (
            ContactSheetDecoder,
            ContactSheetState,
        )

        self._contact_sheet_state: ContactSheetState = ContactSheetState()
        self._contact_sheet_decoder = ContactSheetDecoder(self._video_sources)

    # ------------------------------------------------------------------ Lifecycle

    def run(self, initial_path: Path | None = None) -> int:
        self._window.show()
        # Hand the splash off to the main window so it fades as soon
        # as the real UI becomes the active paintable widget — avoids
        # the brief blank-frame gap an unconditional close can cause.
        # No-op outside a QApplication context.
        from img_player import splash
        splash.close(self._window)
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
        # Annotations are saved by the ``_prompt_save_annotations``
        # callback fired from MainWindow.closeEvent — the user is
        # asked explicitly whether to overwrite the existing sidecar
        # (or create a new one). Nothing to do here at shutdown.
        # Crash / kill paths bypass closeEvent and lose annotations
        # — same as before this change since _shutdown didn't run
        # in those cases either.

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
        # Side-tab selection (Color vs Comments) and view-mode toggle
        # (frames vs timecode) live OUTSIDE saveState — store them
        # explicitly so the user gets back exactly the layout they
        # left. ``QTabWidget.currentIndex`` and the view menu's
        # checked QAction aren't covered by Qt's dock-state blob.
        self._prefs.side_tab_index = self._window.side_tab_index()
        self._prefs.display_timecode = self._window.display_timecode()
        self._prefs.info_band_segments = self._window.info_band_segments()
        # Side panel (Color / Comments) visibility — saveState no
        # longer covers it since the panel was lifted out of the
        # dock system.
        self._prefs.side_panel_visible = self._window._side_dock.isVisible()
        # LayerPanel collapsed state (v1.0).
        panel = getattr(self._window, "_layer_panel", None)
        if panel is not None:
            self._prefs.layer_panel_collapsed = panel.is_collapsed()
        # NB: the channel menu's active selection is intentionally
        # NOT persisted across runs. Each newly loaded sequence opens
        # on its first channel group (RGB / beauty pass) so the user
        # never sees a stale "albedo" pick from a previous sequence
        # carry over to a fresh one.
        self._status_timer.stop()
        self._wait_timer.stop()
        self._cache_bar_timer.stop()
        self._scrub_debounce.stop()
        # Stop the auto-reload file watcher — frees the OS-level
        # handles held on layer source directories so the next
        # process launch doesn't see "files locked" issues.
        try:
            self._source_watcher.stop()
        except Exception:  # pragma: no cover — defensive
            log.exception("source watcher stop failed (non-fatal)")
        self._controller.shutdown()
        # If the disk-cache writer has a backlog, show a small
        # "Flushing disk cache..." label so the user sees the few-
        # second pause is intentional and not a hang. Threshold of
        # 5 is empirical: smaller backlogs drain too fast for the
        # widget to even paint, just adding visual noise.
        flush_label = None
        pending = 0
        try:
            pending = self._cache.disk_cache_pending_writes()
        except Exception:  # pragma: no cover — defensive
            pending = 0
        if pending > 5:
            try:
                from PySide6.QtCore import Qt

                from img_player.ui.flush_indicator import FlushIndicator

                flush_label = FlushIndicator(pending, parent=self._window)
                flush_label.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
                flush_label.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
                flush_label.show()
                QApplication.processEvents()
            except Exception:  # pragma: no cover — UI is best effort at exit
                log.exception("flush indicator setup failed (non-fatal)")
                flush_label = None

        def _on_flush_progress(remaining: int) -> None:
            if flush_label is None:
                return
            try:
                flush_label.update_remaining(remaining)
                QApplication.processEvents()
            except Exception:  # pragma: no cover
                pass

        try:
            self._cache.shutdown(disk_progress_callback=_on_flush_progress)
        finally:
            if flush_label is not None:
                try:
                    flush_label.close()
                except Exception:  # pragma: no cover
                    pass
        # Close every open video decoder. Important on Windows where
        # PyAV holds an OS file handle on the container; without this
        # close the next session reload could see "file in use".
        try:
            self._video_sources.shutdown()
        except Exception:  # pragma: no cover — best effort
            log.exception("video sources shutdown failed")
        # Close the audio output (stops the feeder thread + closes
        # the sounddevice stream + closes the active AudioSource).
        try:
            self._audio_output.close()
        except Exception:  # pragma: no cover — best effort
            log.exception("audio output close failed")
        # Drop python-level refs to anything that could still pin big
        # numpy arrays from the cache, then force a collection. Numpy
        # buffers backed by malloc are released on dict.clear() above,
        # but an explicit ``gc.collect()`` guarantees the cycle
        # collector also walks the (small) graph of QObject parents
        # and disposes any leftover arrays held by transient closures
        # — keeps the visible "process exit" RAM curve clean.
        try:
            self._last_displayed = None
            gc.collect()
        except Exception:  # pragma: no cover — best effort
            pass

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
        """Cross-component signal wiring.

        Split into thematic sub-methods so each block is small enough
        to scan at a glance. Order doesn't matter for correctness
        (signal connections are independent) but follows the rough
        data-flow order: runtime monitor → viewport → controller →
        window → annotations.
        """
        self._wire_runtime_monitor()
        self._wire_viewport()
        self._wire_controller()
        self._wire_layer_stack()
        self._wire_main_window()
        self._wire_channel_menu()
        self._wire_compare()
        self._wire_color_and_zoom()
        self._wire_annotations()
        self._wire_keyboard_shortcuts()

    def _wire_runtime_monitor(self) -> None:
        """Runtime monitor (slice 5) → status bar. The monitor emits
        French user-facing strings; we route them straight to
        set_status so the user sees plain language whenever the
        watchdog catches a degraded condition.
        """
        self._runtime_monitor.playback_struggle.connect(self._window.set_status)
        self._runtime_monitor.memory_pressure.connect(self._window.set_status)
        self._runtime_monitor.frame_pacing_drop.connect(self._window.set_status)

    def _wire_viewport(self) -> None:
        """GL viewport signals: GPU late-bind + tile-isolate."""
        # The viewport emits gpu_renderer_detected exactly once per
        # session, on its first ``initializeGL``, carrying the real
        # ``glGetString(GL_RENDERER)``. We then re-run the auto-tune
        # with the actual GPU classification (slice 4 late-bind).
        self._window.viewer.gl.gpu_renderer_detected.connect(
            self._on_gpu_renderer_detected
        )

    def _wire_controller(self) -> None:
        """Controller → UI updates."""
        self._controller.frame_changed.connect(self._on_frame_changed)
        self._controller.state_changed.connect(self._on_state_changed)

    def _wire_layer_stack(self) -> None:
        """LayerStack signals → cache pre-fetch + viewport refresh.

        The cache hooks the same signals to invalidate its own
        contents, but invalidation alone leaves the viewport on
        whatever frame was last uploaded. This wiring drives the
        active redisplay: when the topmost-visible layer changes
        (œil toggle, reorder, layer added / removed), we
        re-prefetch around the playhead and re-pipe the display so
        the user sees the new content immediately rather than
        having to scrub the timeline.
        """
        self._layer_stack.layers_changed.connect(
            self._refresh_after_stack_change,
        )
        # Keep the file-watcher's directory list synced with the
        # current layer stack — every add / remove / replace fires
        # layers_changed, so this single connection is enough.
        self._layer_stack.layers_changed.connect(
            self._refresh_source_watcher,
        )
        # Close VideoSource handles for layers that just left the
        # stack — separate slot from the redisplay because the order
        # matters: close FIRST, redisplay AFTER (close on a removed
        # layer is the only place we can tell PyAV the file handle is
        # no longer needed; doing it post-redisplay would leak across
        # session swaps on Windows).
        self._layer_stack.layers_changed.connect(
            self._close_orphan_video_sources,
        )
        # Refresh the active audio source whenever the stack composition,
        # visibility, or per-layer audio fields change. Cheap when the
        # active layer is unchanged (just updates gain).
        self._layer_stack.layers_changed.connect(self._refresh_active_audio)
        self._layer_stack.visibility_changed.connect(
            lambda _id: self._refresh_active_audio(),
        )
        # ``layer_modified`` covers offset / trim drags that don't
        # change *which* layer is active but DO change the source-time
        # ↔ master-time mapping. Use the reseeking variant so the
        # audio jumps to the new alignment once. The plain
        # ``_refresh_active_audio`` on frame_changed must not reseek
        # on every tick (would thrash the ring buffer → stutter).
        self._layer_stack.layer_modified.connect(
            lambda _id: self._reseek_active_audio_for_layer_change(),
        )
        self._layer_stack.visibility_changed.connect(
            lambda _id: self._refresh_after_stack_change(),
        )
        self._layer_stack.layer_modified.connect(
            lambda _id: self._refresh_after_stack_change(),
        )
        self._layer_stack.focus_changed.connect(
            self._on_layer_focus_changed,
        )
        # Direct timeline → layer-bar playhead path. Without this the
        # layer bars only see playhead updates after the scrub frame
        # has round-tripped through the controller (frame_requested →
        # _on_scrub_requested → controller.seek → frame_changed →
        # _on_frame_changed → panel.set_playhead). On fast drags the
        # timeline cursor updates synchronously in its mouseMove (it
        # mutates ``_current`` itself before emitting), so the layer
        # bars visibly trail behind. Connecting the same signal
        # directly to the panel skips the round-trip and the two
        # cursors stay in lockstep — the controller path still fires
        # but ``set_playhead`` is idempotent on equal frames.
        panel = getattr(self._window, "_layer_panel", None)
        if panel is not None:
            self._window.timeline.frame_requested.connect(panel.set_playhead)
            self._window.viewer.gl.frame_requested.connect(panel.set_playhead)
            # Selection drives the bottom status bar's "Selected
            # layers" readout — the user clicked rows in the panel to
            # single them out, so the status bar follows that rather
            # than competing with transient ``set_status`` messages.
            panel.selection_changed.connect(
                self._on_layer_selection_changed,
            )
            # NB: timeline ↔ layer-bar alignment used to be a runtime
            # signal (``bar_inset_changed`` → ``set_content_insets``).
            # That's been replaced by ``MasterTimelinePanel`` which
            # holds both widgets under one layout that pins them to
            # the same horizontal axis — no signal needed.

    def _wire_main_window(self) -> None:
        """MainWindow signals → controller / app handlers."""
        w = self._window
        w.play_toggled.connect(self._on_play_toggled)
        w.step_clicked.connect(self._controller.step)
        w.jump_to_ends.connect(self._on_jump_to_ends)
        w.frame_requested.connect(self._on_scrub_requested)
        # Scrub gesture lifecycle — switch video decoders into a
        # keyframe-only fast-seek mode during drag (cuts ~5-15 ms
        # per seek down to ~1-3 ms on long-GOP H.264 / H.265), and
        # back to precise seeks on release with a re-request at the
        # final frame so the landing is exact.
        w.scrub_started.connect(self._on_scrub_started)
        w.scrub_finished.connect(self._on_scrub_finished)
        w.open_requested.connect(self._open_path)
        w.add_layer_requested.connect(self._on_add_layer_requested)
        w.save_session_requested.connect(self._on_save_session_requested)
        w.open_session_requested.connect(self._on_open_session_requested)
        # Export (v0.5.0) — both menu and transport button route here.
        w.export_requested.connect(self._open_export_dialog)
        w.transport.export_clicked.connect(self._open_export_dialog)
        # Save Frame As… (v1.2) — quick WYSIWYG snapshot. Menu only —
        # no transport button, the keyboard shortcut (Ctrl+Alt+S) is
        # the primary entry for power users.
        w.save_frame_requested.connect(self._open_save_frame_dialog)
        # New / Reload (v0.5.1) — same shape, two routes each.
        w.new_sequence_requested.connect(self._on_new_sequence)
        w.reload_sequence_requested.connect(self._on_reload_sequence)
        w.transport.reload_clicked.connect(self._on_reload_sequence)
        w.force_reload_sequence_requested.connect(self._on_force_reload_sequence)
        w.clear_cache_requested.connect(self._on_clear_cache_action)

        # Contact sheet — toggle + settings band wiring. The grid /
        # labels signals carry the new values directly; ``-1`` /
        # ``-1`` from the band means "auto".
        w.contact_sheet_toggle_requested.connect(self._on_contact_sheet_toggle)
        w.contact_sheet_grid_changed.connect(self._on_contact_sheet_grid_changed)
        w.contact_sheet_labels_toggled.connect(self.set_contact_sheet_labels)
        w.contact_sheet_divisor_changed.connect(
            self._on_contact_sheet_divisor_changed,
        )
        # Transport bar's contact-sheet button — same toggle flow as
        # the View menu entry. The chevron menu is populated lazily
        # on each open so checkmarks reflect the live state.
        w.transport.contact_sheet_toggled.connect(
            self._on_contact_sheet_toggle,
        )
        w.transport.contact_sheet_menu.aboutToShow.connect(
            self._build_transport_contact_sheet_menu,
        )
        # Edit menu — wire to the same chained handlers the keyboard
        # shortcut (Ctrl+Z / Ctrl+Shift+Z) uses, so menu and shortcut
        # produce identical behaviour (annotations first, layer
        # stack fallback).
        w.undo_requested.connect(self._undo_annotation)
        w.redo_requested.connect(self._redo_annotation)
        w.fps_changed.connect(self._on_fps_changed)
        w.direction_play_requested.connect(self._on_direction_play)
        w.mark_in_requested.connect(self._on_mark_in)
        w.mark_out_requested.connect(self._on_mark_out)
        w.set_in_at_requested.connect(self._on_set_in_at)
        w.set_out_at_requested.connect(self._on_set_out_at)
        w.clear_in_out_requested.connect(
            lambda: self._controller.set_in_out(None, None),
        )
        w.loop_mode_requested.connect(self._controller.set_loop_mode)
        # NB: the global ``show_transparency_toggled`` /
        # ``alpha_is_straight_toggled`` signals are gone — T and αS
        # are per-row buttons in the layer panel that mutate the
        # focused layer's state directly via ``LayerStack.update``.
        # The cache hooks ``layer_modified`` and reads the per-layer
        # flags during the next decode.
        # Recent-files menu uses callbacks into preferences.
        w.install_recent_provider(
            provider=self._prefs.recent_paths,
            clear_callback=self._prefs.clear_recent,
        )
        # Same hook for ``.session`` files — separate provider so the
        # two recent lists stay independent.
        w.install_recent_session_provider(
            provider=self._prefs.recent_sessions,
            clear_callback=self._prefs.clear_recent_sessions,
        )
        # Annotation save prompt — runs from MainWindow.closeEvent
        # before the window actually closes. Returning False from
        # this callback cancels the close.
        w.set_before_close_callback(self._prompt_save_annotations)

    def _wire_channel_menu(self) -> None:
        """Transport channel menu → app."""
        w = self._window
        # The transport's channel menu emits ``channel_selection_changed``
        # whenever the user picks a different active radio; we bridge
        # straight to ``set_channel_selection`` which handles cache +
        # display.
        w.channel_selection_changed.connect(self._on_channel_selection_changed)
        w.channel_mask_changed.connect(self._on_channel_mask_changed)
        w.transparency_bg_mode_changed.connect(self._on_transparency_bg_mode_changed)
        # Master audio (transport bar popup volume slider). Pushes
        # the gain to ``AudioOutput`` AND persists it so the
        # reviewer's level survives across launches. Mute is
        # implicit: slider==0 → gain=0 → silence in the callback.
        w.master_volume_changed.connect(self._on_master_volume_changed)
        # Hand the channel menu a way to ask the cache for per-group
        # cache-fill data. The menu polls this every 250 ms while it's
        # open so each row's progress pip reflects the alt-channel
        # background prefetch in real time. ``alt_channel_progress``
        # short-circuits to ``{}`` for multi-layer / no-AOV stacks,
        # which makes the pips paint as empty (= cleanly invisible).
        w.transport.channel_menu.set_progress_provider(
            self._cache.alt_channel_progress,
        )

    def _wire_compare(self) -> None:
        """Compare-mode signals: transport button + band + shortcuts."""
        from img_player.compare_handler import (
            set_layer_a,
            set_layer_b,
            set_mode,
            set_seam,
            swap_layers,
            toggle_compare,
            toggle_swap,
        )

        # Transport button → toggle compare on/off.
        self._window.transport.compare_toggled.connect(
            lambda: toggle_compare(self),
        )
        band = self._window.compare_band
        band.layer_a_picked.connect(lambda lid: set_layer_a(self, lid))
        band.layer_b_picked.connect(lambda lid: set_layer_b(self, lid))
        band.mode_picked.connect(lambda mode: set_mode(self, mode))
        band.seam_changed.connect(lambda seam: set_seam(self, seam))
        band.swap_toggled.connect(lambda: toggle_swap(self))
        band.swap_layers_requested.connect(lambda: swap_layers(self))
        band.close_requested.connect(lambda: toggle_compare(self))
        # Keyboard shortcuts (W / Ctrl+W) routed through MainWindow.
        self._window.compare_toggle_requested.connect(
            lambda: toggle_compare(self),
        )
        self._window.compare_swap_layers_requested.connect(
            lambda: swap_layers(self),
        )
        # Mouse-drag in the viewport → moves the seam while compare
        # is active. The filter intercepts left-press / move / release
        # ahead of the GL viewport's normal drag-scrub handler so the
        # gesture doesn't fight the timeline scrub. Held on ``self``
        # so the filter object isn't garbage-collected — Qt only
        # keeps a weak reference via installEventFilter.
        from img_player.compare_handler import _ViewportSeamFilter
        self._compare_viewport_filter = _ViewportSeamFilter(self)
        self._window.viewer.gl.installEventFilter(
            self._compare_viewport_filter,
        )

    def _wire_color_and_zoom(self) -> None:
        """ColorPanel + zoom combo → GL viewport."""
        # Zoom from the combo box → propagate to the GL viewport. The
        # wheel-zoom path (viewport → combo) is wired inside
        # MainWindow so app.py doesn't have to care.
        self._window.zoom_requested.connect(self._on_zoom_requested)
        self._window.exposure_step.connect(self._window.color_panel.bump_exposure)
        self._window.color_panel.color_params_changed.connect(self._on_color_params)
        # Register the OCIO hot-reload entry point for the
        # Preferences dialog. Without this the dialog falls back to a
        # "Restart required" banner — fine, but a restart is no longer
        # actually needed.
        self._window.set_ocio_reload_callback(self.reload_ocio_config)
        # Hand the live DiskCache to MainWindow so the Preferences →
        # Disk cache page can wire its "clear / usage" controls at
        # the running instance (not just persist prefs for next boot).
        self._window.set_disk_cache_handle(self._disk_cache)
        self._window.color_panel.unmarked_exr_save_requested.connect(
            self._on_unmarked_exr_save,
        )
        self._window.color_panel.unmarked_exr_clear_requested.connect(
            self._on_unmarked_exr_clear,
        )

    def _wire_annotations(self) -> None:
        """Annotation toolbar / overlay / store wiring + transport buttons."""
        tb = self._annotation_toolbar
        ov = self._annotation_overlay
        # Toolbar → overlay / app: the toolbar is the UI source of
        # truth for which tool / color / size is active; we forward
        # those to the overlay when they change. Undo / redo / clear
        # dispatch against the current frame's stack.
        tb.tool_changed.connect(ov.set_tool)
        tb.color_changed.connect(ov.set_color)
        tb.size_changed.connect(ov.set_size)
        tb.undo_requested.connect(self._undo_annotation)
        tb.redo_requested.connect(self._redo_annotation)
        tb.clear_requested.connect(self._clear_annotations)
        # Ephemeral mode (v0.4.1) — toolbar drives overlay (routing
        # decision) and manager (fade duration), and we persist the
        # preset on each change so a restart picks up where the user
        # left off.
        tb.ephemeral_mode_changed.connect(self._on_ephemeral_mode_changed)
        tb.ephemeral_duration_changed.connect(self._on_ephemeral_duration_changed)
        # Pen stabilizer (Lazy Mouse) — toolbar slider drives the
        # overlay's smoothing factor, and the level is persisted so
        # the next launch starts with the same setting.
        tb.pen_stabilizer_level_changed.connect(
            self._on_pen_stabilizer_level_changed,
        )
        # Push the initial factor to the overlay so a non-zero
        # restored level takes effect immediately, not only after the
        # user touches the slider.
        self._annotation_overlay.set_pen_stabilizer_factor(
            tb.pen_stabilizer_factor(),
        )
        # Restore the saved ghost state. Fire through the toolbar's
        # public setter so the overlay routing, glyph swap, eraser
        # disabling and border tint all resync via the wired signal.
        if self._prefs.ephemeral_mode_enabled:
            tb.set_ephemeral_mode(True)
        # Persist mode + float position to prefs so the toolbar
        # comes back where the user left it next session.
        tb.mode_changed.connect(self._on_toolbar_mode_changed)
        tb.floating_pos_changed.connect(self._on_toolbar_floating_pos_changed)
        # ✕ close button on the toolbar — same effect as the
        # transport's annotation toggle, so the user has a "this
        # panel takes too much space" exit right next to the pin.
        tb.close_requested.connect(self._toggle_toolbar_visible)
        # Store → timeline + transport: when the set of annotated
        # frames changes, the timeline repaints its markers and the
        # transport's prev/next-annotation buttons re-enable
        # themselves. Comments share the same marker/nav path.
        self._annotation_store.annotated_frames_changed.connect(
            self._on_annotated_frames_changed,
        )
        self._comment_store.commented_frames_changed.connect(
            self._on_annotated_frames_changed,
        )
        # Transport annotation buttons → store / toolbar.
        self._window.transport.annotation_toggle_clicked.connect(
            self._toggle_toolbar_visible,
        )
        self._window.transport.annotation_prev_clicked.connect(
            self._on_annotation_prev,
        )
        self._window.transport.annotation_next_clicked.connect(
            self._on_annotation_next,
        )
        self._window.transport.annotation_show_during_play_toggled.connect(
            self._on_annotation_show_during_play_toggled,
        )
        # Sync the button to the store's persisted flag at startup so
        # the visual matches whatever was saved last session.
        self._window.transport.set_annotation_show_during_play(
            self._annotation_store.show_during_playback,
        )
        # Initialise the overlay from the toolbar's defaults so a user
        # who jumps straight to drawing has the expected color/size.
        ov.set_color(tb.current_color())
        ov.set_size(tb.current_size())

    def _wire_keyboard_shortcuts(self) -> None:
        """Annotation keyboard shortcuts (D/P/E/A/G/[ /]/Undo/Redo).

        Parented to the window so they fire from anywhere in the app,
        not just when the GL viewport has focus. Existing convention
        (main_window.py): blocked while a QLineEdit owns focus, so
        typing a frame number can't toggle the pen.
        """
        from PySide6.QtGui import QKeySequence, QShortcut
        # (key, action) pairs — keeps the table easy to scan and
        # maintain compared to N near-identical QShortcut blocks.
        bindings: list[tuple[QKeySequence, object]] = [
            (QKeySequence("D"), self._toggle_toolbar_visible),
            (QKeySequence("P"), self._toggle_pen_mode),
            (QKeySequence("E"), self._toggle_eraser_mode),
            (QKeySequence("A"), self._toggle_show_annotations_during_play),
            # Ctrl+Z / Ctrl+Shift+Z handled by the Edit menu's
            # QActions (which also carry the shortcut). Registering
            # both here would route Ctrl+Z to two slots → double-undo.
            (QKeySequence("["), self._on_annotation_prev),
            (QKeySequence("]"), self._on_annotation_next),
            # G — toggle ephemeral mode (mnemonic "ghost"). Goes
            # through the toolbar so UI state and overlay routing
            # stay in sync.
            (QKeySequence("G"), self._toggle_ephemeral_mode),
        ]
        for keyseq, slot in bindings:
            QShortcut(keyseq, self._window, activated=slot)

    # ------------------------------------------------------------------ Handlers

    def _on_frame_changed(self, frame: int) -> None:
        """Top-level frame-changed dispatch — fast UI sync first,
        then pick the right display path.

        Split into 4 helpers so the per-frame hot path stays scannable:

        * :meth:`_sync_per_frame_widgets` — fast text / overlay /
          layer-panel updates (queued first so their paint events
          land before the GL viewport's heavyweight one).
        * :meth:`_try_compare_then_video` — compare and video early
          outs; either consumes the frame or returns False.
        * :meth:`_try_display_cached_frame` — cache hit fast path.
        * :meth:`_handle_no_coverage_gap` / :meth:`_handle_cache_miss`
          — terminal paths for the two remaining branches.
        """
        self._sync_per_frame_widgets(frame)
        if self._try_compare_then_video(frame):
            return
        if self._try_display_cached_frame(frame):
            return
        if self._handle_no_coverage_gap(frame):
            return
        self._handle_cache_miss(frame)

    def _sync_per_frame_widgets(self, frame: int) -> None:
        """Fast text / overlay / panel updates that should land
        BEFORE the heavyweight decode + GL upload.

        Order matters: queueing these paint events first means Qt
        processes them ahead of the GL viewport's paint, so the
        bottom info band's "Layer xxx" readout stays in sync with
        the timeline cursor during fast scrub. If we queued them
        last, the GL paint would hog the slot and the QLabels would
        visibly trail by a frame.
        """
        self._refresh_info_band_frames(frame)
        # Re-evaluate the active audio layer — the playhead may have
        # crossed a coverage boundary (entered / exited a clip) which
        # changes whether any layer should be feeding samples. Cheap
        # when the active layer is unchanged: just walks the stack
        # and early-returns.
        self._refresh_active_audio()
        self._window.timeline.set_current_frame(frame)
        # The viewport needs to know the current frame so the next
        # drag-scrub can use it as a base reference.
        self._window.viewer.gl.set_current_frame(frame)
        # Push the master playhead to the layer panel so each
        # LayerBar can draw the playhead line + snap to it during
        # drag.
        panel = getattr(self._window, "_layer_panel", None)
        if panel is not None:
            panel.set_playhead(frame)
        # The annotation overlay paints strokes for the current frame.
        self._annotation_overlay.set_current_frame(frame)
        # The comment panel re-renders its thread for the new frame.
        self._window.comment_panel.set_current_frame(frame)
        # Prev/next-annotation transport buttons depend on the
        # playhead position vs the annotated set — re-evaluate.
        self._refresh_annotation_nav_buttons()

    def _try_compare_then_video(self, frame: int) -> bool:
        """Run contact-sheet, compare-mode and video early-outs in turn.

        Returns ``True`` when one of the paths produced a complete
        frame upload (caller stops there). ``False`` means none was
        applicable / they fell through — caller continues to the
        regular cache lookup.
        """
        # Contact-sheet mode hijacks the upload BEFORE compare —
        # compare with the same enable flag would be ambiguous. The
        # contact sheet's decoder bypasses the master cache entirely
        # and composes a grid from every visible layer, re-aligned
        # to "frame 0".
        if self._contact_sheet_state.is_active() and self._render_contact_sheet(frame):
            self._last_displayed = frame
            self._wait_timer.stop()
            return True

        # Compare mode hijacks the upload entirely: A and B are
        # decoded independently and composed via numpy — the cache
        # + composite pipeline is bypassed.
        if self._compare_state.is_active() and _render_compare(self, frame):
            self._last_displayed = frame
            self._wait_timer.stop()
            return True

        # Video early-out: the topmost-visible layer is a video clip
        # whose long-GOP decode doesn't fit the random-access cache.
        # ``VideoSourceManager`` does seek-then-decode-forward with
        # a single-frame cache.
        displayed = (
            self._layer_stack.topmost_visible_at(frame)
            if self._layer_stack else None
        )
        if displayed is not None and displayed.is_video:
            try:
                arr = self._decode_video_layer(displayed, frame)
            except Exception:
                log.exception("video decode failed at master frame %d", frame)
                return False
            if arr is not None:
                self._display_array(arr)
                self._last_displayed = frame
                self._wait_timer.stop()
                return True
        return False

    def _try_display_cached_frame(self, frame: int) -> bool:
        """Cache hit fast path: if the RAM cache already has this
        frame, upload it and stop the wait timer. Returns ``True``
        when the upload happened, ``False`` on a cache miss."""
        arr = self._cache.get(frame)
        if arr is None:
            return False
        self._display_array(arr)
        self._last_displayed = frame
        self._wait_timer.stop()
        return True

    def _handle_no_coverage_gap(self, frame: int) -> bool:
        """When no layer covers this master frame (e.g. inter-layer
        gap, or playhead before the first scanned frame), upload the
        MISSING FRAME placeholder so the viewport doesn't keep
        showing the previously-displayed image.

        Returns ``True`` when a placeholder was rendered (terminal
        path), ``False`` when the frame IS covered by some layer
        (= regular cache miss, caller continues).
        """
        if not self._layer_stack:
            return False
        if self._layer_stack.topmost_visible_at(frame) is not None:
            return False
        try:
            self._show_gap_placeholder()
        except Exception:
            log.exception("[gap] failed to render placeholder at frame %d", frame)
        self._last_displayed = None
        self._wait_timer.stop()
        return True

    def _handle_cache_miss(self, frame: int) -> None:
        """Cache miss: show the nearest already-decoded fallback when
        playing (keeps motion visible) or start the wait timer when
        parked (so the display snaps to the exact requested frame
        as soon as the prefetcher lands it).
        """
        if self._controller.state.is_playing:
            fallback = self._nearest_cached_fallback(frame)
            if fallback is not None and fallback != self._last_displayed:
                fallback_arr = self._cache.get(fallback)
                if fallback_arr is not None:
                    self._display_array(fallback_arr)
                    self._last_displayed = fallback
        elif not self._wait_timer.isActive():
            self._wait_timer.start()

    def _on_contact_sheet_toggle(self) -> None:
        """Slot for both ``MainWindow.contact_sheet_toggle_requested``
        (= View menu) and ``transport.contact_sheet_toggled`` (=
        toolbar button). Either entry point lands here.

        Flips the state, syncs the View menu checkmark + transport
        button checkmark, and re-renders.
        """
        self.toggle_contact_sheet()
        self._window.set_contact_sheet_enabled(
            self._contact_sheet_state.enabled,
        )
        self._window.transport.set_contact_sheet_checked(
            self._contact_sheet_state.enabled,
        )
        self._sync_contact_sheet_menu_state()

    def _build_transport_contact_sheet_menu(self) -> None:
        """Populate the QMenu attached to the transport bar's
        contact-sheet button on each ``aboutToShow``. Mirrors the
        layout of ``View → Contact sheet settings`` so the user has
        the same options without leaving the toolbar.

        Re-uses the window's settings builder for consistency: the
        QActions there carry the wiring we want; we just clone them
        into the transport menu and trigger the parent action on
        click so signals route through one path.
        """
        from PySide6.QtGui import QAction  # noqa: PLC0415 — UI-only
        from PySide6.QtWidgets import QInputDialog  # noqa: PLC0415 — UI-only
        menu = self._window.transport.contact_sheet_menu
        menu.clear()
        cs = self._contact_sheet_state

        # Header — toggle entry. Useful when the user opened the
        # menu by accident and wants to flip the mode from inside it.
        toggle_act = QAction(
            "Active (contact sheet on)" if cs.enabled else "Activate", self._window,
            checkable=True,
        )
        toggle_act.setChecked(cs.enabled)
        toggle_act.triggered.connect(self._on_contact_sheet_toggle)
        menu.addAction(toggle_act)
        menu.addSeparator()

        # Auto + grid presets.
        auto_act = QAction("Auto (smart)", self._window, checkable=True)
        auto_act.setChecked(cs.cols is None and cs.rows is None)
        auto_act.triggered.connect(
            lambda: self._on_contact_sheet_grid_changed(-1, -1),
        )
        menu.addAction(auto_act)
        presets = (
            ("1 × 2", 1, 2), ("1 × 3", 1, 3), ("1 × 4", 1, 4),
            ("2 × 1", 2, 1), ("2 × 2", 2, 2), ("2 × 3", 2, 3),
            ("3 × 2", 3, 2), ("3 × 3", 3, 3), ("4 × 4", 4, 4),
        )
        for label, c, r in presets:
            act = QAction(label, self._window, checkable=True)
            act.setChecked(cs.cols == c and cs.rows == r)
            act.triggered.connect(
                lambda _chk, cc=c, rr=r: self._on_contact_sheet_grid_changed(cc, rr),
            )
            menu.addAction(act)

        def _ask_custom() -> None:
            cols, ok = QInputDialog.getInt(
                self._window, "Contact sheet — columns",
                "Number of columns:",
                value=cs.cols or 2, min=1, max=16,
            )
            if not ok:
                return
            rows, ok = QInputDialog.getInt(
                self._window, "Contact sheet — rows",
                "Number of rows:",
                value=cs.rows or 2, min=1, max=16,
            )
            if not ok:
                return
            self._on_contact_sheet_grid_changed(cols, rows)

        custom_act = QAction("Custom grid…", self._window)
        custom_act.triggered.connect(_ask_custom)
        menu.addAction(custom_act)
        menu.addSeparator()

        labels_act = QAction(
            "Show labels on tiles", self._window, checkable=True,
        )
        labels_act.setChecked(cs.show_labels)
        labels_act.triggered.connect(
            lambda checked: self.set_contact_sheet_labels(bool(checked)),
        )
        menu.addAction(labels_act)
        menu.addSeparator()

        # Output divisor sub-menu — same presets as the View menu.
        divisor_menu = menu.addMenu("Output size")
        for div, label in (
            (1, "Full (÷1)"), (2, "Half (÷2)"), (3, "Third (÷3)"),
            (4, "Quarter (÷4)"), (6, "Sixth (÷6)"), (8, "Eighth (÷8)"),
        ):
            act = QAction(label, self._window, checkable=True)
            act.setChecked(cs.output_divisor == div)
            act.triggered.connect(
                lambda _chk, d=div: self._on_contact_sheet_divisor_changed(d),
            )
            divisor_menu.addAction(act)

    def _on_contact_sheet_grid_changed(self, cols: int, rows: int) -> None:
        """Slot for ``MainWindow.contact_sheet_grid_changed``.

        ``-1`` / ``-1`` from the menu means "auto"; map to
        ``None`` for the state.
        """
        c: int | None = None if cols < 1 else cols
        r: int | None = None if rows < 1 else rows
        self.set_contact_sheet_grid(c, r)
        self._sync_contact_sheet_menu_state()

    def _on_contact_sheet_divisor_changed(self, divisor: int) -> None:
        """Slot for ``MainWindow.contact_sheet_divisor_changed``."""
        self.set_contact_sheet_output_divisor(divisor)
        self._sync_contact_sheet_menu_state()

    def _sync_contact_sheet_menu_state(self) -> None:
        """Push the current ContactSheetState to the window so the
        settings sub-menu's checkmarks match reality on next open."""
        self._window.set_contact_sheet_grid_state(
            self._contact_sheet_state.cols,
            self._contact_sheet_state.rows,
            self._contact_sheet_state.show_labels,
            self._contact_sheet_state.output_divisor,
        )

    def toggle_contact_sheet(self) -> None:
        """View → Contact sheet toggle entry point.

        Flips ``ContactSheetState.enabled``, drops the per-layer
        decode cache (so a fresh enable doesn't paint stale tiles
        from a previous session), forces a re-display at the
        current frame, and pushes the new state to QSettings so
        the choice persists.

        Also toggles two mode-coupled flags:

        * ``controller.set_always_advance(enabled)`` — bypasses the
          master-cache-stall guard so playback advances regardless
          of whether the regular composite is cached (the contact
          sheet has its own decoder so the master cache emptiness
          is no longer a reason to freeze the playhead).
        * Auto-exits compare mode when entering contact-sheet (the
          two are mutually exclusive — both hijack the GL upload).
        """
        new_enabled = not self._contact_sheet_state.enabled
        self._contact_sheet_state.enabled = new_enabled
        self._contact_sheet_decoder.invalidate()
        # Bypass cache-stall when in contact-sheet (per-layer decoder
        # owns the pixels) and re-engage on exit so regular playback
        # gets its smooth cache-bound behaviour back.
        self._controller.set_always_advance(new_enabled)
        # Auto-exit compare mode: the two are mutually exclusive
        # because both hijack the GL upload in ``_on_frame_changed``.
        if new_enabled and self._compare_state.enabled:
            from img_player.compare_handler import toggle_compare  # noqa: PLC0415
            toggle_compare(self)
        # Persist + re-render at the current frame so the user sees
        # the change immediately.
        self._prefs.contact_sheet_state = self._contact_sheet_state.to_dict()
        # Re-sync the compare button's enabled state — contact sheet
        # entry / exit changes the mutex condition.
        self._sync_compare_band_for_stack_change()
        cur = self._controller.state.current_frame
        self._last_displayed = None
        self._on_frame_changed(cur)

    def set_contact_sheet_grid(
        self, cols: int | None, rows: int | None,
    ) -> None:
        """Update the manual cols / rows pair. Passing ``None`` for
        either falls back to the auto grid (cf. ``effective_grid``).
        Re-renders the current frame so the change is immediate."""
        self._contact_sheet_state.cols = cols
        self._contact_sheet_state.rows = rows
        self._prefs.contact_sheet_state = self._contact_sheet_state.to_dict()
        if self._contact_sheet_state.is_active():
            cur = self._controller.state.current_frame
            self._last_displayed = None
            self._on_frame_changed(cur)

    def set_contact_sheet_labels(self, show: bool) -> None:
        """Toggle the per-tile name overlay."""
        self._contact_sheet_state.show_labels = bool(show)
        self._prefs.contact_sheet_state = self._contact_sheet_state.to_dict()
        if self._contact_sheet_state.is_active():
            cur = self._controller.state.current_frame
            self._last_displayed = None
            self._on_frame_changed(cur)

    def set_contact_sheet_output_divisor(self, divisor: int) -> None:
        """Resize the composite output: ``divisor=1`` = full
        resolution, ``2`` = half on each axis (= 1/4 pixel count),
        etc. Larger values trade detail for compose / upload speed.
        """
        self._contact_sheet_state.output_divisor = max(1, int(divisor))
        self._prefs.contact_sheet_state = self._contact_sheet_state.to_dict()
        if self._contact_sheet_state.is_active():
            cur = self._controller.state.current_frame
            self._last_displayed = None
            self._on_frame_changed(cur)

    def _render_contact_sheet(self, master_frame: int) -> bool:
        """Decode every visible layer at the contact-sheet offset,
        compose them into a grid, push to GL. Returns ``False`` when
        no layer was decodable (caller falls back to the normal
        path so the user still sees something).

        The contact-sheet "frame" is ``master_frame`` interpreted as
        an offset-from-zero, NOT a master-timeline frame: every
        visible layer's tile is decoded at
        ``layer.layer_in + master_frame`` (clamped to the layer's
        trim range) — so layers with different timeline offsets
        look like they all started at master 0.

        Grid + canvas sizing:

        * **Smart grid.** The cols × rows pick uses
          :func:`smart_grid_dimensions` with the GL viewport's
          current aspect — picks the layout that maximises
          per-tile area inside the canvas while keeping the
          composite aspect close to the viewport.
        * **Canvas aspect == viewport aspect.** The composite's
          width/height ratio matches the viewport so the GL
          viewport doesn't add an outer letterbox on top of the
          per-tile letterboxing inside the composite.
        """
        from img_player.contact_sheet import render_contact_sheet  # noqa: PLC0415 — cold path
        layers = [
            layer for layer in self._layer_stack.layers()
            if layer.visible
        ]
        if not layers:
            return False
        decodes = self._contact_sheet_decoder.decode_all(layers, master_frame)
        tiles = [arr for _, arr in decodes]
        if all(arr is None for arr in tiles):
            return False

        # Reference tile = first decoded layer. Drives both the
        # "image_aspect" hint for the smart grid (per-tile aspect)
        # and the output pixel resolution (= one tile's longest
        # dimension, scaled by cols/rows below).
        first_arr = next(arr for arr in tiles if arr is not None)
        src_h, src_w = first_arr.shape[:2]
        image_aspect = src_w / src_h if src_h > 0 else 16 / 9

        # Viewport aspect drives the smart grid + the composite's
        # outer aspect. ``self._window.viewer.gl`` is the QOpenGLWidget
        # — its current widget size is the canvas the user will see.
        gl_widget = self._window.viewer.gl
        vp_w = max(1, gl_widget.width())
        vp_h = max(1, gl_widget.height())
        canvas_aspect = vp_w / vp_h
        cols, rows = self._contact_sheet_state.effective_grid(
            len(layers), image_aspect, canvas_aspect=canvas_aspect,
        )

        # Compose target size: each tile gets ~source resolution
        # divided by the user's chosen ``output_divisor`` (1, 2, 3,
        # 4…). Divisor 1 = full source per tile (= memory- and CPU-
        # expensive on a large stack); 2 = quarter pixel count
        # (~4× faster compose + GL upload). The GL viewport
        # rescales to fit anyway, so smaller is usually fine for
        # review.
        div = max(1, self._contact_sheet_state.output_divisor)
        target_w = max(1, (cols * src_w) // div)
        target_h = max(1, (rows * src_h) // div)

        names = [layer.name for layer, _ in decodes]
        composite = render_contact_sheet(
            tiles,
            names=names,
            cols=cols,
            rows=rows,
            target_w=target_w,
            target_h=target_h,
            show_labels=self._contact_sheet_state.show_labels,
        )
        self._display_array(composite)
        return True

    def _refresh_info_band_frames(self, master_frame: int) -> None:
        """Push local-layer / global-timeline frame readouts to the
        bottom info band. Called from :meth:`_on_frame_changed`.

        Conventions:
        * **Layer** uses the source-frame numbering the user sees on
          disk — for ``shot.0220.png`` the readout shows ``220``,
          not "20th frame of the layer". Upper bound is the layer's
          trimmed source-range last frame (``layer_out``).
        * **Frame** uses the absolute master timeline number — same
          values the timeline ticks and the transport's frame readout
          show. Upper bound is ``last`` (the broad master range's
          last frame, also the rightmost timeline tick).
        """
        band = self._window.viewer.info_band
        # Local: source frame on the topmost visible layer.
        layer = (
            self._layer_stack.topmost_visible_at(master_frame)
            if self._layer_stack else None
        )
        if layer is not None and layer.covers(master_frame):
            band.set_local_frame(
                layer.source_frame_at(master_frame),
                layer.layer_out,
            )
            band.set_layer_name(layer.name)
        else:
            band.set_local_frame(None, None)
            band.set_layer_name(None)
        # Global: absolute master frame.
        panel = getattr(self._window, "_layer_panel", None)
        if panel is not None:
            _first, last = panel.broad_master_range()
        elif self._layer_stack:
            _first, last = self._layer_stack.master_range()
        elif self._controller.sequence is not None:
            last = self._controller.sequence.last_frame
        else:
            last = 0
        if last > 0:
            band.set_global_frame(master_frame, last)
        else:
            band.set_global_frame(None, None)

    def _refresh_status_selected_layers(self) -> None:
        """Push the panel's current selection to the bottom status bar.

        Renders nothing when no layer is selected (the segment goes
        invisible-via-empty-string), so an idle app shows a clean
        status bar rather than a stale placeholder. Multi-select uses
        a mid-dot separator — same convention as the compare-band
        dropdowns — and the same ``"N. name"`` prefix as
        :class:`LayerPanel`'s row column so the user can map "status
        bar text" back to "which row is this" without parsing.
        """
        window = self._window
        if not hasattr(window, "set_selected_layers"):
            # Defensive — older MainWindow stubs in unit tests don't
            # have the helper. No-op rather than crash.
            return
        panel = getattr(window, "_layer_panel", None)
        selected_ids: frozenset[str] = (
            panel.selected_ids() if panel is not None else frozenset()
        )
        if not selected_ids or self._layer_stack is None:
            window.set_selected_layers("")
            return
        parts: list[str] = []
        for i, layer in enumerate(self._layer_stack.layers()):
            if layer.id in selected_ids:
                name = layer.name or "(unnamed)"
                parts.append(f"{i + 1}. {name}")
        window.set_selected_layers(" · ".join(parts))

    def _on_layer_selection_changed(self, _selected_ids) -> None:  # type: ignore[no-untyped-def]
        """``LayerPanel.selection_changed`` → refresh the bottom status
        bar's selected-layers readout. Payload is ignored — we re-pull
        from the panel so the formatting walks the stack in the same
        order the rows are drawn.
        """
        self._refresh_status_selected_layers()

    def _nearest_cached_fallback(self, frame: int) -> int | None:
        """Pick the closest cached frame behind (for forward play) or ahead
        (for reverse play) of ``frame``. Returns ``None`` if the cache
        has no usable candidate.

        **Layer-aware filter.** Multi-layer stacks place different
        layers on different master-frame ranges. Without this filter
        the fallback would grab whichever frame happens to be cached
        nearest, even if its topmost-visible layer differs from the
        target's — producing a visible flicker of "wrong layer pixels"
        while the user scrubs backward into a region the cache hasn't
        decoded yet (the typical case: forward play warms layer A's
        cache, user scrubs back into layer B's territory, fallback
        falls onto an A-frame). We restrict the candidate set to
        frames whose topmost-visible layer matches the target's.
        """
        cached = self._cache.cached_frames()
        if not cached or self._layer_stack is None:
            return None
        target_layer = self._layer_stack.topmost_visible_at(frame)
        if target_layer is None:
            # ``frame`` is in a no-coverage void — caller paints the
            # gap placeholder; no fallback applies.
            return None
        # Filter cached frames to those whose topmost-visible layer is
        # the same as the target's. ``topmost_visible_at`` is cheap
        # (linear walk of the stack), and ``cached_frames()`` typically
        # holds a few hundred entries — total cost stays sub-ms.
        cached_same_layer = [
            f for f in cached
            if self._layer_stack.topmost_visible_at(f) is target_layer
        ]
        if not cached_same_layer:
            return None
        direction = self._controller.state.direction
        if direction >= 0:
            candidates = [f for f in cached_same_layer if f <= frame]
            return max(candidates) if candidates else min(cached_same_layer)
        candidates = [f for f in cached_same_layer if f >= frame]
        return min(candidates) if candidates else max(cached_same_layer)

    def _close_orphan_video_sources(self) -> None:
        _close_orphan_video_sources(self)

    def _refresh_after_stack_change(self) -> None:
        """Re-prefetch + re-display after a LayerStack mutation.

        Called from ``_wire_layer_stack`` on every layers_changed /
        visibility_changed / layer_modified emission. Three cases
        the viewport needs to handle:

        1. The new topmost-visible at the playhead is a different
           layer (e.g. user toggled œil, reordered, or added a
           layer above) → cache was invalidated for that range, we
           queue a fresh prefetch + re-emit ``frame_changed`` so the
           wait-timer falls back / displays once decode lands.
        2. No layer covers the playhead anymore (œil off on the
           only / last covering layer) → the cache won't decode
           anything; we explicitly clear the GL viewport so the
           user sees black instead of the previous frame.
        3. The displayed-layer didn't change (= a non-visual layer
           tweak) → the redisplay is a cheap idempotent no-op.

        Split into 4 helpers so each concern is independently
        scannable: compare-band bookkeeping, navigable-range sync,
        prefetch replan, current-frame redisplay-or-gap.
        """
        self._sync_compare_band_for_stack_change()
        # Selected-layer readout — a stack mutation can renumber rows
        # (insert / remove) or drop a previously-selected layer.
        self._refresh_status_selected_layers()
        # Layer mutation may have invalidated decoder caches (offset
        # change → different pixel for the same master frame).
        self._compare_decoder.invalidate()
        # Contact-sheet decoder lives parallel to compare's and
        # benefits from the same "drop the per-layer slot when
        # anything changed" invariant.
        self._contact_sheet_decoder.invalidate()
        self._sync_navigable_range_to_layer_panel()
        if self._controller.sequence is None:
            return
        # Re-plan prefetch over the FULL navigable range (not just the
        # close window). The MasterFrameCache wiped everything on the
        # ``layers_changed`` signal, so a 35-frame window would leave
        # every frame outside it grey on the cache bar until playback
        # rolls the playhead through them. ``replan_prefetch`` issues
        # a priority-ranked submit for every frame in the nav range;
        # ``request`` dedups against already-cached / already-pending
        # so the call is cheap and idempotent across repeated stack
        # mutations.
        self._controller.replan_prefetch()
        self._redisplay_current_frame_or_show_gap()

    def _sync_compare_band_for_stack_change(self) -> None:
        """Refresh the compare band's dropdown entries + gate the
        transport button on having ≥ 2 layers. Auto-exits compare
        mode when the stack drops below 2 layers (otherwise the
        band's "B" dropdown is stuck on a stale layer id).
        """
        from img_player.compare_handler import (  # noqa: PLC0415 — lazy: cold path
            refresh_band_layers,
            toggle_compare,
        )
        layer_count = len(list(self._layer_stack.layers()))
        # Compare needs ≥ 2 layers AND it can't coexist with
        # contact-sheet (both hijack the GL upload). Greying it out
        # when contact-sheet is active makes the mutex visible to
        # the user rather than letting them click a button that
        # would silently do nothing.
        compare_allowed = (
            layer_count >= 2
            and not self._contact_sheet_state.enabled
        )
        self._window.transport.set_compare_enabled(compare_allowed)
        if not compare_allowed and self._compare_state.enabled:
            toggle_compare(self)
        refresh_band_layers(self)

    def _sync_navigable_range_to_layer_panel(self) -> None:
        """Push the layer-panel's broad master range to timeline /
        controller / GL viewport, and update the timeline's gap /
        disk-availability overlays.

        The broad master range is the union of every layer's
        source-potential — wider than the loaded sequence's bounds
        when later-added layers extend past the original sequence
        end. Without this sync the timeline keeps the loaded
        sequence's range while layer bars use the broad range, and
        the two scrubbers paint at different scales (dragging the
        timeline moves the layer-bar playhead at a visibly different
        speed).
        """
        panel = getattr(self._window, "_layer_panel", None)
        if panel is None or not self._layer_stack:
            return
        first, last = panel.broad_master_range()
        if last > first:
            self._window.timeline.set_range(first, last)
            # Tell the controller the same. Without this its
            # ``_clamp_to_sequence`` (called by ``seek``) caps
            # scrubbing at the controller's held sequence bounds.
            self._controller.set_navigable_range(first, last)
            # Same range to the GL viewport so its drag-scrub
            # clamps at the boundaries — without it the user can
            # drag the cursor past the last frame.
            self._window.viewer.gl.set_navigable_range(first, last)
        # Multi-layer gaps painted in a distinct grey so the user
        # sees "no layer covers these frames" at a glance instead
        # of mistaking them for not-yet-cached slots. Use the broad
        # master range so gaps past the last layer's trimmed OUT
        # also paint grey.
        self._window.timeline.set_gap_frames(
            self._layer_stack.gap_frames(bounds=(first, last)),
        )
        # Disk-tier availability — paint frames that already have a
        # blob on disk in dim orange so the user sees the session is
        # warm before they scrub. Synchronous lookup (~50 ms for a
        # 1000-frame sequence). Cheap relative to the rest of
        # ``_refresh_after_stack_change``.
        try:
            disk_frames = self._cache.disk_available_master_frames()
            self._window.timeline.set_disk_available_frames(disk_frames)
        except Exception:  # pragma: no cover — defensive
            log.exception(
                "disk-available probe failed (non-fatal — timeline "
                "just won't pre-paint)",
            )

    def _redisplay_current_frame_or_show_gap(self) -> None:
        """After cache invalidation, either re-trigger
        :meth:`_on_frame_changed` so the freshly-decoded frame lands,
        or upload the gap placeholder when no layer covers the
        playhead anymore."""
        cur = self._controller.state.current_frame
        if self._layer_stack.topmost_visible_at(cur) is None:
            # No coverage at the playhead — wipe the viewport so the
            # stale image doesn't linger.
            try:
                self._show_gap_placeholder()
            except Exception:
                log.exception("[stack] failed to render placeholder on empty frame")
            self._last_displayed = None
            return
        # Force re-upload even if the master frame number didn't
        # change (the underlying source did).
        self._last_displayed = None
        self._on_frame_changed(cur)

    def _on_layer_focus_changed(self, layer_id: str) -> None:
        """Repopulate the channel menu from the newly-focused layer.

        Each layer carries its own channel selection / layout mode /
        labels-visible (Phase 5a). Switching focus rebuilds the
        menu to reflect the focused layer's saved state without
        firing the menu's ``channel_selection_changed`` signal —
        that would trigger a redundant ``set_channel_selection``
        which would no-op for matching state but also unnecessarily
        invalidate the layer's cache range.

        Empty ``layer_id`` clears focus (= no layers loaded); the
        menu keeps its last state.
        """
        if not layer_id:
            return
        layer = self._layer_stack.find(layer_id)
        if layer is None:
            return
        transport = self._window.transport
        # Block transport's signals during the rebuild so the
        # menu's selection_changed (emitted by both
        # set_available_channels and restore_channel_state) doesn't
        # round-trip back to the layer it just came from.
        transport.blockSignals(True)
        try:
            transport.set_available_channels(layer.sequence.channel_names)
            sel = layer.channel_selection
            if sel is not None:
                # The layer was previously focused and the user picked
                # an explicit channel — restore that pick.
                transport.restore_channel_state(sel.active.label)
            else:
                # Fresh layer: ``set_available_channels`` defaulted the
                # menu to the first group of this sequence. Mirror that
                # onto the layer so its ``channel_selection`` matches
                # what the menu shows. Without this, the cache's
                # signature uses the ``"_"`` (None) fallback while
                # ``alt_channel_progress`` queries with the menu's
                # actual label — they mismatch and the channel button's
                # progress bar reports 0 % even though frames *are*
                # being cached. Direct assignment (not via
                # ``stack.update``) avoids firing ``layer_modified``,
                # which would invalidate a cache range that's empty
                # for a brand-new layer anyway.
                menu_sel = transport.channel_menu.current_selection()
                if menu_sel is not None:
                    layer.channel_selection = menu_sel
        finally:
            transport.blockSignals(False)
        # Sync app-level fallback so legacy export-snapshot code keeps
        # working.
        if layer.channel_selection is not None:
            self._channel_selection = layer.channel_selection

    def _redisplay_current(self) -> None:
        """Re-run the display path on the current frame.

        Used by display-time-only changes (compare-mode toggles,
        per-layer alpha tweaks): no cache invalidation, no decode
        work — just re-pipe whatever's already available through
        the display path so the new param takes effect immediately.

        Compare-mode hijacks the upload when active, so we route
        through its render path first; otherwise fall back to the
        cache lookup. No-op when nothing's cached / no compare A/B
        decoded yet.
        """
        cur = self._controller.state.current_frame
        if self._compare_state.is_active():
            if _render_compare(self, cur):
                return
        arr = self._cache.get(cur)
        if arr is not None:
            self._display_array(arr)

    def _try_display_current_frame(self) -> None:
        frame = self._controller.state.current_frame
        arr = self._cache.get(frame)
        if arr is not None:
            self._display_array(arr)
            self._last_displayed = frame
            self._wait_timer.stop()
            # The Channels panel was retired (replaced by the
            # Comments tab — see ui/main_window.py). Channel info
            # still lives in the transport bar's combo + the four
            # R/G/B/A mute toggles, so the user has full visibility
            # without a dedicated panel.

    def _show_gap_placeholder(self) -> None:
        """Clear the viewport to black for any "no layer covers" frame.

        Reserved cases: gaps between layers, playhead trimmed past a
        layer's OUT, every covering layer toggled hidden, etc. — the
        common factor is that no visible layer claims this master
        frame. The user wants a plain black frame here (= "background"),
        not the rich MISSING FRAME graphic.

        The MISSING FRAME placeholder stays meaningful for its
        original semantic: a covering layer's source file is missing
        on disk (decode failed). That path goes through the cache's
        ``_missing`` set + ``_pre_mark_missing`` / decode-failure
        substitution and shows up via the normal ``cache.get`` →
        ``_display_array`` route — independent from this function.
        """
        self._window.viewer.gl.clear_image()

    def _pick_active_audio_layer(self):  # type: ignore[no-untyped-def]
        return _pick_active_audio_layer(self)

    def _reseek_active_audio_for_layer_change(self) -> None:
        _reseek_active_audio_for_layer_change(self)

    def _refresh_active_audio(self) -> None:
        _refresh_active_audio(self)

    def _current_layer_time(self, layer) -> float | None:  # type: ignore[no-untyped-def]
        return _current_layer_time(self, layer)

    def _decode_video_layer(self, layer, master_frame: int):  # type: ignore[no-untyped-def]
        return _decode_video_layer(self, layer, master_frame)

    def _display_array(self, arr) -> None:  # type: ignore[no-untyped-def]
        """Push a decoded buffer to the GL viewport.

        The viewport only handles RGB/RGBA, so we trim any extra
        channels the cache may have decoded for a multi-channel
        selection that the user later narrowed back to a single
        group. Historical contact-sheet compose path was retired in
        v1.2 — the buffer is uploaded as-is.
        """
        if arr.shape[2] > 4:
            arr = arr[:, :, :4]
        self._window.viewer.gl.set_frame(arr)

    def set_channel_selection(self, selection: ChannelSelection) -> None:
        """Switch to a new channel selection (single + optional tiles)."""
        set_channel_selection(self, selection)

    def _on_state_changed(self, state: PlaybackState) -> None:
        self._window.transport.update_from_state(state)
        # Timeline needs in/out markers and the fps for its timecode labels.
        self._window.timeline.set_in_out(state.in_frame, state.out_frame)
        self._window.timeline.set_fps(state.fps)
        # Bottom info band fps readout follows the controller fps.
        self._window.viewer.info_band.set_fps(state.fps)
        # The layer-panel bars need the master in/out so their drag
        # snap targets reflect the playback range.
        panel = getattr(self._window, "_layer_panel", None)
        if panel is not None:
            panel.set_master_in_out(state.in_frame, state.out_frame)
        # Tell the annotation overlay whether to render: hidden during
        # play unless the show-during-playback toggle is on.
        self._annotation_overlay.set_is_playing(state.is_playing)
        # Drive the audio output. Three cases:
        # - play/pause flip → call play() / pause() and reseek audio
        #   to the current playhead so the user hears from the right
        #   spot, not the residue of a previous run.
        # - large playhead jump (= seek, scrub, J/K-step) → reseek
        #   audio so the feeder picks up at the new time.
        # - small +1 step while playing (= normal tick) → leave the
        #   audio feeder alone; it runs free.
        # set_speed compares session FPS vs the active video layer's
        # native FPS — anything else than 1.0× ratio mutes (option
        # 2(a): no time-stretch).
        active = self._pick_active_audio_layer()
        if active is not None and active.video_metadata is not None \
                and active.video_metadata.fps is not None:
            native = float(active.video_metadata.fps)
            ratio = state.fps / native if native > 0 else 1.0
            self._audio_output.set_speed(ratio)
        else:
            self._audio_output.set_speed(1.0)
        prev_play = getattr(self, "_last_audio_play_state", False)
        prev_frame = getattr(self, "_last_audio_synced_frame", None)
        if state.is_playing != prev_play:
            # Transition: pause → play OR play → pause. On a play
            # transition, seek audio to the current frame so the user
            # hears the right offset (the feeder may have stale data
            # from a previous run).
            if state.is_playing and active is not None:
                t = self._current_layer_time(active)
                if t is not None:
                    self._audio_output.seek(t)
            if state.is_playing:
                self._audio_output.play()
            else:
                self._audio_output.pause()
        else:
            # Same play state — check for a large frame jump (= scrub
            # / step). Tolerance ±2 covers normal forward / reverse
            # play ticks; everything else is a seek.
            if (
                prev_frame is not None
                and abs(state.current_frame - prev_frame) > 2
                and active is not None
            ):
                t = self._current_layer_time(active)
                if t is not None:
                    self._audio_output.seek(t)
        self._last_audio_play_state = state.is_playing
        self._last_audio_synced_frame = state.current_frame

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
        # Reflect the new state on the transport's ✏ button AND the
        # fullscreen bar's twin so the checkable visual matches
        # reality whether the user toggled via D, the toolbar's
        # hide-on-pen-off, the transport button, the fs button, or
        # the toolbar's ✕ close.
        self._window.transport.set_annotation_toggle_active(not was_visible)
        self._window.set_fs_annotation_toggle_active(not was_visible)

    def _on_annotation_prev(self) -> None:
        """``[`` or transport prev button — seek to the highest
        noted frame (annotation OR comment) strictly less than the
        current frame. No-op if none (button is disabled in that
        case but we double-check to keep the keyboard path robust)."""
        cur = self._controller.state.current_frame
        candidates = [f for f in self._noted_frames() if f < cur]
        if candidates:
            self._controller.seek(max(candidates))

    def _on_annotation_next(self) -> None:
        """``]`` or transport next button — seek to the lowest noted
        frame strictly greater than the current frame."""
        cur = self._controller.state.current_frame
        candidates = [f for f in self._noted_frames() if f > cur]
        if candidates:
            self._controller.seek(min(candidates))

    def _noted_frames(self) -> frozenset[int]:
        """Union of frames that carry annotations OR comments.

        The timeline marker and the prev/next transport buttons treat
        both kinds of notes uniformly: a frame with a comment but no
        stroke still gets a marker, and the user can jump to it.
        """
        return (
            self._annotation_store.annotated_frames()
            | self._comment_store.commented_frames()
        )

    def _on_annotated_frames_changed(self) -> None:
        """Pushes the new noted-frames sets to the timeline (markers
        repaint) and re-evaluates whether the transport's prev/next
        buttons are reachable from the current playhead.

        Connected to BOTH the annotation store's
        ``annotated_frames_changed`` and the comment store's
        ``commented_frames_changed`` — either kind of note flipping
        in or out of a frame triggers the same recompute path.

        The timeline draws the two kinds of notes with distinct
        markers (orange triangle for annotations, blue dot for
        comments) so the user can tell them apart at a glance.
        Prev/next nav still works on the union — they're both
        "noted frames the user might want to jump to".
        """
        self._window.timeline.set_annotated_frames(
            self._annotation_store.annotated_frames()
        )
        self._window.timeline.set_commented_frames(
            self._comment_store.commented_frames()
        )
        self._refresh_annotation_nav_buttons()

    def _refresh_annotation_nav_buttons(self) -> None:
        """Enable / disable the transport prev/next buttons based on
        the current playhead position relative to the noted set.
        Called from ``_on_annotated_frames_changed`` (set changed)
        and ``_on_frame_changed`` (playhead moved)."""
        noted = self._noted_frames()
        cur = self._controller.state.current_frame
        prev_avail = any(f < cur for f in noted)
        next_avail = any(f > cur for f in noted)
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
        self._apply_annotation_show_during_play(new)

    def _on_annotation_show_during_play_toggled(self, on: bool) -> None:
        """Slot for the transport's 👁 button — same effect as the
        ``A`` shortcut, but driven by the explicit checked state of
        the button so click + uncheck land deterministically."""
        self._apply_annotation_show_during_play(bool(on))

    def _apply_annotation_show_during_play(self, on: bool) -> None:
        """Single source of truth for the show-annotations-during-play
        flag. Updates the store, repaints the overlay, syncs the
        transport button, and surfaces a status message — so both
        entry points (keyboard and toolbar) produce identical UX."""
        if self._annotation_store.show_during_playback == on:
            # Still re-sync the button in case the toolbar got out
            # of step (e.g. user double-toggled rapidly).
            self._window.transport.set_annotation_show_during_play(on)
            return
        self._annotation_store.show_during_playback = on
        self._annotation_overlay.update()
        self._window.transport.set_annotation_show_during_play(on)
        self._window.set_status(
            f"Annotations pendant lecture : {'visibles' if on else 'masquées'}"
        )

    def _undo_annotation(self) -> None:
        """``Ctrl+Z`` — undo the most recent edit.

        Routing rules, in priority order:

        1. Ephemeral mode active → pull back the last live stroke
           (special case from v0.4.1: ephemerals don't share the
           per-frame undo stack).
        2. The current frame has an annotation undo entry → pop it.
        3. The layer stack has a history entry (= the user just
           added / removed / reordered / toggled / dragged a layer
           — including drop-replace and add-layer drops) → revert it.
        4. Nothing to undo anywhere → status message.

        Falling through to the layer stack means the same Ctrl+Z
        keystroke covers both feature areas. Annotations stay
        prioritised so a stray Ctrl+Z while drawing doesn't
        unexpectedly tear down a layer the user is actively reviewing.
        """
        if self._annotation_toolbar.is_ephemeral_mode():
            if not self._ephemeral_manager.kill_last():
                self._window.set_status(
                    "Éphémère : aucun trait vivant à supprimer"
                )
            return
        frame = self._controller.state.current_frame
        if self._annotation_store.undo(frame):
            return
        if self._layer_stack.can_undo():
            self._layer_stack.undo()
            self._window.set_status("Layer change undone")
            return
        self._window.set_status("Rien à annuler")

    def _redo_annotation(self) -> None:
        """``Ctrl+Y`` / ``Ctrl+Shift+Z`` — redo the most recent undo.

        Same priority chain as :meth:`_undo_annotation`: annotations
        first (for symmetry — if you can undo, you can redo), layer
        stack second. Ephemeral mode swallows redo by design — faded
        strokes don't come back.
        """
        if self._annotation_toolbar.is_ephemeral_mode():
            return
        frame = self._controller.state.current_frame
        if self._annotation_store.redo(frame):
            return
        if self._layer_stack.can_redo():
            self._layer_stack.redo()
            self._window.set_status("Layer change redone")
            return
        self._window.set_status("Rien à rétablir")

    def _clear_annotations(self) -> None:
        """Toolbar's Clear button — context-sensitive (v0.4.1).

        * Persistent mode: wipe every stroke on the current frame,
          each removal landing as its own undo entry.
        * Ephemeral mode: wipe every live ephemeral stroke instantly.
          Not undoable (matches the rest of the ephemeral semantics).
        """
        if self._annotation_toolbar.is_ephemeral_mode():
            count = self._ephemeral_manager.clear_all()
            if count == 0:
                self._window.set_status(
                    "Éphémère : aucun trait vivant à effacer"
                )
            else:
                plural = "s" if count > 1 else ""
                self._window.set_status(
                    f"Éphémère : {count} trait{plural} effacé{plural}"
                )
            return
        frame = self._controller.state.current_frame
        count = self._annotation_store.clear_frame(frame)
        if count == 0:
            self._window.set_status("Annotation : aucune annotation à effacer")
        else:
            plural = "s" if count > 1 else ""
            self._window.set_status(
                f"Annotation : {count} trait{plural} effacé{plural} "
                f"(Ctrl+Z pour annuler)"
            )

    # ------------------------------------------------------------------ Ephemeral wiring (v0.4.1)

    def _on_ephemeral_mode_changed(self, on: bool) -> None:
        """Toolbar's 👻 toggled. Mirror to overlay + status hint and
        persist so the next launch lands in the same mode."""
        self._annotation_overlay.set_ephemeral_mode(on)
        self._prefs.ephemeral_mode_enabled = on
        if on:
            self._window.set_status(
                "Mode éphémère activé — les traits s'effacent tout seuls "
                "(non sauvegardés)"
            )
        else:
            self._window.set_status("Mode persistant rétabli")

    def _on_ephemeral_duration_changed(self, seconds: float) -> None:
        """Toolbar's preset dot clicked. Push duration to the manager
        and persist the preset index for next session."""
        self._ephemeral_manager.set_duration(seconds)
        self._prefs.ephemeral_duration_preset = (
            self._annotation_toolbar.ephemeral_preset_index()
        )

    def _on_pen_stabilizer_level_changed(self, level: int) -> None:
        """Toolbar's stabilizer slider moved. Apply the matching EMA
        factor to the overlay and persist the level for next session.
        """
        self._annotation_overlay.set_pen_stabilizer_factor(
            self._annotation_toolbar.pen_stabilizer_factor(),
        )
        self._prefs.pen_stabilizer_level = level

    def _toggle_ephemeral_mode(self) -> None:
        """``G`` keyboard shortcut — flip the toolbar toggle.

        We go through the toolbar (not directly through the overlay)
        so the toolbar's UI state — checked button, preset bar
        visibility, pen glyph swap, eraser disabling — stays in sync.
        The toolbar emits the change-signal which app.py routes back
        to overlay + status bar via the wiring above.
        """
        new_state = not self._annotation_toolbar.is_ephemeral_mode()
        self._annotation_toolbar.set_ephemeral_mode(new_state)

    def _prompt_save_annotations(self) -> bool:
        """Close-time prompt: ask whether to save review notes
        (annotations + comments — both share the sidecar).

        Called from MainWindow.closeEvent right before the window
        actually closes. Returns:

        * ``True`` — close proceeds (user picked Save or Don't Save).
        * ``False`` — close is cancelled (user picked Cancel).

        The dialog is skipped entirely when:

        * No sequence is open (``_annotations_path`` is ``None``).
        * Both stores are clean (no mutations since the last load /
          save) — nothing meaningful to save, no nag.
        """
        if self._annotations_path is None or self._annotations_basename is None:
            return True
        annotations_dirty = self._annotation_store.is_dirty()
        comments_dirty = self._comment_store.is_dirty()
        if not annotations_dirty and not comments_dirty:
            return True

        existing = self._annotations_path.exists()
        # Phrase the body to reflect what actually changed — gives
        # the user a richer picture of why they're being asked.
        if annotations_dirty and comments_dirty:
            body = "Des annotations et commentaires ont été modifiés."
        elif annotations_dirty:
            body = "Des annotations ont été modifiées."
        else:
            body = "Des commentaires ont été modifiés."

        if existing:
            informative = (
                f"Sauvegarder dans {self._annotations_path} "
                f"écrasera le fichier existant."
            )
        else:
            informative = (
                f"Le fichier {self._annotations_path} sera créé."
            )

        box = QMessageBox(self._window)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Sauvegarder les notes de review ?")
        box.setText(body)
        box.setInformativeText(informative)
        save_btn = box.addButton(
            "Sauvegarder", QMessageBox.ButtonRole.AcceptRole
        )
        discard_btn = box.addButton(
            "Ne pas sauvegarder", QMessageBox.ButtonRole.DestructiveRole
        )
        cancel_btn = box.addButton("Annuler", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save_btn)
        box.exec()
        clicked = box.clickedButton()

        if clicked is cancel_btn:
            return False  # user backed out of the close
        if clicked is save_btn:
            # Save both stores. Order matters: annotations first
            # (they wrote the "frames" sub-tree); save_comments then
            # reads that file and writes "comments" alongside it.
            # Each is best-effort; we record per-store success so a
            # partial failure (e.g. read-only dossier corrupted
            # halfway) leaves the other store's dirty flag intact
            # and the user can re-try.
            if annotations_dirty:
                ok_anno = save_annotations(
                    self._annotations_path,
                    self._annotation_store,
                    basename=self._annotations_basename,
                )
                if ok_anno:
                    self._annotation_store.mark_clean()
                else:
                    log.warning(
                        "[annotations] save failed at close — "
                        "underlying error already logged."
                    )
            if comments_dirty:
                ok_com = save_comments(
                    self._annotations_path,
                    self._comment_store,
                    basename=self._annotations_basename,
                )
                if ok_com:
                    self._comment_store.mark_clean()
                else:
                    log.warning(
                        "[comment] save failed at close — "
                        "underlying error already logged."
                    )
        # Falls through for discard_btn or save_btn (success or fail):
        # in either case we allow the close to proceed.
        return True

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

    def _on_transparency_bg_mode_changed(self, mode: int) -> None:
        """Forward the BG picker to the GL viewport + persist the
        choice. Mode is 0..3 (checker / black / grey / white)."""
        self._window.viewer.gl.set_color_params(transparency_bg_mode=int(mode))
        try:
            self._prefs.transparency_bg_mode = int(mode)
        except Exception:
            log.exception("[prefs] failed to persist transparency_bg_mode")

    def _on_master_volume_changed(self, gain: float) -> None:
        """Transport slider → audio output + persist. Linear gain in
        [0.0, 1.0]."""
        try:
            g = max(0.0, min(1.0, float(gain)))
        except (TypeError, ValueError):
            return
        try:
            self._audio_output.set_master_gain(g)
        except Exception:
            log.exception("[audio] failed to set master gain")
        try:
            self._prefs.master_volume = g
        except Exception:
            log.exception("[prefs] failed to persist master_volume")

    def _on_zoom_requested(self, factor: object) -> None:
        """Forward a zoom request from the combo to the GL viewport.

        ``factor`` is either ``None`` (= fit-to-window) or a float.
        """
        try:
            zoom: float | None = None if factor is None else float(factor)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return
        self._window.viewer.gl.set_zoom(zoom)

    def _on_channel_selection_changed(self, selection: object) -> None:
        """Apply a fresh :class:`ChannelSelection` from the transport menu."""
        on_channel_selection_changed(self, selection)


    def _on_scrub_started(self) -> None:
        """User started dragging the timeline. Flip video decoders
        into keyframe-only fast-seek mode so long-GOP H.264 / H.265
        seeks return a clean keyframe in ~1-3 ms instead of paying
        the full 5-15 ms decode-forward cost on every drag tick.

        Image-sequence layers are untouched — their frame cache
        keeps them snappy already. Only the video path benefits.
        """
        try:
            self._video_sources.set_fast_seek_all(True)
        except Exception:
            log.exception("[scrub] could not enable fast seek on video decoders")

    def _on_scrub_finished(self) -> None:
        """User released the mouse. Restore precise seeks and force
        a re-decode at the final landing frame so the post-scrub
        frame is exact (the cache was cleared on the transition,
        so the next ``decode_at`` call goes through the worker's
        sync path).
        """
        try:
            self._video_sources.set_fast_seek_all(False)
        except Exception:
            log.exception("[scrub] could not disable fast seek on video decoders")
        # Re-paint with a precise decode at the current frame. The
        # cache clear inside ``set_fast_seek_all`` guarantees this
        # request can't return a stale approximate frame.
        try:
            current = int(self._controller.state.current_frame)
            self._show_best_available(current)
        except Exception:
            log.exception("[scrub] post-release re-decode failed")

    def _on_scrub_requested(self, frame: int) -> None:
        """Timeline scrub: update the display immediately from the cache, but
        defer the full seek (which re-does prefetch planning) to coalesce
        rapid slider events.

        Auto-pause during scrub: when the user drags the timeline while
        the controller is playing, the play-tick and the scrub fight
        for the decoder cursor — each fires every ~20 ms in opposite
        directions, the threaded video decoder seeks backward on every
        tick, throughput collapses. Pause on the first scrub event of
        a gesture, then ``_apply_pending_seek`` resumes once the
        debounce window closes (= the user stopped dragging). 20 ms is
        short enough that a flick-of-the-wrist scrub never feels like
        a deliberate stop.
        """
        if self._controller.state.is_playing:
            self._scrub_was_playing = True
            self._controller.pause()
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
        # And the comment panel — same reason; the thread should
        # follow the cursor in real time.
        self._window.comment_panel.set_current_frame(frame)
        # The bottom info band's Layer / Frame readouts also follow
        # the scrub — without this they only refresh after the
        # debounced seek lands and ``frame_changed`` finally fires.
        self._refresh_info_band_frames(frame)
        # Defer the expensive part.
        self._pending_seek = frame
        self._scrub_debounce.start()

    def _apply_pending_seek(self) -> None:
        if self._pending_seek is None:
            # Even with no pending seek, an active scrub-pause may need
            # to resume — the debounce timer fires 20 ms after the last
            # scrub event regardless of whether we coalesced one.
            if getattr(self, "_scrub_was_playing", False):
                self._scrub_was_playing = False
                self._controller.play()
            return
        frame = self._pending_seek
        self._pending_seek = None
        self._controller.seek(frame)
        # Resume playback if we paused for the scrub. Done after the
        # seek so play() picks up at the new playhead, not the
        # pre-scrub one.
        if getattr(self, "_scrub_was_playing", False):
            self._scrub_was_playing = False
            self._controller.play()

    def _show_best_available(self, frame: int) -> None:
        # Contact-sheet early-out — same reasoning as the compare
        # check below. Without this, the scrub path uploads the
        # plain cached composite of the topmost-visible layer, which
        # overwrites the contact-sheet grid until the debounced seek
        # eventually lets ``_on_frame_changed`` repaint it. The user
        # sees a flicker (or, when scrubbing then pressing play, the
        # last scrub upload sticks around for one tick) — exactly the
        # "the cache wins over the contact sheet" symptom reported.
        if self._contact_sheet_state.is_active():
            if self._render_contact_sheet(frame):
                self._last_displayed = frame
                return
            # else: every layer's decode failed — fall through to the
            # cache so the user still sees something instead of black.

        # Compare-mode early-out: same hijack as ``_on_frame_changed``.
        # Without this, scrubbing while compare is active fires
        # ``_show_best_available`` from the scrub fast-path (which
        # bypasses the controller's debounced seek) and uploads the
        # plain cache pixel data — i.e. only layer A, no wipe — until
        # the debounced seek lands and ``_on_frame_changed`` finally
        # paints the compare result. The user sees a flicker between
        # "just A" and "A vs B at the seam" with each scrub tick.
        # Routing through ``render_compare`` here keeps the wipe
        # painted continuously under the cursor.
        if self._compare_state.is_active():
            if _render_compare(self, frame):
                self._last_displayed = frame
                return
            # else: fall through to the cache path so the user still
            # sees something if the compare decode failed.
        # Video layer? Decode synchronously so scrub gives the user
        # frame-accurate feedback under the cursor instead of the
        # MISSING-FRAME placeholder (the cache never has anything for
        # video). Same code path as ``_on_frame_changed``'s video
        # branch — VideoSource caches the last frame internally so
        # repeated calls within the same display interval are free.
        displayed = (
            self._layer_stack.topmost_visible_at(frame)
            if self._layer_stack else None
        )
        if displayed is not None and displayed.is_video:
            try:
                arr_v = self._decode_video_layer(displayed, frame)
                if arr_v is not None:
                    self._last_displayed = frame
                    self._display_array(arr_v)
                    return
            except Exception:
                log.exception("video scrub decode failed at frame %d", frame)
        arr = self._cache.get(frame)
        if arr is not None:
            self._last_displayed = frame
            self._display_array(arr)
            return
        # No-coverage gap. Two distinct sub-cases:
        #
        # * Stack is non-empty but no layer reaches THIS frame — could
        #   be a true compositional gap (between two layers) or, more
        #   commonly, the playhead landing before the first / after
        #   the last scanned frame because the source has missing
        #   files at the boundary. In both cases the user expects
        #   feedback ("there's nothing to show here"), not a silent
        #   black flash. We upload the same MISSING FRAME placeholder
        #   the cache uses — visually unmistakable, and consistent
        #   with how missing-source frames inside the layer's range
        #   already render.
        # * Stack is empty (no sequence loaded at all) — clear to
        #   black, that's the correct "no content" state.
        if (
            self._layer_stack
            and self._layer_stack.topmost_visible_at(frame) is None
        ):
            try:
                self._show_gap_placeholder()
            except Exception:
                log.exception("[gap-scrub] failed to render placeholder at %d", frame)
            self._last_displayed = None
            return
        # Cache miss but the frame is covered — fall back to the
        # nearest decoded frame so the user gets *something* moving
        # under their cursor while the prefetcher catches up.
        fallback = self._nearest_cached_fallback(frame)
        if fallback is None:
            return
        arr = self._cache.get(fallback)
        if arr is None:
            return
        self._last_displayed = fallback
        self._display_array(arr)

    def _on_jump_to_ends(self, direction: int) -> None:
        seq = self._controller.sequence
        if seq is None:
            return
        target = seq.first_frame if direction < 0 else seq.last_frame
        self._controller.seek(target)

    def _on_unmarked_exr_save(self, source: str, view: str) -> None:
        """User clicked "Save as EXR default" in the Color panel.

        Persists the pair to preferences and refreshes the panel's
        status row. Future sequences without colorspace tags will
        auto-pick this pair via :meth:`_guess_source_colorspace`.
        """
        self._prefs.unmarked_exr_source = source
        self._prefs.unmarked_exr_view = view
        self._window.color_panel.set_unmarked_exr_default(source, view)
        self._window.set_status(
            f"EXR default pinned: {source} / {view}",
        )

    def _on_unmarked_exr_clear(self) -> None:
        """Drop the pinned EXR default — auto-detection reverts to the
        industry-standard linear assumption for unmarked EXRs."""
        self._prefs.unmarked_exr_source = None
        self._prefs.unmarked_exr_view = None
        self._window.color_panel.set_unmarked_exr_default(None, None)
        self._window.set_status(
            "EXR default cleared — using industry default (linear)",
        )

    def reload_ocio_config(self) -> dict[str, object]:
        """Hot-reload the OCIO configuration from current preferences.

        Called by :class:`PreferencesDialog` when the user changes the
        OCIO config source (Default / $OCIO env / Custom file). Builds
        a fresh :class:`OCIOManager`, swaps it in, repopulates the
        :class:`ColorPanel` combos, validates persisted colorspace
        prefs against the new config, and triggers a shader rebuild.

        Returns a status dict with:
          * ``config_name``  — name string of the freshly loaded config
          * ``origin``       — ``"file"`` / ``"env"`` / ``"builtin"``
          * ``description``  — human-readable origin (path, $OCIO=…, …)
          * ``source_preserved`` / ``display_preserved`` / ``view_preserved``
            — whether the user's current panel picks survived the swap

        Failure to load a custom config falls back to the built-in
        (already handled inside :class:`OCIOManager`); the returned
        ``description`` reflects that fallback so the dialog can
        surface the warning.
        """
        new_manager = OCIOManager()

        # Drop persisted prefs that no longer reference valid names
        # in the new config. Without this the user would see the
        # ColorPanel snap to a fallback while their on-disk prefs
        # still pointed at a now-gone colorspace, leading to a
        # confusing mismatch on next launch.
        valid_colorspaces = set(new_manager.list_colorspaces())
        valid_displays = set(new_manager.list_displays())
        if self._prefs.source_colorspace and self._prefs.source_colorspace not in valid_colorspaces:
            self._prefs.source_colorspace = None
        if self._prefs.display and self._prefs.display not in valid_displays:
            self._prefs.display = None
            # View is keyed off the display, so it goes too if display does.
            self._prefs.view = None
        elif self._prefs.view and self._prefs.display:
            valid_views = set(new_manager.list_views(self._prefs.display))
            if self._prefs.view not in valid_views:
                self._prefs.view = None

        # Same sanity check for the unmarked-EXR override.
        if self._prefs.unmarked_exr_source and self._prefs.unmarked_exr_source not in valid_colorspaces:
            self._prefs.unmarked_exr_source = None
            self._prefs.unmarked_exr_view = None

        self._ocio = new_manager
        # ColorPanel.reload_from_manager fires color_params_changed at
        # the end with the final validated triple — that signal is
        # already wired to _on_color_params, which rebuilds the GPU
        # shader against the new manager. So no manual rebuild here:
        # the wiring takes care of it.
        result = self._window.color_panel.reload_from_manager(new_manager)

        # Re-sync the unmarked-EXR status row since we may have
        # cleared the underlying prefs above.
        self._window.color_panel.set_unmarked_exr_default(
            self._prefs.unmarked_exr_source,
            self._prefs.unmarked_exr_view,
        )

        return {
            "config_name": new_manager.config.getName() or "(unnamed)",
            "origin": new_manager.source.origin,
            "description": new_manager.source.description,
            **result,
        }

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

    # ------------------------------------------------------------------ New / Reload (v0.5.1)

    def _on_new_sequence(self) -> None:
        """File → New (Ctrl+N): clear the loaded sequence without
        resetting the rest of the UI.

        Tools (toolbar, color panel, FPS, view, ephemeral mode,
        annotations toolbar visibility) keep their state — only the
        viewport, the cache, the in-memory annotation/comment data
        and the timeline get wiped. The user can then File → Open a
        different sequence with all their preferences still in place.
        """
        # Stop any ongoing playback first; ticking a detached cache
        # would just spin no-ops.
        self._controller.pause()
        # Exit compare mode if it was active. Without this the
        # CompareBand stays visible after the wipe, pointing at
        # ids that are about to be invalidated by the layer-stack
        # reset below — and the user has no obvious way to dismiss
        # the band since File → New conceptually resets the project.
        if self._compare_state.enabled:
            self._compare_state.enabled = False
            self._compare_state.layer_a_id = None
            self._compare_state.layer_b_id = None
            self._compare_decoder.invalidate()
            self._window.viewer.gl.clear_compare()
            self._window.transport.set_compare_checked(False)
            self._window.set_compare_band_visible(False)
        # Contact-sheet decoder lives parallel to compare's and
        # benefits from the same "drop the per-layer slot when
        # anything changed" invariant.
        self._contact_sheet_decoder.invalidate()
        # File → New is a project-load entry point too — re-tune the
        # cache budget so the next project opened from this empty
        # state benefits from any RAM the user has freed in the
        # meantime.
        self._retune_for_current_ram()
        self._controller._sequence = None  # noqa: SLF001 — there's no public detach
        # ``cache.detach()`` empties the LayerStack which cascades
        # via signals to the cache's clear() and the LayerPanel
        # rebuild. With FrameCache the call simply clears its
        # internal state (no stack involvement).
        self._cache.detach()
        # Clear in-memory annotation + comment data (their sidecar
        # path tracking goes too).
        self._annotation_store.load_from_dict({})
        self._comment_store.load_from_dict({})
        self._annotations_path = None
        self._annotations_basename = None
        # Drop live ephemeral strokes (they were image-anchored to
        # the previous sequence).
        self._ephemeral_manager.clear_all()
        # Reset the viewport — drop the image entirely so the user
        # gets the same "no sequence loaded" look as at first
        # launch. We deliberately avoid pushing any placeholder
        # buffer (a centred grey square would falsely suggest
        # something is loaded).
        try:
            self._window.viewer.gl.clear_image()
        except Exception:
            log.exception("[new] failed to clear viewport")
        # Clear timeline UI.
        self._window.timeline.set_range(0, 0)
        self._window.timeline.set_cached_frames(frozenset())
        self._window.timeline.set_missing_frames(frozenset())
        self._window.timeline.set_annotated_frames(frozenset())
        self._window.timeline.set_commented_frames(frozenset())
        # Wipe the channel menu so the previous sequence's groups
        # don't linger after the cache is detached. Without this the
        # button keeps its old caption ("albedo", "RGB +2", …) and
        # the dropdown still lists the AOVs of a sequence that no
        # longer exists.
        self._window.transport.clear_channels()
        # Re-disable the actions that need a loaded sequence.
        if hasattr(self._window, "_export_act"):
            self._window._export_act.setEnabled(False)  # noqa: SLF001
        if hasattr(self._window, "_save_frame_act"):
            self._window._save_frame_act.setEnabled(False)  # noqa: SLF001
        if hasattr(self._window, "_reload_act"):
            self._window._reload_act.setEnabled(False)  # noqa: SLF001
        if hasattr(self._window, "_force_reload_act"):
            self._window._force_reload_act.setEnabled(False)  # noqa: SLF001
        self._window.transport.set_export_enabled(False)
        self._window.transport.set_reload_enabled(False)
        # Reset the current-session pointer + title bar.
        # ``set_current_session_path(None)`` rewrites the title to
        # the bare "Flick Player" baseline.
        self._window.set_current_session_path(None)
        self._window.set_status("No sequence loaded — File → Open to load one.")

    def _refresh_source_watcher(self) -> None:
        """Re-sync the auto-reload watcher's directory list to the live stack.

        Called on every ``layers_changed`` signal. Cheap diff inside
        :meth:`SourceWatcher.set_watched_layers` — only the delta is
        passed to Qt's QFileSystemWatcher.
        """
        watcher = getattr(self, "_source_watcher", None)
        if watcher is None:
            return
        try:
            watcher.set_watched_layers(self._layer_stack.layers())
        except Exception:  # pragma: no cover — defensive, watcher is best-effort
            log.exception("source watcher refresh failed (non-fatal)")

    def _on_source_watcher_fired(self) -> None:
        """File-watcher debounce ticked — trigger the smart reload.

        Routes through the same path as a manual Ctrl+R so we get the
        mtime-diff for free: unchanged frames stay hot in RAM, the
        re-rendered ones get re-decoded. Disk-cache entries for the
        old mtime stay until LRU evicts them; their keys won't match
        the new mtime, so they can't serve stale pixels.

        Squashed to a no-op when no sequence is loaded (e.g. the
        watcher fires during teardown while we still have a stale
        directory in the list).
        """
        if self._controller.sequence is None:
            return
        try:
            self._on_reload_sequence()
        except Exception:  # pragma: no cover — defensive
            log.exception("auto-reload from source watcher failed (non-fatal)")

    def _on_reload_sequence(self) -> None:
        """Reload (Ctrl+R / 🔄): smart re-scan.

        Re-globs the source folder for the current sequence's pattern,
        diffs file mtimes vs the cached snapshot, drops only the
        frames that changed (or vanished), keeps the rest. Frames
        that newly appeared on disk become eligible for prefetch;
        frames that disappeared get the missing-checkerboard
        placeholder on next access.
        """
        from img_player.sequence.scanner import rescan as _rescan
        seq = self._controller.sequence
        if seq is None:
            self._window.set_status("Reload: no sequence loaded.")
            return
        try:
            new_seq = _rescan(seq)
        except Exception as err:
            log.exception("[reload] rescan failed")
            self._window.set_status(f"Reload failed: {err}")
            return
        kept, dropped, missing = self._cache.reload(new_seq)
        # Hand the new sequence info to the controller so its
        # frame-range views (in/out, last_frame) reflect any
        # newly-arrived frames.
        self._controller._sequence = new_seq  # noqa: SLF001
        self._window.update_sequence_info(new_seq)
        # Use the layer panel's BROAD master range (= union of every
        # layer's source potential) — same range the rest of the app
        # uses (``_refresh_after_stack_change`` after layer_modified,
        # the layer bars themselves, etc.). Mixing master_range()
        # and broad_master_range() across surfaces caused the
        # timeline cursor and the layer-bar playhead to land at
        # different fractions when the user trimmed a layer's tail.
        panel = getattr(self._window, "_layer_panel", None)
        if panel is not None and self._layer_stack:
            # The layer's ``sequence`` reference was just mutated by
            # ``cache.reload`` (no signal fired, by design — keeps
            # the mtime-kept frames). Re-sync the bars manually so
            # they pick up any new ``broad`` range from the new
            # sequence's first/last frame.
            panel.sync_bar_geometry()
            first, last = panel.broad_master_range()
        else:
            first, last = new_seq.first_frame, new_seq.last_frame
        if last > first:
            self._window.timeline.set_range(first, last)
            self._controller.set_navigable_range(first, last)
            self._window.viewer.gl.set_navigable_range(first, last)
        else:
            self._window.timeline.set_range(
                new_seq.first_frame, new_seq.last_frame,
            )
        # Push the freshly-rebuilt missing set straight to the
        # timeline so the user sees the red slots without waiting
        # for the next 200 ms _refresh_cache_bar tick.
        self._window.timeline.set_cached_frames(self._cache.cached_frames())
        self._window.timeline.set_missing_frames(self._cache.missing_frames())
        # Re-prime the prefetch ring around the current playhead —
        # in master coords, so a moved layer prefetches its OWN
        # range rather than the (now mismatched) source range.
        # ``replan_prefetch`` issues priority-by-distance from the
        # playhead (same path as ``seek`` / stack-change). The
        # earlier ``request_range(first, last)`` walked frames in
        # iteration order, anchoring decoding at ``first`` instead
        # of the cursor — visible to the user as "the cache bar
        # fills from the middle, not from where I'm parked".
        cur = self._controller.state.current_frame
        self._cache.set_current_frame(cur)
        self._controller.replan_prefetch()
        # Refresh the on-screen image: the user expects the viewport
        # to update right after reload — either the old missing
        # placeholder is replaced with freshly decoded data, or a
        # cached frame whose source file vanished now shows the
        # red checkerboard. Routing through ``_on_frame_changed``
        # reuses the existing cache-hit / wait-timer fallback path.
        self._on_frame_changed(cur)
        added = len(new_seq.frames) - (len(seq.frames) - dropped - missing)
        added = max(0, added)
        self._window.set_status(
            f"Reload: {kept} kept, {dropped} re-decoded, {missing} missing"
            + (f", {added} new" if added else "")
        )

    def _on_clear_cache_action(self) -> None:
        """Image → Clear cache… (Ctrl+Alt+Shift+R).

        Wipes both tiers of the cache after a confirmation dialog:

        1. **RAM** (``MasterFrameCache.clear``) — every decoded frame
           in memory is dropped. The next paint will re-fetch from
           the disk cache (if still present) or re-decode from the
           source files.
        2. **Disk** (``DiskCache.clear``) — every persisted blob is
           removed, the SQLite index is wiped. The next session
           restarts from the source files.

        Distinct from Reload (force) which only nukes the RAM tier
        and re-decodes immediately: this one ALSO clears the
        persistent state, so future sessions don't benefit from the
        previous warm-up. After the clear, ``replan_prefetch`` is
        called so playback resumes via fresh decodes if a sequence
        is loaded.
        """
        from PySide6.QtWidgets import QMessageBox  # noqa: PLC0415 — UI-only

        # The dialog mentions both tiers so the user knows what
        # they're signing up for — clearing the disk cache is the
        # destructive part (RAM clears itself naturally at app
        # shutdown anyway).
        body = (
            "This wipes BOTH cache tiers:\n\n"
            "  • RAM master cache — every decoded frame in memory.\n"
            "  • Persistent disk cache — every blob on disk.\n\n"
            "The next playback will re-decode from the source files. "
            "Future sessions won't benefit from the previously warmed "
            "disk cache until they replay the shots.\n\n"
            "Clear now?"
        )
        reply = QMessageBox.warning(
            self._window,
            "Clear cache?",
            body,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # RAM tier — always present.
        try:
            self._cache.clear()
        except Exception:  # pragma: no cover — defensive
            log.exception("[clear-cache] RAM cache clear failed")

        # Disk tier — present only when the user hasn't disabled it
        # via Preferences > Disk cache > Enabled. ``getattr`` keeps
        # this safe even on a future build where ``_disk_cache``
        # might not be wired.
        disk_cache = getattr(self._cache, "_disk_cache", None)
        freed_bytes = 0
        if disk_cache is not None:
            try:
                freed_bytes = disk_cache.clear()
            except Exception:  # pragma: no cover — defensive
                log.exception("[clear-cache] disk cache clear failed")
                QMessageBox.critical(
                    self._window,
                    "Clear failed",
                    "RAM cache cleared, but the disk cache clear "
                    "raised. Check the log for details.",
                )
                return

        # Re-trigger prefetch so playback resumes cleanly if a
        # sequence is loaded. No-op when the controller has no
        # sequence (= startup state, never opened anything).
        if self._controller.sequence is not None:
            self._controller.replan_prefetch()

        freed_gb = freed_bytes / (1024 ** 3)
        if disk_cache is None:
            self._window.set_status("Cache cleared — RAM only (disk cache disabled).")
        elif freed_gb >= 0.01:
            self._window.set_status(
                f"Cache cleared — RAM + {freed_gb:.2f} GB freed on disk."
            )
        else:
            self._window.set_status("Cache cleared — RAM + disk (both were empty).")

    def _on_force_reload_sequence(self) -> None:
        """Reload (force) — Ctrl+Shift+R / File → Reload (force).

        Drops every cached frame and re-decodes from scratch, ignoring
        the mtime-diff that the smart reload uses. The smart reload is
        right 99 % of the time, but for the rare case where files were
        overwritten without an mtime bump (``cp -p``, restore from
        backup, slow Drive sync that updated content but not the
        timestamp) the user gets a nuclear option to re-read everything.

        Implementation = ``cache.clear()`` then route through the
        regular smart-reload path. The clear empties every cached
        frame; the smart reload's mtime diff has nothing left to pop
        and just refreshes the per-layer path / mtime indexes against
        the rescanned sequence (= what we want, without duplicating
        the target-resolution logic).
        """
        seq = self._controller.sequence
        if seq is None:
            self._window.set_status("Reload (force): no sequence loaded.")
            return
        self._cache.clear()
        self._on_reload_sequence()
        # Override the smart-reload status (which would say
        # "0 kept, 0 re-decoded" after a clear — confusing) with
        # something that reflects what actually happened.
        self._window.set_status(
            "Reload (force): cache cleared, full re-decode in progress…"
        )

    # ------------------------------------------------------------------ Export (v0.5.0)

    def _open_export_dialog(self) -> None:
        """File → Export… (or 💾 transport button) — open the dialog,
        kick off the worker on accept."""
        from img_player.export_handler import open_export_dialog
        open_export_dialog(self)

    def _open_save_frame_dialog(self) -> None:
        """File → Save Frame As… (Ctrl+Alt+S) — quick WYSIWYG
        snapshot of the current viewer with optional
        annotations / overlay toggles."""
        from img_player.save_frame_handler import open_save_frame_dialog
        open_save_frame_dialog(self)

    def _on_export_finished(self, output_path: str, frames: int, duration_s: float) -> None:
        self._window.set_status(
            f"Exported {frames} frames to {output_path} in {duration_s:.1f} s"
        )

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

    def _on_set_in_at(self, frame: int) -> None:
        """Ctrl-click drag on the timeline LEFT of the cursor →
        place / drag the in-point at ``frame``. Clamped so it never
        ends up past the current out-point (keeps in ≤ out)."""
        out = self._controller.state.out_frame
        if out is not None and frame > out:
            frame = out
        self._controller.set_in_out(frame, out)

    def _on_set_out_at(self, frame: int) -> None:
        """Ctrl-click drag RIGHT of the cursor → place / drag the
        out-point at ``frame``. Clamped so it never ends up before
        the current in-point."""
        in_f = self._controller.state.in_frame
        if in_f is not None and frame < in_f:
            frame = in_f
        self._controller.set_in_out(in_f, frame)

    def _retune_for_current_ram(self) -> None:
        """Re-apply runtime constraints against the live RuntimeState.

        Called at every project-load entry point so a user who freed
        memory between Flick launch and project open gets the updated
        cache budget without restarting. Symmetrically, if the user
        has loaded other apps that ate RAM, the new project starts
        with a smaller, safer budget — eviction kicks in once and the
        playback that follows is honest about what fits.

        The "ceiling" is the pre-runtime-constraint tune the user
        actually asked for at boot (CLI + profile + heuristics). We
        prefer ``_desired_tune`` (set by the late-bind GPU re-tune,
        so it reflects the real renderer) over ``_boot_tune`` (= the
        boot-time tune resolved before the GL context exists).

        No-op when the new budget is within ~100 MB of the current
        one — avoids spurious status messages on tiny RAM jitter.
        """
        if not hasattr(self._cache, "set_budget"):
            return
        ceiling = self._desired_tune or self._boot_tune
        if ceiling is None:
            return
        from img_player.perf.runtime_state import (
            RuntimeState,
            apply_runtime_constraints,
        )
        state = RuntimeState.snapshot()
        retuned = apply_runtime_constraints(ceiling, state)
        new_budget = int(retuned.cache_gb * 1024**3)
        old_budget = self._cache._budget  # noqa: SLF001 — internal int read
        # Threshold: 100 MB diff to avoid noise on small RAM swings.
        if abs(new_budget - old_budget) < 100 * 1024**2:
            return
        self._cache.set_budget(new_budget)
        old_gb = old_budget / 1024**3
        new_gb = new_budget / 1024**3
        if new_gb > old_gb:
            log.info(
                "[retune] cache budget grown: %.1f → %.1f GB "
                "(RAM dispo : %.1f GB)",
                old_gb, new_gb, state.available_ram_gb,
            )
            self._window.set_status(
                f"Cache élargi à {new_gb:.1f} GB "
                f"(RAM dispo : {state.available_ram_gb:.1f} GB)."
            )
        else:
            log.info(
                "[retune] cache budget reduced: %.1f → %.1f GB "
                "(RAM dispo : %.1f GB)",
                old_gb, new_gb, state.available_ram_gb,
            )
            self._window.set_status(
                f"Cache réduit à {new_gb:.1f} GB "
                f"(mémoire système plus tendue)."
            )

    def _open_path(self, paths: list[Path] | Path) -> None:
        """Scan one or more ``paths`` off the main thread.

        Always replaces the current sequence. The "add layer"
        semantic is handled by a separate signal
        (:meth:`_on_add_layer_requested`) — drops on the viewer area
        fire this one, drops on the layer panel area fire the
        add-layer one. No more modal Add / Replace / Cancel dialog;
        the spatial disambiguation does the same job without an
        interruption.

        Multi-source drops: when more than one folder / file is
        provided, the picker is shown with a hierarchical tree (one
        bold header per folder, sequences listed below) so the user
        can tick exactly which sequences to load. The first ticked
        sequence becomes the active sequence (replaces the
        controller's binding) and the remainder are appended as
        layers in pick order.

        Confirmation: when a sequence is already loaded (or layers
        exist in the stack), prompt before wiping the current state.
        """
        path_list = [paths] if isinstance(paths, Path) else list(paths)
        if not path_list:
            return
        primary = path_list[0]

        # Video file? mp4 / mov / … drops create video layer(s) that
        # bypass the per-frame OIIO cache and pull pixels via PyAV
        # (see ``VideoSourceManager``).
        # Three cases:
        #   1. Single video → replace-load via ``_open_video_path``.
        #   2. Multi-video drop → first video replace-loads, rest
        #      append as layers (mirrors multi-folder image-sequence
        #      drops).
        #   3. Mixed drop (videos + image-sequence folders) →
        #      currently unsupported; surface a status hint and route
        #      the image-sequence portion only. Mixing both in one
        #      gesture would need a unified picker variant.
        from img_player.media import is_video_file
        video_paths = [
            p for p in path_list if p.is_file() and is_video_file(p)
        ]
        non_video_paths = [p for p in path_list if p not in video_paths]
        if video_paths and not non_video_paths:
            primary = video_paths[0]
            if self._is_replace_destructive():
                if not self._confirm_replace(primary):
                    self._window.set_status("Replace annulé.")
                    return
            self._open_video_path(primary)
            for extra in video_paths[1:]:
                self._add_video_layer(extra)
            if len(video_paths) > 1:
                self._window.set_status(
                    f"Loaded {primary.name} + "
                    f"{len(video_paths) - 1} additional video layer"
                    f"{'s' if len(video_paths) > 1 else ''}."
                )
            return
        if video_paths and non_video_paths:
            self._window.set_status(
                "Mixing videos and image sequences in one drop is "
                "not yet supported — loaded only the image sequences."
            )
            # Fall through with non_video_paths only.
            path_list = non_video_paths
            primary = path_list[0]

        # Project file? A ``.session`` drop is a "load this whole
        # project" gesture, not a sequence open. Route to the session
        # loader so the LayerStack, Color panel and recent-sessions
        # list all update. If the drop also contains other paths we
        # ignore them — mixing a session with loose sequences in one
        # drop has no sane semantic. Same destructive-replace
        # confirmation as a regular sequence drop, since loading a
        # session also wipes the current stack.
        session_paths = [p for p in path_list if p.suffix.lower() == ".session"]
        if session_paths:
            session_path = session_paths[0]
            if self._is_replace_destructive():
                if not self._confirm_replace(session_path):
                    self._window.set_status("Replace annulé.")
                    return
            if len(path_list) > 1:
                self._window.set_status(
                    f"Loading session {session_path.name} "
                    f"(other dropped items ignored)."
                )
            self._on_open_session_requested(session_path)
            return

        if self._is_replace_destructive():
            if not self._confirm_replace(primary):
                self._window.set_status("Replace annulé.")
                return
        # Re-snapshot ambient RAM and resize the cache before the
        # scan starts. If the user closed Chrome / Premiere between
        # Flick launch and now, the new project enjoys a roomier
        # cache; if they opened more apps, the budget shrinks safely
        # before we load fresh frames into it.
        self._retune_for_current_ram()
        from img_player.scan_handler import open_path, open_paths
        if len(path_list) == 1:
            open_path(self, primary)
        else:
            open_paths(self, path_list)

    def _open_video_path(self, path: Path) -> None:
        from img_player.media_handler import open_video_path
        open_video_path(self, path)

    def _is_replace_destructive(self) -> bool:
        """True when a Replace would wipe state the user might want
        to keep — a loaded sequence and/or layers in the stack."""
        if self._controller.sequence is not None:
            return True
        if self._layer_stack and len(self._layer_stack) > 0:
            return True
        return False

    def _confirm_replace(self, path: Path) -> bool:
        """Modal Yes/Cancel — returns True iff the user confirms.

        Inventory of what's about to be lost is built dynamically:
        layer count, sequence name, dirty annotation badge. The user
        sees what they'd discard before deciding.
        """
        layer_count = (
            len(self._layer_stack) if self._layer_stack is not None else 0
        )
        seq = self._controller.sequence
        seq_name = seq.display_pattern() if seq is not None else None

        bullets: list[str] = []
        if layer_count > 0:
            bullets.append(
                f"• {layer_count} layer"
                f"{'s' if layer_count > 1 else ''} "
                f"(offsets, trims, sélection de canaux)"
            )
        elif seq_name is not None:
            bullets.append(f"• La séquence courante : {seq_name}")
        if self._annotation_store.is_dirty():
            bullets.append("• Les annotations non sauvegardées")
        inventory = "\n".join(bullets) if bullets else (
            "• L'état courant du player"
        )

        box = QMessageBox(self._window)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Remplacer la séquence ?")
        box.setText(
            f"Charger <b>{path.name}</b> va remplacer ce qui est "
            f"actuellement ouvert."
        )
        box.setInformativeText(
            "Ce remplacement va supprimer :\n\n"
            f"{inventory}\n\n"
            "Pour ajouter sans remplacer, drop sur le panel des layers "
            "(zone teal) au lieu du viewport (zone orange)."
        )
        replace_btn = box.addButton(
            "Remplacer", QMessageBox.ButtonRole.DestructiveRole,
        )
        cancel_btn = box.addButton(
            "Annuler", QMessageBox.ButtonRole.RejectRole,
        )
        box.setDefaultButton(cancel_btn)
        box.exec()
        return box.clickedButton() is replace_btn

    def _on_add_layer_requested(self, paths: list[Path] | Path) -> None:
        """File → Add layer… handler — appends one or more layers to
        the stack without replacing the existing sequence.

        Single-path call (file menu, programmatic) loads directly via
        the legacy ``add_layer`` helper; multi-source drops route
        through ``add_layers`` which shows the grouped picker first.

        Video files (mp4 / mov / …) are split out and added as
        :meth:`Layer.from_video` directly — the OIIO-driven scan path
        can't handle video containers. Mixed drops (videos + image
        sequences in the same gesture) work: each video lands as its
        own layer, the image sequences flow through the normal
        scan / picker.
        """
        path_list = [paths] if isinstance(paths, Path) else list(paths)
        if not path_list:
            return
        # Sessions describe an entire LayerStack, not a single layer.
        # Dropping one on the layer panel has no useful semantic —
        # surface a status hint and ignore. The user can still drop
        # the same file on the viewer to load it as a project.
        if any(p.suffix.lower() == ".session" for p in path_list):
            self._window.set_status(
                "Session files can't be added as a layer — drop on the "
                "viewer to load the project."
            )
            return
        # Split video files out — they take the dedicated
        # Layer.from_video path; everything else goes through the
        # OIIO scan / picker as before.
        from img_player.media import is_video_file
        video_paths = [
            p for p in path_list if p.is_file() and is_video_file(p)
        ]
        other_paths = [p for p in path_list if p not in video_paths]
        added = 0
        for vp in video_paths:
            if self._add_video_layer(vp):
                added += 1
        if other_paths:
            from img_player.scan_handler import add_layer, add_layers
            if len(other_paths) == 1:
                add_layer(self, other_paths[0])
            else:
                add_layers(self, other_paths)
        if video_paths and not other_paths:
            self._window.set_status(
                f"Added {added} video layer{'s' if added != 1 else ''}."
            )

    def _add_video_layer(self, path: Path) -> bool:
        from img_player.media_handler import add_video_layer
        return add_video_layer(self, path)

    def _on_save_session_requested(self, path: Path) -> None:
        """File → Save session… — write the full LayerStack to a
        ``.session`` JSON file."""
        from img_player.layers.session import ColorState, save_session
        # Snapshot the global Color panel — the OCIO triple + viewing
        # tweaks travel with the session so a re-open restores the
        # exact look the user shipped with. Without this, opening a
        # saved Rec709-deliverable session would inherit whatever
        # display/view the player is currently set to (e.g. ACEScg
        # left from a prior project).
        src, display, view, exposure, gamma = (
            self._window.color_panel.current_params()
        )
        color_state = ColorState(
            source_colorspace=src or None,
            display=display or None,
            view=view or None,
            exposure=float(exposure),
            gamma=float(gamma),
        )
        # Persist compare-mode state too so a Ctrl+S during an
        # active compare round-trips back into the same overlay
        # on next open.
        compare_payload = (
            self._compare_state.to_dict() if self._compare_state.enabled else None
        )
        try:
            save_session(
                self._layer_stack, path,
                color_state=color_state,
                compare_state=compare_payload,
            )
        except Exception as err:
            log.exception("[session] save failed for %s", path)
            self._window.set_status(f"Save session failed: {err}")
            return
        self._window.set_status(f"Session saved to {path}")
        # Track this session in the Open Recent Session list — the
        # user just declared interest in coming back to it.
        self._prefs.push_recent_session(path)
        # Tell the window this is now the "current" session — the
        # next Ctrl+S overwrites this file silently instead of
        # popping the file picker. Updates the title bar too.
        self._window.set_current_session_path(path)

    def _apply_session_color_state(self, color_state) -> None:  # type: ignore[no-untyped-def]
        """Push a saved ColorState onto the live Color panel.

        Each combo is set with a "if available" guard — the OCIO
        config on the loading machine may not expose the same display
        / view names the saving machine had. When a combo entry is
        missing we keep the current value silently and log a warning,
        which is friendlier than refusing to load the session.

        The exposure / gamma spinboxes accept any float, so they're
        always restored.

        Setting via ``setCurrentText`` triggers the panel's standard
        change signals → re-emits ``color_params_changed`` →
        rebuilds the OCIO shader, exactly as if the user had clicked
        the combos manually.
        """
        panel = self._window.color_panel
        cs = color_state

        # Per-kind warning when the saved value isn't in the current
        # combo's list — kept as a closure so we can format the
        # message uniformly across the three (source / display / view)
        # cases.
        def _warn_missing(kind: str, requested: str, current: str) -> None:
            messages = {
                "source": "saved source colorspace %r not in current OCIO config — keeping %r",
                "display": "saved display %r not available — keeping %r",
                "view": "saved view %r not available for display — keeping %r",
            }
            log.warning("[session] " + messages[kind], requested, current)

        panel.apply_state(
            source_colorspace=cs.source_colorspace or None,
            display=cs.display or None,
            view=cs.view or None,
            exposure=cs.exposure,
            gamma=cs.gamma,
            on_missing=_warn_missing,
        )

    def _on_open_session_requested(self, path: Path) -> None:
        """File → Open session… — replace the LayerStack from a
        previously saved ``.session`` file."""
        # Same per-session cache re-tune as the regular Open path.
        self._retune_for_current_ram()
        from img_player.layers.session import load_session
        try:
            result = load_session(self._layer_stack, path)
        except Exception as err:
            log.exception("[session] load failed for %s", path)
            self._window.set_status(f"Open session failed: {err}")
            return
        msg = f"Session loaded: {result.loaded} layers"
        if result.skipped:
            msg += f" ({result.skipped} skipped)"
        self._window.set_status(msg)
        # Restore the global Color panel state if the session shipped
        # one (v2+). v1 sessions and sessions saved without a color
        # block leave the panel as-is — same legacy behaviour.
        if result.color_state is not None:
            self._apply_session_color_state(result.color_state)
        # Restore compare-mode if the session shipped one. Build the
        # state from the dict, sync the band/transport to it, then
        # call into the toggle path if it should be enabled. Setting
        # ``enabled=False`` first ensures the toggle flips it on
        # cleanly rather than racing the auto-pick.
        if result.compare_state is not None:
            from img_player.compare import CompareState
            from img_player.compare_handler import (
                refresh_band_layers,
                toggle_compare,
            )
            restored = CompareState.from_dict(result.compare_state)
            self._compare_state = restored
            self._compare_decoder.invalidate()
            # Contact-sheet decoder lives parallel to compare's and
            # benefits from the same "drop the per-layer slot when
            # anything changed" invariant.
            self._contact_sheet_decoder.invalidate()
            refresh_band_layers(self)
            band = self._window.compare_band
            band.set_mode(restored.mode)
            band.set_seam(restored.seam)
            if restored.enabled:
                # ``toggle_compare`` flips ``enabled`` from False to
                # True — we just patched it above to the saved value
                # before flipping; reset to False so the call enables
                # it cleanly.
                self._compare_state.enabled = False
                toggle_compare(self)
        # Track in Open Recent Session — same trigger as a save: the
        # user just used the file, so it deserves a slot in the list.
        if result.loaded > 0:
            self._prefs.push_recent_session(path)
            # This is now the "current" session: subsequent Ctrl+S
            # overwrites it in place. Updates the title bar so the
            # user always knows which session they're working in.
            self._window.set_current_session_path(path)
        # Point the controller at the focused layer's sequence so
        # the timeline range + scrubbing have a target. We bypass
        # ``controller.load_sequence`` here on purpose: that call
        # would call ``cache.attach(seq)`` which replaces the
        # LayerStack with a single layer — wiping the session we
        # just loaded.
        focused = self._layer_stack.focused()
        if focused is None:
            return
        self._controller._sequence = focused.sequence  # noqa: SLF001
        self._window.update_sequence_info(focused.sequence)
        # ``update_sequence_info`` set the timeline range to the
        # focused layer's own first/last — but the LayerPanel uses
        # ``broad_master_range`` (the union of every layer's source
        # potential), so the two scrubbers end up on different scales
        # and the playhead lands at different x positions on each.
        # Re-run the post-stack-change sync so the timeline picks up
        # the broad range and the controller / GL navigable bounds
        # match. Same call the layer-stack signals fire normally —
        # we re-trigger it explicitly here because session load
        # already emitted ``layers_changed`` BEFORE
        # ``update_sequence_info`` overwrote the timeline range.
        self._refresh_after_stack_change()
        first = self._layer_stack.master_range()[0]
        self._controller.seek(first)

    def _enrich_with_header(self, seq: SequenceInfo) -> SequenceInfo:
        """Fill in channel_names / width / height from the first frame's
        header if the scanner skipped them. Returns the same seq when
        already populated.

        Thin wrapper over :func:`enrich_with_header` — keeps the live-
        flow's success-log (which the session restore path doesn't
        emit) while sharing the actual probe logic.
        """
        enriched = enrich_with_header(seq)
        if enriched is not seq and enriched.channel_names:
            channels = enriched.channel_names
            log.info(
                "header probe: %d channels (%s), %sx%s",
                len(channels),
                ", ".join(channels[:8]) + ("…" if len(channels) > 8 else ""),
                enriched.width, enriched.height,
            )
        return enriched

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
            unmarked_exr_source=self._prefs.unmarked_exr_source,
        )
        # When the unmarked-EXR override fires, also pin the matching
        # view (if the user paired one). Without this the source would
        # change but the view would stay on whatever the auto-classifier
        # picks — defeating half the override's purpose.
        used_unmarked_exr_override = (
            source_result.colorspace is not None
            and "user override" in source_result.reason
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
                # If the source was set by the unmarked-EXR override AND
                # the user paired a view with it, honour that pairing.
                # Otherwise fall through to the standard category-based
                # picker.
                view_override = (
                    self._prefs.unmarked_exr_view
                    if used_unmarked_exr_override
                    else None
                )
                if view_override and view_override in available_views:
                    self._window.color_panel._view_combo.setCurrentText(view_override)
                    view_msg = f" → view: {view_override} (user override)"
                    log.info(
                        "auto-detect: view = %s (user override)", view_override,
                    )
                else:
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

    def _refresh_cache_bar(self) -> None:
        if self._controller.sequence is None:
            return
        self._window.timeline.set_cached_frames(self._cache.cached_frames())
        self._window.timeline.set_missing_frames(self._cache.missing_frames())
        # Push the active channel's cache fill onto the channel
        # button so the bar paints over the closed dropdown too —
        # not just inside the menu rows. Cheap: ``alt_channel_progress``
        # reuses the same single-pass scan the menu polls.
        progress = self._cache.alt_channel_progress()
        active_label = self._window.transport.channel_menu.active_label
        cached, total = progress.get(active_label, (0, 0))
        self._window.transport.channel_button.set_active_progress(
            (cached / total) if total > 0 else -1.0,
        )

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
        ram_budget_gb = stats.bytes_budget / 1024**3
        # Current free system RAM — reported alongside cache RAM so
        # the user can tell whether the headroom they see is the cache
        # being well below its budget, or the OS itself running tight.
        # ``psutil.virtual_memory().available`` is the canonical
        # cross-platform "RAM the OS could hand out without swapping"
        # number; cheap to call (one syscall).
        sys_avail_gb: float | None
        try:
            import psutil
            sys_avail_gb = psutil.virtual_memory().available / 1024**3
        except Exception:
            sys_avail_gb = None

        self._window.status_right.setText(
            format_perf_html(
                cache_n=stats.frames_cached,
                cache_total=cache_total,
                cache_ratio=cache_ratio,
                fps_effective=eff,
                fps_target=state.fps,
                ram_gb=ram_gb,
                ram_budget_gb=ram_budget_gb,
                sys_avail_gb=sys_avail_gb,
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

    # Side-tab selection (Color vs Comments) — falls outside of
    # saveState's coverage, restore explicitly. set_side_tab_index
    # clamps against the current tab count, so an old preference
    # value pointing at a tab that no longer exists is a no-op
    # rather than a crash.
    app._window.set_side_tab_index(prefs.side_tab_index)

    # View mode (frames vs timecode) — same reasoning, the View
    # menu's QAction state isn't part of saveState. The setter
    # routes through the same slot the user click triggers, so the
    # timeline + transport's frame display update accordingly.
    app._window.set_display_timecode(prefs.display_timecode)

    # Info-band visible-segments — restored from prefs so the user's
    # right-click toggles persist across runs.
    app._window.set_info_band_segments(prefs.info_band_segments)

    # Side panel (Color / Comments) visibility — explicit pref now
    # that the panel was lifted out of the dock system.
    app._window._side_dock.setVisible(prefs.side_panel_visible)
    # NB: transparency and alpha convention used to live on global
    # preferences; they're now per-layer fields auto-detected from
    # the source extension at ``Layer.from_sequence``. No global
    # restore step needed — the focus_changed handler syncs the
    # transport buttons / view actions to whichever layer is focused.

    # LayerPanel collapsed state (v1.0). The widget itself owns the
    # toggle button; we just sync the boolean at boot.
    panel = getattr(app._window, "_layer_panel", None)
    if panel is not None:
        panel.set_collapsed(prefs.layer_panel_collapsed)

    # Contact-sheet state (v1.5.14) — restore the persisted grid /
    # labels / enabled flag. The View menu's QAction is synced via
    # ``MainWindow.set_contact_sheet_enabled`` after we mutate the
    # underlying state, so the checkmark matches reality.
    from img_player.contact_sheet import ContactSheetState  # noqa: PLC0415
    try:
        cs_dict = prefs.contact_sheet_state
        app._contact_sheet_state = ContactSheetState.from_dict(cs_dict)
        app._window.set_contact_sheet_enabled(
            app._contact_sheet_state.enabled,
        )
        # Sync the transport bar's contact-sheet toggle too so the
        # toolbar checkmark matches the restored state.
        app._window.transport.set_contact_sheet_checked(
            app._contact_sheet_state.enabled,
        )
        # Sync the controller's always-advance flag — without this,
        # restoring an "enabled" contact-sheet state across launches
        # would leave the controller in its cache-stall default
        # state, and the user would see playback freeze on cold
        # cache despite the contact sheet being active.
        app._controller.set_always_advance(
            app._contact_sheet_state.enabled,
        )
        app._sync_contact_sheet_menu_state()
    except Exception:  # pragma: no cover — defensive
        log.exception("[contact_sheet] failed to restore prefs (using defaults)")

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

    # Seed the unmarked-EXR override status row so the user sees what's
    # currently pinned without having to open / re-pick anything.
    app._window.color_panel.set_unmarked_exr_default(
        prefs.unmarked_exr_source,
        prefs.unmarked_exr_view,
    )

    # Transparency-background pick — restore the GL viewport's uniform
    # AND the transport's BG button so what the user sees matches what
    # the menu reports.
    bg_mode = int(prefs.transparency_bg_mode)
    app._window.viewer.gl.set_color_params(transparency_bg_mode=bg_mode)
    app._window.transport.set_transparency_bg_mode(bg_mode)

    # Master audio — push the persisted volume into the audio output
    # AND the transport bar UI so the slider position matches the
    # actual gain being applied on the first frame the user plays.
    # Mute is implicit: a saved volume of 0 will naturally silence
    # the output and show the muted glyph on the button.
    try:
        master_vol = float(prefs.master_volume)
        app._audio_output.set_master_gain(master_vol)
        app._window.transport.set_master_volume(master_vol)
    except Exception:
        log.exception("[prefs] failed to restore master audio state")

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

    ``QScreen.colorSpace()`` isn't exposed on every PySide6 build /
    platform combo (notably absent from PySide6 6.11 on Windows even
    though the C++ API exists), so we ``getattr`` it and gracefully
    bail to ``None`` when missing — same outcome as a screen with no
    classifiable profile.
    """
    from PySide6.QtGui import QColorSpace

    screen = qapp.primaryScreen() if qapp is not None else None
    if screen is None:
        return None
    color_space_fn = getattr(screen, "colorSpace", None)
    if color_space_fn is None:
        return None
    try:
        qcs = color_space_fn()
    except Exception:
        return None
    if qcs is None or not qcs.isValid():
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
    boot_tune: PerformanceTune | None = None,
) -> int:
    """Public entry point used by ``python -m img_player``.

    ``cli_args`` propagates the parsed argparse Namespace down to
    ``ImgPlayerApp`` so the late-bind perf tune (slice 4) can re-apply
    user overrides at the same precedence as the boot pipeline. Older
    callers that don't pass it fall through to plain auto-tune, which
    is also fine.

    ``boot_tune`` is the *pre-runtime-constraint* tune resolved at boot
    (compute_tune → profile → CLI overrides). Stored on the app so the
    per-session re-tune (``app._retune_for_current_ram``) can recompute
    the cache budget against the live ``RuntimeState`` whenever the
    user opens a new project — letting them benefit from freed RAM
    without restarting Flick.
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
        boot_tune=boot_tune,
    )
    return app.run(initial_path=initial_path)
