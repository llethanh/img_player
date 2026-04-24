"""Qt application bootstrap: builds the main window, cache, controller and wires them."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
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
DEFAULT_CACHE_BUDGET_BYTES = 4 * 1024**3  # 4 GB
DEFAULT_NUM_WORKERS = 4


class ImgPlayerApp:
    """Owns every long-lived object (cache, controller, window) and their wiring."""

    def __init__(self, argv: list[str]) -> None:
        self._qapp = QApplication.instance() or QApplication(argv)

        self._ocio = OCIOManager()
        self._cache = FrameCache(
            budget_bytes=DEFAULT_CACHE_BUDGET_BYTES,
            num_workers=DEFAULT_NUM_WORKERS,
        )
        self._controller = PlayerController(self._cache)
        self._window = MainWindow(self._ocio)

        # A light-touch status timer so we can surface cache hit/miss info.
        self._status_timer = QTimer(self._window)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._refresh_status)

        self._wire()

        # Push the initial color params so the GL shader is ready before any
        # frame arrives.
        self._window.color_panel.emit_current()

    # ------------------------------------------------------------------ Lifecycle

    def run(self, initial_path: Path | None = None) -> int:
        self._window.show()
        self._status_timer.start()
        if initial_path is not None:
            self._open_path(initial_path)
        exit_code = int(self._qapp.exec())
        self._shutdown()
        return exit_code

    def _shutdown(self) -> None:
        self._status_timer.stop()
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
        self._window.frame_requested.connect(self._controller.seek)
        self._window.open_requested.connect(self._open_path)
        self._window.exposure_step.connect(self._window.color_panel.bump_exposure)

        # ColorPanel -> GL viewport
        self._window.color_panel.color_params_changed.connect(self._on_color_params)

    # ------------------------------------------------------------------ Handlers

    def _on_frame_changed(self, frame: int) -> None:
        arr = self._cache.get(frame)
        if arr is not None:
            if arr.shape[2] > 4:
                arr = arr[:, :, :4]  # viewport only handles RGB/RGBA today
            self._window.viewer.gl.set_frame(arr)
        self._window.timeline.set_current_frame(frame)

    def _on_state_changed(self, state: PlaybackState) -> None:
        self._window.transport.update_from_state(state)

    def _on_play_toggled(self) -> None:
        if self._controller.state.is_playing:
            self._controller.pause()
        else:
            self._controller.play()

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
        try:
            seq = scan(path)
        except SequenceNotFoundError as err:
            QMessageBox.warning(self._window, "Cannot open", str(err))
            return
        log.info("loaded sequence: %s (%d frames)", seq.display_pattern(), seq.frame_count)
        self._window.update_sequence_info(seq)
        self._guess_source_colorspace(seq)
        self._controller.load_sequence(seq)
        self._window.set_status(
            f"Loaded {seq.display_pattern()} ({seq.frame_count} frames, {seq.width}x{seq.height})"
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


def run_gui(argv: list[str] | None = None, initial_path: Path | None = None) -> int:
    """Public entry point used by ``python -m img_player``."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    app = ImgPlayerApp(argv or sys.argv)
    return app.run(initial_path=initial_path)
