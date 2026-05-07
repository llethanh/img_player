"""The :class:`ExportEngine` — orchestrates render → write across a frame range.

This is the loop body. The :class:`ExportWorker` runs it on a Qt
thread and pipes the engine's progress callbacks through Qt
signals; tests can also drive the engine synchronously.

The engine is deliberately Qt-free at its API surface — the only
Qt dependency comes from the renderer (QImage / QPainter for the
annotation bake). That keeps the loop testable without spinning
up a QThread.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from img_player.color.ocio_manager import OCIOManager
from img_player.export.renderer import (
    CompareRenderContext,
    FrameRenderer,
    RenderContext,
)
from img_player.export.settings import ExportSettings
from img_player.export.writers import BaseWriter, build_writer
from img_player.sequence.channels import ChannelSelection
from img_player.sequence.models import SequenceInfo

log = logging.getLogger(__name__)


# Codecs reject odd dimensions. We round UP (ceiling-to-even) on
# output_size when the user picks a custom resolution that lands odd.
def _even(n: int) -> int:
    return n if n % 2 == 0 else n + 1


@dataclass
class EngineResult:
    """Returned by :meth:`ExportEngine.run`. Carries enough state for
    the success / cancel / fail messages in the UI."""

    output_path: Path
    frames_written: int
    duration_s: float
    canceled: bool = False


class ExportEngine:
    """Owns one export run: open writer, loop frames, close writer."""

    def __init__(
        self,
        settings: ExportSettings,
        sequence: SequenceInfo,
        annotation_store,
        ocio_manager: OCIOManager | None,
        *,
        source_colorspace: str | None,
        display: str | None,
        view: str | None,
        sidecar_source: Path | None = None,
        channel_selection: ChannelSelection | None = None,
        compare: CompareRenderContext | None = None,
    ) -> None:
        self._settings = settings
        self._sequence = sequence
        # Capture the live channel state so the export reproduces the
        # exact channel the user has on screen. ``None`` falls back to
        # the legacy default-channels path.
        self._channel_selection = channel_selection
        # The CPU OCIO processor is built once at engine setup. If
        # OCIO isn't available or the user disabled the transform we
        # leave it at None and the renderer skips the colour step.
        ocio_proc = None
        if ocio_manager is not None and settings.apply_display_transform:
            ocio_proc = self._build_cpu_processor(
                ocio_manager,
                source_colorspace=source_colorspace,
                display=display,
                view=view,
            )
        # Compare overlay only honoured when the user ticked the
        # bake-compare option AND the caller passed a live snapshot.
        # Either condition flipping off → None → renderer falls back
        # to the single-sequence path.
        compare_ctx = compare if settings.bake_compare else None
        ctx = RenderContext(
            sequence=sequence,
            annotation_store=annotation_store if settings.bake_annotations else None,
            ocio_cpu_processor=ocio_proc,
            channel_selection=channel_selection,
            compare=compare_ctx,
        )
        self._renderer = FrameRenderer(ctx, settings)
        self._sidecar_source = sidecar_source
        # Cancel flag set by the worker thread. Plain attribute reads
        # are atomic in CPython.
        self._cancel = False
        # The writer is built lazily inside ``run()`` so a failed
        # OCIO setup doesn't leave a half-opened video container on
        # disk — we want the construction to fail BEFORE we touch any
        # output file.
        self._writer: BaseWriter | None = None

    # ------------------------------------------------------------------ Public

    @property
    def total_frames(self) -> int:
        return self._settings.total_frames

    def cancel(self) -> None:
        """Mark the engine for cancellation. The next loop iteration
        notices and returns. Idempotent."""
        self._cancel = True

    def discard_partial_output(self) -> None:
        """Delete whatever the writer wrote before cancellation.

        Called by the export orchestrator when the user opts to
        discard partial files in the cancel-confirmation dialog.
        Safe to call after :meth:`run` returned with
        ``canceled=True`` — the writer was just closed cleanly
        there, so its file list / output path is still valid.
        Video writers always discard regardless (a mid-encode
        container is unreadable anyway); the prompt happens for
        image-sequence writers only.
        """
        if self._writer is None:
            return
        try:
            self._writer.abort()
        except Exception:  # pragma: no cover — defensive
            log.exception("[export] discard_partial_output failed")

    def run(
        self,
        progress_cb: Callable[[int, int, float], None] | None = None,
    ) -> EngineResult:
        """Synchronously execute the export.

        ``progress_cb(current_frame, total, fps_running)`` is invoked
        AFTER each successful frame write. ``current_frame`` is
        1-based for the user's mental model ("frame 247 / 500").

        Raises :exc:`Exception` on any unrecoverable I/O error — the
        worker catches and routes to a ``failed`` Qt signal.
        """
        settings = self._settings
        settings.validate()

        out_w, out_h = self._resolve_output_size()
        out_fps = self._resolve_output_fps()

        # User-overridden basename wins; otherwise fall back to the
        # source sequence's base_name (legacy behaviour). Strips
        # trailing separators / spaces so a stem like ``render._``
        # produces ``render.0001.png`` rather than ``render.._0001.png``.
        custom = (settings.basename or "").strip()
        if custom:
            basename = custom.rstrip("._- ") or "export"
        else:
            basename = self._sequence.base_name.rstrip("._-") or "export"
        self._writer = build_writer(settings, basename=basename)
        self._writer.open(settings, out_w, out_h, out_fps)

        start = time.monotonic()
        frames_written = 0
        try:
            for i in range(settings.total_frames):
                if self._cancel:
                    log.info("[export] canceled at frame %d / %d",
                             i, settings.total_frames)
                    # Close the writer cleanly so partial files stay
                    # readable on disk. The orchestrator decides
                    # whether to keep or discard them via
                    # :meth:`discard_partial_output` after asking the
                    # user — auto-deleting here would steal that
                    # choice.
                    self._writer.close()
                    return EngineResult(
                        output_path=self._writer.output_path(),
                        frames_written=frames_written,
                        duration_s=time.monotonic() - start,
                        canceled=True,
                    )
                source_frame = settings.in_frame + i
                arr = self._renderer.render(source_frame, (out_w, out_h))
                self._writer.write_frame(arr, i)
                frames_written += 1
                if progress_cb is not None:
                    elapsed = max(1e-6, time.monotonic() - start)
                    fps_running = frames_written / elapsed
                    progress_cb(frames_written, settings.total_frames, fps_running)
            self._writer.close()
            # Optional sidecar copy.
            if (
                settings.bake_annotations
                and settings.copy_sidecar
                and self._sidecar_source is not None
                and self._sidecar_source.exists()
            ):
                try:
                    target = settings.output_dir / self._sidecar_source.name
                    shutil.copyfile(self._sidecar_source, target)
                except OSError:
                    log.exception(
                        "[export] failed to copy sidecar %s", self._sidecar_source,
                    )
            return EngineResult(
                output_path=self._writer.output_path(),
                frames_written=frames_written,
                duration_s=time.monotonic() - start,
                canceled=False,
            )
        except Exception:
            # Best-effort cleanup of the partial output.
            try:
                if self._writer is not None:
                    self._writer.abort()
            except Exception:  # pragma: no cover — defensive
                log.exception("[export] secondary error during abort")
            raise

    # ------------------------------------------------------------------ Internals

    def _resolve_output_size(self) -> tuple[int, int]:
        """Source-or-explicit + even-dimensions guard for video."""
        if self._settings.width is not None and self._settings.height is not None:
            w = self._settings.width
            h = self._settings.height
        else:
            w = self._sequence.width or 1920
            h = self._sequence.height or 1080
        if self._settings.is_video:
            w = _even(w)
            h = _even(h)
        return w, h

    def _resolve_output_fps(self) -> float:
        if self._settings.fps is not None and self._settings.fps > 0:
            return float(self._settings.fps)
        return float(self._sequence.fps_default or 24.0)

    @staticmethod
    def _build_cpu_processor(
        manager: OCIOManager,
        *,
        source_colorspace: str | None,
        display: str | None,
        view: str | None,
    ):
        """Build a CPU display-view processor from the OCIO manager.

        Falls back to ``getDefaultCpuProcessor`` on the resolved
        :class:`PyOpenColorIO.Processor`. Returns ``None`` if any
        dependency is missing — the renderer treats that as
        "skip the colour step".
        """
        try:
            src = source_colorspace or manager.role("scene_linear") or "Linear Rec.709 (sRGB)"
            disp = display or manager.default_display()
            v = view or manager.default_view(disp)
            proc = manager.get_display_view_processor(src, disp, v)
            return proc.getDefaultCPUProcessor()
        except Exception:
            log.exception("[export] failed to build OCIO CPU processor; baking raw")
            return None
