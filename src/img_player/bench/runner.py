"""Headless-ish benchmark driver for img_player.

The runner reuses the real :class:`ImgPlayerApp` (window, GL viewport,
controller, cache) — we want to measure *the actual playback path*, not a
synthetic micro-benchmark. We just attach a state machine that:

1. Waits for the sequence to load.
2. Optionally waits a number of warmup frames for the cache to fill.
3. Hits "play".
4. Counts complete passes, then asks Qt to quit.
5. Aggregates samples and dumps a JSON report.

If the user wants to inspect the run visually they can pass ``--show`` —
otherwise the window still appears (Qt requires a display to make a GL
context) but we close it as soon as the run is over.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from PySide6.QtCore import QTimer

from img_player.app import (
    DEFAULT_CACHE_BUDGET_BYTES,
    DEFAULT_NUM_WORKERS,
    ImgPlayerApp,
)
from img_player.bench import recorder
from img_player.bench.summarize import (
    BenchContext,
    build_report,
    format_summary,
    write_report,
)
from img_player.player.state import LoopMode

log = logging.getLogger(__name__)


class BenchmarkSession:
    """Glue between Qt signals and the benchmark state machine."""

    def __init__(
        self,
        app: ImgPlayerApp,
        *,
        passes: int,
        warmup_frames: int,
        target_fps: float,
        output_path: Path,
    ) -> None:
        self._app = app
        self._passes = max(1, passes)
        self._warmup = max(0, warmup_frames)
        self._target_fps = target_fps
        self._output_path = output_path

        self._sequence_loaded = False
        self._warmup_done = False
        self._passes_seen = 0
        self._last_frame: int | None = None
        # Maximum wall-clock time to wait for the cache to warm up before
        # giving up and starting playback anyway. Otherwise a too-large
        # warmup target on a slow disk would hang the bench forever.
        self._warmup_deadline_s = 30.0
        self._warmup_started_at: float | None = None

        # Tick the watcher every 50 ms — fine grained enough to start playback
        # promptly after warmup, coarse enough to add no measurable load.
        self._poll = QTimer()
        self._poll.setInterval(50)
        self._poll.timeout.connect(self._tick)

        # We listen to frame_changed to count passes (the controller wraps
        # to the in-frame at end of pass when LoopMode.LOOP is active).
        self._app._controller.frame_changed.connect(self._on_frame_changed)

    # ------------------------------------------------------------------ Lifecycle

    def start(self) -> None:
        """Called once the Qt event loop is running."""
        # Force LOOP mode and the requested FPS.
        self._app._controller.set_loop_mode(LoopMode.LOOP)
        self._app._controller.set_fps(self._target_fps)
        self._poll.start()

    def _tick(self) -> None:
        seq = self._app._controller.sequence
        if seq is None:
            return  # scan still pending

        if not self._sequence_loaded:
            self._sequence_loaded = True
            self._warmup_started_at = time.monotonic()
            log.info("bench: sequence loaded (%d frames), warming up to %d frames",
                     seq.frame_count, self._warmup)

        if not self._warmup_done:
            cached = self._app._cache.cached_frames()
            elapsed = time.monotonic() - (self._warmup_started_at or time.monotonic())
            if len(cached) >= self._warmup or elapsed > self._warmup_deadline_s:
                self._warmup_done = True
                if elapsed > self._warmup_deadline_s:
                    log.warning("bench: warmup timeout (%.1fs) — proceeding with %d cached frames",
                                elapsed, len(cached))
                else:
                    log.info("bench: warmup done (%d frames cached in %.1fs) — starting playback",
                             len(cached), elapsed)

                # Enable the recorder *now*, just before play() — anything
                # before this is warmup noise we don't want in the stats.
                recorder.enable()
                self._app._controller.play()

    def _on_frame_changed(self, frame: int) -> None:
        if not self._warmup_done:
            return
        seq = self._app._controller.sequence
        if seq is None:
            return
        if self._last_frame is not None and self._last_frame > frame and frame == seq.first_frame:
            # Wrap-around: a pass just completed.
            self._passes_seen += 1
            log.info("bench: pass %d/%d done", self._passes_seen, self._passes)
            if self._passes_seen >= self._passes:
                self._finish()
        self._last_frame = frame

    def _finish(self) -> None:
        self._poll.stop()
        recorder.disable()
        self._app._controller.pause()

        seq = self._app._controller.sequence
        ticks, paints, decodes = recorder.take_samples()

        if seq is None:
            log.error("bench: no sequence loaded — nothing to report")
            self._app._qapp.quit()
            return

        # If scan was metadata-less (probe=False), pull resolution from the
        # first paint sample we collected — by then we've actually decoded
        # at least one frame, so the GL viewport knows the pixel size.
        width = seq.width or 0
        height = seq.height or 0
        if (not width or not height) and paints:
            width = paints[0].width or width
            height = paints[0].height or height
        channels = (
            paints[0].channels if paints
            else (len(seq.channel_names) if seq.channel_names else 4)
        )

        ctx = BenchContext(
            sequence_label=seq.display_pattern(),
            frame_count=seq.frame_count,
            width=width,
            height=height,
            channels=channels,
            target_fps=self._target_fps,
            cache_budget_bytes=self._app._cache._budget,  # noqa: SLF001
            num_workers=self._app._cache._pool._num_workers,  # noqa: SLF001
            passes_played=self._passes_seen,
            warmup_frames=self._warmup,
        )
        report = build_report(ctx, ticks, paints, decodes)
        write_report(self._output_path, report)

        # Print to stdout (the user will see this in the terminal) and to log.
        summary = format_summary(report)
        sys.stdout.write(summary)
        sys.stdout.flush()
        log.info("bench: report written to %s", self._output_path)

        # Quit the Qt event loop. Use singleShot so any in-flight signals drain.
        QTimer.singleShot(50, self._app._qapp.quit)


def run_benchmark(
    path: Path,
    *,
    passes: int = 3,
    warmup_frames: int = 30,
    target_fps: float = 24.0,
    output: Path | None = None,
    cache_budget_bytes: int = DEFAULT_CACHE_BUDGET_BYTES,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> int:
    """Bootstraps the app, plays N passes, and writes a JSON report."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output or Path("perf") / f"bench_{timestamp}.json"

    app = ImgPlayerApp(
        sys.argv,
        cache_budget_bytes=cache_budget_bytes,
        num_workers=num_workers,
    )
    session = BenchmarkSession(
        app,
        passes=passes,
        warmup_frames=warmup_frames,
        target_fps=target_fps,
        output_path=output_path,
    )
    # The Qt event loop drives everything from here. start() is called once
    # the event loop is running so timers arm correctly.
    QTimer.singleShot(0, session.start)
    return app.run(initial_path=path)
