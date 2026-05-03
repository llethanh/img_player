"""Synchronous PyAV video decoder with seek-then-decode-forward strategy.

A :class:`VideoSource` wraps an ``av.container.Container`` for one
video file and answers ``frame_at_time(t) -> ndarray`` queries. It is
intentionally **synchronous**: a higher-level component (the future
``VideoDecoderRunner``) will hold one of these on a worker thread and
push results to the cache. Keeping the decoder itself synchronous
makes seek semantics testable without QThread infrastructure.

Strategy:

* **Forward decode (cheap)**: when the requested time is at or after
  the last decoded frame, just keep pulling packets — long-GOP H.264
  / H.265 streams are decode-forward friendly, this is the hot path.
* **Backward seek (costly)**: when the requested time is before the
  last decoded frame, ``container.seek()`` to the requested PTS
  (FFmpeg lands at the keyframe at or before, "backward" mode), then
  decode-forward to find the frame whose PTS bracket the target.

A single-frame cache (last decoded frame + its PTS) makes scrubbing
within a frame's display duration a no-op and stops the decoder from
re-decoding the same frame on every paint event.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import numpy as np

from img_player.media.video_probe import VideoMetadata, probe_video

log = logging.getLogger(__name__)


class VideoSource:
    """Open a video file once, answer ``frame_at_time(t)`` queries.

    Not thread-safe — one instance per worker thread. ``close()`` (or
    the context manager protocol) releases the container; calling
    ``frame_at_time`` after close raises.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._meta = probe_video(self._path)
        if not self._meta.has_video:
            raise ValueError(f"No video stream in {self._path}")

        # Imported lazily so module import is light — same pattern as
        # in ``video_probe`` for headless / non-video code paths.
        import av  # type: ignore[import-untyped]

        self._container = av.open(str(self._path))
        # Pick the first (= primary) video stream. Multi-track files
        # are rare in VFX delivery; if needed later, expose a stream
        # index parameter.
        self._stream = self._container.streams.video[0]
        # ``thread_type="AUTO"`` lets FFmpeg pick FRAME/SLICE threading
        # per codec — H.264 typically defaults to slice, ProRes to frame.
        # Either is well-suited to a player workload (low latency on
        # individual decodes, cores stay busy on long-GOP streams).
        self._stream.thread_type = "AUTO"

        # Cache of the most recently decoded frame: (pts_seconds,
        # ndarray). Holding one frame is enough — repeated paints of
        # the same time hit this cache, while a forward step always
        # advances past it. Bigger caches (an LRU) duplicate what the
        # MasterFrameCache already provides at the layer level.
        self._last_pts: float | None = None
        self._last_frame: np.ndarray | None = None
        # Persistent decoder generator — PyAV's ``container.decode()``
        # returns a generator that pulls packets from the demuxer; once
        # exhausted (EOF or interrupted via ``break``/``return``), we
        # need a fresh one. Recreated on seek and on EOF; survives
        # across ``frame_at_time`` calls so a sequence of forward queries
        # decodes through the stream without redundant seeks.
        self._decoder = self._container.decode(self._stream)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> VideoMetadata:
        return self._meta

    @property
    def fps(self) -> Fraction:
        # The probe always populates ``fps`` for files that have a
        # video stream, but the type system can't see that — assert
        # at the access point so the failure mode is obvious if a
        # truly weird container slips through.
        assert self._meta.fps is not None, "VideoSource requires a known fps"
        return self._meta.fps

    @property
    def duration_seconds(self) -> float:
        # Same logic as ``fps``: probe + metadata invariant on open.
        if self._meta.duration_seconds is not None:
            return self._meta.duration_seconds
        # Fallback: derive from frame_count and fps when the container
        # didn't report a clean duration.
        if self._meta.frame_count is not None and self._meta.fps is not None:
            return self._meta.frame_count / float(self._meta.fps)
        raise RuntimeError(f"Cannot determine duration of {self._path}")

    @property
    def width(self) -> int:
        assert self._meta.width is not None
        return self._meta.width

    @property
    def height(self) -> int:
        assert self._meta.height is not None
        return self._meta.height

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def frame_at_time(self, t_seconds: float) -> np.ndarray:
        """Return the RGB24 frame whose presentation interval contains ``t``.

        Output is an ``(H, W, 3)`` ``uint8`` ndarray in RGB order. The
        future OCIO input transform will consume this; for now the
        viewport's existing path treats it as already-display-ready.

        Clamps ``t`` to ``[0, duration)``: requesting before-start
        returns the first frame, after-end returns the last decoded
        frame on file (matches the "freeze on tail" behaviour of
        common players).
        """
        if self._container is None:
            raise RuntimeError("VideoSource is closed")

        # Clamp to the valid range so ``seek`` doesn't bounce off the
        # end and the caller doesn't have to special-case boundaries.
        t = max(0.0, t_seconds)
        if self._meta.duration_seconds is not None:
            # Stay strictly less than duration so the last frame's PTS
            # (≤ duration - 1/fps) is reachable.
            t = min(t, max(0.0, self._meta.duration_seconds - 1e-9))

        # Cache hit: same display interval as the last decoded frame.
        # The "interval" check uses 1/fps as the upper bound — within
        # that window, the same frame is the correct answer.
        if self._last_frame is not None and self._last_pts is not None:
            interval = 1.0 / float(self.fps)
            if self._last_pts <= t < self._last_pts + interval:
                return self._last_frame

        # If the request is before our current decode position, we
        # need a backwards seek (FFmpeg lands at or before the
        # requested PTS on a keyframe boundary).
        # Tolerance: a fraction of a frame interval, so float jitter
        # in PTS arithmetic doesn't trigger spurious seeks.
        interval = 1.0 / float(self.fps)
        needs_seek = (
            self._last_pts is None
            or t < self._last_pts - 0.5 * interval
        )
        if needs_seek:
            self._seek_to(t)

        # Decode-forward until we land on a frame whose PTS interval
        # contains ``t``, or we run out of frames. The ``_decoder``
        # generator persists across calls so consecutive forward
        # queries don't re-seek; it is only recreated on seek (above)
        # or after EOF (below).
        retried = False
        while True:
            try:
                frame = next(self._decoder)
            except StopIteration:
                # Generator exhausted — end of stream. Return the last
                # frame we have on file (freeze-on-tail), or, if we've
                # truly never decoded anything for this source, raise.
                if self._last_frame is not None:
                    return self._last_frame
                raise RuntimeError(
                    f"No decodable frame found at t={t} in {self._path}"
                )
            except Exception as exc:  # PyAV EOFError, codec hiccups
                # PyAV occasionally raises av.error.EOFError on the
                # final packet of a short clip — treat as end of
                # stream, same recovery as StopIteration above.
                if "End of file" in str(exc) or "EOF" in type(exc).__name__:
                    if self._last_frame is not None:
                        return self._last_frame
                    raise RuntimeError(
                        f"No decodable frame found at t={t} in {self._path}"
                    ) from exc
                raise

            pts = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None
                else 0.0
            )
            if pts <= t < pts + interval:
                target_frame = frame.to_ndarray(format="rgb24")
                self._last_pts = pts
                self._last_frame = target_frame
                return target_frame
            if pts > t + interval:
                # Overshot — the right frame was before this one and
                # we've already consumed it. Re-seek once and try
                # again. ``retried`` guards against infinite loops if
                # the seek lands past ``t`` (can happen when ``t`` is
                # before the first keyframe).
                if retried:
                    target_frame = frame.to_ndarray(format="rgb24")
                    self._last_pts = pts
                    self._last_frame = target_frame
                    return target_frame
                self._seek_to(t)
                retried = True
                continue
            # pts < t — keep decoding forward until we bracket the target.

    def _seek_to(self, t_seconds: float) -> None:
        """Seek the demuxer to the keyframe at or before ``t``.

        Uses ``container.seek`` rather than ``stream.seek`` because
        the container-level call respects the ``backward`` flag
        across all streams (audio + video stay aligned), while
        per-stream seek can leave audio dangling.
        """
        # ``container.seek`` takes microseconds in AV_TIME_BASE units.
        target_us = int(t_seconds * 1_000_000)
        # ``backward=True`` is the keyframe-at-or-before semantic.
        # ``any_frame=False`` forces it to a keyframe (otherwise the
        # decoder would have to re-derive the keyframe boundary).
        self._container.seek(target_us, backward=True, any_frame=False)
        # ``seek`` flushes the decoder, but the per-frame state we
        # cache (last_pts) becomes stale until the next decode lands
        # — invalidate it so the bracketing logic above doesn't
        # short-circuit on a now-incorrect cache.
        self._last_pts = None
        self._last_frame = None
        # Recreate the persistent decoder generator — the previous one
        # is bound to the demuxer state pre-seek and would yield
        # stale packets (or raise) post-seek.
        self._decoder = self._container.decode(self._stream)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None  # type: ignore[assignment]
            self._last_frame = None
            self._last_pts = None

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Defensive — explicit ``close`` is preferred (the GC timing
        # of FFmpeg containers can hold file handles longer than
        # expected on Windows).
        try:
            self.close()
        except Exception:
            pass
