"""Thread-safe sample collector.

Designed so that the per-frame hooks add **near-zero overhead** when the
recorder is disabled (the common case): a single ``if not _ENABLED`` branch.

When enabled, samples are appended to per-kind ``deque`` objects guarded by
a single lock. ``take_samples()`` atomically swaps in fresh empty deques
and returns the previous batch, which the runner then aggregates.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

# ----------------------------------------------------------------------- Sample dataclasses


@dataclass(frozen=True, slots=True)
class TickSample:
    """One playback tick from the controller's QTimer."""

    t_ms: float          # monotonic ms since recorder enable()
    requested_frame: int
    cache_hit: bool      # was the requested frame already decoded at tick time?
    pending_decodes: int # number of in-flight + queued decode jobs


@dataclass(frozen=True, slots=True)
class PaintSample:
    """One ``paintGL()`` execution from the GL viewport."""

    t_ms: float          # monotonic ms since recorder enable()
    displayed_frame: int # what the controller asked the viewport to show
    upload_us: float     # glTexImage2D / glTexSubImage2D wall-clock
    paint_us: float      # whole paintGL body (clear + uniforms + draw + upload)
    width: int
    height: int
    channels: int


@dataclass(frozen=True, slots=True)
class DecodeSample:
    """One frame decoded by a worker thread."""

    t_ms: float          # monotonic ms when the decode completed
    frame: int
    decode_ms: float     # duration of the OIIO read_frame call
    nbytes: int          # size of the decoded numpy array


# ----------------------------------------------------------------------- Module state

# A lock guards all mutable state. Recorder is process-wide singleton on purpose:
# benchmarking is a development tool, not a feature shipped to end users.
_LOCK = threading.Lock()
_ENABLED = False
_T0 = 0.0
_TICKS: Deque[TickSample] = deque()
_PAINTS: Deque[PaintSample] = deque()
_DECODES: Deque[DecodeSample] = deque()


def is_enabled() -> bool:
    """Fast read of the recorder state. No lock — Python attribute reads are atomic."""
    return _ENABLED


def enable() -> None:
    """Start recording. Resets the time origin and any previously held samples."""
    global _ENABLED, _T0, _TICKS, _PAINTS, _DECODES
    with _LOCK:
        _T0 = time.monotonic()
        _TICKS = deque()
        _PAINTS = deque()
        _DECODES = deque()
        _ENABLED = True


def disable() -> None:
    global _ENABLED
    with _LOCK:
        _ENABLED = False


def reset() -> None:
    """Drop currently held samples without changing the enabled state."""
    with _LOCK:
        _TICKS.clear()
        _PAINTS.clear()
        _DECODES.clear()


def _now_ms() -> float:
    return (time.monotonic() - _T0) * 1000.0


# ----------------------------------------------------------------------- Hooks

def record_tick(requested_frame: int, cache_hit: bool, pending_decodes: int) -> None:
    if not _ENABLED:
        return
    s = TickSample(
        t_ms=_now_ms(),
        requested_frame=requested_frame,
        cache_hit=cache_hit,
        pending_decodes=pending_decodes,
    )
    with _LOCK:
        _TICKS.append(s)


def record_paint(
    displayed_frame: int,
    upload_us: float,
    paint_us: float,
    width: int,
    height: int,
    channels: int,
) -> None:
    if not _ENABLED:
        return
    s = PaintSample(
        t_ms=_now_ms(),
        displayed_frame=displayed_frame,
        upload_us=upload_us,
        paint_us=paint_us,
        width=width,
        height=height,
        channels=channels,
    )
    with _LOCK:
        _PAINTS.append(s)


def record_decode(frame: int, decode_ms: float, nbytes: int) -> None:
    if not _ENABLED:
        return
    s = DecodeSample(
        t_ms=_now_ms(),
        frame=frame,
        decode_ms=decode_ms,
        nbytes=nbytes,
    )
    with _LOCK:
        _DECODES.append(s)


# ----------------------------------------------------------------------- Drain

def take_samples() -> tuple[list[TickSample], list[PaintSample], list[DecodeSample]]:
    """Atomically swap in fresh deques and return the captured ones as lists."""
    global _TICKS, _PAINTS, _DECODES
    with _LOCK:
        ticks, paints, decodes = _TICKS, _PAINTS, _DECODES
        _TICKS = deque()
        _PAINTS = deque()
        _DECODES = deque()
    return list(ticks), list(paints), list(decodes)
