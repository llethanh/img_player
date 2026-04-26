"""Benchmark instrumentation for img_player.

This package adds an optional, low-overhead recording layer that captures
per-frame metrics during playback. It is enabled only when the runner is
active (cf. :func:`enable`); otherwise every hook is a fast no-op.

Public API:
    enable()/disable() — toggle recording globally.
    is_enabled()       — fast check for hook callsites.
    record_*()         — typed hooks called from controller / app / GL viewport.
    take_samples()     — atomically dequeue all samples (used by the runner).

The data flow is one-way: hot-path code only ever appends; the runner reads
the samples after the playback session has finished.
"""

from img_player.bench.recorder import (
    DecodeSample,
    PaintSample,
    TickSample,
    disable,
    enable,
    is_enabled,
    record_decode,
    record_paint,
    record_tick,
    reset,
    take_samples,
)

__all__ = [
    "DecodeSample",
    "PaintSample",
    "TickSample",
    "disable",
    "enable",
    "is_enabled",
    "record_decode",
    "record_paint",
    "record_tick",
    "reset",
    "take_samples",
]
