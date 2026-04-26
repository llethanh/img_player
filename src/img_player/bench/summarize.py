"""Aggregate raw bench samples into a human + machine readable report.

The runner collects three streams (ticks, paints, decodes). This module
turns them into:

* A console-friendly text summary (``format_summary``).
* A JSON-serialisable dict (``build_report``) that we dump to disk so that
  a future run can compare deltas vs this baseline.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from img_player.bench.recorder import DecodeSample, PaintSample, TickSample


# ----------------------------------------------------------------------- Helpers

def _percentile(values: list[float], q: float) -> float:
    """Return the q-th percentile (0..100). Empty input -> NaN."""
    if not values:
        return float("nan")
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def _stats(values: list[float]) -> dict[str, float]:
    """Reduce a list to a JSON-serialisable stats dict."""
    if not values:
        return {
            "count": 0, "mean": float("nan"),
            "p50": float("nan"), "p95": float("nan"), "p99": float("nan"),
            "min": float("nan"), "max": float("nan"),
        }
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "min": min(values),
        "max": max(values),
    }


def _stats_line(name: str, unit: str, s: dict[str, float]) -> str:
    """One line of pre-aggregated stats, padded for column alignment."""
    if not s.get("count"):
        return f"  {name:<22} (no samples)"
    return (
        f"  {name:<22} "
        f"n={int(s['count']):<5}  "
        f"mean={s['mean']:7.2f}  "
        f"p50={s['p50']:7.2f}  "
        f"p95={s['p95']:7.2f}  "
        f"p99={s['p99']:7.2f}  "
        f"max={s['max']:7.2f} {unit}"
    )


# ----------------------------------------------------------------------- Aggregation

@dataclass
class BenchContext:
    """Static metadata about the run, attached to the report."""

    sequence_label: str
    frame_count: int
    width: int
    height: int
    channels: int
    target_fps: float
    cache_budget_bytes: int
    num_workers: int
    passes_played: int
    warmup_frames: int


def build_report(
    ctx: BenchContext,
    ticks: list[TickSample],
    paints: list[PaintSample],
    decodes: list[DecodeSample],
) -> dict[str, Any]:
    """Distill samples + context into a structured report dict."""
    # Inter-tick gaps tell us the *real* fps the play loop achieves.
    inter_tick_ms = [b.t_ms - a.t_ms for a, b in zip(ticks[:-1], ticks[1:])]
    effective_fps = (
        1000.0 / statistics.fmean(inter_tick_ms) if inter_tick_ms else float("nan")
    )

    # Cache hit rate at tick time — a "miss" is a frame the controller wanted
    # to display but that wasn't ready (counts as a drop in the player UI).
    if ticks:
        hit_count = sum(1 for t in ticks if t.cache_hit)
        cache_hit_rate = hit_count / len(ticks)
    else:
        cache_hit_rate = float("nan")

    upload_us = [p.upload_us for p in paints]
    paint_us = [p.paint_us for p in paints]
    inter_paint_ms = [b.t_ms - a.t_ms for a, b in zip(paints[:-1], paints[1:])]
    effective_paint_fps = (
        1000.0 / statistics.fmean(inter_paint_ms) if inter_paint_ms else float("nan")
    )

    decode_ms = [d.decode_ms for d in decodes]
    decoded_bytes = sum(d.nbytes for d in decodes)

    return {
        "schema_version": 1,
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "context": {
            "sequence_label": ctx.sequence_label,
            "frame_count": ctx.frame_count,
            "resolution": [ctx.width, ctx.height],
            "channels": ctx.channels,
            "target_fps": ctx.target_fps,
            "cache_budget_bytes": ctx.cache_budget_bytes,
            "num_workers": ctx.num_workers,
            "passes_played": ctx.passes_played,
            "warmup_frames": ctx.warmup_frames,
        },
        "tick": {
            "samples": len(ticks),
            "effective_fps": effective_fps,
            "cache_hit_rate": cache_hit_rate,
            "inter_tick_ms": _stats(inter_tick_ms),
        },
        "paint": {
            "samples": len(paints),
            "effective_fps": effective_paint_fps,
            "upload_us": _stats(upload_us),
            "paint_us": _stats(paint_us),
            "inter_paint_ms": _stats(inter_paint_ms),
        },
        "decode": {
            "samples": len(decodes),
            "decode_ms": _stats(decode_ms),
            "total_bytes": decoded_bytes,
        },
    }


def format_summary(report: dict[str, Any]) -> str:
    """Render the report as a compact console string."""
    ctx = report["context"]
    tick = report["tick"]
    paint = report["paint"]
    decode = report["decode"]

    lines = [
        "",
        "=" * 78,
        f"img_player benchmark — {ctx['sequence_label']}",
        f"  resolution : {ctx['resolution'][0]} × {ctx['resolution'][1]} × {ctx['channels']} ch",
        f"  frames     : {ctx['frame_count']}  ×  {ctx['passes_played']} passes",
        f"  target fps : {ctx['target_fps']:.3f}",
        f"  cache      : {ctx['cache_budget_bytes'] / 1024**3:.1f} GiB / "
        f"{ctx['num_workers']} workers",
        "-" * 78,
        "Tick (controller QTimer):",
        f"  effective fps  : {tick['effective_fps']:7.3f}  "
        f"(target {ctx['target_fps']:.3f})",
        f"  cache hit rate : {tick['cache_hit_rate'] * 100:6.2f} %",
        _stats_line("inter-tick gap", "ms", tick["inter_tick_ms"]),
        "-" * 78,
        "Paint (paintGL body):",
        f"  effective fps  : {paint['effective_fps']:7.3f}",
        _stats_line("upload", "µs", paint["upload_us"]),
        _stats_line("paint total", "µs", paint["paint_us"]),
        _stats_line("inter-paint gap", "ms", paint["inter_paint_ms"]),
        "-" * 78,
        "Decode (worker pool):",
        _stats_line("decode time", "ms", decode["decode_ms"]),
        f"  total decoded   : "
        f"{decode['total_bytes'] / 1024**2:.0f} MiB ({decode['samples']} frames)",
        "=" * 78,
        "",
    ]
    return "\n".join(lines)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
