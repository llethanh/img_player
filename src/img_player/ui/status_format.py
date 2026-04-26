"""Pure-function helpers for the right side of the status bar.

The status bar's right block shows three perf indicators (cache, fps,
RAM) with conditional coloured dots. Keeping the formatting and colour
logic here — separated from the Qt widgets — makes the rules unit-
testable without standing up a `QApplication`.

Usage from ``app._refresh_status``::

    html = format_perf_html(
        cache_n=42, cache_total=90, cache_ratio=0.46,
        fps_effective=23.8, fps_target=24.0,
        ram_gb=2.6,
    )
    main_window.status_right.setText(html)

The label must be configured with ``setTextFormat(Qt.TextFormat.RichText)``
and a monospace font (use ``F.mono(F.SIZE_XS)``) for the dots and
numbers to align nicely.
"""

from __future__ import annotations

from img_player.ui.theme import H


# ----------------------------------------------------------------------- Thresholds

# Effective FPS / target FPS thresholds:
#   >= _FPS_OK   → green, "ça tient"
#   >= _FPS_WARN → amber, "ça fléchit, l'œil voit le drop"
#   <  _FPS_WARN → red,   "ça décroche, on lâche des frames"
_FPS_OK = 0.95
_FPS_WARN = 0.80

# Cache fill (bytes_used / bytes_budget). Above this the cache is well
# served and we show the green dot. Below, no dot — the user understands
# from the missing dot that the cache is still ramping up. We don't
# escalate to amber/red here because a half-full cache is normal during
# warm-up and not actually a problem.
_CACHE_FULL = 0.80


# ----------------------------------------------------------------------- Dot colour helpers


def fps_dot_color(effective: float | None, target: float) -> str | None:
    """Pick the colour of the dot in front of the live-fps indicator.

    Returns a hex string (``H.CACHE_BAR``, ``H.ACCENT`` or
    ``H.MARKER_IO``), or ``None`` when the dot should not be rendered
    at all (paused / unknown — there's no meaningful colour for "no
    fps to show").
    """
    if effective is None or target <= 0:
        return None
    ratio = effective / target
    if ratio >= _FPS_OK:
        return H.CACHE_BAR
    if ratio >= _FPS_WARN:
        return H.ACCENT
    return H.MARKER_IO


def cache_dot_color(ratio: float) -> str | None:
    """Pick the colour of the dot in front of the cache indicator.

    Green when the cache is well-filled, no dot otherwise (the absence
    is a deliberate "not yet warm" signal — quieter than amber/red).
    """
    if ratio >= _CACHE_FULL:
        return H.CACHE_BAR
    return None


# ----------------------------------------------------------------------- HTML format


def _dot_span(color: str | None) -> str:
    """Render either a coloured dot or a transparent placeholder of the
    same width so columns stay aligned with or without a dot."""
    if color is None:
        # U+2003 EM SPACE keeps the column width stable.
        return "<span style='color:transparent'>●</span>"
    return f"<span style='color:{color}'>●</span>"


def format_perf_html(
    *,
    cache_n: int,
    cache_total: int,
    cache_ratio: float,
    fps_effective: float | None,
    fps_target: float,
    ram_gb: float,
) -> str:
    """Build the rich-text HTML rendered in the right status label.

    The output has three space-separated chunks; each chunk starts with
    a coloured dot (or an invisible placeholder when there's nothing to
    flag) so the spacing stays consistent regardless of dot presence.
    """
    cache_dot = _dot_span(cache_dot_color(cache_ratio))
    fps_dot = _dot_span(fps_dot_color(fps_effective, fps_target))
    fps_text = f"{fps_effective:.1f} fps" if fps_effective is not None else "— fps"

    return (
        f"{cache_dot} cache {cache_n}/{cache_total}"
        f" &nbsp;&nbsp; {fps_dot} {fps_text}"
        f" &nbsp;&nbsp; RAM {ram_gb:.1f} GB"
    )
