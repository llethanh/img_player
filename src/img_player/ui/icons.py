"""Inline SVG icon factory — single source for transport / panel icons.

Icons are stored as small XML strings with a ``{color}`` placeholder.
At call time they're rendered into a ``QPixmap`` via ``QSvgRenderer``
and wrapped as ``QIcon``. Two reasons we keep them inline rather than
shipping ``.svg`` files:

* No PyInstaller ``datas`` plumbing — ``importlib.resources`` doesn't
  work nicely from a frozen bundle without explicit hooks. Strings in
  Python source travel for free.
* Coloured icons. The play button needs ``ACCENT`` orange, the rest
  use ``TEXT_PRIMARY``. Doing this with on-disk SVG files would
  require either editing them on the fly or shipping one file per
  colour.

The icons themselves match the mockup's geometry (16×16 viewBox).

Usage::

    from img_player.ui.icons import make_icon
    from img_player.ui.theme import H

    play_icon  = make_icon("play",  color=H.ACCENT)
    stop_icon  = make_icon("stop")  # default = TEXT_PRIMARY
    pause_icon = make_icon("pause")
"""

from __future__ import annotations

from functools import lru_cache

from PySide6.QtCore import QByteArray, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

from img_player.ui.theme import H

# ----------------------------------------------------------------------- Templates

# Each template uses a 16×16 viewBox so they all line up on the same grid.
# The ``{color}`` placeholder gets ``str.format``-ed at call time.
# Geometry comes straight from `ui_mockup.html` — keep it aligned.
_TEMPLATES: dict[str, str] = {
    # Triangle pointing right.
    "play": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<polygon points="4,2 14,8 4,14" fill="{color}"/>'
        "</svg>"
    ),
    # Mirror of "play" — same triangle pointing left for reverse play.
    "play_reverse": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<polygon points="12,2 2,8 12,14" fill="{color}"/>'
        "</svg>"
    ),
    # Two vertical bars (the natural counterpart to play). Geometry
    # chosen so the two bars together occupy roughly the same visual
    # mass as the play triangle.
    "pause": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="4" y="2" width="3" height="12" fill="{color}"/>'
        '<rect x="9" y="2" width="3" height="12" fill="{color}"/>'
        "</svg>"
    ),
    # Rounded square.
    "stop": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="3" y="3" width="10" height="10" rx="1" fill="{color}"/>'
        "</svg>"
    ),
    # Step backward: main triangle pointing left + smaller dim triangle
    # to suggest "skip".
    "prev": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<polygon points="12,2 4,8 12,14" fill="{color}"/>'
        '<polygon points="6,2 4,8 6,14" fill="{color}" opacity="0.5"/>'
        "</svg>"
    ),
    # Step forward, mirror of "prev".
    "next": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<polygon points="4,2 12,8 4,14" fill="{color}"/>'
        '<polygon points="10,2 12,8 10,14" fill="{color}" opacity="0.5"/>'
        "</svg>"
    ),
    # Jump to first frame: vertical bar on the left + leftward triangle.
    "first": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="2" y="2" width="2" height="12" rx="1" fill="{color}"/>'
        '<polygon points="14,2 6,8 14,14" fill="{color}"/>'
        "</svg>"
    ),
    # Jump to last frame, mirror of "first".
    "last": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="12" y="2" width="2" height="12" rx="1" fill="{color}"/>'
        '<polygon points="2,2 10,8 2,14" fill="{color}"/>'
        "</svg>"
    ),
    # Compare two layers — a rectangle split vertically by a seam,
    # left half filled and labelled ``A``, right half outlined and
    # labelled ``B``. Reads instantly as "compare two sides". The
    # ``A`` punches dark on the filled half (hard-coded #1A1A1A so
    # contrast is preserved regardless of the dynamic icon colour);
    # the ``B`` rides the icon colour against the empty half. The
    # outer rect is enlarged to nearly fill the 16×16 viewBox so the
    # glyphs have room to breathe at small render sizes.
    "compare": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        # Larger content — fills ~75% of the viewBox vertically so
        # the mark reads at small icon sizes. The rect occupies the
        # full width minus a tiny inset; A / B labels grow
        # proportionally so they're legible at the target render
        # sizes.
        '<rect x="1.4" y="2.6" width="13.2" height="10.8" rx="1.2" '
        'fill="none" stroke="{color}" stroke-width="1.4"/>'
        '<rect x="2.1" y="3.3" width="5.9" height="9.4" '
        'fill="{color}"/>'
        '<text x="5.0" y="10.4" font-family="Segoe UI, Arial, sans-serif" '
        'font-size="7" font-weight="700" fill="#1A1A1A" '
        'text-anchor="middle">A</text>'
        '<text x="11.0" y="10.4" font-family="Segoe UI, Arial, sans-serif" '
        'font-size="7" font-weight="700" fill="{color}" '
        'text-anchor="middle">B</text>'
        # Central seam — small overshoot above and below the rect
        # so it reads as a draggable handle going through the
        # image.
        '<line x1="8" y1="1.6" x2="8" y2="14.4" '
        'stroke="{color}" stroke-width="1.6" stroke-linecap="square"/>'
        "</svg>"
    ),
    # Contact sheet — a 2×2 grid of small filled rectangles
    # surrounded by a stroked outer frame, reading instantly as
    # "tiled view" / "grid of thumbnails". Same outer rectangle
    # geometry as the compare icon so the two review-mode toggles
    # look like siblings in the transport bar.
    "grid": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="1.4" y="2.6" width="13.2" height="10.8" rx="1.2" '
        'fill="none" stroke="{color}" stroke-width="1.4"/>'
        # Top-left cell.
        '<rect x="2.6" y="3.8" width="5.0" height="3.8" '
        'fill="{color}"/>'
        # Top-right cell.
        '<rect x="8.4" y="3.8" width="5.0" height="3.8" '
        'fill="{color}"/>'
        # Bottom-left cell.
        '<rect x="2.6" y="8.4" width="5.0" height="3.8" '
        'fill="{color}"/>'
        # Bottom-right cell.
        '<rect x="8.4" y="8.4" width="5.0" height="3.8" '
        'fill="{color}"/>'
        "</svg>"
    ),
    # Vertical compare — same shape as the ``compare`` icon above
    # but without the A/B labels: just the two halves of an image
    # separated by a vertical seam handle that slightly overshoots
    # at top and bottom. Used by the compare-band's "Vert" mode
    # toggle.
    "compare_vert": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="1.4" y="2.6" width="13.2" height="10.8" rx="1.2" '
        'fill="none" stroke="{color}" stroke-width="1.4"/>'
        '<rect x="2.1" y="3.3" width="5.9" height="9.4" '
        'fill="{color}"/>'
        '<line x1="8" y1="1.6" x2="8" y2="14.4" '
        'stroke="{color}" stroke-width="1.6" stroke-linecap="square"/>'
        "</svg>"
    ),
    # Horizontal compare — same outer rect but the split runs
    # horizontally: top half filled, bottom half empty, seam handle
    # overshoots left and right.
    "compare_horiz": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="1.4" y="2.6" width="13.2" height="10.8" rx="1.2" '
        'fill="none" stroke="{color}" stroke-width="1.4"/>'
        '<rect x="2.1" y="3.3" width="11.8" height="4.4" '
        'fill="{color}"/>'
        '<line x1="0.6" y1="8" x2="15.4" y2="8" '
        'stroke="{color}" stroke-width="1.6" stroke-linecap="square"/>'
        "</svg>"
    ),
    # Opacity / blend — two rounded squares overlapping, with the
    # intersection filled solid to suggest the "common alpha"
    # region between layers. Pure outline elsewhere keeps the icon
    # readable at small sizes.
    "opacity": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="1.4" y="1.4" width="9.2" height="9.2" rx="1.4" '
        'fill="none" stroke="{color}" stroke-width="1.4"/>'
        '<rect x="5.4" y="5.4" width="9.2" height="9.2" rx="1.4" '
        'fill="none" stroke="{color}" stroke-width="1.4"/>'
        '<rect x="5.4" y="5.4" width="5.2" height="5.2" '
        'fill="{color}"/>'
        "</svg>"
    ),
    # Hamburger / dock toggle. Three horizontal bars at y=3, y=7, y=11
    # (height 2 each → centres at 4 / 8 / 12), so the gaps above/below
    # each bar are identical. The Unicode glyph U+2630 we used to
    # render here had inconsistent spacing across system fonts; an
    # SVG primitive is the only way to guarantee even bars.
    "menu": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="3" y="3" width="10" height="2" rx="1" fill="{color}"/>'
        '<rect x="3" y="7" width="10" height="2" rx="1" fill="{color}"/>'
        '<rect x="3" y="11" width="10" height="2" rx="1" fill="{color}"/>'
        "</svg>"
    ),
    # ===== Annotation toolbar — line-art set =============================
    # All line-art icons share the same conventions:
    # * stroke (not fill) at 1.4 px — readable at 16 px and beyond,
    #   matches the visual weight of the modern review tools the
    #   user pointed to as a reference.
    # * Round caps + round joins so corners breathe (SVG default
    #   "miter" gets ugly on small icons).
    # * fill="none" everywhere except where a small filled accent
    #   reinforces the silhouette (the pin's body).
    "pen": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="none" '
        'stroke="{color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">'
        # body diagonal
        '<path d="M3.5 12.5 L11.5 4.5"/>'
        # tip cap (a small triangle at the top-right end of the body)
        '<path d="M10.5 3.5 L13.5 6.5 L11.5 8.5 L8.5 5.5 Z"/>'
        # short ground line under the tip — suggests the pen has just
        # left a stroke
        '<path d="M2 14 L4 14"/>'
        "</svg>"
    ),
    "eraser": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="none" '
        'stroke="{color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">'
        # main body — a parallelogram (rectangle rotated -20° around its centre)
        '<path d="M4.2 11.8 L9.5 4.0 L13.0 6.5 L7.7 14.3 Z"/>'
        # division line between the rubber tip (upper-left) and the
        # holder (lower-right) — half the way across the parallelogram
        '<path d="M6.85 8.0 L10.35 10.5"/>'
        "</svg>"
    ),
    "undo": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="none" '
        'stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
        # ¾ arc: starts at top-left, sweeps clockwise to the right
        '<path d="M 3.5 6 A 4.5 4.5 0 1 1 2.5 9.5"/>'
        # arrowhead — open chevron pointing down-left at the arc start
        '<polyline points="2,3.5 3.5,6 6,5"/>'
        "</svg>"
    ),
    "redo": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="none" '
        'stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M 12.5 6 A 4.5 4.5 0 1 0 13.5 9.5"/>'
        '<polyline points="14,3.5 12.5,6 10,5"/>'
        "</svg>"
    ),
    "pin": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="none" '
        'stroke="{color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">'
        # Thumbtack viewed from the side — head + flared shoulders +
        # vertical needle. Stroke-only outline so the silhouette
        # matches the rest of the line-art set.
        '<path d="M 8 2 L 11 5 L 11 9 L 13 11 L 9 11 L 8 14 L 7 11 L 3 11 L 5 9 L 5 5 Z"/>'
        "</svg>"
    ),
    "trash": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="none" '
        'stroke="{color}" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">'
        # Lid (top horizontal bar with a small handle ridge) and the
        # bin body (rectangle with two vertical bars suggesting
        # corrugation). The Clear button uses this — visually
        # decisive about "remove everything on this frame".
        '<path d="M3 4 L13 4"/>'
        '<path d="M6 4 L6 2.5 L10 2.5 L10 4"/>'
        '<path d="M5 4 L5.5 13.5 L10.5 13.5 L11 4"/>'
        '<path d="M7 7 L7 11 M9 7 L9 11"/>'
        "</svg>"
    ),
    # Four corner brackets pointing outward — universal "expand to
    # fullscreen" cue (matches YouTube / VLC / etc.).
    "fullscreen_enter": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="none" '
        'stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M2 6 L2 2 L6 2"/>'
        '<path d="M14 6 L14 2 L10 2"/>'
        '<path d="M2 10 L2 14 L6 14"/>'
        '<path d="M14 10 L14 14 L10 14"/>'
        "</svg>"
    ),
    # Mirror — corners pointing INWARD = "exit fullscreen, contract".
    "fullscreen_exit": (
        '<svg viewBox="0 0 16 16" xmlns="http://www.w3.org/2000/svg" fill="none" '
        'stroke="{color}" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M6 2 L6 6 L2 6"/>'
        '<path d="M10 2 L10 6 L14 6"/>'
        '<path d="M6 14 L6 10 L2 10"/>'
        '<path d="M10 14 L10 10 L14 10"/>'
        "</svg>"
    ),
}


# ----------------------------------------------------------------------- Factory


def _device_pixel_ratio() -> float:
    """Return the active screen's DPR if a QApplication exists, else 1.0.

    Used to render icons at 2× on hi-DPI displays so they stay sharp
    when Qt scales them down.
    """
    app = QApplication.instance()
    if app is None:
        return 1.0
    screen = app.primaryScreen()
    if screen is None:
        return 1.0
    return float(screen.devicePixelRatio())


@lru_cache(maxsize=64)
def _render_pixmap(name: str, color: str, size: int) -> QPixmap:
    """Internal: render the named SVG template into a pixmap of the
    given logical ``size`` at the system DPR."""
    template = _TEMPLATES[name]
    xml = template.format(color=color)
    renderer = QSvgRenderer(QByteArray(xml.encode("utf-8")))
    dpr = _device_pixel_ratio()
    physical = max(1, round(size * dpr))
    pixmap = QPixmap(physical, physical)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    pixmap.setDevicePixelRatio(dpr)
    return pixmap


@lru_cache(maxsize=64)
def make_icon(
    name: str,
    color: str = H.TEXT_PRIMARY,
    size: int = 18,
    disabled_color: str | None = None,
) -> QIcon:
    """Return a ``QIcon`` for a named template, painted in ``color``.

    Parameters
    ----------
    name:
        One of the keys in ``_TEMPLATES`` (e.g. ``"play"``).
    color:
        Hex string used as the SVG ``fill``. Default is the charter's
        primary text colour. Pass ``H.ACCENT`` for the play button.
    size:
        Logical pixel size of the resulting icon. Hi-DPI handling is
        automatic — on a 200 % display we render at ``size * dpr`` and
        attach the DPR to the pixmap so Qt downscales cleanly.
    disabled_color:
        Optional hex string used to paint the icon in its disabled
        state. Default ``None`` → Qt grays the Normal pixmap (= the
        usual "icon on a disabled QPushButton" look). Pass e.g.
        ``H.ACCENT_DIM`` to keep a coloured silhouette when disabled
        instead of losing the icon's identity to grayscale.

    Caching: the icon **and** its underlying pixmap are both memoized
    on ``(name, color, size, disabled_color)``. We expect at most a
    few dozen unique combinations across the whole app — callers
    that pass identical args get the **same** ``QIcon`` instance back
    (``is``-equal), which saves both the SVG render and the icon
    construction. Don't ``addPixmap`` to a returned icon — you'd
    mutate the shared instance.
    """
    pixmap = _render_pixmap(name, color, size)
    icon = QIcon(pixmap)
    if disabled_color is not None:
        disabled_pix = _render_pixmap(name, disabled_color, size)
        icon.addPixmap(
            disabled_pix, QIcon.Mode.Disabled, QIcon.State.Off,
        )
        icon.addPixmap(
            disabled_pix, QIcon.Mode.Disabled, QIcon.State.On,
        )
    return icon


def available_names() -> tuple[str, ...]:
    """Lightweight introspection helper, mostly for tests."""
    return tuple(_TEMPLATES.keys())
