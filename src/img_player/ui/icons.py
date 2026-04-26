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
def make_icon(name: str, color: str = H.TEXT_PRIMARY, size: int = 18) -> QIcon:
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

    Caching: the icon is memoized on ``(name, color, size)``. We expect
    at most a few dozen unique combinations across the whole app.
    """
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
    return QIcon(pixmap)


def available_names() -> tuple[str, ...]:
    """Lightweight introspection helper, mostly for tests."""
    return tuple(_TEMPLATES.keys())
