"""Generate a "Missing Frame" QPixmap for the sequence player.

Drop-in version of the standalone module the user designed: damier
greyscale + chromatic aberration + Qt overlay (4-corner crosshairs,
central "MISSING FRAME" boxed text, vignette).

Used by :mod:`missing_placeholder` to feed the GL viewport's
float32 RGBA pipeline — so the new visual flows through the same
multi-layer compositing path as before, no GL-side changes needed.

The "Big Shoulders Display ExtraBold" font is loaded automatically
at import time if a bundled .ttf is found next to the module (under
``assets/fonts/``). Falls back silently to Arial otherwise so the
module stays usable on any machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)


# ------------------------------------------------------------
# Font handling — auto-load bundled .ttf if present
# ------------------------------------------------------------
_FONT_FAMILY = "Arial"  # fallback


def load_custom_font(ttf_path: str) -> bool:
    """Register a TrueType font file for use by the missing-frame
    overlay. Returns ``True`` on success."""
    global _FONT_FAMILY
    fid = QFontDatabase.addApplicationFont(str(ttf_path))
    if fid >= 0:
        families = QFontDatabase.applicationFontFamilies(fid)
        if families:
            _FONT_FAMILY = families[0]
            return True
    return False


def _try_autoload_default_font() -> None:
    """Look for ``BigShouldersDisplay-ExtraBold.ttf`` in the project's
    ``assets/fonts/`` folder and register it. No-op (and no warning)
    if the file isn't shipped — the Arial fallback handles that case
    gracefully."""
    # Resolve relative to this file: src/img_player/cache/missing_frame.py
    # → src/img_player/assets/fonts/<file>.
    base = Path(__file__).resolve().parent.parent / "assets" / "fonts"
    candidates = [
        # Variable-font release — covers all weights / optical sizes
        # in a single .ttf. Comma in the filename is intentional
        # (Google Fonts naming convention for the axis list).
        base / "BigShoulders-VariableFont_opsz,wght.ttf",
        # Static fallbacks (older Big Shoulders releases).
        base / "BigShouldersDisplay-ExtraBold.ttf",
        base / "BigShouldersDisplay-ExtraBold.otf",
    ]
    for path in candidates:
        if path.is_file():
            load_custom_font(str(path))
            return


# Loading at module import requires a live QApplication; QFontDatabase
# raises a warning otherwise. We guard with a hasattr check so this
# import is safe even from headless / pre-app contexts (tests, etc.).
try:
    from PySide6.QtWidgets import QApplication

    if QApplication.instance() is not None:
        _try_autoload_default_font()
except Exception:  # pragma: no cover — import safety net
    pass


def ensure_font_loaded() -> None:
    """Idempotent hook the app can call after constructing
    ``QApplication`` to make sure the bundled font is registered."""
    if _FONT_FAMILY == "Arial":
        _try_autoload_default_font()


# ------------------------------------------------------------
# Checkerboard
# ------------------------------------------------------------
def _draw_checkerboard(arr: np.ndarray, w: int, h: int) -> None:
    sz = max(min(w, h) // 20, 8)
    cols = w // sz + 2
    rows = h // sz + 2
    ox = -((cols * sz - w) // 2)
    oy = -((rows * sz - h) // 2)
    # RGBA — alpha included so the (3,) → (4,) broadcast doesn't blow
    # up when assigning into the (h, w, 4) buffer below.
    c1 = np.array([176, 176, 176, 255], dtype=np.uint8)  # #B0B0B0
    c2 = np.array([202, 202, 202, 255], dtype=np.uint8)  # #CACACA
    for r in range(rows):
        for c in range(cols):
            x0 = max(0, ox + c * sz)
            y0 = max(0, oy + r * sz)
            x1 = min(w, ox + (c + 1) * sz)
            y1 = min(h, oy + (r + 1) * sz)
            if x1 <= x0 or y1 <= y0:
                continue
            arr[y0:y1, x0:x1] = c1 if (r + c) % 2 == 0 else c2


# ------------------------------------------------------------
# Chromatic aberration (vectorised numpy)
# ------------------------------------------------------------
def _apply_chromatic_aberration(
    arr: np.ndarray, w: int, h: int, strength: float = 0.007,
) -> np.ndarray:
    cx, cy = w / 2.0, h / 2.0
    max_d = np.sqrt(cx ** 2 + cy ** 2)
    max_off = max(min(w, h) * strength, 2.0)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xs - cx
    dy = ys - cy
    d = np.sqrt(dx ** 2 + dy ** 2)
    t = d / max_d
    strength_map = t ** 2 * max_off

    safe_d = np.where(d == 0, 1, d)
    nx = dx / safe_d
    ny = dy / safe_d

    rx = np.clip((xs + nx * strength_map).round().astype(int), 0, w - 1)
    ry = np.clip((ys + ny * strength_map).round().astype(int), 0, h - 1)
    bx = np.clip((xs - nx * strength_map * 0.6).round().astype(int), 0, w - 1)
    by = np.clip((ys - ny * strength_map * 0.6).round().astype(int), 0, h - 1)

    out = arr.copy()
    out[:, :, 0] = arr[ry, rx, 0]   # R
    out[:, :, 2] = arr[by, bx, 2]   # B
    return out


# ------------------------------------------------------------
# Qt overlay (corner crosshairs, central text, vignette)
# ------------------------------------------------------------
def _draw_overlay(pixmap: QPixmap, w: int, h: int) -> None:
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # 4-corner registration crosshairs
    ms = min(w, h) * 0.045
    mg = ms * 0.9
    pen = QPen(QColor(255, 255, 255, 140))
    pen.setWidthF(max(0.8, w / 900))
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    for cx, cy in [(mg, mg), (w - mg, mg), (mg, h - mg), (w - mg, h - mg)]:
        painter.drawLine(QPointF(cx - ms, cy), QPointF(cx + ms, cy))
        painter.drawLine(QPointF(cx, cy - ms), QPointF(cx, cy + ms))
        painter.drawEllipse(QPointF(cx, cy), ms * 0.28, ms * 0.28)

    # Central box
    fs = max(min(w * 0.075, h * 0.13), 12)
    sub_fs = fs * 0.3
    bw = min(w * 0.7, fs * 9)
    bh = fs * 2.2
    bx = (w - bw) / 2
    by = (h - bh) / 2

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(8, 8, 8, 122))
    painter.drawRoundedRect(QRectF(bx, by, bw, bh), 3, 3)

    line_pen = QPen(QColor(255, 255, 255, 56))
    line_pen.setWidthF(0.8)
    painter.setPen(line_pen)
    painter.drawLine(QPointF(bx + 20, by),      QPointF(bx + bw - 20, by))
    painter.drawLine(QPointF(bx + 20, by + bh), QPointF(bx + bw - 20, by + bh))

    # Main text
    font_main = QFont(_FONT_FAMILY)
    font_main.setPixelSize(int(fs))
    font_main.setWeight(QFont.Weight.ExtraBold)
    font_main.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 105)
    painter.setFont(font_main)
    painter.setPen(QColor(255, 255, 255, 245))
    painter.drawText(
        QRectF(bx, by, bw, bh * 0.58),
        int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom),
        "MISSING FRAME",
    )

    # Subtitle
    font_sub = QFont(_FONT_FAMILY)
    font_sub.setPixelSize(int(sub_fs))
    font_sub.setWeight(QFont.Weight.DemiBold)
    font_sub.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 110)
    painter.setFont(font_sub)
    painter.setPen(QColor(255, 255, 255, 97))
    painter.drawText(
        QRectF(bx, by + bh * 0.62, bw, bh * 0.38),
        int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
        "SEQUENCE ERROR  ·  FRAME NOT FOUND",
    )

    # Vignette
    grad = QRadialGradient(w / 2, h / 2, max(w, h) * 0.72)
    grad.setColorAt(0,    QColor(0, 0, 0, 0))
    grad.setColorAt(0.55, QColor(0, 0, 0, 7))
    grad.setColorAt(1,    QColor(0, 0, 0, 122))
    painter.setBrush(QBrush(grad))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawRect(0, 0, w, h)

    painter.end()


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def generate_missing_frame(width: int, height: int) -> QPixmap:
    """Generate a 'Missing Frame' QPixmap at the requested size."""
    width = max(2, int(width))
    height = max(2, int(height))

    # 1. Greyscale checkerboard on a numpy RGBA array
    arr = np.zeros((height, width, 4), dtype=np.uint8)
    arr[:, :, 3] = 255
    _draw_checkerboard(arr, width, height)

    # 2. Chromatic aberration
    arr = _apply_chromatic_aberration(arr, width, height, strength=0.007)
    arr = np.ascontiguousarray(arr)

    # 3. Convert to QPixmap (copy so the QImage owns its buffer)
    img = QImage(
        arr.data, width, height, width * 4, QImage.Format.Format_RGBA8888,
    ).copy()
    pixmap = QPixmap.fromImage(img)

    # 4. Qt overlay
    _draw_overlay(pixmap, width, height)
    return pixmap


def generate_missing_frame_rgba_float(width: int, height: int) -> np.ndarray:
    """Same as :func:`generate_missing_frame` but returns the result as
    an ``H×W×4`` float32 RGBA array in [0, 1] — the format the GL
    viewport / multi-layer compositor consumes directly."""
    pixmap = generate_missing_frame(width, height)
    qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    w = qimg.width()
    h = qimg.height()
    bpl = qimg.bytesPerLine()
    ptr = qimg.constBits()
    # ``constBits()`` returns a memoryview-compatible buffer in
    # PySide6 — np.frombuffer reads through it without copying.
    raw = np.frombuffer(ptr, dtype=np.uint8, count=bpl * h).reshape(h, bpl)
    arr = raw[:, : w * 4].reshape(h, w, 4).copy()
    return arr.astype(np.float32) / 255.0
