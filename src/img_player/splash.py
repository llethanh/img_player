"""Boot-time splash screen, backed by Qt's :class:`QSplashScreen`.

Reads a static PNG from ``src/img_player/assets/splash.png`` and
shows it as a frameless always-on-top window for the duration of
boot. Three helpers (``init`` / ``update`` / ``close``) so callers
can sprinkle status updates without ``if`` ladders; every helper
degrades to a no-op outside a QApplication context (CLI scan paths,
headless tests, …).

The splash only becomes visible once PySide6 finishes importing
(~1-2 s into boot on a cold launch). That trade-off is the cost of
a DPI-aware Qt-rendered splash — earlier iterations of this module
tried PyInstaller's bootloader Splash() (instant, but Tk on Windows
isn't DPI-aware so the splash visibly shrunk mid-boot) and a
PowerShell + WPF external launcher (instant, DPI-aware, but
spawn-then-handshake added moving parts and edge cases). Net
preference after evaluation: a plain Qt splash that's a tick late
but visually clean and behaves like the rest of the app.
"""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Module-level so ``update`` and ``close`` can find what ``init``
# brought up. ``None`` means "no splash is showing" — every helper
# guards on this so a missed ``init`` doesn't crash anything.
_qt_splash = None  # type: ignore[var-annotated]


def init() -> None:
    """Bring up the splash window, if a QApplication is alive.

    Idempotent: a second call while a splash already exists is a
    no-op. Silent failure modes — no QApplication, missing PNG asset,
    PySide6 import error — all return without raising so the boot
    sequence keeps moving even when the splash can't render.
    """
    global _qt_splash
    if _qt_splash is not None:
        return
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QApplication, QSplashScreen
    except ImportError:
        log.debug("PySide6 not available; skipping splash.")
        return
    if QApplication.instance() is None:
        log.debug("No QApplication; skipping splash.")
        return

    asset = Path(__file__).parent / "assets" / "splash.png"
    if not asset.is_file():
        log.debug("Splash asset missing at %s; skipping splash.", asset)
        return
    pixmap = QPixmap(str(asset))
    if pixmap.isNull():
        log.debug("Failed to load splash pixmap from %s.", asset)
        return
    # The asset is rendered at 2× its logical size by
    # ``tools/regen_splash.py`` (960×520 physical for a 480×260
    # logical splash). Flagging the pixmap as 2× tells Qt to draw
    # it at half its bitmap size in *logical* pixels — on a 1×
    # monitor it downsamples crisply, on a 2× Hi-DPI monitor it
    # paints 1:1 at native density. Without this flag Qt upscaled
    # the splash on Hi-DPI displays and the text read soft /
    # pixelated.
    pixmap.setDevicePixelRatio(2.0)

    _qt_splash = QSplashScreen(pixmap, Qt.WindowType.WindowStaysOnTopHint)
    _qt_splash.show()
    _qt_splash.raise_()
    # Force the splash to actually paint before any heavy work
    # blocks the event loop — otherwise the user can see the click
    # register but the splash flashes for a frame before disappearing
    # under heavier import work.
    QApplication.processEvents()


def update(message: str) -> None:
    """Repaint the splash with a new bottom-left status message.

    No-op when ``init`` was never called (or failed silently)."""
    if _qt_splash is None:
        return
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    _qt_splash.showMessage(
        message,
        # Bottom-left band, white text — mirrors the area the static
        # PNG asset reserves for a dynamic status string.
        int(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft),
        Qt.GlobalColor.white,
    )
    QApplication.processEvents()


def close(window=None) -> None:  # type: ignore[no-untyped-def]
    """Dismiss the splash. Pass ``window`` to fade as it appears.

    With a ``window`` argument we use ``QSplashScreen.finish(window)``,
    which keeps the splash visible until ``window`` becomes the
    active paintable widget — preventing the brief blank-frame gap
    an unconditional close can produce on slow first paints.
    """
    global _qt_splash
    if _qt_splash is None:
        return
    try:
        if window is not None:
            _qt_splash.finish(window)
        else:
            _qt_splash.close()
    except Exception:  # pragma: no cover — defensive
        log.debug("Error closing splash", exc_info=True)
    _qt_splash = None


def is_active() -> bool:
    """``True`` when the splash is currently showing. Useful for
    log breadcrumbs and tests."""
    return _qt_splash is not None
