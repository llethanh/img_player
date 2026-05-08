"""Boot-time splash screen — splits responsibility across two backends.

External launcher mode (``flick.bat`` → ``splash_launcher.ps1``):
    The PowerShell launcher paints a WPF splash within ~200 ms of
    the user's double-click and then spawns ``FlickPlayer.exe`` with
    ``FLICK_LAUNCHER=1`` set in the environment. We detect that env
    var and stay out of the way: no QSplashScreen, no overlap. When
    Flick's MainWindow finally shows, :func:`close` writes a marker
    file (``%TEMP%\\flick_ready.flag``) the launcher polls — that
    closes the WPF window cleanly.

Direct-launch mode (``python -m img_player``, or double-clicking
``FlickPlayer.exe`` without the wrapper):
    No external launcher, no marker handshake. We bring up a Qt
    :class:`QSplashScreen` reading the same PNG asset — DPI-aware,
    no shrinking, but only visible after PySide6 finishes importing
    (~1-2 s). Status updates are forwarded to the QSplashScreen as
    boot milestones tick over.

The previous PyInstaller-bootloader (``pyi_splash``) backend was
removed in this refactor: it boots fast but uses Tcl/Tk, which is
not DPI-aware on Windows and visibly shrinks the splash mid-boot.
The PowerShell launcher fills the same instant-feedback role
without the cosmetic glitch.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# Marker file the PowerShell launcher polls. Stays in lockstep with
# the path baked into ``splash_launcher.ps1``; a shared constant
# lives here (Python side) since this is the side that *writes* it.
_READY_MARKER = Path(tempfile.gettempdir()) / "flick_ready.flag"


def _under_external_launcher() -> bool:
    """``True`` when ``flick.bat`` / ``splash_launcher.ps1`` spawned us."""
    return os.environ.get("FLICK_LAUNCHER") == "1"


# Qt fallback splash. Stored at module level so :func:`update` and
# :func:`close` can find what :func:`init` brought up.
_qt_splash = None  # type: ignore[var-annotated]


def init() -> None:
    """Bring up the in-process splash, if appropriate.

    No-op under the external PS launcher (it's already showing one).
    Otherwise tries to construct a :class:`QSplashScreen`; silent
    failure modes (no QApplication, missing PNG, PySide6 import
    error) all return without raising so the boot sequence keeps
    moving.
    """
    global _qt_splash
    if _under_external_launcher():
        return
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

    _qt_splash = QSplashScreen(pixmap, Qt.WindowType.WindowStaysOnTopHint)
    _qt_splash.show()
    _qt_splash.raise_()
    QApplication.processEvents()


def update(message: str) -> None:
    """Repaint the in-process splash with a new bottom-band status.

    No-op under the external launcher — the WPF splash is static
    artwork (matches Photoshop / Maya / Resolve convention). For the
    Qt fallback, the message lands in the bottom-left band, white
    text, same area the previous PIL-rendered PNG reserved.
    """
    if _under_external_launcher():
        return
    if _qt_splash is None:
        return
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication
    except ImportError:
        return
    _qt_splash.showMessage(
        message,
        int(Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft),
        Qt.GlobalColor.white,
    )
    QApplication.processEvents()


def close(window=None) -> None:  # type: ignore[no-untyped-def]
    """Dismiss the splash. Idempotent.

    Two responsibilities packed in:
      * Under the external launcher: write the ready marker so the
        WPF splash polls one last tick and goes away.
      * For the Qt fallback: ``finish(window)`` if a window is given
        (avoids the brief blank-frame gap of a flat ``close``),
        otherwise plain ``close``.
    """
    global _qt_splash
    if _under_external_launcher():
        try:
            _READY_MARKER.write_text("ready", encoding="utf-8")
        except Exception:  # pragma: no cover — defensive
            log.debug("Failed to write ready marker", exc_info=True)
        return
    if _qt_splash is None:
        return
    try:
        if window is not None:
            _qt_splash.finish(window)
        else:
            _qt_splash.close()
    except Exception:  # pragma: no cover — defensive
        log.debug("Error closing Qt splash", exc_info=True)
    _qt_splash = None


def is_active() -> bool:
    """``True`` when *some* splash is presumed to be visible.

    Best-effort: under the external launcher we can't query the WPF
    window's state from here, so we assume "yes" until ``close`` has
    written the ready marker. Useful for log breadcrumbs / tests.
    """
    if _under_external_launcher():
        return not _READY_MARKER.is_file()
    return _qt_splash is not None
