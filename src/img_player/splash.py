"""Wrapper around PyInstaller's ``pyi_splash`` module so callers
don't have to guard the import everywhere.

``pyi_splash`` is auto-injected into the bundled .exe by PyInstaller
when the spec carries a ``Splash`` block. In dev mode (``python -m
img_player``) the import simply fails — every helper here becomes a
no-op so calling code can sprinkle ``update("…")`` and ``close()``
at boot milestones without ``if`` ladders.
"""

from __future__ import annotations

try:  # pragma: no cover — only succeeds inside a PyInstaller bundle
    import pyi_splash  # type: ignore[import-not-found]
except ImportError:
    pyi_splash = None  # type: ignore[assignment]


def update(message: str) -> None:
    """Push ``message`` into the splash's bottom status band.

    No-op outside a PyInstaller bundle, or after :func:`close` has
    fired (the bootloader rejects further updates once it disposes
    the window — wrapped here so callers don't have to track it).
    """
    if pyi_splash is None:
        return
    try:
        pyi_splash.update_text(message)
    except Exception:  # pragma: no cover — defensive
        # ``update_text`` raises on a closed splash; treat as
        # silent no-op so a late milestone (e.g. an OCIO retry)
        # doesn't crash startup.
        pass


def close() -> None:
    """Dismiss the splash. Idempotent. Call once the main window
    is about to be shown so the user sees the real UI immediately."""
    if pyi_splash is None:
        return
    try:
        pyi_splash.close()
    except Exception:  # pragma: no cover — defensive
        pass


def is_active() -> bool:
    """``True`` when running inside a PyInstaller bundle that
    bundled the splash. Useful for log breadcrumbs."""
    return pyi_splash is not None
