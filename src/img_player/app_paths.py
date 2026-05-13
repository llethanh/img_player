"""Canonical app-data paths + migration from the legacy ``img_player`` name.

History: the project's internal package + binary were both called
``img_player`` until v1.5.8. The user-visible name was "Flick Player"
all along, but the on-disk directories under ``%APPDATA%`` and
``%LOCALAPPDATA%`` reused the package name — confusing for an artist
looking at their file system.

v1.5.9 renames every user-visible folder to ``FlickPlayer`` (no space:
matches ``FlickPlayer.exe``, avoids shell-quoting headaches) while
keeping the QSettings org/app name intact (renaming would lose every
user's window geometry, recent files, etc. — too invasive a break).

This module is the single source of truth for those paths and runs the
one-shot legacy-rename migration on first import. Other modules call
:func:`user_prefs_dir`, :func:`disk_cache_dir`, etc. instead of
hardcoding paths.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Canonical on-disk folder name. Matches ``FlickPlayer.exe`` and the
# bundle directory ``FlickPlayer_v<X.Y.Z>``.
APP_DIR_NAME = "FlickPlayer"

# Legacy directory name still found under ``%APPDATA%`` /
# ``%LOCALAPPDATA%`` on machines that ran v1.5.8 or earlier. The
# migration below renames it once per location.
_LEGACY_APP_DIR_NAME = "img_player"


def _appdata_roaming_root() -> Path:
    """Per-user ROAMING app-data root (follows users across machines)."""
    if sys.platform == "win32":
        env = os.environ.get("APPDATA")
        if env:
            return Path(env)
        return Path.home() / "AppData" / "Roaming"
    if sys.platform == "darwin":  # pragma: no cover — non-Windows
        return Path.home() / "Library" / "Application Support"
    xdg = os.environ.get("XDG_CONFIG_HOME")  # pragma: no cover
    return Path(xdg) if xdg else Path.home() / ".config"  # pragma: no cover


def _appdata_local_root() -> Path:
    """Per-user LOCAL app-data root (machine-bound, not roamed)."""
    if sys.platform == "win32":
        env = os.environ.get("LOCALAPPDATA")
        if env:
            return Path(env)
        return Path.home() / "AppData" / "Local"
    if sys.platform == "darwin":  # pragma: no cover — non-Windows
        return Path.home() / "Library" / "Caches"
    xdg = os.environ.get("XDG_CACHE_HOME")  # pragma: no cover
    return Path(xdg) if xdg else Path.home() / ".cache"  # pragma: no cover


def user_prefs_dir() -> Path:
    """Where ``flick.toml`` (user prefs) lives. Roamed."""
    return _appdata_roaming_root() / APP_DIR_NAME


def disk_cache_default_dir() -> Path:
    """Default location for the on-disk frame cache. Local (not roamed)."""
    return _appdata_local_root() / APP_DIR_NAME / "disk_cache"


def calibration_profile_path() -> Path:
    """Performance calibration JSON. Local."""
    return _appdata_local_root() / APP_DIR_NAME / "profile.json"


def log_dir() -> Path:
    """Where ``flick.log`` lives. Local (high churn, not worth roaming)."""
    if sys.platform == "win32":
        return _appdata_local_root() / APP_DIR_NAME
    # XDG state dir is the canonical home for log files on Linux.
    if sys.platform.startswith("linux"):  # pragma: no cover
        xdg_state = os.environ.get("XDG_STATE_HOME")
        if xdg_state:
            return Path(xdg_state) / APP_DIR_NAME
        return Path.home() / ".local" / "state" / APP_DIR_NAME
    # macOS: alongside the cache.
    return _appdata_local_root() / APP_DIR_NAME  # pragma: no cover


# ============================================================================
# One-shot legacy directory rename
# ============================================================================


def _migrate_one(legacy_dir: Path, current_dir: Path) -> bool:
    """Rename ``legacy_dir`` → ``current_dir`` if the legacy exists and
    the current doesn't. Returns ``True`` if a migration happened.

    Best-effort: any OS error is logged and swallowed — we'd rather
    have Flick boot with a "fresh" appdata than fail to start because
    of a permission glitch on an old folder.
    """
    if not legacy_dir.is_dir():
        return False
    if current_dir.exists():
        # User already has the new dir — keep both and let the legacy
        # one fade. We don't auto-merge (would risk overwriting newer
        # files with older ones).
        return False
    try:
        legacy_dir.rename(current_dir)
        log.info(
            "App-data migration: renamed %s → %s",
            legacy_dir, current_dir,
        )
        return True
    except OSError as err:
        log.warning(
            "App-data migration: could not rename %s → %s (%s). "
            "Old folder left in place; new data lands in %s.",
            legacy_dir, current_dir, err, current_dir,
        )
        return False


_migrated_once = False


def migrate_legacy_dirs_once() -> None:
    """Rename ``img_player`` → ``FlickPlayer`` in every appdata root.

    Called from app boot (``__main__.main``) so the migration runs
    BEFORE any module touches the user prefs / disk cache / log file.
    Idempotent + cheap after the first call.
    """
    global _migrated_once
    if _migrated_once:
        return
    _migrated_once = True

    pairs = [
        (
            _appdata_roaming_root() / _LEGACY_APP_DIR_NAME,
            _appdata_roaming_root() / APP_DIR_NAME,
        ),
        (
            _appdata_local_root() / _LEGACY_APP_DIR_NAME,
            _appdata_local_root() / APP_DIR_NAME,
        ),
    ]
    for legacy, current in pairs:
        _migrate_one(legacy, current)
