"""Site-wide preference defaults loaded from a TOML file at the app root.

For enterprise deployment: ship the Flick bundle, drop a ``flick.toml``
next to the executable, and every user launching this build gets the
studio's preferred defaults (OCIO config, disk-cache path, etc.)
without having to configure QSettings on each machine.

The file is **optional**. When absent, the app uses its hardcoded
defaults. When present, every value in it becomes the new
hardcoded-equivalent default; the user can still override individual
values via File → Preferences (their QSettings choice always wins at
runtime — the site config only changes what they see as "the default"
on a fresh profile or after resetting prefs).

Resolution order for the file location:

  1. ``$FLICK_SITE_CONFIG`` environment variable (absolute path to a
     ``.toml`` file). Useful for testing and for studios that prefer a
     centralised path on a network share.
  2. Next to the executable in frozen / PyInstaller bundle mode
     (``Path(sys.executable).parent / "flick.toml"``).
  3. Next to the package in dev mode (= repo root).
  4. Nothing → empty config, hardcoded defaults apply.

Format
------

Plain TOML with sectioned keys. Values are looked up via dotted-key
notation, e.g. ``site_config().get("color.ocio_builtin_uri")``. See
``flick.toml.example`` at the repo root for the full schema.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+ stdlib
except ImportError:  # pragma: no cover — Python <3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef, import-not-found]

log = logging.getLogger(__name__)

_SITE_FILE_NAME = "flick.toml"


def _resolve_site_path() -> Path | None:
    """Find the active site config file or return ``None`` if none.

    Logs the resolved path at INFO level so a studio admin can confirm
    Flick picked up the right file (visible in ``flick.log``).
    """
    # 1) Explicit env override — wins over everything else. Lets ops
    #    point at a fileshare path without renaming bundled artefacts.
    env = os.environ.get("FLICK_SITE_CONFIG")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        log.warning(
            "FLICK_SITE_CONFIG=%s does not point at a readable file — "
            "falling back to bundle-relative search.",
            env,
        )

    # 2) PyInstaller / frozen bundle: live next to the .exe so a studio
    #    can drop the toml right into the dist folder.
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        p = exe_dir / _SITE_FILE_NAME
        if p.is_file():
            return p

    # 3) Dev mode: live at the repo root (one level above src/).
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent.parent  # src/img_player/ → src → repo
    p = repo_root / _SITE_FILE_NAME
    if p.is_file():
        return p

    return None


class SiteConfig:
    """Read-only view over the parsed TOML data.

    Construction is cheap (just stores the dict); the I/O cost lives
    in :func:`site_config`. Tests can construct one directly with a
    pre-built dict, bypassing the file resolution entirely.
    """

    def __init__(self, data: dict[str, Any], source: Path | None = None) -> None:
        self._data = data or {}
        self._source = source

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up a nested key like ``"color.ocio_builtin_uri"``.

        Returns ``default`` when any segment is missing OR when an
        intermediate node isn't a dict (= malformed config). This is
        deliberately lenient so a botched studio toml degrades to
        hardcoded defaults rather than crashing the app.
        """
        cursor: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    @property
    def source(self) -> Path | None:
        """Where the data was loaded from. ``None`` if no file existed."""
        return self._source

    @property
    def is_empty(self) -> bool:
        return not self._data


_cached: SiteConfig | None = None


def site_config() -> SiteConfig:
    """Process-singleton accessor for the site config.

    Loaded lazily on first call, then cached for the lifetime of the
    process. Subsequent edits to the file don't take effect until
    restart — same semantics as a Windows policy file. The cache is
    NOT cleared by tests (each test that needs a custom site config
    should patch this function directly or use the ``invalidate``
    helper below).
    """
    global _cached
    if _cached is not None:
        return _cached
    path = _resolve_site_path()
    if path is None:
        log.debug("No flick.toml site config found — using built-in defaults.")
        _cached = SiteConfig({}, source=None)
        return _cached
    try:
        with path.open("rb") as fh:
            data = tomllib.load(fh)
    except Exception as err:
        log.warning(
            "flick.toml at %s failed to parse (%s) — ignoring, using built-in defaults.",
            path, err,
        )
        _cached = SiteConfig({}, source=None)
        return _cached
    log.info("Loaded site config from %s", path)
    _cached = SiteConfig(data, source=path)
    return _cached


def invalidate_cache() -> None:
    """Force the next :func:`site_config` call to re-read the file.

    Test-only helper; the live app loads the file exactly once at
    boot and never reloads (deliberately, to mirror policy-file
    semantics where mid-session changes shouldn't affect a running
    process).
    """
    global _cached
    _cached = None
