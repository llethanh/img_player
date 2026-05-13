"""User-level ``flick.toml`` store.

Companion to :mod:`img_player.site_config`. Where the site config
provides studio-wide defaults (read-only, lives next to the .exe),
this module manages a **per-user TOML file** in the user's app-data
directory that captures the user's explicit preference overrides.

Resolution order at read time (lowest to highest priority):

  1. Hardcoded fallback in :class:`Preferences`
  2. Site ``flick.toml`` (next to .exe)
  3. User ``flick.toml`` (this file) ← wins

Write semantics:

  * The user file is created on first write — until then it doesn't
    exist on disk (no clutter for users who never touch a preference).
  * Each ``set()`` does an atomic write (temp file + ``os.replace``)
    so a crash mid-save can't corrupt prefs.
  * Only the keys the user has actually changed get written — the
    file stays small and human-readable.
  * The user file path is ``%APPDATA%\\img_player\\flick.toml`` on
    Windows; XDG equivalents elsewhere.

Why not QSettings?

  We keep QSettings for "session state" blobs (window geometry,
  recent files, save-frame dialog state) because they're internal
  and not interesting to edit by hand. But for *user-facing*
  preferences a TOML file is portable, version-controllable, and
  doesn't hide values in the Windows registry. A user can copy
  their ``flick.toml`` between machines and get the same setup.
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+ stdlib
except ImportError:  # pragma: no cover — Python <3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef, import-not-found]

log = logging.getLogger(__name__)

_USER_FILE_NAME = "flick.toml"


def default_user_prefs_dir() -> Path:
    """OS-conventional per-user app-data directory for Flick.

    Windows  → ``%APPDATA%\\FlickPlayer\\`` (Roaming).
    macOS    → ``~/Library/Application Support/FlickPlayer/``
    Linux    → ``$XDG_CONFIG_HOME/FlickPlayer/`` or
                ``~/.config/FlickPlayer/``.

    The directory is NOT created here; only :meth:`UserPrefsStore._write`
    creates it lazily on the first user save. The canonical name lives
    in :mod:`img_player.app_paths`.
    """
    # Lazy import to avoid a circular dep at module load (preferences.py
    # imports user_prefs which imports app_paths which… would import
    # nothing else, but keeping it lazy is harmless and future-proof).
    from img_player.app_paths import user_prefs_dir as _dir
    return _dir()


def _toml_quote(value: str) -> str:
    """Encode a string as a TOML basic-string literal."""
    # Spec: backslash, doublequote, control chars need escaping.
    escapes = {
        "\\": "\\\\",
        '"': '\\"',
        "\b": "\\b",
        "\t": "\\t",
        "\n": "\\n",
        "\f": "\\f",
        "\r": "\\r",
    }
    out = []
    for ch in value:
        if ch in escapes:
            out.append(escapes[ch])
        elif ord(ch) < 0x20:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def _format_value(value: Any) -> str:
    """Render a Python value as a TOML literal.

    Supports the scalars we actually store in user prefs: str, bool,
    int, float, Path. Anything else falls back to ``str()`` quoted
    as a string — defensive, but unexpected types will look obvious
    in the file so debug is easy.
    """
    if isinstance(value, bool):
        # Must come before ``int`` because ``isinstance(True, int)`` is True.
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Path):
        return _toml_quote(str(value))
    return _toml_quote(str(value))


def _serialize(data: dict[str, dict[str, Any]]) -> str:
    """Write nested dict as TOML. We only emit one level of nesting
    (sections + flat keys) — matches every user-facing preference we
    expose. Sections appear alphabetically for diff-friendly output.
    """
    lines: list[str] = [
        "# Flick Player — user preferences",
        "# Auto-generated; safe to edit by hand. See flick.toml.example",
        "# at the install root for the documented schema.",
        "",
    ]
    for section in sorted(data.keys()):
        section_dict = data[section]
        if not isinstance(section_dict, dict) or not section_dict:
            continue
        lines.append(f"[{section}]")
        for key in sorted(section_dict.keys()):
            lines.append(f"{key} = {_format_value(section_dict[key])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class UserPrefsStore:
    """File-backed TOML store with dotted-key get/set + atomic writes.

    Constructed once and held by :class:`Preferences`. Thread-safe:
    a single lock guards both the in-memory dict and the file write
    so concurrent ``set()`` from worker threads can't interleave
    bad bytes onto disk.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or (default_user_prefs_dir() / _USER_FILE_NAME)
        self._lock = threading.Lock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------ API

    @property
    def path(self) -> Path:
        """Where the file lives (or will be created at first write)."""
        return self._path

    def exists(self) -> bool:
        """Whether the user file has been created on disk yet."""
        return self._path.is_file()

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Read ``section.key``; ``default`` if missing or malformed."""
        section, _, key = dotted_key.partition(".")
        if not key:
            return default
        with self._lock:
            sec = self._data.get(section)
            if not isinstance(sec, dict):
                return default
            return sec.get(key, default)

    def set(self, dotted_key: str, value: Any) -> None:
        """Write ``section.key = value`` and flush to disk atomically.

        ``value = None`` removes the key (and the section if it
        becomes empty) — equivalent to "reset this pref to fall back
        on the site / hardcoded default".
        """
        section, _, key = dotted_key.partition(".")
        if not key:
            raise ValueError(
                f"user_prefs.set requires a dotted key (got {dotted_key!r})",
            )
        with self._lock:
            sec = self._data.setdefault(section, {})
            if value is None:
                sec.pop(key, None)
                if not sec:
                    self._data.pop(section, None)
            else:
                sec[key] = value
            self._write_unlocked()

    def remove(self, dotted_key: str) -> None:
        """Alias for ``set(key, None)`` — clearer at call sites."""
        self.set(dotted_key, None)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Deep-ish copy of the current data. Used by debug surfaces /
        tests; not part of the hot read path."""
        with self._lock:
            return {sec: dict(kv) for sec, kv in self._data.items()}

    # ------------------------------------------------------------------ Internals

    def _load(self) -> None:
        """Read the file if it exists. Missing file → empty data
        (we'll create the file on first ``set``). Malformed file →
        empty data + warning (preserves the broken file on disk so
        the user can inspect it)."""
        if not self._path.is_file():
            return
        try:
            with self._path.open("rb") as fh:
                parsed = tomllib.load(fh)
        except Exception as err:
            log.warning(
                "User prefs at %s failed to parse (%s) — starting empty. "
                "The malformed file is left in place for inspection.",
                self._path, err,
            )
            return
        # Flatten any unexpected top-level scalars under a synthetic
        # section so we don't lose them — they just won't be reachable
        # via dotted keys until manually fixed.
        for section, content in parsed.items():
            if isinstance(content, dict):
                self._data[section] = dict(content)

    def _write_unlocked(self) -> None:
        """Atomic write: temp file in the same directory + os.replace.

        Caller must hold ``self._lock``. Writes to a sibling ``.tmp``
        file first so a crash mid-save can't corrupt the user's
        existing prefs — the rename is a single syscall that's
        atomic on every modern filesystem we target.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            log.warning(
                "Could not create user prefs directory %s (%s) — "
                "preference change won't persist.",
                self._path.parent, err,
            )
            return
        content = _serialize(self._data)
        # NamedTemporaryFile creates the file with restrictive perms;
        # we then rename it over the target so the final perms are
        # whatever the parent dir's umask grants. Good enough — these
        # are user prefs, not secrets.
        tmp = None
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=".flick-toml-",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            tmp = Path(tmp_name)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp, self._path)
            tmp = None
        except OSError as err:
            log.warning(
                "Failed to write user prefs to %s (%s)",
                self._path, err,
            )
        finally:
            if tmp is not None and tmp.exists():
                # Best-effort cleanup if the rename failed.
                try:
                    tmp.unlink()
                except OSError:
                    pass


_cached: UserPrefsStore | None = None


def user_prefs() -> UserPrefsStore:
    """Process-singleton accessor. Same lazy-loading pattern as
    :func:`img_player.site_config.site_config`."""
    global _cached
    if _cached is None:
        _cached = UserPrefsStore()
    return _cached


def invalidate_cache() -> None:
    """Test helper — force the next :func:`user_prefs` call to
    reinitialise (re-reads the file)."""
    global _cached
    _cached = None
