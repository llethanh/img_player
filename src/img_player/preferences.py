"""Persistent user preferences, stored via Qt's ``QSettings`` backend.

On Windows this lives in the registry under
``HKCU\\Software\\img_player\\img_player``. On macOS/Linux it's a standard
config file. Use :class:`Preferences` from app code — never touch QSettings
directly elsewhere.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings

_ORG = "img_player"
_APP = "img_player"
_RECENT_LIMIT = 10


class Preferences:
    """Typed, app-shaped wrapper around QSettings.

    Every property round-trips through the platform's settings store. Writes
    are flushed immediately; reads are cheap.
    """

    def __init__(self) -> None:
        self._s = QSettings(_ORG, _APP)

    # ------------------------------------------------------------------ Last / recent

    @property
    def last_path(self) -> Path | None:
        raw = self._s.value("session/last_path")
        if not raw:
            return None
        return Path(str(raw))

    @last_path.setter
    def last_path(self, value: Path | None) -> None:
        if value is None:
            self._s.remove("session/last_path")
        else:
            self._s.setValue("session/last_path", str(value))

    def recent_paths(self) -> list[Path]:
        raw = self._s.value("session/recent", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [Path(str(p)) for p in raw if p]

    def push_recent(self, path: Path) -> None:
        existing = [str(p) for p in self.recent_paths()]
        spath = str(path)
        # Keep most-recent-first, deduplicated.
        existing = [p for p in existing if p != spath]
        existing.insert(0, spath)
        self._s.setValue("session/recent", existing[:_RECENT_LIMIT])

    def clear_recent(self) -> None:
        self._s.remove("session/recent")

    # ------------------------------------------------------------------ Playback / color

    @property
    def fps(self) -> float:
        val = self._s.value("playback/fps", 24.0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 24.0

    @fps.setter
    def fps(self, value: float) -> None:
        self._s.setValue("playback/fps", float(value))

    @property
    def source_colorspace(self) -> str | None:
        raw = self._s.value("color/source")
        return str(raw) if raw else None

    @source_colorspace.setter
    def source_colorspace(self, value: str | None) -> None:
        if value:
            self._s.setValue("color/source", value)
        else:
            self._s.remove("color/source")

    @property
    def display(self) -> str | None:
        raw = self._s.value("color/display")
        return str(raw) if raw else None

    @display.setter
    def display(self, value: str | None) -> None:
        if value:
            self._s.setValue("color/display", value)
        else:
            self._s.remove("color/display")

    @property
    def view(self) -> str | None:
        raw = self._s.value("color/view")
        return str(raw) if raw else None

    @view.setter
    def view(self, value: str | None) -> None:
        if value:
            self._s.setValue("color/view", value)
        else:
            self._s.remove("color/view")

    # ------------------------------------------------------------------ Window geometry

    @property
    def window_geometry(self) -> bytes | None:
        raw = self._s.value("window/geometry")
        if raw is None:
            return None
        return bytes(raw) if isinstance(raw, (bytes, bytearray)) else None

    @window_geometry.setter
    def window_geometry(self, value: bytes | None) -> None:
        if value:
            self._s.setValue("window/geometry", value)
        else:
            self._s.remove("window/geometry")

    @property
    def window_state(self) -> bytes | None:
        """Qt's serialised dock layout (``QMainWindow.saveState()``).

        Stores the visibility, position, size and floating state of
        every QDockWidget in the window — so the right-hand "Panels"
        dock comes back collapsed/floating/whatever the user left it
        when they re-open the app.
        """
        raw = self._s.value("window/state")
        if raw is None:
            return None
        return bytes(raw) if isinstance(raw, (bytes, bytearray)) else None

    @window_state.setter
    def window_state(self, value: bytes | None) -> None:
        if value:
            self._s.setValue("window/state", value)
        else:
            self._s.remove("window/state")
