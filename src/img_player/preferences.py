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

    # ------------------------------------------------------------------ Side-tabs + display mode

    @property
    def side_tab_index(self) -> int:
        """Index of the active tab in the right-side dock (Color = 0,
        Comments = 1). ``QMainWindow.saveState()`` covers the dock's
        position / floating / visibility but NOT a child QTabWidget's
        currentIndex — hence the explicit pref.
        """
        try:
            return int(self._s.value("side_tab/index", 0))
        except (TypeError, ValueError):
            return 0

    @side_tab_index.setter
    def side_tab_index(self, value: int) -> None:
        try:
            self._s.setValue("side_tab/index", int(value))
        except (TypeError, ValueError):
            return

    @property
    def display_timecode(self) -> bool:
        """``True`` if the View → Show timecode toggle was on at last
        close. The Ctrl+T action mirrors this in the menu state.
        """
        raw = self._s.value("view/display_timecode", False)
        if isinstance(raw, str):
            return raw.lower() in ("true", "1", "yes")
        return bool(raw)

    @display_timecode.setter
    def display_timecode(self, value: bool) -> None:
        self._s.setValue("view/display_timecode", bool(value))

    # ------------------------------------------------------------------ Annotation toolbar (slice 3)

    @property
    def annotation_toolbar_mode(self) -> str:
        """``"float"`` (overlay on the viewport) or ``"dock"`` (right side).

        Default is ``"float"`` — the lighter-touch mode for first-time
        discovery. Persisted across sessions so the user's choice
        sticks.
        """
        raw = self._s.value("annotation_toolbar/mode", "float")
        return raw if raw in ("float", "dock") else "float"

    @annotation_toolbar_mode.setter
    def annotation_toolbar_mode(self, value: str) -> None:
        if value not in ("float", "dock"):
            value = "float"
        self._s.setValue("annotation_toolbar/mode", value)

    @property
    def annotation_toolbar_pos(self) -> tuple[int, int]:
        """``(x, y)`` position of the toolbar when in float mode.

        Coordinates relative to the GL viewport (top-left = 0,0).
        Default is ``(12, 12)`` — comfortable margin from the corner
        without covering the most common region of interest.
        """
        x = self._s.value("annotation_toolbar/x")
        y = self._s.value("annotation_toolbar/y")
        try:
            return (int(x), int(y)) if x is not None and y is not None else (12, 12)
        except (TypeError, ValueError):
            return (12, 12)

    @annotation_toolbar_pos.setter
    def annotation_toolbar_pos(self, value: tuple[int, int]) -> None:
        try:
            x, y = int(value[0]), int(value[1])
        except (TypeError, ValueError, IndexError):
            return
        self._s.setValue("annotation_toolbar/x", x)
        self._s.setValue("annotation_toolbar/y", y)

    @property
    def annotation_toolbar_visible(self) -> bool:
        """Whether to show the toolbar at startup. Default: hidden."""
        raw = self._s.value("annotation_toolbar/visible", False)
        if isinstance(raw, str):
            return raw.lower() in ("true", "1", "yes")
        return bool(raw)

    @annotation_toolbar_visible.setter
    def annotation_toolbar_visible(self, value: bool) -> None:
        self._s.setValue("annotation_toolbar/visible", bool(value))
