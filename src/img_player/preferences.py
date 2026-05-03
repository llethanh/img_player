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

    # ---- Recent .session files (multi-layer setups) -----------------
    # Same shape as ``recent_paths`` above but for ``.session`` JSON
    # files (saved multi-layer setups). Kept in a separate QSettings
    # key so the two recent lists don't bleed into each other in the
    # File menu.

    def recent_sessions(self) -> list[Path]:
        raw = self._s.value("session_files/recent", [])
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [Path(str(p)) for p in raw if p]

    def push_recent_session(self, path: Path) -> None:
        existing = [str(p) for p in self.recent_sessions()]
        spath = str(path)
        existing = [p for p in existing if p != spath]
        existing.insert(0, spath)
        self._s.setValue("session_files/recent", existing[:_RECENT_LIMIT])

    def clear_recent_sessions(self) -> None:
        self._s.remove("session_files/recent")

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

    @property
    def side_panel_visible(self) -> bool:
        """Whether the right-hand Color/Comments panel is visible.

        Used to live in :meth:`QMainWindow.saveState` (when the panel
        was a real QDockWidget); promoted to an explicit pref now
        that the panel is a plain widget nested inside the central
        layout — saveState doesn't see it anymore.
        """
        raw = self._s.value("view/side_panel_visible", True)
        if isinstance(raw, str):
            return raw.lower() in ("true", "1", "yes")
        return bool(raw)

    @side_panel_visible.setter
    def side_panel_visible(self, value: bool) -> None:
        self._s.setValue("view/side_panel_visible", bool(value))

    # Transparency / alpha convention previously lived here as global
    # prefs. They moved to ``Layer.alpha_composite`` /
    # ``Layer.alpha_is_straight`` (per-layer, auto-detected from the
    # source extension in ``Layer.from_sequence``). The QSettings
    # values are now ignored — old keys stay in the user's INI but
    # nothing reads them.

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

    # ------------------------------------------------------------------ Layer panel (v1.0)

    @property
    def layer_panel_collapsed(self) -> bool:
        """Whether the multi-layer panel below the timeline is folded
        away. Default: ``False`` (= visible) so first-run users see
        the new feature."""
        raw = self._s.value("layer_panel/collapsed", False)
        if isinstance(raw, str):
            return raw.lower() in ("true", "1", "yes")
        return bool(raw)

    @layer_panel_collapsed.setter
    def layer_panel_collapsed(self, value: bool) -> None:
        self._s.setValue("layer_panel/collapsed", bool(value))

    # ------------------------------------------------------------------ Ephemeral annotations (v0.4.1)

    @property
    def ephemeral_duration_preset(self) -> int:
        """Index of the active ephemeral fade preset.

        ``0`` = court (~2 s), ``1`` = moyen (~5 s, default),
        ``2`` = long (~10 s). Persisted across sessions so the user's
        last-picked rhythm survives an app restart. We only persist the
        preset *index* — the seconds-mapping lives in the toolbar code,
        keeping this preference free of "magic numbers" that would
        drift if the mapping changes.
        """
        try:
            v = int(self._s.value("ephemeral/duration_preset", 1))
        except (TypeError, ValueError):
            return 1
        return v if v in (0, 1, 2) else 1

    @ephemeral_duration_preset.setter
    def ephemeral_duration_preset(self, value: int) -> None:
        # Silent reject for out-of-range values — same defensive
        # pattern as side_tab_index above. A bad value in QSettings
        # shouldn't crash the app at boot.
        try:
            v = int(value)
        except (TypeError, ValueError):
            return
        if v not in (0, 1, 2):
            return
        self._s.setValue("ephemeral/duration_preset", v)

    # ------------------------------------------------------------------ Pen stabilizer (Lazy Mouse)

    @property
    def pen_stabilizer_level(self) -> int:
        """Index of the pen stabilizer (Lazy Mouse) preset.

        ``0`` = off (default — line follows the cursor exactly),
        ``1`` = medium (light filtering of hand tremor),
        ``2`` = strong (line trails noticeably behind the cursor for
        ultra-clean review annotations). Persisted so the user's
        chosen smoothing survives an app restart.
        """
        try:
            v = int(self._s.value("annotate/stabilizer_level", 0))
        except (TypeError, ValueError):
            return 0
        return v if v in (0, 1, 2) else 0

    @pen_stabilizer_level.setter
    def pen_stabilizer_level(self, value: int) -> None:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return
        if v not in (0, 1, 2):
            return
        self._s.setValue("annotate/stabilizer_level", v)

    @property
    def ephemeral_mode_enabled(self) -> bool:
        """Whether ghost mode is on. Persisted so the user finds the
        toolbar in the same state as when they closed the app — handy
        for someone who works mostly in ephemeral mode and would
        otherwise have to press G after every restart. Default off:
        a fresh user lands on the persistent (saving) mode."""
        raw = self._s.value("ephemeral/mode_enabled", False)
        if isinstance(raw, str):
            return raw.lower() in ("true", "1", "yes")
        return bool(raw)

    @ephemeral_mode_enabled.setter
    def ephemeral_mode_enabled(self, value: bool) -> None:
        self._s.setValue("ephemeral/mode_enabled", bool(value))

    # ------------------------------------------------------------------ Channel menu (Slice 2 of contact-sheet)

    @property
    def channel_layout_mode(self) -> str:
        """Grid shape preference for the contact sheet ("Auto" /
        "1×N" / "N×1" / "2×2" / "3×3" / "4×4"). Defaults to "Auto"
        — matches the menu's footer combo default."""
        raw = self._s.value("channel/layout_mode", "Auto")
        return str(raw) if raw else "Auto"

    @channel_layout_mode.setter
    def channel_layout_mode(self, value: str) -> None:
        self._s.setValue("channel/layout_mode", str(value))

    @property
    def channel_active_label(self) -> str:
        """Label of the radio-selected channel group at last shutdown.

        Empty string when no sequence had ever been opened in this
        QSettings (= fresh user). The transport restores this on
        boot via :meth:`TransportBar.restore_channel_state` — silent
        fallback when the label doesn't exist in the new sequence's
        groups (different EXR conventions)."""
        return str(self._s.value("channel/active_label", "") or "")

    @channel_active_label.setter
    def channel_active_label(self, value: str) -> None:
        self._s.setValue("channel/active_label", str(value))

    @property
    def channel_tile_labels(self) -> tuple[str, ...]:
        """Last set of contact-sheet tiles (group labels). Empty
        tuple = single-mode (no tiles checked)."""
        raw = self._s.value("channel/tile_labels", "")
        if not raw:
            return ()
        if isinstance(raw, list):
            # QSettings backends sometimes round-trip lists natively;
            # accept either to keep the API forgiving.
            return tuple(str(x) for x in raw)
        # Pipe-separated — labels can contain spaces but never a
        # pipe in practice (channel names are EXR-conformant).
        return tuple(s for s in str(raw).split("|") if s)

    @channel_tile_labels.setter
    def channel_tile_labels(self, labels: tuple[str, ...] | list[str]) -> None:
        joined = "|".join(str(l) for l in labels)
        self._s.setValue("channel/tile_labels", joined)

    @property
    def channel_labels_visible(self) -> bool:
        """Whether the per-tile name chip is baked onto contact-sheet
        composites. Default ``True`` — first-run discoverability ("which
        tile is which?") matters more than visual purity."""
        raw = self._s.value("channel/labels_visible", True)
        if isinstance(raw, str):
            return raw.lower() in ("true", "1", "yes")
        return bool(raw)

    @channel_labels_visible.setter
    def channel_labels_visible(self, value: bool) -> None:
        self._s.setValue("channel/labels_visible", bool(value))

    # ------------------------------------------------------------------ Export dialog (v0.5.0)

    @property
    def export_settings(self) -> dict[str, object]:
        """Round-trip the last-used export dialog settings.

        Stored as flat keys under ``export/...`` so QSettings keeps
        them in plain INI on macOS / Linux. The dialog calls
        :meth:`ExportSettings.from_prefs_dict` with this on open and
        :meth:`ExportSettings.to_prefs_dict` on accept. Defaults are
        produced by :class:`ExportSettings` itself when a key is
        missing — no defaults baked in here so the source of truth
        stays in one place.
        """
        keys = (
            "output_dir",
            "start_frame",
            "format_key",
            "width",
            "height",
            "fps",
            "apply_display_transform",
            "bake_annotations",
            "copy_sidecar",
            "jpg_quality",
            "exr_compression",
            "video_crf",
            "prores_profile",
            "h26x_preset",
        )
        out: dict[str, object] = {}
        for key in keys:
            raw = self._s.value(f"export/{key}")
            if raw is not None:
                out[key] = raw
        return out

    @export_settings.setter
    def export_settings(self, data: dict[str, object]) -> None:
        for key, value in data.items():
            self._s.setValue(f"export/{key}", value)
