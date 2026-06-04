"""Persistent user preferences, stored via Qt's ``QSettings`` backend.

On Windows this lives in the registry under
``HKCU\\Software\\img_player\\img_player``. On macOS/Linux it's a standard
config file. Use :class:`Preferences` from app code — never touch QSettings
directly elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings

from img_player._value_coerce import qsettings_bool
from img_player.site_config import site_config
from img_player.user_prefs import user_prefs

_ORG = "img_player"
_APP = "img_player"
_RECENT_LIMIT = 10


# Sentinel for "no value layered above the hardcoded default". Distinct
# from ``None`` since ``None`` is itself a valid stored value for some
# preferences (e.g. ``ocio_config_path``).
_UNSET = object()


def _site_default(dotted_key: str, hardcoded: Any) -> Any:
    """Return ``flick.toml[dotted_key]`` from the SITE config, or
    ``hardcoded`` if absent. Used for keys that aren't routed through
    the user TOML store."""
    return site_config().get(dotted_key, hardcoded)


def _layered_default(dotted_key: str, hardcoded: Any) -> Any:
    """Three-tier preference lookup with the user TOML on top.

    Used by every user-facing preference getter. Resolution order:

        user flick.toml  →  site flick.toml  →  hardcoded fallback

    User changes flow through :func:`_set_user_pref` which writes
    only to the user TOML; the site TOML is read-only at runtime.
    The site config supplies the studio-wide default; the user
    config holds individual overrides.
    """
    user_value = user_prefs().get(dotted_key, _UNSET)
    if user_value is not _UNSET:
        return user_value
    return site_config().get(dotted_key, hardcoded)


def _set_user_pref(dotted_key: str, value: Any) -> None:
    """Write a user override to the user TOML. ``None`` removes the
    key so the next read falls back on site / hardcoded. Called by
    every user-facing preference setter — keeps the routing in one
    place so a future refactor (e.g. add a "lock" flag from the site
    config) only has to be patched here."""
    user_prefs().set(dotted_key, value)


# Local alias kept for backwards-compat inside this file. The canonical
# implementation lives in :mod:`_value_coerce` so that pure modules
# (e.g. :mod:`export.settings`) can share it without pulling in Qt.
_qbool = qsettings_bool


def _layered_bool(dotted_key: str, default: bool) -> bool:
    """``_layered_default`` + :func:`qsettings_bool` in one call.

    The site / user TOML stores Python booleans natively, but legacy
    QSettings keys (and TOML files hand-edited by users) sometimes
    round-trip as ``"true"``/``"false"`` strings. ``qsettings_bool``
    folds both into a real bool with the right fallback semantics.
    """
    return qsettings_bool(_layered_default(dotted_key, default), default)


def _layered_int(dotted_key: str, default: int) -> int:
    """``_layered_default`` + ``int`` coerce with fallback. Returns
    ``default`` when the stored value can't be parsed (corrupt TOML,
    user hand-edit gone wrong, …)."""
    raw = _layered_default(dotted_key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _qsettings_dict(qsettings: QSettings, prefix: str, keys: tuple[str, ...]) -> dict[str, object]:
    """Read a fixed set of QSettings keys under ``prefix/`` into a dict.

    Used by :attr:`Preferences.export_settings` and
    :attr:`Preferences.save_frame_settings` — same dance both sides:
    iterate the known keys, copy non-``None`` values into a plain
    dict for the dialog. Defaults stay with the consumer.
    """
    out: dict[str, object] = {}
    for key in keys:
        raw = qsettings.value(f"{prefix}/{key}")
        if raw is not None:
            out[key] = raw
    return out


def _qsettings_set_dict(qsettings: QSettings, prefix: str, data: dict[str, object]) -> None:
    """Write every ``(key, value)`` from ``data`` under ``prefix/``."""
    for key, value in data.items():
        qsettings.setValue(f"{prefix}/{key}", value)


# Keys migrated from QSettings to the user TOML in v1.5.8. The mapping
# is ``(qsettings_key, toml_dotted_key, coerce_fn)``. ``coerce_fn``
# converts the raw QSettings return into the Python type the TOML
# should hold — QSettings returns strings for booleans on disk-backed
# stores, ints on Windows registry, etc. Kept here so the migration
# step has one canonical list rather than duplicating the type knowledge
# scattered across the getters.
_LEGACY_QSETTINGS_MIGRATIONS: tuple[tuple[str, str, Any], ...] = (
    ("color/ocio_config_mode", "color.ocio_config_mode", str),
    ("color/ocio_config_path", "color.ocio_config_path", str),
    ("color/ocio_builtin_uri", "color.ocio_builtin_uri", str),
    ("disk_cache/enabled", "disk_cache.enabled", lambda v: _qbool(v, default=True)),
    ("disk_cache/path", "disk_cache.path", str),
    ("disk_cache/budget_gb", "disk_cache.budget_gb", lambda v: max(0, int(v))),
    ("disk_cache/compression", "disk_cache.compression", lambda v: _qbool(v, default=True)),
)


# Module-level latch so the migration runs at most once per process.
_legacy_migration_done = False

# QSettings flag remembering that the one-shot user-prefs migration
# already happened on this user account. Bump if the migration logic
# changes substantially and old data needs to be re-imported.
_MIGRATION_VERSION = 1
_MIGRATION_FLAG_KEY = "_user_prefs_migration_version"


def _migrate_legacy_qsettings_once(qs: QSettings) -> None:
    """Copy v1.5.7-era QSettings values for the user-facing keys into
    the new user TOML — but exactly ONCE per user account, ever.

    The "once ever" guarantee is critical: if the migration re-ran on
    every launch, deleting the user TOML to fall back on the site
    default would do nothing (the next launch would re-create the
    file from the stale QSettings values). We persist a version flag
    in QSettings the first time it runs so subsequent launches are
    no-ops, regardless of whether the user has since deleted the
    user TOML.

    Conservative: we don't delete the QSettings entries themselves
    (preserves rollback to an older Flick build, and other internal
    QSettings keys live in the same scope). We just stop reading
    from them once the migration flag is set.
    """
    global _legacy_migration_done
    if _legacy_migration_done:
        return
    _legacy_migration_done = True

    # Persistent flag: skip if a previous launch already migrated.
    # ``qs.value`` returns the QSettings-side default when the key is
    # absent — ``0`` means "never migrated". Cast defensively because
    # registry-backed QSettings can return int OR string.
    try:
        prior = int(qs.value(_MIGRATION_FLAG_KEY, 0) or 0)
    except (TypeError, ValueError):
        prior = 0
    if prior >= _MIGRATION_VERSION:
        return

    store = user_prefs()
    migrated = []
    for qkey, toml_key, coerce in _LEGACY_QSETTINGS_MIGRATIONS:
        # Don't clobber a user TOML value (e.g. user already hand-edited).
        if store.get(toml_key, _UNSET) is not _UNSET:
            continue
        raw = qs.value(qkey)
        if raw is None:
            continue
        try:
            value = coerce(raw)
        except (TypeError, ValueError):
            continue
        if value is None or value == "":
            continue
        store.set(toml_key, value)
        migrated.append(toml_key)

    # Mark migration done EVEN IF nothing was migrated — first launch
    # of a brand-new user has nothing in QSettings, but should still
    # be exempt from re-running this routine forever.
    qs.setValue(_MIGRATION_FLAG_KEY, _MIGRATION_VERSION)

    if migrated:
        import logging
        logging.getLogger(__name__).info(
            "Migrated %d legacy QSettings preference(s) into %s: %s",
            len(migrated), store.path, ", ".join(migrated),
        )


class Preferences:
    """Typed, app-shaped wrapper around QSettings + per-user TOML.

    Most properties round-trip through QSettings (window geometry,
    recent files, ephemeral UI state). The "user-facing" preferences
    — color management + disk cache settings — go through a layered
    store: per-user ``flick.toml`` on top, site-wide ``flick.toml``
    in the middle, hardcoded fallback at the bottom. See
    :mod:`img_player.user_prefs` for the user store and
    :mod:`img_player.site_config` for the site config.
    """

    def __init__(self) -> None:
        self._s = QSettings(_ORG, _APP)
        # One-shot migration of v1.5.7-era QSettings values for the
        # user-facing keys into the new user TOML. Idempotent and
        # cheap after the first call thanks to the module-level latch.
        _migrate_legacy_qsettings_once(self._s)

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

    @property
    def transparency_bg_mode(self) -> int:
        """Background drawn under transparent pixels in the viewport.
        ``0 = checker`` (default), ``1 = black``, ``2 = mid-grey``,
        ``3 = white``. Persisted across sessions so the user's choice
        survives restarts.
        """
        try:
            v = int(self._s.value("color/transparency_bg_mode", 0))
        except (TypeError, ValueError):
            return 0
        return v if 0 <= v <= 3 else 0

    @transparency_bg_mode.setter
    def transparency_bg_mode(self, value: int) -> None:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return
        if not 0 <= v <= 3:
            return
        self._s.setValue("color/transparency_bg_mode", v)

    # ---- Master audio (transport bar volume slider + mute) ----------
    @property
    def master_volume(self) -> float:
        """Master linear gain (0.0 silent → 1.0 unity). Persisted so
        the reviewer's preferred level survives across launches —
        nothing more annoying than blasting full volume on every
        cold start when you'd dialled it in on the previous run."""
        try:
            v = float(self._s.value("audio/master_volume", 1.0))
        except (TypeError, ValueError):
            return 1.0
        return max(0.0, min(1.0, v))

    @master_volume.setter
    def master_volume(self, value: float) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        self._s.setValue("audio/master_volume", max(0.0, min(1.0, v)))

    # ---- Default profile for unmarked EXRs ---------------------------
    # Studios that bake their display transform into EXR (or write EXR
    # without a colorspace tag) need a project-wide override so the
    # auto-detector doesn't silently fall through to the EXR-as-linear
    # convention. These prefs are consulted by ``detect_source_colorspace``
    # only when the EXR has no explicit tag *and* no chromaticities;
    # tagged files keep working without surprises.

    # ---- OCIO config source ----------------------------------------
    # User-visible override of how the OCIO config is resolved at
    # boot. ``"default"`` forces the library's built-in (ignoring
    # ``$OCIO``); ``"env"`` keeps the historical behaviour of reading
    # ``$OCIO``; ``"custom"`` loads ``ocio_config_path``. Changes
    # require a restart — the GPU shader, color panel and cached
    # processors are all keyed on the active config.

    @property
    def ocio_config_mode(self) -> str:
        raw = _layered_default("color.ocio_config_mode", "default")
        return raw if raw in ("default", "env", "custom") else "default"

    @ocio_config_mode.setter
    def ocio_config_mode(self, value: str) -> None:
        if value not in ("default", "env", "custom"):
            value = "default"
        _set_user_pref("color.ocio_config_mode", value)

    @property
    def ocio_config_path(self) -> str | None:
        raw = _layered_default("color.ocio_config_path", None)
        return str(raw) if raw else None

    @ocio_config_path.setter
    def ocio_config_path(self, value: str | None) -> None:
        _set_user_pref(
            "color.ocio_config_path",
            str(value) if value else None,
        )

    # ---- Built-in OCIO config selection ---------------------------------
    # When ``ocio_config_mode == "default"``, this URI picks WHICH of the
    # OCIO library's bundled configs to load. The shipped default is the
    # ACES 1.3 CG config — it matches the view-transform family used by
    # Nuke / Maya / OpenRV in the vast majority of studios (the older
    # "RRT + ODT" curve). The newer ACES 2.0 CG config is also bundled
    # but gives visibly different highlights / blues, so it's opt-in
    # rather than default to avoid surprising users coming from a
    # 1.x pipeline.

    _DEFAULT_OCIO_BUILTIN_URI = "ocio://cg-config-v2.2.0_aces-v1.3_ocio-v2.4"

    @property
    def ocio_builtin_uri(self) -> str:
        raw = _layered_default(
            "color.ocio_builtin_uri", self._DEFAULT_OCIO_BUILTIN_URI,
        )
        return str(raw) if raw else self._DEFAULT_OCIO_BUILTIN_URI

    @ocio_builtin_uri.setter
    def ocio_builtin_uri(self, value: str) -> None:
        _set_user_pref(
            "color.ocio_builtin_uri",
            str(value) if value else None,
        )

    @property
    def unmarked_exr_source(self) -> str | None:
        """Source colorspace to apply on EXRs without any colorspace
        signal in their metadata. ``None`` means "use the standard EXR
        scene_linear fallback" (industry default)."""
        raw = self._s.value("color/unmarked_exr_source")
        return str(raw) if raw else None

    @unmarked_exr_source.setter
    def unmarked_exr_source(self, value: str | None) -> None:
        if value:
            self._s.setValue("color/unmarked_exr_source", value)
        else:
            self._s.remove("color/unmarked_exr_source")

    @property
    def unmarked_exr_view(self) -> str | None:
        """View to pair with :attr:`unmarked_exr_source` on tag-less EXRs.
        Only consulted when the source override fired (= the file would
        have hit the EXR scene_linear fallback)."""
        raw = self._s.value("color/unmarked_exr_view")
        return str(raw) if raw else None

    @unmarked_exr_view.setter
    def unmarked_exr_view(self, value: str | None) -> None:
        if value:
            self._s.setValue("color/unmarked_exr_view", value)
        else:
            self._s.remove("color/unmarked_exr_view")

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
    def prefs_dialog_section(self) -> int:
        """Index of the active section in the Preferences dialog's
        sidebar (General = 0, Color Management = 1, Disk cache = 2…).

        Persisted so the dialog reopens on the same section the user
        was reading last — both within a session (close + reopen the
        dialog) and across app launches. The dialog clamps the
        returned value to its actual section count at open time, so
        a future refactor that drops sections doesn't crash on an
        out-of-range index pulled from a stale prefs file.
        """
        try:
            return int(self._s.value("prefs_dialog/section", 0))
        except (TypeError, ValueError):
            return 0

    @prefs_dialog_section.setter
    def prefs_dialog_section(self, value: int) -> None:
        try:
            self._s.setValue("prefs_dialog/section", int(value))
        except (TypeError, ValueError):
            return

    @property
    def display_timecode(self) -> bool:
        """``True`` if the View → Show timecode toggle was on at last
        close. The Ctrl+T action mirrors this in the menu state.
        """
        raw = self._s.value("view/display_timecode", False)
        return _qbool(raw)

    @display_timecode.setter
    def display_timecode(self, value: bool) -> None:
        self._s.setValue("view/display_timecode", bool(value))

    @property
    def burnin_enabled(self) -> bool:
        """``True`` if the View → Show burnins toggle was on at last
        close. The Ctrl+B action and the View menu entry mirror this
        in the menu state, and the overlay reads it at boot."""
        return _qbool(self._s.value("view/burnin_enabled", False))

    @burnin_enabled.setter
    def burnin_enabled(self, value: bool) -> None:
        self._s.setValue("view/burnin_enabled", bool(value))

    @property
    def burnin_template_slug(self) -> str:
        """Active burnin template slug. Defaults to ``"default"`` —
        the shipped builtin that prints sequence + frame counter on
        the top bar and user + date on the bottom. Unknown slugs
        (e.g. a user template that has been deleted) fall back to
        the builtin at boot rather than rendering an empty overlay.
        Pre-1.7 prefs that still hold ``"dailies_default"`` /
        ``"minimal"`` / ``"studio_banner"`` are resolved by the
        burnin loader's ``resolve_slug`` shim — no migration is
        needed here."""
        raw = self._s.value("view/burnin_template_slug", "default")
        return str(raw) if raw else "default"

    @burnin_template_slug.setter
    def burnin_template_slug(self, value: str) -> None:
        self._s.setValue("view/burnin_template_slug", str(value or ""))

    @property
    def burnin_shared_dir(self) -> str:
        """Optional path to a shared burnin-templates directory —
        typically a network share that everyone on a project points
        at, so the team converges on the same library without
        manually emailing JSONs around. Empty string when no shared
        dir is configured; the editor's "Shared folder…" toolbar
        button is what writes here. Templates in this dir surface
        in the View → Active burnin template submenu alongside
        local user templates and the builtin."""
        raw = self._s.value("view/burnin_shared_dir", "")
        return str(raw) if raw else ""

    @burnin_shared_dir.setter
    def burnin_shared_dir(self, value: str) -> None:
        self._s.setValue("view/burnin_shared_dir", str(value or ""))

    @property
    def side_panel_visible(self) -> bool:
        """Whether the right-hand Color/Comments panel is visible.

        Used to live in :meth:`QMainWindow.saveState` (when the panel
        was a real QDockWidget); promoted to an explicit pref now
        that the panel is a plain widget nested inside the central
        layout — saveState doesn't see it anymore.
        """
        raw = self._s.value("view/side_panel_visible", True)
        return _qbool(raw)

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
        return _qbool(raw)

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
        return _qbool(raw)

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
        return _qbool(raw)

    @ephemeral_mode_enabled.setter
    def ephemeral_mode_enabled(self, value: bool) -> None:
        self._s.setValue("ephemeral/mode_enabled", bool(value))

    # NB: ``channel_active_label`` was retired in v1.2 alongside the
    # other channel-menu prefs (``channel/tile_labels``,
    # ``channel/layout_mode``, ``channel/labels_visible``). The active
    # channel is no longer persisted across runs — each newly loaded
    # sequence opens on its first group so the user can't see a
    # stale pick carry over from a previous, unrelated sequence.
    # Existing QSettings keys stay as harmless leftovers.

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
            "basename",
            "format_key",
            "width",
            "height",
            "fps",
            "apply_display_transform",
            "bake_annotations",
            "copy_sidecar",
            "bake_compare",
            "jpg_quality",
            "exr_compression",
            "video_crf",
            "prores_profile",
            "h26x_preset",
            "missing_frame_policy",
        )
        return _qsettings_dict(self._s, "export", keys)

    @export_settings.setter
    def export_settings(self, data: dict[str, object]) -> None:
        _qsettings_set_dict(self._s, "export", data)

    # ------------------------------------------------------------------ Save Frame As (v1.2)

    @property
    def save_frame_settings(self) -> dict[str, object]:
        """Round-trip the last-used "Save Frame As…" dialog state.

        Stored under ``save_frame/...`` keys: ``output_dir`` (parent
        directory the user last picked), ``format`` (file extension
        without dot, e.g. ``"png"``), ``with_annotations`` (bool),
        ``bake_compare`` (bool — whether to keep the A/B wipe in the
        capture when compare is active). Defaults are picked by the
        dialog itself when a key is missing so the source of truth
        stays in one place.

        The HUD / brackets / decorative overlays are always excluded
        from the capture (= UI chrome, not content) so there is no
        ``with_overlay`` toggle to persist.
        """
        keys = ("output_dir", "format", "with_annotations", "bake_compare")
        return _qsettings_dict(self._s, "save_frame", keys)

    @save_frame_settings.setter
    def save_frame_settings(self, data: dict[str, object]) -> None:
        _qsettings_set_dict(self._s, "save_frame", data)

    # ------------------------------------------------------------------ Contact sheet (v1.5.14)

    @property
    def contact_sheet_state(self) -> dict[str, object]:
        """Round-trip the last contact-sheet config (enabled flag,
        grid dims, label toggle). Same dict shape as
        :meth:`ContactSheetState.to_dict` — the app instantiates the
        state from this on boot.

        ``cols`` / ``rows`` are stored as strings (``"None"`` for
        auto, the integer otherwise) since QSettings has no native
        ``int | None``; the parsing in
        :meth:`ContactSheetState.from_dict` understands both.
        """
        keys = (
            "enabled",
            "cols",
            "rows",
            "show_labels",
            "output_divisor",
            "label_size",
        )
        return _qsettings_dict(self._s, "contact_sheet", keys)

    @contact_sheet_state.setter
    def contact_sheet_state(self, data: dict[str, object]) -> None:
        # Normalise ``None`` → the string "None" so the QSettings
        # round-trip preserves the auto-grid marker (a stored ``None``
        # comes back as the empty string on POSIX .conf files,
        # ambiguous with "user wrote 0").
        normalised = {
            key: ("None" if value is None else value)
            for key, value in data.items()
        }
        _qsettings_set_dict(self._s, "contact_sheet", normalised)

    # ------------------------------------------------------------------ Disk cache (v1.5)

    @property
    def disk_cache_enabled(self) -> bool:
        """When ``True``, the cache writes evicted RAM frames to disk
        and looks them up before re-decoding on subsequent sessions.

        Default ``True`` — the disk cache is a strict performance win
        on a typical review workflow (open the same shot day after
        day). Users on machines with tight SSD budgets can opt out
        from Preferences → Disk cache.
        """
        return _layered_bool("disk_cache.enabled", True)

    @disk_cache_enabled.setter
    def disk_cache_enabled(self, value: bool) -> None:
        _set_user_pref("disk_cache.enabled", bool(value))

    @property
    def disk_cache_path(self) -> Path | None:
        """Where to store the on-disk frame cache. ``None`` means use
        the default location (``%LOCALAPPDATA%\\img_player\\disk_cache\\``
        on Windows; XDG-standard equivalent elsewhere).

        Users can pick a custom path in Preferences when they want
        the cache on a faster drive (NVMe scratch) or a different
        partition with more headroom.
        """
        raw = _layered_default("disk_cache.path", None)
        if not raw:
            return None
        return Path(str(raw))

    @disk_cache_path.setter
    def disk_cache_path(self, value: Path | str | None) -> None:
        _set_user_pref(
            "disk_cache.path",
            str(value) if value is not None else None,
        )

    @property
    def disk_cache_budget_gb(self) -> int:
        """Soft upper bound on disk-cache size, in **gigabytes**.

        ``0`` = unlimited (the cache only grows when the user
        explicitly clears it). Default 50 GB — enough room for ~2 000
        4K frames. Values are stored as int GB to keep the
        Preferences spinner UI simple; the cache converts to bytes
        internally.
        """
        return max(0, _layered_int("disk_cache.budget_gb", 50))

    @disk_cache_budget_gb.setter
    def disk_cache_budget_gb(self, value: int) -> None:
        _set_user_pref("disk_cache.budget_gb", max(0, int(value)))

    # ------------------------------------------------------------------ Network staging

    @property
    def network_staging_enabled(self) -> bool:
        """Network-source staging cache. When on, image-sequence
        layers opened from network shares (UNC, mapped drives) get
        background-copied to a local SSD staging dir; reads then
        decode from the local copy. Measured ~3× cold-decode
        speedup on AOV-heavy Maya EXR over SMB because the readers
        do many small reads that SMB can't pipeline well. Default
        ``True`` — flipping off forces every read to go direct to
        the network share."""
        return _layered_bool("network_staging.enabled", True)

    @network_staging_enabled.setter
    def network_staging_enabled(self, value: bool) -> None:
        _set_user_pref("network_staging.enabled", bool(value))

    @property
    def network_staging_path(self) -> Path | None:
        """Optional explicit path for the staging root. Empty / unset
        → :func:`app_paths.network_staging_default_dir` (i.e.
        ``%LOCALAPPDATA%\\FlickPlayer\\staging``). Pin to a faster
        SSD if your %LOCALAPPDATA% lives on a slow drive."""
        raw = _layered_default("network_staging.path", "")
        return Path(str(raw)) if raw else None

    @network_staging_path.setter
    def network_staging_path(self, value: Path | str | None) -> None:
        _set_user_pref(
            "network_staging.path",
            "" if value is None else str(value),
        )

    @property
    def network_staging_budget_gb(self) -> int:
        """Maximum size (GB) the staging cache may grow to. When
        exceeded, the LRU sequence directory is evicted whole. Default
        50 GB — fits ~5 typical 23 GB Maya AOV sequences."""
        return max(0, _layered_int("network_staging.budget_gb", 50))

    @network_staging_budget_gb.setter
    def network_staging_budget_gb(self, value: int) -> None:
        _set_user_pref("network_staging.budget_gb", max(0, int(value)))

    # ------------------------------------------------------------------ Video cache

    @property
    def video_cache_budget_gb(self) -> int:
        """RAM budget per VideoSource — the per-video-layer LRU
        cache that holds **uint8 RGBA** decoded frames and is fed
        by the background prefetch worker. Default 8 GB matches the
        image-sequence cache budget. Per-frame footprint at common
        resolutions: 3.5 MB (720p), 8 MB (1080p), 14 MB (1440p),
        32 MB (4K). Multiply by the number of concurrent video
        layers to estimate total RAM use; the OS reclaims it when
        a layer closes.

        At the 8 GB default that caches ~2280 frames at 720p, ~1020
        at 1080p, ~570 at 1440p, or ~256 at 4K (= 38, 17, 9.5,
        4.3 seconds at 60 fps respectively). The setting takes
        effect at the next video open; restarting Flick is not
        required, but already-open video layers keep their
        previously-allocated budget until closed."""
        return max(0, _layered_int("video_cache.budget_gb", 8))

    @video_cache_budget_gb.setter
    def video_cache_budget_gb(self, value: int) -> None:
        _set_user_pref("video_cache.budget_gb", max(0, int(value)))

    @property
    def disk_cache_compression(self) -> bool:
        """Legacy bool view of :attr:`disk_cache_compression_mode`.

        Pre-v1.5.x, compression was a simple on/off flag. The newer
        tri-state ("none" / "lz4" / "lz4hc") supersedes it; this
        bool stays for backward compat with any caller that hasn't
        migrated. Returns ``True`` whenever the mode is anything
        other than ``"none"``.
        """
        return self.disk_cache_compression_mode != "none"

    @disk_cache_compression.setter
    def disk_cache_compression(self, value: bool) -> None:
        # Map bool → matching mode string. ``True`` lands on fast
        # LZ4 (the historical default); ``False`` on "none".
        self.disk_cache_compression_mode = "lz4" if value else "none"

    @property
    def disk_cache_compression_mode(self) -> str:
        """Compression mode for new disk-cache writes — one of
        ``"none"``, ``"lz4"``, ``"lz4hc"``.

        Trade-offs:

        * **"none"** — no compression. Fastest reads & writes; blobs
          are ~50 MB per 4K frame. Worth picking only on fast NVMe
          where I/O is essentially free and the ~5 ms lz4 decode
          shows up in the profile.
        * **"lz4"** (default) — fast LZ4 at compression level 1.
          ~5 ms decode per 4K frame, ~25 MB on disk. The universal
          safe choice on any disk type.
        * **"lz4hc"** — LZ4 High Compression at level 12. Same
          decoder as fast LZ4 (~5 ms / 4K), produces ~30 % smaller
          blobs (~17 MB / 4K). Encoder is ~3× slower than fast LZ4
          but each blob is written once and read many times across
          sessions, so the encode cost amortises immediately.

        Setting any value only affects **new writes**; existing
        entries stay readable since both LZ4 sub-modes share the
        same on-disk format and the "none" path uses a distinct
        magic prefix that the reader auto-detects.

        Unknown stored values (e.g. a corrupt prefs file) fall
        back silently to ``"lz4"``.
        """
        from img_player.cache.disk_cache import COMPRESSION_MODES  # noqa: PLC0415
        raw_value = _layered_default("disk_cache.compression_mode", "")
        raw = str(raw_value) if raw_value else ""
        if raw in COMPRESSION_MODES:
            return raw
        # No mode stored — read the legacy bool (for users upgrading
        # from a version that only had the on/off flag) and map.
        legacy_bool = _layered_bool("disk_cache.compression", True)
        return "lz4" if legacy_bool else "none"

    @disk_cache_compression_mode.setter
    def disk_cache_compression_mode(self, value: str) -> None:
        from img_player.cache.disk_cache import COMPRESSION_MODES  # noqa: PLC0415
        mode = value if value in COMPRESSION_MODES else "lz4"
        _set_user_pref("disk_cache.compression_mode", mode)
        # Keep the legacy bool in sync so any old caller still
        # reading it sees the right value. Once nothing references
        # ``disk_cache.compression`` we can drop this line.
        _set_user_pref("disk_cache.compression", mode != "none")
