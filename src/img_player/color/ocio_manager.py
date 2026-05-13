"""Wraps an OpenColorIO config: loads it from env or a built-in, exposes queries."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import PyOpenColorIO as ocio

log = logging.getLogger(__name__)

# Built-in config URI used as a fallback when no preference is set or
# we can't read prefs at boot. ``ocio://default`` resolves to whatever
# the installed OCIO library considers the current recommended (ACES
# 2.0 CG config in OCIO 2.5.x). The app-level default — set via
# ``Preferences.ocio_builtin_uri`` and threaded through
# :meth:`_resolve_default` — points at the ACES 1.3 CG config to match
# Nuke / Maya / OpenRV out of the box. This constant is the
# last-ditch fallback.
DEFAULT_BUILTIN_URI = "ocio://default"


@dataclass(frozen=True)
class BuiltinConfigInfo:
    """Display-ready metadata for an OCIO library built-in config.

    Surfaced via :meth:`OCIOManager.list_builtin_configs` so the
    Preferences > Color Management page can populate a dropdown
    without needing to import PyOpenColorIO itself.
    """

    uri: str                # ``ocio://<name>`` URI suitable for CreateFromBuiltinConfig
    name: str               # registry name (the part after ``ocio://``)
    ui_name: str            # human-readable label from the OCIO registry
    aces_family: str        # "1.3" or "2.0" — extracted from the name
    kind: str               # "cg" or "studio"
    recommended: bool       # OCIO's own "recommended for new users" flag
    library_default: bool   # True for the entry ``ocio://default`` resolves to


@dataclass(frozen=True)
class OCIOSource:
    """How the config was obtained, for user feedback / debugging."""

    origin: str  # 'env', 'builtin', 'file'
    description: str


class OCIOManager:
    """Loads an OCIO config and exposes the bits we need.

    Resolution order:
      1. Explicit `config` argument
      2. User preference (mode = ``custom`` / ``env`` / ``default``)
      3. ``$OCIO`` environment variable (legacy fallback when no pref is set)
      4. Built-in fallback (`ocio://default`)
    """

    def __init__(self, config: ocio.Config | None = None) -> None:
        if config is not None:
            self._config = config
            self._source = OCIOSource("explicit", "provided by caller")
        else:
            self._config, self._source = self._resolve_default()

    # ------------------------------------------------------------------ Queries

    @property
    def config(self) -> ocio.Config:
        return self._config

    @property
    def source(self) -> OCIOSource:
        return self._source

    def list_colorspaces(self) -> list[str]:
        return [cs.getName() for cs in self._config.getColorSpaces()]

    def list_displays(self) -> list[str]:
        return list(self._config.getDisplays())

    def list_views(self, display: str) -> list[str]:
        return list(self._config.getViews(display))

    def default_display(self) -> str:
        return str(self._config.getDefaultDisplay())

    def default_view(self, display: str | None = None) -> str:
        display = display or self.default_display()
        return str(self._config.getDefaultView(display))

    def role(self, role_name: str) -> str | None:
        """Return the colorspace name for a given role (e.g. 'scene_linear')."""
        cs = self._config.getColorSpace(role_name)
        return cs.getName() if cs is not None else None

    # ------------------------------------------------------------------ Processors

    def get_processor(self, src: str, dst: str) -> ocio.Processor:
        """Return a CPU/GPU processor from `src` to `dst` colorspace."""
        return self._config.getProcessor(src, dst)

    def get_display_view_processor(self, src: str, display: str, view: str) -> ocio.Processor:
        """Processor from a working colorspace to a display's view transform."""
        return self._config.getProcessor(src, display, view, ocio.TRANSFORM_DIR_FORWARD)

    # ------------------------------------------------------------------ Internals

    @classmethod
    def _resolve_default(cls) -> tuple[ocio.Config, OCIOSource]:
        # User preference (set via File → Preferences → Color Management)
        # takes precedence over the historical $OCIO behaviour. Lazy
        # import + try/except keeps OCIOManager unit-testable without a
        # QApplication and tolerant of QSettings read errors at boot.
        mode, custom_path = cls._read_pref_override()

        if mode == "custom" and custom_path:
            try:
                cfg = ocio.Config.CreateFromFile(custom_path)
                return cfg, OCIOSource("file", f"custom: {custom_path}")
            except ocio.Exception as err:
                log.warning(
                    "Failed to load custom OCIO config %s (%s). "
                    "Falling back to builtin.",
                    custom_path,
                    err,
                )
                cfg = ocio.Config.CreateFromBuiltinConfig(DEFAULT_BUILTIN_URI)
                return cfg, OCIOSource(
                    "builtin",
                    f"{DEFAULT_BUILTIN_URI} (custom load failed)",
                )

        if mode == "default":
            # User-selectable builtin URI (default = ACES 1.3 CG to
            # match Nuke / Maya / OpenRV defaults). Fall back to
            # ``ocio://default`` if the saved URI is unparseable —
            # better than crashing the boot.
            builtin_uri = cls._read_pref_builtin_uri() or DEFAULT_BUILTIN_URI
            try:
                cfg = ocio.Config.CreateFromBuiltinConfig(builtin_uri)
                return cfg, OCIOSource("builtin", builtin_uri)
            except ocio.Exception as err:
                log.warning(
                    "Failed to load builtin OCIO config %s (%s). "
                    "Falling back to %s.",
                    builtin_uri, err, DEFAULT_BUILTIN_URI,
                )
                cfg = ocio.Config.CreateFromBuiltinConfig(DEFAULT_BUILTIN_URI)
                return cfg, OCIOSource(
                    "builtin",
                    f"{DEFAULT_BUILTIN_URI} ({builtin_uri} load failed)",
                )

        # mode == "env" (or unset): keep the historical $OCIO lookup.
        env_path = os.environ.get("OCIO")
        if env_path:
            try:
                cfg = ocio.Config.CreateFromFile(env_path)
                return cfg, OCIOSource("env", f"$OCIO = {env_path}")
            except ocio.Exception as err:  # pragma: no cover — depends on user's env
                log.warning(
                    "Failed to load OCIO from $OCIO=%s (%s). Falling back to builtin.",
                    env_path,
                    err,
                )

        cfg = ocio.Config.CreateFromBuiltinConfig(DEFAULT_BUILTIN_URI)
        return cfg, OCIOSource("builtin", DEFAULT_BUILTIN_URI)

    @staticmethod
    def _read_pref_override() -> tuple[str | None, str | None]:
        """Return ``(mode, custom_path)`` from QSettings, or ``(None, None)``
        if Preferences can't be read (no QApplication, import error, …).

        Kept defensive on purpose: the OCIO config is loaded very early
        in the boot sequence and we don't want any setting-store hiccup
        to take the app down before the splash even appears.
        """
        try:
            from img_player.preferences import Preferences

            prefs = Preferences()
            return prefs.ocio_config_mode, prefs.ocio_config_path
        except Exception as err:  # pragma: no cover — defensive
            log.debug("Could not read OCIO prefs (%s). Using legacy resolution.", err)
            return None, None

    @staticmethod
    def _read_pref_builtin_uri() -> str | None:
        """Return the user's preferred builtin URI (or ``None`` if prefs
        can't be read). Used by :meth:`_resolve_default` when
        ``mode == 'default'``."""
        try:
            from img_player.preferences import Preferences

            return Preferences().ocio_builtin_uri
        except Exception as err:  # pragma: no cover — defensive
            log.debug("Could not read OCIO builtin URI pref (%s).", err)
            return None

    @staticmethod
    def list_builtin_configs() -> list[BuiltinConfigInfo]:
        """Enumerate every OCIO library built-in config available on
        the running OCIO install.

        The Preferences > Color Management page calls this to populate
        the "Built-in config" dropdown. We do the parsing here (extract
        ACES family + kind from the registry name) so the UI layer
        stays free of OCIO imports.

        Returns an empty list if the registry can't be queried (very
        old OCIO build, broken install) — the UI falls back to a hard-
        coded single-entry list in that case.
        """
        out: list[BuiltinConfigInfo] = []
        try:
            registry = ocio.BuiltinConfigRegistry()
            entries = list(registry.getBuiltinConfigs())
        except Exception as err:  # pragma: no cover — OCIO older than 2.2
            log.warning("Could not enumerate OCIO builtin configs (%s).", err)
            return out

        for entry in entries:
            # Each entry is a tuple ``(name, ui_name, recommended, isDefault)``.
            try:
                name, ui_name, recommended, is_default = entry
            except (ValueError, TypeError):
                # Defensive against future OCIO API tuple changes.
                continue
            # Parse the registry name. Examples:
            #   "cg-config-v2.2.0_aces-v1.3_ocio-v2.4"
            #   "studio-config-v4.0.0_aces-v2.0_ocio-v2.5"
            kind = "studio" if name.startswith("studio-config") else "cg"
            if "aces-v2." in name or "aces-v2_" in name:
                aces_family = "2.0"
            elif "aces-v1." in name or "aces-v1_" in name:
                aces_family = "1.3"
            else:
                aces_family = "unknown"
            out.append(
                BuiltinConfigInfo(
                    uri=f"ocio://{name}",
                    name=name,
                    ui_name=str(ui_name),
                    aces_family=aces_family,
                    kind=kind,
                    recommended=bool(recommended),
                    library_default=bool(is_default),
                ),
            )
        return out
