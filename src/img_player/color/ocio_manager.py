"""Wraps an OpenColorIO config: loads it from env or a built-in, exposes queries."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import PyOpenColorIO as ocio

log = logging.getLogger(__name__)

# Built-in config URI. "ocio://default" maps to whatever the installed
# OCIO library considers the current recommended default (ACES 2.0 CG
# config in OCIO 2.5.x).
DEFAULT_BUILTIN_URI = "ocio://default"


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
            cfg = ocio.Config.CreateFromBuiltinConfig(DEFAULT_BUILTIN_URI)
            return cfg, OCIOSource("builtin", DEFAULT_BUILTIN_URI)

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
