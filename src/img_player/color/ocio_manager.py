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
      2. `$OCIO` environment variable
      3. Built-in fallback (`ocio://default`)
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
