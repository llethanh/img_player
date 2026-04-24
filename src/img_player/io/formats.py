"""Queries OpenImageIO for the set of image file extensions it can read."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import OpenImageIO as oiio


@lru_cache(maxsize=1)
def supported_extensions() -> frozenset[str]:
    """Return the set of extensions OIIO can read, lowercase with leading dot.

    Example: {'.exr', '.png', '.jpg', '.dpx', ...}
    """
    raw = oiio.get_string_attribute("extension_list")
    exts: set[str] = set()
    for format_entry in raw.split(";"):
        if ":" not in format_entry:
            continue
        _, ext_list = format_entry.split(":", 1)
        for ext in ext_list.split(","):
            ext = ext.strip().lower()
            if ext:
                exts.add(f".{ext}")
    return frozenset(exts)


def is_supported(path: Path | str) -> bool:
    return Path(path).suffix.lower() in supported_extensions()
