"""Regenerate ``src/img_player/assets/splash.png`` from the current
``__version__``.

Called both as a script (``python tools/regen_splash.py``) and as a
module from ``img_player.spec`` so the splash PNG is always in sync
with the version stamped on the .exe. The asset is tracked in git;
running this script after a version bump and committing the new PNG
keeps the splash truthful between builds.

The repo also has ``build/splash/preview_splash.py`` (gitignored —
dev sandbox for tweaking the layout). This module is the canonical
build-time path.
"""

from __future__ import annotations

import re
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_version(root: Path | None = None) -> str:
    """Pull ``__version__`` from ``src/img_player/__init__.py``.

    Falls back to ``"unknown"`` if the regex fails — the splash still
    renders, just with a placeholder string. Same single-source-of-
    truth approach as ``img_player.spec``.
    """
    if root is None:
        root = _project_root()
    init = root / "src" / "img_player" / "__init__.py"
    text = init.read_text(encoding="utf-8")
    match = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    return match.group(1) if match else "unknown"


def regenerate(root: Path | None = None) -> Path:
    """Write a fresh ``splash.png`` to the assets dir and return the
    path. Safe to call repeatedly — overwrites the existing PNG.

    Rendered at 2× the logical splash size (960×520 physical for a
    480×260 logical splash) so a Hi-DPI display can show it 1:1
    without upscaling, and a 1× display gets a clean downscale.
    :mod:`img_player.splash` flags the loaded pixmap with
    ``devicePixelRatio(2.0)`` so Qt picks the matching draw size on
    each screen.
    """
    from PIL import Image, ImageDraw, ImageFont

    if root is None:
        root = _project_root()
    version = read_version(root)
    asset_path = root / "src" / "img_player" / "assets" / "splash.png"

    # Render at SCALE × logical size; the runtime loader sets DPR =
    # SCALE so Qt draws at logical px on Hi-DPI. Bumping SCALE here
    # (e.g. 3 for 4K-native displays) keeps the layout identical and
    # only sharpens the result.
    SCALE = 2
    logical_w, logical_h = 480, 260
    width, height = logical_w * SCALE, logical_h * SCALE
    img = Image.new("RGB", (width, height), color=(20, 22, 26))
    draw = ImageDraw.Draw(img)

    tile_size = 112 * SCALE
    tile_x = (width - tile_size) // 2
    tile_y = 16 * SCALE
    draw.rounded_rectangle(
        (tile_x, tile_y, tile_x + tile_size, tile_y + tile_size),
        radius=14 * SCALE,
        fill=(40, 42, 48),
        outline=(56, 56, 60),
        width=max(1, SCALE),
    )

    icon_path = root / "src" / "img_player" / "assets" / "icons" / "flick.ico"
    if icon_path.is_file():
        icon = Image.open(icon_path)
        icon_px = 88 * SCALE
        try:
            icon = icon.resize((icon_px, icon_px), Image.LANCZOS)
        except Exception:
            pass
        if icon.mode != "RGBA":
            icon = icon.convert("RGBA")
        img.paste(
            icon,
            (
                tile_x + (tile_size - icon_px) // 2,
                tile_y + (tile_size - icon_px) // 2,
            ),
            icon,
        )

    try:
        title_font = ImageFont.truetype("arialbd.ttf", 22 * SCALE)
        version_font = ImageFont.truetype("arial.ttf", 12 * SCALE)
    except OSError:
        title_font = ImageFont.load_default()
        version_font = ImageFont.load_default()

    title = "Flick Player"
    version_label = f"v{version}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    version_bbox = draw.textbbox((0, 0), version_label, font=version_font)
    title_w = title_bbox[2] - title_bbox[0]
    version_w = version_bbox[2] - version_bbox[0]
    draw.text(
        ((width - title_w) // 2, 132 * SCALE),
        title, fill=(232, 144, 28), font=title_font,
    )
    draw.text(
        ((width - version_w) // 2, 162 * SCALE),
        version_label, fill=(138, 138, 142), font=version_font,
    )
    # NB: no baked-in status text. ``splash.update`` paints the runtime
    # status via ``QSplashScreen.showMessage`` which draws on top of the
    # pixmap rather than replacing it — if we bake a placeholder string
    # here it stays visible as a ghost under the live message, giving
    # the user two overlapping "Loading…" lines.

    asset_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(asset_path, "PNG")
    return asset_path


if __name__ == "__main__":
    p = regenerate()
    print(f"Wrote {p}")
