# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for img_player — produces a standalone Windows folder.

Build with:
    pyinstaller img_player.spec --noconfirm

Output goes to ``dist/FlickPlayer_v<version>/`` where ``<version>`` is
read from ``src/img_player/__init__.py``. Copy that whole folder to
the target machine — it ships its own Python, OpenImageIO,
OpenColorIO and Qt6, so nothing needs to be installed.
"""

import re
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

PROJECT_ROOT = Path(SPECPATH)  # noqa: F821 — SPECPATH is injected by PyInstaller


def _read_version() -> str:
    """Pull ``__version__`` from ``src/img_player/__init__.py`` via a
    cheap regex (no real import — the source tree we're bundling
    isn't on ``sys.path`` yet at spec-eval time).

    Falls back to ``"unknown"`` so a version-stripping accident in
    ``__init__.py`` doesn't fail the whole build.
    """
    init_path = PROJECT_ROOT / "src" / "img_player" / "__init__.py"
    try:
        text = init_path.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    match = re.search(r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', text)
    return match.group(1) if match else "unknown"


VERSION = _read_version()


# ----------------------------------------------------------------------- Datas

# Package datas: GLSL shaders that are loaded via importlib.resources.files()
# at import time. PyInstaller doesn't pick those up automatically — we have
# to list them.
shader_dir = PROJECT_ROOT / "src" / "img_player" / "render" / "shaders"
shader_datas = [
    (str(shader_dir / "vertex.glsl"), "img_player/render/shaders"),
    (str(shader_dir / "fragment_template.glsl"), "img_player/render/shaders"),
]

# Fonts — ``cache.missing_frame.register_bundled_font`` resolves to
# ``src/img_player/assets/fonts/`` via Path(__file__) arithmetic. That
# pattern survives PyInstaller as long as the ``assets/`` tree is
# bundled to the matching destination.
fonts_dir = PROJECT_ROOT / "src" / "img_player" / "assets" / "fonts"
font_datas = [
    (str(fonts_dir / f.name), "img_player/assets/fonts")
    for f in fonts_dir.glob("*") if f.is_file()
]

# App icon — same .ico the EXE() block embeds as the executable's
# resource, also shipped as a data file so the runtime
# ``setWindowIcon`` call (in ``ImgPlayerApp._build_qt_runtime``) can
# resolve it via Path arithmetic. Without this the bundled .exe shows
# the embedded icon in Explorer / taskbar but Qt's window-frame icon
# falls back to the default Qt logo.
icons_dir = PROJECT_ROOT / "src" / "img_player" / "assets" / "icons"
icon_datas = [
    (str(icons_dir / f.name), "img_player/assets/icons")
    for f in icons_dir.glob("*") if f.is_file()
]

# OCIO ships built-in configs *inside* the shared library (ocio://default
# resolves to ACES 2.0 CG). No external files to bundle. But we still want
# to grab any data files the PyOpenColorIO Python package may carry.
ocio_datas = collect_data_files("PyOpenColorIO", include_py_files=False)

# OpenImageIO same idea — most plugins are statically linked into the main
# DLL on conda-forge Windows builds, but we still let the helper scoop up
# anything tagged as "data".
oiio_datas = collect_data_files("OpenImageIO", include_py_files=False)

# sounddevice's pip wheel ships a bundled PortAudio DLL inside
# ``_sounddevice_data/portaudio-binaries/libportaudio64bit.dll`` and
# imports it via ``ctypes.CDLL`` at module load. PyInstaller's hooks
# don't always relocate this folder reliably; ship it explicitly.
sd_datas = []
try:
    import _sounddevice_data  # type: ignore[import-untyped]
    sd_data_root = Path(next(iter(_sounddevice_data.__path__)))
    for p in sd_data_root.rglob("*"):
        if not p.is_file():
            continue
        rel_parent = p.relative_to(sd_data_root).parent
        dest = "_sounddevice_data"
        if str(rel_parent) not in ("", "."):
            dest = f"_sounddevice_data/{rel_parent.as_posix()}"
        sd_datas.append((str(p), dest))
except ImportError:
    pass


# ----------------------------------------------------------------------- Binaries

# These are the native libraries that need to ship next to the .exe. The
# `collect_dynamic_libs` helper walks the package directory and pulls every
# .dll/.pyd it finds — that's what gets us libOpenImageIO, libOpenEXR, all
# the codec backends, libOpenColorIO, expat, yaml-cpp, etc.
oiio_bins = collect_dynamic_libs("OpenImageIO")
ocio_bins = collect_dynamic_libs("PyOpenColorIO")
# PyAV (conda-forge build) does NOT ship FFmpeg next to the ``av``
# package — the DLLs live in the conda env's ``Library/bin/`` and
# ``collect_dynamic_libs("av")`` returns nothing. We have to grab
# them by hand. The list covers the FFmpeg core (avcodec / avformat
# / avutil / sw{resample,scale} / avfilter / avdevice) plus every
# codec backend conda-forge's ffmpeg pulls in: H.264 (x264, openh264),
# H.265 (x265), AV1 (aom, dav1d), VP9 (vpx is part of ffmpeg), audio
# (opus, vorbis/ogg, lame), and image (avif, webp).
import sys
av_bins = []
conda_bin = Path(sys.prefix) / "Library" / "bin"
if conda_bin.is_dir():
    _dll_prefixes = (
        "avcodec-", "avformat-", "avutil-", "avdevice-", "avfilter-",
        "swresample-", "swscale-", "postproc-",
        "libx264", "libx265", "openh264-",
        "aom", "dav1d", "libvpl",
        "opus", "ogg", "vorbis",
        "lame", "mp3lame",
        "avif", "libwebp",
        "lcms2", "zimg", "vmaf",
        "libxml2", "libxslt",
        # SDL3 is loaded transitively by FFmpeg's avdevice on
        # conda-forge win-64 builds since 2026-Q2. PyInstaller's
        # static analyser doesn't see this dep (it's resolved by
        # Windows at LoadLibrary time), so without an explicit
        # prefix here the bundle ships avdevice-XX.dll but no
        # SDL3.dll → "Failed loading SDL3 library" MessageBox the
        # moment a video layer is touched in the .exe.
        "sdl3",
    )
    for f in conda_bin.glob("*.dll"):
        name = f.name.lower()
        if any(name.startswith(p.lower()) for p in _dll_prefixes):
            av_bins.append((str(f), "."))
    # Belt-and-suspenders: PyInstaller's helper sometimes finds extras
    # tied to the ``av`` package itself (build-specific layouts).
    av_bins += collect_dynamic_libs("av")

# PySide6 has its own opinionated layout that PyInstaller's official
# hook usually handles. BUT under the conda-forge install (which is
# our supported env), the Qt6 + Shiboken DLLs live in
# ``$CONDA_PREFIX/Library/bin/`` rather than next to the
# ``PySide6/__init__.py`` — and the official hook only walks the
# package directory. Result: the bundle ships ``QtCore.cp311-...pyd``
# but no ``Qt6Core.dll``, no ``shiboken6.cp311-...dll`` → the .exe
# fails at startup with "DLL load failed while importing Shiboken".
# Pull every Qt6*.dll + the pyside6/shiboken6 runtime from
# ``Library/bin/`` ourselves to fix it.
pyside_bins: list[tuple[str, str]] = []
if conda_bin.is_dir():
    _qt_prefixes = (
        "qt6",
        "shiboken6.",
        "pyside6.",
        "pyside6qml.",
        # Qt6 transitive deps shipped under Library/bin (zstd, double-
        # conversion, harfbuzz, freetype, brotli, pcre2, icudt/icuuc/
        # icuin, libssl/libcrypto). The PyInstaller PE walker would
        # normally chase these from Qt6Core.dll itself, but since the
        # DLLs aren't where it expects, the chase never starts.
        "double-conversion",
        "harfbuzz",
        "freetype",
        "brotli",
        "pcre2",
        "icudt", "icuuc", "icuin", "icuio", "icutu",
        "libssl", "libcrypto",
        "zstd",
        "libpng", "libjpeg", "tiff",
        "zlib",
    )
    for f in conda_bin.glob("*.dll"):
        name = f.name.lower()
        if any(name.startswith(p.lower()) for p in _qt_prefixes):
            pyside_bins.append((str(f), "."))
    # Qt platform / style / image-format plugins live under
    # Library/plugins/ in the conda layout. PyInstaller's PySide6
    # hook expects them under PySide6/plugins/ — bundle them by hand
    # to the path Qt looks them up at runtime
    # (``_internal/PySide6/plugins/<group>/``).
    qt_plugins_root = Path(sys.prefix) / "Library" / "plugins"
    if qt_plugins_root.is_dir():
        for plugin_dll in qt_plugins_root.rglob("*.dll"):
            rel = plugin_dll.relative_to(qt_plugins_root).parent
            dest = f"PySide6/plugins/{rel.as_posix()}" if str(rel) != "." else "PySide6/plugins"
            pyside_bins.append((str(plugin_dll), dest))


# ----------------------------------------------------------------------- Hidden imports

# These are imports done dynamically (lazy imports inside __main__, OIIO
# plugin discovery, OCIO Python helpers). PyInstaller's static analyser
# can miss them, so we list them explicitly.
hidden = []
hidden += collect_submodules("img_player")
hidden += collect_submodules("OpenImageIO")
hidden += collect_submodules("PyOpenColorIO")
# PyAV is heavily Cython-based with per-codec submodules loaded at
# decode-time. Pull the whole tree so demuxer / decoder lookups don't
# fail with ImportError mid-playback.
hidden += collect_submodules("av")
# sounddevice + its CFFI shim. ``_sounddevice`` is the compiled bridge.
hidden += ["sounddevice", "_sounddevice", "_cffi_backend", "cffi"]
hidden += [
    "OpenGL.platform.win32",
    "OpenGL.arrays.ctypesarrays",
    "OpenGL.arrays.numpymodule",
    "OpenGL.arrays.lists",
    "OpenGL.arrays.numbers",
    "OpenGL.arrays.strings",
    "OpenGL.arrays.formathandler",
    "OpenGL.GL.shaders",
]


# ----------------------------------------------------------------------- Build

block_cipher = None

a = Analysis(  # noqa: F821 — Analysis is injected by PyInstaller
    ["src/img_player/__main__.py"],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=oiio_bins + ocio_bins + av_bins + pyside_bins,
    datas=shader_datas + font_datas + icon_datas + ocio_datas + oiio_datas + sd_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    # Runtime hook executed *before* __main__.py — forces our bundled
    # _internal/ to win the DLL lookup race against system-wide VFX
    # tooling (mrViewer, Nuke, RV, etc. that put their bin/ on PATH).
    runtime_hooks=[str(PROJECT_ROOT / "pyi_rth_dll_priority.py")],
    # Trim Qt translations + tests we never load — saves ~80 MB.
    excludes=[
        "tkinter",
        "PySide6.Qt3DAnimation",
        "PySide6.Qt3DCore",
        "PySide6.Qt3DExtras",
        "PySide6.Qt3DInput",
        "PySide6.Qt3DLogic",
        "PySide6.Qt3DRender",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
        "PySide6.QtDesigner",
        "PySide6.QtHelp",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "PySide6.QtNetwork",
        "PySide6.QtPositioning",
        "PySide6.QtQml",
        "PySide6.QtQuick",
        "PySide6.QtQuick3D",
        "PySide6.QtQuickWidgets",
        "PySide6.QtRemoteObjects",
        "PySide6.QtSensors",
        "PySide6.QtSerialPort",
        "PySide6.QtSpatialAudio",
        "PySide6.QtSql",
        # NOTE: PySide6.QtSvg is needed by ui/icons.py (transport icons,
        # burger menu, etc.) — keep it bundled.
        "PySide6.QtTest",
        "PySide6.QtWebChannel",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebSockets",
        "PySide6.QtXml",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)  # noqa: F821


# ----------------------------------------------------------------------- Splash
# Build-time-generated PNG: dark background + Flick icon + title +
# version number, with a reserved bottom band for the dynamic status
# text PyInstaller's bootloader paints over via ``pyi_splash.update_text``.
# The user sees the splash *before* the Python interpreter even starts
# (that's the point — bridge the 2-3 s import wait so they don't doubt
# their click registered).
def _build_splash_png() -> Path:
    """Render the splash PNG into the PyInstaller build dir and
    return its absolute path. Idempotent — overwrites on every
    rebuild so a version bump propagates immediately."""
    from PIL import Image, ImageDraw, ImageFont
    out_dir = PROJECT_ROOT / "build" / "splash"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "splash.png"
    width, height = 480, 240
    img = Image.new("RGB", (width, height), color=(20, 22, 26))  # BG_DEEP
    draw = ImageDraw.Draw(img)

    # Lighter tile behind the icon — the film-perf border on the
    # Flick icon is near-black and disappears against the splash's
    # BG_DEEP. A slightly raised panel (BG_RAISED-ish) gives the
    # icon a frame to stand against.
    tile_size = 112
    tile_x = (width - tile_size) // 2
    tile_y = 16
    draw.rounded_rectangle(
        (tile_x, tile_y, tile_x + tile_size, tile_y + tile_size),
        radius=14,
        fill=(40, 42, 48),       # BG_RAISED-ish
        outline=(56, 56, 60),    # BORDER_DEFAULT
        width=1,
    )

    # Drop the icon centred on the tile. PIL reads .ico containers;
    # pick the largest available size for crisp downscale.
    icon_path = PROJECT_ROOT / "src" / "img_player" / "assets" / "icons" / "flick.ico"
    if icon_path.is_file():
        icon = Image.open(icon_path)
        try:
            icon = icon.resize((88, 88), Image.LANCZOS)
        except Exception:
            pass
        # Convert to RGBA so alpha pastes correctly over the tile.
        if icon.mode != "RGBA":
            icon = icon.convert("RGBA")
        img.paste(
            icon,
            (tile_x + (tile_size - 88) // 2, tile_y + (tile_size - 88) // 2),
            icon,
        )

    # Title + version centred under the icon.
    try:
        title_font = ImageFont.truetype("arialbd.ttf", 22)
        version_font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        title_font = ImageFont.load_default()
        version_font = ImageFont.load_default()
    title = "Flick Player"
    version = f"v{VERSION}"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    version_bbox = draw.textbbox((0, 0), version, font=version_font)
    title_w = title_bbox[2] - title_bbox[0]
    version_w = version_bbox[2] - version_bbox[0]
    draw.text(
        ((width - title_w) // 2, 132),
        title, fill=(232, 144, 28), font=title_font,  # ACCENT
    )
    draw.text(
        ((width - version_w) // 2, 162),
        version, fill=(138, 138, 142), font=version_font,  # TEXT_SECONDARY
    )
    # Bottom band reserved for ``pyi_splash.update_text`` — leave it
    # empty here, the bootloader paints over it at runtime. We just
    # draw a thin separator line so the live status reads as a
    # distinct row rather than floating arbitrarily.
    draw.line([(40, 200), (width - 40, 200)], fill=(56, 56, 60), width=1)
    img.save(out_path, "PNG")
    return out_path


splash_png_path = _build_splash_png()
# ``text_pos`` is the top-left anchor of the dynamic status string.
# The splash is 480 px wide; anchoring at x=40 leaves ~400 px of
# usable width — fits "Initialising OpenColorIO…" and friends with
# room to spare without needing centred-text gymnastics that
# ``pyi_splash`` doesn't natively support.
splash = Splash(  # noqa: F821
    str(splash_png_path),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(40, 215),
    text_size=11,
    text_color="white",
    text_default="Loading…",
    minify_script=True,
    always_on_top=True,
    rundir=None,
)


exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    splash,
    splash.binaries,
    [],
    exclude_binaries=True,
    name="FlickPlayer",
    icon=str(PROJECT_ROOT / "src" / "img_player" / "assets" / "icons" / "flick.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,            # UPX often trips antivirus heuristics — leave off.
    console=False,        # No console window for the bundled .exe.
                          # Logs land in a rotating file under
                          # %LOCALAPPDATA%\img_player\flick.log
                          # (set up by ``__main__._setup_logging``).
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    # Version-stamp the bundle directory so successive builds don't
    # overwrite each other on the Drive (= the user can keep a
    # ``FlickPlayer_v1.1.0`` next to a ``FlickPlayer_v1.2.0`` and
    # diff regressions, hand the older one to a reviewer who needs
    # it, etc.). The ``.exe`` inside keeps its bare ``FlickPlayer``
    # name so existing Windows shortcuts / file associations don't
    # break across versions.
    name=f"FlickPlayer_v{VERSION}",
)
