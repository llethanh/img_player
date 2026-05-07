# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for img_player — produces a standalone Windows folder.

Build with:
    pyinstaller img_player.spec --noconfirm

Output goes to ``dist/img_player/``. Copy that whole folder to the target
machine — it ships its own Python, OpenImageIO, OpenColorIO and Qt6, so
nothing needs to be installed.
"""

from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

PROJECT_ROOT = Path(SPECPATH)  # noqa: F821 — SPECPATH is injected by PyInstaller


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

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FlickPlayer",
    icon=str(PROJECT_ROOT / "src" / "img_player" / "assets" / "icons" / "flick.ico"),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,            # UPX often trips antivirus heuristics — leave off.
    console=True,         # Keep the console window for now: shows logs +
                          # bench output. Switch to False once we trust it.
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
    name="FlickPlayer",
)
