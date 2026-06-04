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

# Splash PNG — loaded at boot via ``QSplashScreen`` from
# ``src/img_player/splash.py``. Regenerate it now so the version
# stamp on the splash matches whatever ``__version__`` was bumped
# to before this build. Otherwise the bundled splash drifts behind
# real releases (the "1.4.2 install that boots showing v1.4.0"
# bug we hit on 1.4.1 and 1.4.2). Falls back to whatever PNG is
# on disk if the regen helper / Pillow blow up — better an old
# splash than no splash.
splash_asset = PROJECT_ROOT / "src" / "img_player" / "assets" / "splash.png"
try:
    import sys as _sys
    _tools_dir = str(PROJECT_ROOT / "tools")
    if _tools_dir not in _sys.path:
        _sys.path.insert(0, _tools_dir)
    from regen_splash import regenerate as _regen_splash  # type: ignore[import-untyped]
    _regen_splash(PROJECT_ROOT)
    print(f"[spec] regenerated splash PNG at v{VERSION}")
except Exception as _err:  # noqa: BLE001 — best-effort, never fail the build over the splash
    print(f"[spec] splash regen skipped ({_err}); using existing PNG if any.")
splash_datas = (
    [(str(splash_asset), "img_player/assets")] if splash_asset.is_file() else []
)

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
# HISTORICAL NOTE — kept here as a warning for future me:
#
# An earlier version of this spec mirrored the conda-forge
# ``$CONDA_PREFIX\Library\bin\`` directory directly into the bundle
# root (the ``_oiio_extra_bins`` block) to fix an
# ``ImportError: DLL load failed while importing OpenImageIO`` that
# fired the moment ``__main__`` booted. At that time conda's
# OpenImageIO Python package shipped a bare ``.pyd`` linked to the
# UNVERSIONED libs in ``Library/bin/`` (``OpenImageIO.dll``,
# ``Iex.dll``, ``Imath.dll``, ``OpenEXR.dll``, etc.) — so the spec
# had to hand-pull those.
#
# As of conda-forge 2026-Q2 the package layout switched: the .pyd
# now links to VERSIONED filenames (``Iex_v_3_3_5_OpenImageIO_v3_1.dll``,
# ``OpenEXR_v_3_3_5_OpenImageIO_v3_1.dll``, etc.) that ship in
# ``Lib/site-packages/OpenImageIO/bin/``, and the package's
# ``__init__.py`` does ``os.add_dll_directory("bin")`` so Windows
# finds them at import time. ``collect_dynamic_libs("OpenImageIO")``
# already picks those up.
#
# Mirroring ``Library/bin/`` into the bundle root in addition to
# that creates a NAME-COLLISION TRAP: Windows resolves the .pyd's
# bare references first via its application directory (= the bundle
# root = ``_internal/``), finds the unversioned ``OpenImageIO.dll``,
# loads it — and that DLL was built against a different runtime
# than the .pyd expects, so the symbol lookup fails with
# ``La procédure spécifiée est introuvable``. v1.8.4's first build
# shipped this bug; removing the conda mirror fixes it.
#
# If a future conda layout change brings the unversioned scheme
# back, expect a fresh import failure right here and re-introduce
# a targeted copy (filtered to ONLY the prefixes actually needed).
# Don't blanket-mirror ``Library/bin/`` — that's the trap.
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


# stdlib ``sqlite3`` lives in two pieces on conda-Python:
#   * ``Lib/sqlite3/_sqlite3.pyd``     — Python binding (PyInstaller picks up via stdlib)
#   * ``Library/bin/sqlite3.dll``      — the real SQLite engine (PyInstaller MISSES this)
# The ``.pyd`` delay-loads ``sqlite3.dll`` at import time. PyInstaller's
# PE walker only follows imports listed in the .pyd's directly-bound
# import table; delay-loads from conda's Library/bin/ aren't found.
# Result: the bundle ships ``_sqlite3.pyd`` but no ``sqlite3.dll`` and
# the first ``import sqlite3`` at runtime crashes with
# ``ImportError: DLL load failed while importing _sqlite3``. We hit
# this in v1.8.0 — disk_cache.py is the first module that lands here,
# so the .exe died on startup. Add ``sqlite3.dll`` to the bundle by
# hand to fix it permanently.
stdlib_bins: list[tuple[str, str]] = []
if conda_bin.is_dir():
    sqlite_dll = conda_bin / "sqlite3.dll"
    if sqlite_dll.is_file():
        stdlib_bins.append((str(sqlite_dll), "."))

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
# PyOpenEXR — io/reader.py's fast-path for cold-network EXR decode
# (~30 % faster than OIIO's wrapper on AOV-heavy multipart Maya
# EXRs). Falls back to OIIO when the import or decode fails, but
# we always want the wheel bundled in the standalone .exe. The C++
# OpenEXR DLLs ship with OIIO's transitive deps already.
hidden += collect_submodules("OpenEXR")
hidden += collect_dynamic_libs("OpenEXR")
# PyAV is heavily Cython-based with per-codec submodules loaded at
# decode-time. Pull the whole tree so demuxer / decoder lookups don't
# fail with ImportError mid-playback.
#
# ``collect_submodules("av")`` walks the package's Python sources but
# silently skips compiled-only ``.pyd`` modules that are NOT imported
# at the top of ``av/__init__.py`` — and PyAV uses several of those
# as lazy-loaded utility modules (``bytesource``, ``dictionary``,
# ``utils``, ``opaque``, ``container.pyio``, ``filter.link``,
# ``video.reformatter``). Without explicit hints they're missing
# from the bundle, and the first ``av.open(mp4)`` crashes with
# ``ImportError: No module named 'av.bytesource'``. Listing them
# here as forced hidden imports — PyInstaller will then bundle the
# matching ``.pyd`` files automatically.
hidden += collect_submodules("av")
hidden += [
    "av.bytesource",
    "av.dictionary",
    "av.utils",
    "av.opaque",
    "av.container.pyio",
    "av.filter.link",
    "av.video.reformatter",
]
# sounddevice + its CFFI shim. ``_sounddevice`` is the compiled bridge.
hidden += ["sounddevice", "_sounddevice", "_cffi_backend", "cffi"]
# lz4 is loaded lazily inside ``img_player.cache.disk_cache`` (wrapped in
# a try / except so the disk cache degrades gracefully to stdlib zlib).
# Without this hint PyInstaller's static scan misses ``lz4.frame`` and
# the bundle silently falls back to the slow path even though the dep
# is in the env. The full submodule set is small.
hidden += collect_submodules("lz4")
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
    binaries=oiio_bins + ocio_bins + av_bins + pyside_bins + stdlib_bins,
    datas=shader_datas + font_datas + icon_datas + splash_datas + ocio_datas + oiio_datas + sd_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    # Runtime hook executed *before* __main__.py — forces our bundled
    # _internal/ to win the DLL lookup race against system-wide VFX
    # tooling (mrViewer, Nuke, RV, etc. that put their bin/ on PATH).
    runtime_hooks=[str(PROJECT_ROOT / "pyi_rth_dll_priority.py")],
    # Trim Qt translations + tests we never load — saves ~80 MB.
    # ``tkinter`` is excluded: the splash now goes through the
    # external PowerShell launcher (``splash_launcher.ps1`` + WPF),
    # so we no longer need PyInstaller's Tk-backed Splash() block,
    # and shedding the Tcl/Tk runtime saves another ~5 MB.
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


# Splash is now external: ``splash_launcher.ps1`` (driven from
# ``flick.bat``) paints a WPF window before Python starts, then
# spawns ``FlickPlayer.exe`` with ``FLICK_LAUNCHER=1`` set. The
# Python side detects the env var and stays quiet (writes a ready
# marker to ``%TEMP%\flick_ready.flag`` when the main window
# appears, which the launcher polls to dismiss its splash). See
# ``src/img_player/splash.py`` for the runtime hooks.


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

# Post-COLLECT: drop a tiny version marker next to the .exe so the
# bundle is self-identifying at a glance — both the filename
# ("Flick Player v1.4.2.txt") and the file body carry the version,
# which is what a reviewer wants to copy / paste into a bug
# report. The bundle dir was just populated by COLLECT above; we
# can just touch the file now.
_bundle_dir = PROJECT_ROOT / "dist" / f"FlickPlayer_v{VERSION}"
if _bundle_dir.is_dir():
    _marker = _bundle_dir / f"Flick Player v{VERSION}.txt"
    try:
        _marker.write_text(f"Flick Player v{VERSION}\n", encoding="utf-8")
        print(f"[spec] wrote version marker {_marker.name}")
    except Exception as _err:  # noqa: BLE001 — never fail the build on a marker file
        print(f"[spec] could not write version marker ({_err})")

    # Copy the site-config template next to the .exe so a deployer
    # can see the schema right there in the bundle. Renaming this
    # file to ``flick.toml`` activates studio-wide preference
    # defaults on the next launch — see site_config.py for the
    # resolution order.
    _site_example = PROJECT_ROOT / "flick.toml.example"
    if _site_example.is_file():
        try:
            import shutil as _shutil
            _shutil.copyfile(_site_example, _bundle_dir / "flick.toml.example")
            print("[spec] copied flick.toml.example into bundle")
        except Exception as _err:  # noqa: BLE001
            print(f"[spec] could not copy flick.toml.example ({_err})")
