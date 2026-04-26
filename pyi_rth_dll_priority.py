"""PyInstaller runtime hook: force the bundle's `_internal/` to win the
Windows DLL lookup race.

Why this exists: when img_player is deployed on a workstation that
already has VFX tooling installed (Nuke, RV, mrViewer, DaVinci, Maya,
Houdini…), those apps put their `bin/` folder on `PATH` — which means
their copy of `OpenEXR.dll`, `Imath.dll`, `OpenImageIO.dll` etc. wins
the lookup against ours. Symptom: the cryptic "ImportError: DLL load
failed while importing OpenImageIO: La procédure spécifiée est
introuvable." (we expect a function symbol that the loaded — wrong
version — DLL doesn't export).

Three layers of defence, in increasing order of strength:

1. Prepend `_internal/` to ``PATH`` for legacy ``SearchPath`` callers.
2. Register `_internal/` with the Win 3.8+ ``AddDllDirectory`` API so
   ``LoadLibraryEx`` with ``LOAD_LIBRARY_SEARCH_USER_DIRS`` finds it.
3. **The actual fix**: pre-load the OpenEXR / OIIO / OCIO chain by
   absolute path with ``LOAD_WITH_ALTERED_SEARCH_PATH``, so that any
   subsequent ``LoadLibrary("OpenEXR.dll")`` issued from inside our
   loaded native modules returns *our* in-memory handle instead of
   re-resolving the name through the system search path.
"""

from __future__ import annotations

import ctypes
import os
import sys
from ctypes import wintypes


# ---- LoadLibraryExW glue --------------------------------------------------

# When given an absolute path, this flag makes Windows search the loaded
# DLL's *own* directory first when resolving its dependencies — exactly
# what we need so that loading our OpenEXR.dll pulls in our Imath.dll,
# not the one mrViewer left on PATH.
_LOAD_WITH_ALTERED_SEARCH_PATH = 0x00000008


def _preload(path: str) -> bool:
    """Load `path` so subsequent LoadLibrary by bare name finds this copy.

    Returns True if the load succeeded.
    """
    if not os.path.isfile(path):
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.LoadLibraryExW.restype = wintypes.HMODULE
        kernel32.LoadLibraryExW.argtypes = [
            wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD,
        ]
        handle = kernel32.LoadLibraryExW(path, None, _LOAD_WITH_ALTERED_SEARCH_PATH)
        if handle:
            return True
    except OSError:
        pass
    # Fallback: ctypes.WinDLL uses LoadLibraryW with default search;
    # still better than nothing if LoadLibraryExW failed.
    try:
        ctypes.WinDLL(path)
        return True
    except OSError:
        return False


# ---- Patch entry point ----------------------------------------------------

# Every native lib our process is likely to share a base name with
# something already on the workstation's PATH (mrViewer, Nuke, RV…).
# Order matters: dependencies first, even though
# LOAD_WITH_ALTERED_SEARCH_PATH usually handles transitive resolution
# correctly — we keep the explicit ordering as a belt-and-suspenders
# safety net for older Windows / unusual driver stacks.
_PRELOAD_CHAIN: tuple[str, ...] = (
    # Qt6 — mrViewer 6 / Nuke / DaVinci Resolve all ship their own Qt6
    # build with possibly-different ABI. Pre-load ours first.
    # Order: Core -> Gui -> Widgets -> OpenGL -> OpenGLWidgets so each
    # link in the dependency chain is already resident before the next
    # one is loaded. LOAD_WITH_ALTERED_SEARCH_PATH below would handle
    # this anyway, but the explicit order is cheaper than re-trying.
    "Qt6Core.dll",
    "Qt6Network.dll",
    "Qt6Gui.dll",
    "Qt6Widgets.dll",
    "Qt6OpenGL.dll",
    "Qt6OpenGLWidgets.dll",
    "Qt6Svg.dll",
    # OpenEXR family
    "Imath.dll",
    "Iex.dll",
    "IlmThread.dll",
    "OpenEXRCore.dll",
    "OpenEXR.dll",
    # Color management
    "OpenColorIO.dll",
    "yaml-cpp.dll",
    "expat.dll",
    "pystring.dll",
    # Image codecs the OIIO plugin layer will pull in
    "tiff.dll",
    "libpng16.dll",
    "libpng.dll",
    "jpeg62.dll",
    "turbojpeg.dll",
    "libjpeg-turbo.dll",
    "Lerc.dll",
    "libdeflate.dll",
    "zlib.dll",
    "zlib1.dll",
    "raw.dll",
    "libraw.dll",
    "libheif.dll",
    "libwebp.dll",
    "aom.dll",
    "SvtAv1Enc.dll",
    "x265.dll",
    # OIIO itself, last
    "OpenImageIO_Util.dll",
    "OpenImageIO.dll",
)


def _patch_dll_search() -> None:
    if sys.platform != "win32":
        return
    if not getattr(sys, "frozen", False):
        return  # not running inside a PyInstaller bundle

    bundle_dir = os.path.dirname(sys.executable)
    internal_dir = os.path.join(bundle_dir, "_internal")
    if not os.path.isdir(internal_dir):
        return

    # 1. Prepend to PATH (legacy SearchPath fallback).
    os.environ["PATH"] = internal_dir + os.pathsep + os.environ.get("PATH", "")

    # 2. Modern Win 3.8+ AddDllDirectory.
    add = getattr(os, "add_dll_directory", None)
    if add is not None:
        try:
            add(internal_dir)
        except OSError:
            pass

    # 3. Pre-load the OpenEXR / OIIO / OCIO chain so subsequent
    #    LoadLibrary calls by name resolve to our in-memory handles.
    for name in _PRELOAD_CHAIN:
        _preload(os.path.join(internal_dir, name))


_patch_dll_search()
