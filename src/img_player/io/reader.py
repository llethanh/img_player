"""Single-frame image decoding on top of OpenImageIO."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

# Disable OIIO's PATH-based DLL search before importing it. Without
# this OIIO's __init__.py iterates every PATH entry and registers
# them via os.add_dll_directory — which on machines with other VFX
# tools installed (mrViewer, Nuke runtime, RV, etc.) means OIIO's
# transitive deps (libheif, OpenEXR, libavif…) get satisfied from
# whichever bundled copy lands first in PATH order. The version
# mismatch then surfaces as "DLL load failed: La procédure spécifiée
# est introuvable" at import time.
#
# With this flag at "0" OIIO uses Python's standard DLL resolution,
# which conda's python launcher has already pointed at the env's
# Library/bin via os.add_dll_directory — exactly the directory that
# holds the matching DLLs. Setting it here (before the import) keeps
# the fix in tree, so users don't have to remember to "set OIIO_…"
# in every shell.
os.environ.setdefault("OIIO_LOAD_DLLS_FROM_PATH", "0")

import numpy as np
import OpenImageIO as oiio

log = logging.getLogger(__name__)


class FrameReadError(RuntimeError):
    """Raised when a frame cannot be decoded (missing file, bad format, ...)."""


# Network-staging hook. The App installs a callable at boot that
# turns a network path into a local-staged path (or returns ``None``
# if not yet staged). When set, ``read_frame`` consults it BEFORE
# opening the file — staged frames decode ~3× faster than direct
# network reads because EXR / DPX / TIFF libs do many small reads
# the SMB protocol can't pipeline well. See
# :mod:`img_player.cache.network_staging` for the manager.
_staging_lookup: "Callable[[Path], Path | None] | None" = None  # type: ignore[name-defined]


def set_staging_lookup(lookup) -> None:  # type: ignore[no-untyped-def]
    """Install (or clear by passing ``None``) the staging lookup
    callable. The App wires this at boot to ``self._staging.
    staged_path_for``. Tests can install a fake to drive
    ``read_frame`` deterministically."""
    global _staging_lookup
    _staging_lookup = lookup


# Local Callable import — kept here so the file-top imports stay
# minimal (we already have collections.abc.Sequence from above).
from collections.abc import Callable  # noqa: E402


def configure_oiio(threads: int | None = None) -> int:
    """Set the OpenImageIO global thread pool size.

    OIIO uses a *single* shared thread pool across the whole process. Our
    own decode worker pool runs ``N`` Python workers that each call into
    OIIO in parallel — OIIO's pool is what lets each individual decode
    parallelise (e.g. across EXR scanlines or channels).

    Sizing rule of thumb:
    * ``None`` (default) → use ``os.cpu_count()``. OIIO is free to use
      every logical core, but it shares them across all in-flight decodes
      so we don't actually create N×workers threads.
    * Pass an explicit number to constrain it (useful when sharing the
      machine with other heavy work).

    Returns the value that was actually applied so the caller can log it.
    """
    desired = threads if threads is not None else (os.cpu_count() or 1)
    desired = max(1, int(desired))
    try:
        oiio.attribute("threads", desired)
    except Exception:
        log.exception("OIIO: failed to set 'threads' attribute to %d", desired)
        return -1

    # Best-effort: also nudge the EXR plugin specifically. Older OIIO
    # builds may not expose this attribute; ignore failures.
    try:
        oiio.attribute("exr_threads", desired)
    except Exception:
        pass

    log.info("OIIO threads attribute set to %d (was reading defaults)", desired)
    return desired


def read_frame(
    path: Path | str,
    channels: Sequence[str] | None = None,
    *,
    as_half: bool = True,
) -> np.ndarray:
    """Decode a frame to a float HxWxC numpy array.

    Values are returned in the file's native color space — no OCIO transform
    is applied here. That's the job of the render layer.

    Parameters
    ----------
    path:
        Filesystem path to the image file.
    channels:
        Optional list of channel names to keep. When ``None`` (default), we
        pick a minimal sensible subset for *display* — the named R/G/B/A
        channels when they exist, otherwise the first up to four channels.
        This is critical for multichannel EXRs (AOVs, depth, normals,
        cryptomattes): reading only the beauty can be 3-4x smaller in RAM
        and faster to decode than pulling every layer.
    as_half:
        When True (default) decodes to ``float16`` — half the RAM and half
        the PCIe upload bandwidth, which matters for realtime 4K playback.
        Precision is still fine for display (EXR's native format anyway).
        Set to False to force float32 output.

    Raises
    ------
    FrameReadError
        When the file is missing, unreadable, or doesn't contain the
        requested channels.
    """
    path = Path(path)
    if not path.exists():
        raise FrameReadError(f"File not found: {path}")

    # Network staging hook: if the path is on a network share and
    # we've already bulk-copied the file to a local SSD staging
    # cache, redirect the read to the local copy. Measured ~3×
    # speedup on a real Maya AOV EXR over SMB because OIIO and
    # PyOpenEXR both do many small reads that SMB can't pipeline
    # well; the bulk copy + local-fast reads is a net win even on
    # first access. See :mod:`img_player.cache.network_staging`.
    if _staging_lookup is not None:
        staged = _staging_lookup(path)
        if staged is not None:
            path = staged

    # EXR dispatch — choose between PyOpenEXR and OIIO based on
    # where the file lives.
    #
    # Measured on the AOV-heavy Maya CHARS sequence (1920×900,
    # 158 raw / 50 grouped channels, 232 MB per frame, zips):
    #
    #   from M:\ (SMB)  : PyOpenEXR 1.0 s   OIIO 1.9 s   → use PyOpenEXR
    #   from local SSD : PyOpenEXR 1.0 s   OIIO 0.14 s  → use OIIO
    #
    # Why the asymmetry: PyOpenEXR decodes ALL channels eagerly into
    # per-key ndarrays — fast on SMB (it makes one optimal bulk read
    # of the whole file) but slow on local (CPU-bound decompress of
    # data we don't need). OIIO does the opposite: many small reads
    # to skip AOVs and decode only the requested ones — terrible on
    # SMB (RTT-bound), great on local SSD.
    #
    # With the staging cache in play, network-source frames get
    # bulk-copied to local; from that point on we should switch to
    # OIIO. The path translation above (``_staging_lookup``) means
    # ``path`` is already the LOCAL staged copy at this point, so
    # the check below correctly routes staged frames through OIIO.
    if path.suffix.lower() == ".exr":
        from img_player.cache.network_staging import is_network_path  # noqa: PLC0415
        if is_network_path(path):
            # On-network read: PyOpenEXR is the fast path.
            try:
                arr = _read_exr_pyopenexr(path, channels, as_half)
                if arr is not None:
                    return arr
            except Exception:  # noqa: BLE001 — fall through to OIIO
                log.debug(
                    "PyOpenEXR fast-path failed for %s; falling back to OIIO",
                    path, exc_info=True,
                )

    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FrameReadError(f"Failed to open {path}: {oiio.geterror()}")

    oiio_type = oiio.HALF if as_half else oiio.FLOAT
    numpy_dtype = np.float16 if as_half else np.float32

    try:
        spec = inp.spec()
        available = list(spec.channelnames)
        selected = _resolve_channels(available, channels, path)
        indices = [available.index(c) for c in selected]

        # Fast path: contiguous channel range → ask OIIO to decode only that
        # slice, skipping the AOV layers entirely.
        if _is_contiguous(indices):
            chbegin = indices[0]
            chend = indices[-1] + 1
            pixels = inp.read_image(chbegin, chend, oiio_type)
        else:
            # Rare: non-contiguous selection. Read everything and subset.
            pixels = inp.read_image(oiio_type)
            pixels = pixels[..., indices]

        if pixels is None:
            raise FrameReadError(f"{path}: OIIO read returned None ({inp.geterror()})")

        arr = np.asarray(pixels, dtype=numpy_dtype)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        # Single-channel readout (e.g. just "Z" or just "A") needs to be
        # broadcast to RGB so the GL viewport's RGBA pipeline can show
        # it as monochrome. We use np.broadcast_to (zero-copy) followed
        # by ascontiguousarray only if the upload path needs contiguous
        # memory — typical OpenGL pipelines handle the broadcasted
        # (strided) view fine for read-only data.
        if arr.shape[2] == 1:
            arr = np.broadcast_to(arr, (arr.shape[0], arr.shape[1], 3))
            # Make it contiguous so glTexSubImage2D doesn't choke on
            # the zero-stride dimension.
            arr = np.ascontiguousarray(arr)
        return arr
    finally:
        inp.close()


def read_header(path: Path | str) -> oiio.ImageSpec:
    """Read only the file header (no pixel data). Cheap."""
    path = Path(path)
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise FrameReadError(f"Failed to open header for {path}: {oiio.geterror()}")
    try:
        return inp.spec()
    finally:
        inp.close()


# Attributes we care about for colorspace auto-detection. Listed in
# priority order: explicit colorspace tags first (most reliable), then
# chromaticities (gamut signature), finally OIIO's own normalisation.
_COLOR_METADATA_ATTRS: tuple[str, ...] = (
    # ICC profile bakers tend to set these.
    "ICCProfile:cprt",
    "ICCProfile:desc",
    # OIIO's own classification of the file's encoding.
    "oiio:ColorSpace",
    # Standard EXR header attribute — 8 floats describing R/G/B/W
    # primaries. Lets us match against ACES AP0 / AP1, Rec.709,
    # Rec.2020, P3 etc. by gamut signature.
    "chromaticities",
)


# Substrings that flag an attribute as "carries a colorspace name".
# Generic on purpose: covers Nuke (``nuke/input/colorspace``), Arnold
# (``exr/arnold/color_space``), V-Ray, Renderman, Houdini, custom
# pipelines, and anything else that follows the convention of putting
# ``colorspace`` / ``color_space`` / ``colourspace`` in the attr name.
# Trades whack-a-mole for a broader sweep. The auto-detector is the
# one that decides priority among the matches it gets.
_COLOR_KEY_PATTERNS: tuple[str, ...] = (
    "colorspace",   # most renderers (camelCase or lowercase)
    "color_space",  # snake_case variants (Arnold, etc.)
    "colourspace",  # British spelling occasionally shows up
)


def read_color_metadata(path: Path | str) -> dict[str, object]:
    """Return whatever colour-related attributes the file's header
    advertises. Cheap — opens the file but doesn't decode pixels.

    Two sweeps:

    1. **Hardcoded standard attrs** (ICC, OIIO classification,
       chromaticities) — fixed names that don't follow the
       ``...colorspace...`` convention.
    2. **Pattern sweep over every header attribute** — any name whose
       lowercased form contains ``colorspace`` / ``color_space`` /
       ``colourspace``. Catches renderer-specific tags (Nuke, Arnold,
       V-Ray, Houdini…) without us having to enumerate them
       individually.

    Values come straight from OIIO so they may be strings, tuples of
    floats, or other types depending on the attribute.
    """
    path = Path(path)
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        # We don't raise — auto-detection is best-effort, so a
        # failure to read metadata just means "no metadata, fall
        # back to extension".
        log.debug("read_color_metadata: cannot open %s (%s)", path, oiio.geterror())
        return {}
    try:
        spec = inp.spec()
        meta: dict[str, object] = {}
        for attr in _COLOR_METADATA_ATTRS:
            value = spec.getattribute(attr)
            if value is not None:
                meta[attr] = value
        # Sweep all extra attribs for anything carrying a colorspace
        # name. ``extra_attribs`` is OIIO's catch-all bag for
        # renderer-emitted metadata that doesn't have a standard slot.
        for param in getattr(spec, "extra_attribs", ()):
            try:
                name = param.name
                value = param.value
            except AttributeError:
                continue
            if not isinstance(name, str) or value is None:
                continue
            lower = name.lower()
            if any(pat in lower for pat in _COLOR_KEY_PATTERNS):
                # Don't clobber an entry the hardcoded pass already
                # placed (its name might be the canonical form OIIO
                # uses for ``getattribute``).
                meta.setdefault(name, value)
        return meta
    finally:
        inp.close()


# ---------------------------------------------------------------------------
# PyOpenEXR fast-path
# ---------------------------------------------------------------------------

# Lazy import — OpenEXR is an optional speed-up. If it's not installed
# (CI, lite builds, dev envs without conda's openimageio extras), the
# ``read_frame`` flow falls back to the OIIO path. We probe the import
# at module-load time and cache the verdict so the hot path doesn't pay
# an ``ImportError`` catch on every call.
try:  # noqa: SIM105 — explicit "import to a sentinel" idiom
    import OpenEXR as _OpenEXR  # type: ignore[import-untyped]
    _PYOPENEXR_AVAILABLE = True
except ImportError:
    _OpenEXR = None  # type: ignore[assignment]
    _PYOPENEXR_AVAILABLE = False


# Channel keys we treat as the "RGB beauty pass" when picking a default
# subset. PyOpenEXR groups multi-component channels into a single dict
# key (e.g. ``"RGBA"`` is one key with a (H, W, 4) ndarray). The order
# is preference-ordered: an explicit ``"RGB"`` group wins over ``"RGBA"``
# (skip the alpha plane for ~25 % less RAM + PCIe), then any 3-channel
# layer that starts with ``RGB`` (Maya / Arnold ``RGBA_backLight`` etc.).
_PYEXR_BEAUTY_KEYS: tuple[str, ...] = ("RGB", "RGBA")


def _read_exr_pyopenexr(
    path: Path, channels: Sequence[str] | None, as_half: bool,
) -> np.ndarray | None:
    """Decode an EXR via PyOpenEXR — the fast cold-network path.

    Returns the decoded ndarray (H, W, C) in the requested half/float
    dtype, OR ``None`` if PyOpenEXR isn't available so the caller
    falls back to OIIO. Raises on any other failure so the caller's
    try/except also routes to OIIO.

    Strategy:
    * The ``OpenEXR.File`` constructor opens the file and decodes
      every channel into per-key ndarrays — the C++ reader is
      ~30 % faster than OIIO's wrapper on this codepath.
    * We pluck the requested channels (or the default ``RGB`` /
      ``RGBA`` group) and stack into a single (H, W, C) array.
    * For an explicit channel list (caller passed
      ``channels=["R","G","B"]``) we still try to satisfy it
      directly off the grouped channel dict — most production EXRs
      have R/G/B available individually OR via the ``RGB`` /
      ``RGBA`` group key.

    Multipart EXRs: only the FIRST part is read. Multi-part is rare
    in our review pipeline; the OIIO fallback handles it.
    """
    if not _PYOPENEXR_AVAILABLE or _OpenEXR is None:
        return None

    numpy_dtype = np.float16 if as_half else np.float32

    with _OpenEXR.File(str(path)) as exr_file:
        parts = exr_file.parts
        if not parts:
            return None
        part = parts[0]
        ch_dict = part.channels  # {key: Channel} — pixels live on Channel.pixels

        # --- Pick which channels to return ---
        if channels is None:
            # Default: prefer "RGB" group (no alpha = faster, smaller),
            # fall back to "RGBA" then any 3-RGB layer name.
            picked = None
            for key in _PYEXR_BEAUTY_KEYS:
                if key in ch_dict:
                    picked = key
                    break
            if picked is None:
                # No grouped key — look for individual R/G/B
                if all(c in ch_dict for c in ("R", "G", "B")):
                    return _stack_individual(
                        ch_dict, ("R", "G", "B"), numpy_dtype,
                    )
                # Last resort: take the first key
                first_key = next(iter(ch_dict), None)
                if first_key is None:
                    return None
                return _stack_individual(
                    ch_dict, (first_key,), numpy_dtype,
                )
            return np.asarray(ch_dict[picked].pixels, dtype=numpy_dtype)

        # --- Explicit channel list ---
        requested = list(channels)
        # Try the exact concatenation first ("RGBA" or "RGB").
        joined = "".join(requested)
        if joined in ch_dict:
            return np.asarray(ch_dict[joined].pixels, dtype=numpy_dtype)
        # Slice-from-grouped fast path: if every requested name is a
        # contiguous prefix of a grouped key (e.g. ``["R","G","B"]``
        # requested but the file groups them as ``"RGBA"``), take a
        # plain numpy slice of the grouped array. Zero per-plane
        # allocation, just a view + dtype cast at the end.
        for group_key in ("RGBA", "RGB"):
            if group_key in ch_dict and len(requested) <= len(group_key):
                if all(
                    i < len(group_key) and group_key[i] == name
                    for i, name in enumerate(requested)
                ):
                    grouped = np.asarray(
                        ch_dict[group_key].pixels, dtype=numpy_dtype,
                    )
                    return np.ascontiguousarray(grouped[..., :len(requested)])
        # Fall back to per-channel access (each channel as its own
        # array, then stack). Slower but handles arbitrary picks.
        return _stack_individual(ch_dict, requested, numpy_dtype)


def _stack_individual(
    ch_dict, names: Sequence[str], dtype: type,
) -> np.ndarray:
    """Stack per-channel ndarrays into one (H, W, C) array.

    Raises ``KeyError`` if any requested name isn't in ``ch_dict`` —
    the caller's try/except then routes to OIIO. Channels in the
    grouped layout (e.g. ``"RGBA"``) are detected and unpacked so
    the caller can pass ``("R", "G", "B")`` even when PyOpenEXR
    grouped them.
    """
    planes: list[np.ndarray] = []
    for name in names:
        if name in ch_dict:
            arr = np.asarray(ch_dict[name].pixels, dtype=dtype)
            if arr.ndim == 3:
                # Already a multi-component channel (e.g. "RGBA" key) —
                # spread its planes.
                for i in range(arr.shape[2]):
                    planes.append(arr[..., i])
            else:
                planes.append(arr)
            continue
        # Try to find the name inside a grouped key. For "R" with an
        # "RGBA" group: index 0; "G": 1; "B": 2; "A": 3.
        for group_key in ("RGBA", "RGB"):
            if group_key in ch_dict and name in group_key:
                idx = group_key.index(name)
                grouped = np.asarray(
                    ch_dict[group_key].pixels, dtype=dtype,
                )
                planes.append(grouped[..., idx])
                break
        else:
            raise KeyError(f"channel {name!r} not in EXR")
    if not planes:
        raise KeyError("no channels resolved")
    return np.stack(planes, axis=-1)


# ---------------------------------------------------------------------------
# Channel resolver (shared by both readers)
# ---------------------------------------------------------------------------


def _resolve_channels(
    available: list[str], requested: Sequence[str] | None, path: Path
) -> list[str]:
    if requested is None:
        # Prefer R/G/B (3 channels) over R/G/B/A (4 channels) when both
        # available — alpha is almost always uniform 1.0 on opaque
        # renders and reading it forces the EXR ``zips`` decoder
        # through ~8× more I/O on AOV-heavy files. See the matching
        # group-builder rationale in ``sequence/channels.py`` for the
        # measured numbers. Callers that need alpha (compositing,
        # straight-alpha layers, sprite PNGs) pass an explicit
        # ``requested`` selection — the group dispatcher does that
        # whenever the user picks the ``RGBA`` group from the
        # channel menu.
        rgb = [c for c in ("R", "G", "B") if c in available]
        if len(rgb) == 3:
            return rgb
        preferred = [c for c in ("R", "G", "B", "A") if c in available]
        if preferred:
            return preferred
        if not available:
            raise FrameReadError(f"{path}: no channels in spec")
        return available[: min(4, len(available))]

    missing = [c for c in requested if c not in available]
    if missing:
        raise FrameReadError(
            f"{path}: requested channels not found: {missing}. Available: {available}"
        )
    return list(requested)


def _is_contiguous(indices: list[int]) -> bool:
    if not indices:
        return False
    return indices == list(range(indices[0], indices[-1] + 1))
