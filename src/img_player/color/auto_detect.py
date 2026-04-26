"""Auto-detect the source colorspace from image metadata.

Detection is best-effort: we'd rather correctly identify 80 % of
sequences automatically and let the user override the rest, than
guess wrong on edge cases. The detector returns a colorspace name
**that exists in the active OCIO config**, plus a short reason
string that the UI surfaces in the status bar so the user can
quickly verify (or correct) the choice.

Detection cascade — first match wins:

    1. **Explicit colorspace tag** — `colorSpaceName`,
       `nuke/input/colorspace`. These are written by OCIO-aware
       renderers / Nuke and are essentially ground truth.
    2. **OIIO's own classification** — `oiio:ColorSpace`. Reliable
       for sRGB/Linear from non-EXR formats; less precise for EXR.
    3. **Chromaticities (gamut signature)** — match the EXR
       `chromaticities` attribute against the canonical primaries
       of ACES AP0 / AP1, Rec.709, Rec.2020, DCI-P3, Display-P3.
    4. **Extension fallback** — `.exr → scene_linear`,
       `.png/.jpg/.tga → sRGB`, `.dpx/.cin → Cineon`.

If everything fails we return ``(None, "no signal in metadata")`` and
the caller decides what to do (typically: keep whatever the user
last selected, or surface a warning).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


# ----------------------------------------------------------------------- Detection result


@dataclass(frozen=True)
class DetectionResult:
    """Outcome of one round of detection.

    ``colorspace`` is ``None`` when nothing matched; ``reason``
    always carries a short human-readable explanation suitable for
    the status bar.
    """

    colorspace: str | None
    reason: str


# ----------------------------------------------------------------------- Canonical chromaticities

# (red.x, red.y, green.x, green.y, blue.x, blue.y, white.x, white.y)
# Values come from the official spec sheets (ACES TB-2014-004, ITU-R
# BT.709-6, BT.2020-2, SMPTE RP 431-2). The match tolerance is loose
# enough (±0.005) to absorb rounding differences in different export
# pipelines, tight enough to never confuse two adjacent gamuts.
_GAMUTS: tuple[tuple[str, tuple[float, ...]], ...] = (
    # ACES AP1 — what most studios call "ACEScg".
    ("ACEScg", (0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767)),
    # ACES AP0 — the wider primary set; "ACES2065-1" / "ACES" in OCIO.
    ("ACES2065-1", (0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.077, 0.32168, 0.33767)),
    # Rec.709 / sRGB — same primaries, only the EOTF differs.
    ("Rec.709", (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.329)),
    # Rec.2020 — UHD.
    ("Rec.2020", (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.329)),
    # DCI-P3 — theatrical projection (D63 white).
    ("DCI-P3", (0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.314, 0.351)),
    # Display P3 — Apple-style (D65 white).
    ("Display P3", (0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.329)),
)
_CHROMATICITY_TOLERANCE = 0.005


def _chromaticities_match(
    measured: tuple[float, ...], expected: tuple[float, ...]
) -> bool:
    if len(measured) != len(expected):
        return False
    return all(abs(m - e) <= _CHROMATICITY_TOLERANCE for m, e in zip(measured, expected))


def _identify_gamut(chromaticities: tuple[float, ...]) -> str | None:
    """Return the canonical gamut name (e.g. ``"ACEScg"``) for a set
    of chromaticity primaries, or ``None`` if no gamut matches."""
    for name, expected in _GAMUTS:
        if _chromaticities_match(chromaticities, expected):
            return name
    return None


# ----------------------------------------------------------------------- Name normalisation

def _normalize(name: str) -> str:
    """Collapse a colorspace name to a fuzzy-match key.

    Studios shuffle ACEScg / ACES - ACEScg / aces_cg / etc. — we
    strip spaces, dashes, dots and underscores, lowercase, then look
    for substring containment. Good enough for the dozen common
    names we care about.
    """
    return (
        name.lower()
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace(".", "")
    )


def _find_colorspace(canonical: str, available: Iterable[str]) -> str | None:
    """Pick the OCIO-config name that best matches ``canonical``.

    We do substring matching after normalisation: e.g. canonical
    ``"ACEScg"`` (norm: ``acescg``) will match any of:
      - ``"ACES - ACEScg"`` (norm: ``acesacescg``) — contains acescg
      - ``"ACEScg"``
      - ``"acescg_linear"``
    """
    target = _normalize(canonical)
    for cs in available:
        if target in _normalize(cs):
            return cs
    return None


# ----------------------------------------------------------------------- Aliases

# Maps "what the metadata says" → "canonical name we look up in the
# OCIO config". Kept tiny on purpose — only the cases we know occur
# in the wild. Anything else falls through and we try a direct
# substring match against the config.
_OIIO_COLORSPACE_ALIASES: dict[str, str] = {
    "linear": "scene_linear",          # OIIO's generic "linear" tag
    "lin_rec709": "Rec.709",
    "lin_srgb": "Linear sRGB",
    "g22_rec709": "Rec.709",
    "g18_rec709": "Rec.709",
    "rec709": "Rec.709",
    "rec.709": "Rec.709",
    "srgb": "sRGB",
    "acescg": "ACEScg",
    "aces": "ACES2065-1",
    "aces2065-1": "ACES2065-1",
}


# ----------------------------------------------------------------------- Public API


def detect_source_colorspace(
    metadata: dict[str, object],
    extension: str,
    available_colorspaces: Iterable[str],
    *,
    scene_linear_role: str | None = None,
) -> DetectionResult:
    """Pick a source colorspace based on file metadata + extension.

    ``metadata`` is whatever ``io.reader.read_color_metadata`` returned
    (may be empty). ``extension`` is the file extension including the
    dot, lowercased. ``available_colorspaces`` is the list of names
    the active OCIO config exposes; we never return a name that isn't
    in this list.

    ``scene_linear_role``, when provided, is the colorspace the active
    OCIO config has assigned to the ``scene_linear`` *role*. We use it
    as the EXR fallback in preference to substring-matching against
    ``"linear"`` — which is too greedy: it would match
    ``Linear AdobeRGB`` etc. on the way to ``Linear Rec.709``.

    Returns a :class:`DetectionResult`. ``colorspace`` is ``None`` if
    no signal could be turned into an existing OCIO name.
    """
    available = list(available_colorspaces)

    # 1. Explicit tag (highest priority — the file is telling us).
    for key in ("colorSpaceName", "nuke/input/colorspace", "nuke/input/colourspace"):
        raw = metadata.get(key)
        if isinstance(raw, str) and raw:
            cs = _find_colorspace(raw, available)
            if cs is not None:
                return DetectionResult(cs, f"file metadata: {key}={raw!r}")
            # The file claims a colorspace that isn't in our config —
            # surface that, the user might want to fix their config.
            return DetectionResult(None, f"file metadata claims {raw!r} but it is not in the config")

    # 2. OIIO's normalised tag.
    oiio_cs = metadata.get("oiio:ColorSpace")
    if isinstance(oiio_cs, str) and oiio_cs:
        normalized_oiio = oiio_cs.lower().strip()
        canonical = _OIIO_COLORSPACE_ALIASES.get(normalized_oiio, oiio_cs)
        cs = _find_colorspace(canonical, available)
        if cs is not None:
            return DetectionResult(cs, f"oiio:ColorSpace={oiio_cs!r}")

    # 3. Chromaticities (gamut signature).
    chroma = metadata.get("chromaticities")
    if isinstance(chroma, (tuple, list)) and len(chroma) == 8:
        try:
            tup = tuple(float(v) for v in chroma)
        except (TypeError, ValueError):
            tup = ()
        if tup:
            gamut = _identify_gamut(tup)
            if gamut is not None:
                cs = _find_colorspace(gamut, available)
                if cs is not None:
                    return DetectionResult(cs, f"chromaticities match {gamut}")

    # 4. Extension fallback.
    ext = extension.lower()
    if ext == ".exr":
        # Prefer the OCIO scene_linear role if we have it — that's
        # what the active config considers "linear scene-referred".
        # Then try the literal "scene_linear" name (some configs
        # expose it directly), then specific Rec.709 / sRGB linear
        # variants. "Linear" alone would be too greedy
        # (matches AdobeRGB, ProPhoto, etc.).
        if scene_linear_role and scene_linear_role in available:
            return DetectionResult(
                scene_linear_role,
                f"EXR default — scene_linear role ({scene_linear_role})",
            )
        for candidate in (
            "scene_linear",
            "Linear Rec.709",
            "Linear Rec 709",
            "Linear BT.709",
            "Linear sRGB",
        ):
            cs = _find_colorspace(candidate, available)
            if cs is not None:
                return DetectionResult(cs, f"EXR default — assumed {candidate}")
    elif ext in (".png", ".jpg", ".jpeg", ".tga", ".bmp"):
        for candidate in ("sRGB Encoded Rec.709", "sRGB", "Gamma 2.2 Encoded Rec.709"):
            cs = _find_colorspace(candidate, available)
            if cs is not None:
                return DetectionResult(cs, f"extension {ext} — assumed {candidate}")
    elif ext in (".dpx", ".cin"):
        for candidate in ("Cineon", "Log Film"):
            cs = _find_colorspace(candidate, available)
            if cs is not None:
                return DetectionResult(cs, f"extension {ext} — assumed {candidate}")

    # 5. Total whiff.
    return DetectionResult(None, "no signal in metadata")


# ----------------------------------------------------------------------- Display detection

# Map "what Qt's QColorSpace tells us about the screen" → "canonical
# name we look up in the OCIO config's list of displays". Kept tiny:
# a screen is one of a handful of well-known colorspaces, and the
# rare custom ICC profiles fall through to the sRGB fallback (which
# is what Qt itself uses when it can't introspect a screen).
_QT_NAMED_COLORSPACE_TO_DISPLAY: dict[str, str] = {
    "srgb": "sRGB",
    "srgblinear": "sRGB",          # sRGB primaries, linear EOTF — matches sRGB display
    "displayp3": "P3-D65",
    "adobergb": "AdobeRGB",
    "prophotorgb": "ProPhoto",
    "bt2020": "Rec.2020",
    "bt2100pq": "Rec.2020",        # PQ HDR uses Rec.2020 primaries
    "bt2100hlg": "Rec.2020",       # HLG HDR same primaries
}


def detect_display(
    qt_named_hint: str | None,
    available_displays: Iterable[str],
) -> DetectionResult:
    """Pick an OCIO display name based on what Qt told us about the
    monitor.

    ``qt_named_hint`` is the screen's named colorspace as a plain
    string, lowercased (``"srgb"``, ``"displayp3"``, ``"adobergb"``…).
    ``None`` covers the case where the screen has a custom ICC
    profile that Qt couldn't classify — we fall back to sRGB (the
    safe default for desktop monitors).

    Returns a :class:`DetectionResult`. Like ``detect_source_colorspace``,
    we only return display names that actually exist in the config.
    """
    available = list(available_displays)
    if not available:
        return DetectionResult(None, "OCIO config exposes no displays")

    if qt_named_hint:
        canonical = _QT_NAMED_COLORSPACE_TO_DISPLAY.get(qt_named_hint.lower())
        if canonical is not None:
            cs = _find_display(canonical, available)
            if cs is not None:
                return DetectionResult(
                    cs, f"screen colorspace: {qt_named_hint} → {canonical}"
                )

    # Fallback: most desktop monitors are sRGB-equivalent, and the
    # OCIO config almost always exposes an sRGB display. Use it as
    # a deliberate default — better than a random first display.
    cs = _find_display("sRGB", available)
    if cs is not None:
        reason = "fallback: sRGB (no screen hint)" if not qt_named_hint else (
            f"fallback: sRGB (Qt reported {qt_named_hint!r}, no OCIO match)"
        )
        return DetectionResult(cs, reason)

    # Truly weird config without sRGB — return the first display so
    # the app at least has *something* to render with.
    return DetectionResult(
        available[0], f"fallback: first available display ({available[0]})"
    )


def _find_display(canonical: str, available: Iterable[str]) -> str | None:
    """Same fuzzy matching as :func:`_find_colorspace`, just renamed
    for clarity at call sites — displays and colorspaces are
    different OCIO concepts even though the matching trick is the
    same string-normalisation."""
    return _find_colorspace(canonical, available)


# ----------------------------------------------------------------------- View detection

# Classify a source colorspace name into one of three buckets that
# matter for picking the right view:
#
#   * scene  — linear scene-referred (ACES, ACEScg, scene_linear,
#              Linear Rec.709, etc.). Wants tone mapping.
#   * display — already display-referred (sRGB, Rec.709 with EOTF
#              applied, Output - sRGB). Wants Raw / Un-tone-mapped
#              to avoid doubling up the tone map.
#   * log    — log-encoded (Cineon, ARRI Log_C, RED Log…). Wants a
#              "Log to display" / Un-tone-mapped that delogs the
#              source first.

SourceCategory = str  # "scene" / "display" / "log" / "unknown"


def classify_source_colorspace(source: str) -> SourceCategory:
    """Categorise a colorspace name into scene / display / log /
    unknown for view-selection purposes."""
    norm = source.lower()
    # Log-encoded: Cineon, ARRI Log_C, RED Log3G10, slog3, etc.
    if "log" in norm or "cineon" in norm:
        return "log"
    # Scene-referred:
    #   - anything containing "linear" (e.g. "Linear sRGB",
    #     "Linear Rec.709", "scene_linear")
    #   - anything ACES that *isn't* "Output - ACES …" (which is
    #     display-referred ACES output)
    if "linear" in norm or "scene_linear" in norm:
        return "scene"
    if "aces" in norm and "output" not in norm and "display" not in norm:
        return "scene"
    # Display-referred: sRGB / Rec.709 / Output / Display naming.
    if (
        "srgb" in norm
        or "rec.709" in norm
        or "rec709" in norm
        or "output" in norm
        or "display" in norm
        or "gamma" in norm
    ):
        return "display"
    return "unknown"


# View preferences per source category, in priority order. We try each
# in turn against the available views; the first one that fuzzy-matches
# wins. Studios use different naming conventions ("ACES 1.0 - SDR
# Video" vs "ACES SDR" vs "Default"), so we list a handful per slot.
_VIEW_PREFERENCES: dict[SourceCategory, tuple[str, ...]] = {
    "scene": (
        "ACES 1.0 SDR-video",  # ACES 1.0 OCIO config
        "ACES 1.0 - SDR Video",
        "ACES SDR",
        "ACES 2.0 SDR-100nit",  # ACES 2.0 / ACES Studio config
        "Filmic",                # Blender-style
        "Standard",              # nuke-default-config name
    ),
    "display": (
        "Raw",
        "Un-tone-mapped",
        "None",
        "sRGB",
    ),
    "log": (
        "Un-tone-mapped",
        "Raw",
    ),
    "unknown": (),
}


def detect_view(
    source_colorspace: str | None,
    available_views: Iterable[str],
    *,
    default_view: str | None = None,
) -> DetectionResult:
    """Pick the OCIO view that suits a given source.

    The point: applying ACES tone mapping ("ACES 1.0 SDR-video") on a
    scene-linear source is correct — applying it on already-display-
    referred sRGB doubles up the curve and bouillonifies the image.
    So we look at the source's nature and pick a view in the right
    family.

    Returns a :class:`DetectionResult`. Falls back to
    ``default_view`` (typically the OCIO config's default for the
    chosen display) when nothing better can be picked.
    """
    available = list(available_views)
    if not available:
        return DetectionResult(None, "no views available")

    if source_colorspace is None:
        if default_view and default_view in available:
            return DetectionResult(default_view, "no source — using config default")
        return DetectionResult(available[0], "no source — using first available")

    category = classify_source_colorspace(source_colorspace)
    candidates = _VIEW_PREFERENCES.get(category, ())
    for canonical in candidates:
        cs = _find_colorspace(canonical, available)
        if cs is not None:
            return DetectionResult(
                cs, f"source is {category}-referred → {canonical}"
            )

    if default_view and default_view in available:
        return DetectionResult(
            default_view,
            f"source is {category}-referred — falling back to config default ({default_view})",
        )
    return DetectionResult(
        available[0],
        f"source is {category}-referred — no preference matched, picked first ({available[0]})",
    )
