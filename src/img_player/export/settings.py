"""Immutable :class:`ExportSettings` dataclass + format catalog.

All export choices live here as a single, validated value object.
The dialog produces one; the engine consumes one. Serialised to
:class:`Preferences` so the next session opens with the user's last
choices pre-filled.

Design principle: this module is **pure** — no Qt, no I/O. Tests
import it without spinning up a QApplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path


class ExportSettingsError(ValueError):
    """Raised by :meth:`ExportSettings.validate` on bad config."""


class ExportFormatKind(Enum):
    """Coarse routing category — which writer subclass handles this format."""

    IMAGE_SEQUENCE = "image_sequence"
    VIDEO = "video"


class MissingFramePolicy(Enum):
    """How the engine handles a hole in the source sequence.

    * ``ABORT`` — raise on the first missing frame so the user sees
      the sequence is incomplete (default — matches the legacy
      behaviour and surfaces silent data loss).
    * ``BLACK`` — write a solid black frame at the export resolution.
      Keeps timing intact for video / contact-sheet review when the
      user knows there are gaps and just wants the playable file.
    * ``PLACEHOLDER`` — write the "MISSING FRAME" placeholder visual
      (greyscale damier + crosshairs + label). Preserves timing AND
      makes the gap unmistakable on screen — best for review copies.
    """

    ABORT = "abort"
    BLACK = "black"
    PLACEHOLDER = "placeholder"


# Allowed string keys for round-trip through QSettings.
_MISSING_FRAME_POLICY_VALUES = {p.value for p in MissingFramePolicy}


@dataclass(frozen=True)
class ExportFormat:
    """One row in the format catalog.

    Holds everything the dialog needs to populate its dropdowns
    (label, file extension) and what the writer needs to encode
    (codec name, container, ffmpeg pix_fmt, default colorspace bake
    suggestion). Defining them as a tuple of ``ExportFormat`` keeps
    the catalog easy to extend without touching the writer code —
    add a row, add a branch in the writer's ``_configure_codec``.
    """

    key: str                     # internal id, e.g. "h264_mp4"
    label: str                   # UI label, e.g. "H.264 (MP4)"
    extension: str               # ".mp4"
    kind: ExportFormatKind
    # --- video-only fields (None for image seq) ---
    codec: str | None = None     # ffmpeg codec name, e.g. "libx264"
    pix_fmt: str | None = None   # default ffmpeg pix_fmt
    # --- image-only fields ---
    bit_depth: int | None = None  # 8/16/32 — fed to OIIO
    supports_alpha: bool = True
    # Whether the format default to display-baked color (sRGB, what
    # you see) or linear passthrough (the raw working data). The
    # dialog reads this to set the "Apply display transform"
    # checkbox initial state per format.
    display_bake_default: bool = True
    # Soft user-facing description (tooltip / help line).
    description: str = ""


# ---------------------------------------------------------------- Image catalog

# 4 formats per the user's MVP scope (Q2: D = PNG + JPG + EXR + TIFF).
# EXR + TIFF default to linear passthrough (VFX pipeline expectation);
# PNG + JPG default to display-baked sRGB (review / share).
AVAILABLE_IMAGE_FORMATS: tuple[ExportFormat, ...] = (
    ExportFormat(
        key="png",
        label="PNG (8-bit, lossless)",
        extension=".png",
        kind=ExportFormatKind.IMAGE_SEQUENCE,
        bit_depth=8,
        supports_alpha=True,
        display_bake_default=True,
        description="Universal, lossless, sRGB. Good for share & review.",
    ),
    ExportFormat(
        key="jpg",
        label="JPEG (8-bit, lossy)",
        extension=".jpg",
        kind=ExportFormatKind.IMAGE_SEQUENCE,
        bit_depth=8,
        supports_alpha=False,
        display_bake_default=True,
        description="Smallest files, lossy, no alpha. Quick previews.",
    ),
    ExportFormat(
        key="exr",
        label="OpenEXR (half float, VFX)",
        extension=".exr",
        kind=ExportFormatKind.IMAGE_SEQUENCE,
        bit_depth=16,
        supports_alpha=True,
        display_bake_default=False,  # linear passthrough by default
        description="Half-float linear, ZIP compression. The VFX standard.",
    ),
    ExportFormat(
        key="tiff",
        label="TIFF (16-bit, archival)",
        extension=".tif",
        kind=ExportFormatKind.IMAGE_SEQUENCE,
        bit_depth=16,
        supports_alpha=True,
        display_bake_default=False,
        description="16-bit per channel, LZW compression. Archive grade.",
    ),
)

# ---------------------------------------------------------------- Video catalog

# Q2 + Q3: H.264 / H.265 / ProRes / DNxHR / VP9 / FFV1 (lossless) / v210
# (10-bit broadcast "uncompressed"). rawvideo dropped per Q3=B.
AVAILABLE_VIDEO_FORMATS: tuple[ExportFormat, ...] = (
    ExportFormat(
        key="h264_mp4",
        label="H.264 (MP4) — universal share",
        extension=".mp4",
        kind=ExportFormatKind.VIDEO,
        codec="libx264",
        pix_fmt="yuv420p",
        display_bake_default=True,
        description="Plays everywhere — Discord, mail, web players.",
    ),
    ExportFormat(
        key="h265_mp4",
        label="H.265 / HEVC (MP4) — modern, smaller",
        extension=".mp4",
        kind=ExportFormatKind.VIDEO,
        codec="libx265",
        pix_fmt="yuv420p",
        display_bake_default=True,
        description="Same quality at ~half the bitrate. Newer players only.",
    ),
    ExportFormat(
        key="prores_422_mov",
        label="ProRes 422 HQ (MOV) — review VFX",
        extension=".mov",
        kind=ExportFormatKind.VIDEO,
        codec="prores_ks",
        pix_fmt="yuv422p10le",
        display_bake_default=True,
        description="Industry-standard editorial codec. 10-bit 4:2:2.",
    ),
    ExportFormat(
        key="prores_4444_mov",
        label="ProRes 4444 (MOV) — alpha-aware",
        extension=".mov",
        kind=ExportFormatKind.VIDEO,
        codec="prores_ks",
        pix_fmt="yuva444p10le",
        display_bake_default=True,
        description="ProRes with alpha. 10-bit 4:4:4:4.",
    ),
    ExportFormat(
        key="dnxhr_mov",
        label="DNxHR (MOV) — Avid / broadcast",
        extension=".mov",
        kind=ExportFormatKind.VIDEO,
        codec="dnxhd",
        pix_fmt="yuv422p10le",
        display_bake_default=True,
        description="Avid-friendly. 10-bit 4:2:2.",
    ),
    ExportFormat(
        key="vp9_webm",
        label="VP9 (WebM) — modern web",
        extension=".webm",
        kind=ExportFormatKind.VIDEO,
        codec="libvpx-vp9",
        pix_fmt="yuv420p",
        display_bake_default=True,
        description="Open codec, plays in modern browsers.",
    ),
    ExportFormat(
        key="ffv1_mkv",
        label="FFV1 (MKV) — lossless archive",
        extension=".mkv",
        kind=ExportFormatKind.VIDEO,
        codec="ffv1",
        pix_fmt="yuv422p10le",
        display_bake_default=True,
        description="Bit-exact lossless. ~3× smaller than uncompressed.",
    ),
    ExportFormat(
        key="v210_mov",
        label="v210 10-bit (MOV) — broadcast uncompressed",
        extension=".mov",
        kind=ExportFormatKind.VIDEO,
        codec="v210",
        pix_fmt="yuv422p10le",
        display_bake_default=True,
        description="Standard broadcast 10-bit YUV422 uncompressed.",
    ),
)

# Concatenation for "find by key" lookups.
ALL_FORMATS: tuple[ExportFormat, ...] = AVAILABLE_IMAGE_FORMATS + AVAILABLE_VIDEO_FORMATS


def format_by_key(key: str) -> ExportFormat:
    """Look up an :class:`ExportFormat` by its ``key``. Raises ``KeyError``
    on unknown keys — callers shouldn't be guessing format ids."""
    for fmt in ALL_FORMATS:
        if fmt.key == key:
            return fmt
    raise KeyError(f"Unknown export format key: {key!r}")


# ---------------------------------------------------------------- Resolution presets

# (label, width, height). ``None``/``None`` = "Source — keep input
# resolution untouched" — the renderer skips the resize step.
RESOLUTION_PRESETS: tuple[tuple[str, int | None, int | None], ...] = (
    ("Source", None, None),
    ("4K UHD (3840×2160)", 3840, 2160),
    ("1080p (1920×1080)", 1920, 1080),
    ("720p (1280×720)", 1280, 720),
    ("Custom…", 0, 0),  # sentinel — width/height come from user fields
)

# (label, fps). ``None`` = source FPS.
FPS_PRESETS: tuple[tuple[str, float | None], ...] = (
    ("Source", None),
    ("23.976", 23.976),
    ("24", 24.0),
    ("25", 25.0),
    ("29.97", 29.97),
    ("30", 30.0),
    ("60", 60.0),
    ("Custom…", 0.0),
)

# EXR compression options (advanced section).
EXR_COMPRESSIONS: tuple[str, ...] = ("none", "rle", "zip", "zips", "piz", "dwaa", "dwab")

# ProRes profile mapping (FFmpeg's prores_ks profile param).
# 0 = Proxy, 1 = LT, 2 = 422 SQ, 3 = 422 HQ, 4 = 4444, 5 = 4444 XQ.
PRORES_PROFILES: tuple[tuple[str, int], ...] = (
    ("Proxy", 0),
    ("LT", 1),
    ("422", 2),
    ("422 HQ", 3),
    ("4444", 4),
    ("4444 XQ", 5),
)


# ============================================================================
# The settings dataclass
# ============================================================================


@dataclass(frozen=True)
class ExportSettings:
    """Every choice the user can make in the export dialog.

    Frozen so the engine can stash a reference safely. The dialog
    rebuilds a new instance on every change.
    """

    # ---- Output ----------------------------------------------------
    output_dir: Path
    # First frame number to use in the output filename for image
    # sequences. Lets you renumber a 1001-1500 source down to
    # 0001-0500 (or up). Ignored for video.
    start_frame: int = 1

    # ---- Format ----------------------------------------------------
    format_key: str = "png"  # one of ``ALL_FORMATS[i].key``

    # ---- Range -----------------------------------------------------
    in_frame: int = 0
    out_frame: int = 0  # inclusive

    # ---- Resolution ------------------------------------------------
    # ``None``/``None`` → keep source. Otherwise both are positive
    # ints and the renderer resizes (Lanczos) to that size.
    width: int | None = None
    height: int | None = None

    # ---- Frame rate (video only) ----------------------------------
    fps: float | None = None  # ``None`` → source fps

    # ---- Color -----------------------------------------------------
    # When True, the OCIO display transform is baked into pixels.
    # When False, the linear working buffer is written as-is (only
    # makes sense for EXR/TIFF; for PNG/JPG/video the visible result
    # would be near-black).
    apply_display_transform: bool = True

    # ---- Annotations ----------------------------------------------
    bake_annotations: bool = True
    # When ``bake_annotations`` is True, this controls whether we
    # also drop a copy of the sidecar JSON next to the export so the
    # recipient can reload them in img_player. When False, the export
    # is "view-only".
    copy_sidecar: bool = False

    # ---- Missing-frame handling -----------------------------------
    # Default = ABORT to match the legacy behaviour: an incomplete
    # sequence shouldn't silently produce a corrupt-looking export
    # unless the user explicitly opts in.
    missing_frame_policy: "MissingFramePolicy" = field(
        default_factory=lambda: MissingFramePolicy.ABORT,
    )

    # ---- Format-specific (Advanced) -------------------------------
    jpg_quality: int = 95           # 1..100
    exr_compression: str = "zip"    # one of EXR_COMPRESSIONS
    video_crf: int = 18             # H.264/H.265/VP9 quality (lower = better)
    prores_profile: int = 3         # default: 422 HQ (matches the "human" label)
    # H.264 / H.265 encoder preset — speed/efficiency tradeoff.
    h26x_preset: str = "medium"

    # ---- Computed properties ---------------------------------------

    @property
    def fmt(self) -> ExportFormat:
        """The :class:`ExportFormat` row for ``format_key``."""
        return format_by_key(self.format_key)

    @property
    def is_video(self) -> bool:
        return self.fmt.kind == ExportFormatKind.VIDEO

    @property
    def is_image_sequence(self) -> bool:
        return self.fmt.kind == ExportFormatKind.IMAGE_SEQUENCE

    @property
    def total_frames(self) -> int:
        """Inclusive frame count of the export range."""
        return max(0, self.out_frame - self.in_frame + 1)

    # ---- Validation ------------------------------------------------

    def validate(self) -> None:
        """Raise :class:`ExportSettingsError` on bad config.

        Catches the obvious user errors before the engine starts:
        empty dir, range inverted, jpg quality out of bounds, etc.
        Format-format mismatches (e.g. ProRes profile passed to mp4)
        are simply ignored — the writer reads only the field it
        cares about, so leftover values from previous sessions don't
        block an export.
        """
        if not self.output_dir or str(self.output_dir).strip() == "":
            raise ExportSettingsError("Output folder is required.")
        if self.in_frame > self.out_frame:
            raise ExportSettingsError(
                f"Range is inverted: in_frame={self.in_frame} "
                f"> out_frame={self.out_frame}"
            )
        if self.is_image_sequence and self.start_frame < 0:
            raise ExportSettingsError(
                f"Start frame must be ≥ 0, got {self.start_frame}"
            )
        if self.width is not None and self.width <= 0:
            raise ExportSettingsError(f"Width must be > 0, got {self.width}")
        if self.height is not None and self.height <= 0:
            raise ExportSettingsError(f"Height must be > 0, got {self.height}")
        if (self.width is None) != (self.height is None):
            raise ExportSettingsError(
                "Width and height must both be set or both be None (Source)."
            )
        if self.fps is not None and self.fps <= 0:
            raise ExportSettingsError(f"FPS must be > 0, got {self.fps}")
        if not (1 <= self.jpg_quality <= 100):
            raise ExportSettingsError(
                f"JPG quality must be in [1, 100], got {self.jpg_quality}"
            )
        if self.exr_compression not in EXR_COMPRESSIONS:
            raise ExportSettingsError(
                f"EXR compression {self.exr_compression!r} not in {EXR_COMPRESSIONS}"
            )
        if not (0 <= self.video_crf <= 51):
            raise ExportSettingsError(
                f"Video CRF must be in [0, 51], got {self.video_crf}"
            )
        valid_profiles = {p[1] for p in PRORES_PROFILES}
        if self.prores_profile not in valid_profiles:
            raise ExportSettingsError(
                f"ProRes profile must be one of {sorted(valid_profiles)}, "
                f"got {self.prores_profile}"
            )

    # ---- Serialization for prefs ----------------------------------

    def to_prefs_dict(self) -> dict[str, object]:
        """Flatten to a JSON-friendly dict for QSettings."""
        return {
            "output_dir": str(self.output_dir),
            "start_frame": self.start_frame,
            "format_key": self.format_key,
            "width": self.width if self.width is not None else 0,
            "height": self.height if self.height is not None else 0,
            "fps": self.fps if self.fps is not None else 0.0,
            "apply_display_transform": self.apply_display_transform,
            "bake_annotations": self.bake_annotations,
            "copy_sidecar": self.copy_sidecar,
            "jpg_quality": self.jpg_quality,
            "exr_compression": self.exr_compression,
            "video_crf": self.video_crf,
            "prores_profile": self.prores_profile,
            "h26x_preset": self.h26x_preset,
            "missing_frame_policy": self.missing_frame_policy.value,
        }

    @classmethod
    def from_prefs_dict(
        cls,
        data: dict[str, object],
        *,
        in_frame: int,
        out_frame: int,
    ) -> ExportSettings:
        """Inverse of :meth:`to_prefs_dict`. ``in_frame`` / ``out_frame``
        come from the controller (current playback range) — we don't
        persist them, since they are session-specific."""
        def _opt_int(v: object) -> int | None:
            try:
                iv = int(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return iv if iv > 0 else None

        def _opt_float(v: object) -> float | None:
            try:
                fv = float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
            return fv if fv > 0 else None

        def _safe_int(v: object, default: int) -> int:
            """Coerce to int, falling back to ``default`` on garbage —
            QSettings round-trips can produce strings like 'banana'
            if a hand-edited INI feeds back through."""
            try:
                return int(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return default

        def _safe_str(v: object, default: str, allowed: tuple[str, ...] | None = None) -> str:
            try:
                s = str(v) if v is not None else default
            except Exception:  # pragma: no cover — defensive
                return default
            if allowed is not None and s not in allowed:
                return default
            return s

        return cls(
            output_dir=Path(str(data.get("output_dir", ""))),
            start_frame=_safe_int(data.get("start_frame", 1), 1),
            format_key=_safe_str(data.get("format_key", "png"), "png"),
            in_frame=in_frame,
            out_frame=out_frame,
            width=_opt_int(data.get("width", 0)),
            height=_opt_int(data.get("height", 0)),
            fps=_opt_float(data.get("fps", 0.0)),
            apply_display_transform=bool(data.get("apply_display_transform", True)),
            bake_annotations=bool(data.get("bake_annotations", True)),
            copy_sidecar=bool(data.get("copy_sidecar", False)),
            jpg_quality=_safe_int(data.get("jpg_quality", 95), 95),
            exr_compression=_safe_str(
                data.get("exr_compression", "zip"), "zip", EXR_COMPRESSIONS,
            ),
            video_crf=_safe_int(data.get("video_crf", 18), 18),
            prores_profile=_safe_int(data.get("prores_profile", 3), 3),
            h26x_preset=_safe_str(data.get("h26x_preset", "medium"), "medium"),
            missing_frame_policy=MissingFramePolicy(
                _safe_str(
                    data.get("missing_frame_policy", MissingFramePolicy.ABORT.value),
                    MissingFramePolicy.ABORT.value,
                    tuple(_MISSING_FRAME_POLICY_VALUES),
                )
            ),
        )

    def with_changes(self, **kwargs: object) -> ExportSettings:
        """Convenience around :func:`dataclasses.replace`."""
        return replace(self, **kwargs)


# ============================================================================
# Estimated size helpers (used by the dialog's "~340 MB" label)
# ============================================================================


# Empirical bytes-per-pixel estimates for image formats at typical
# content. Wildly inaccurate at the byte level — we want order of
# magnitude only.
_IMAGE_BPP_ESTIMATE = {
    "png": 2.0,    # 8-bit RGBA, ~50% compressible
    "jpg": 0.6,    # quality 95, ~80% compressible
    "exr": 4.0,    # 16-bit half RGBA, ZIP ~50%
    "tiff": 4.0,   # 16-bit RGBA, LZW ~50%
}

# Empirical bitrates (bits/s/megapixel/fps) for video codecs.
# i.e. expected_bitrate = bpp_per_mpx * width*height/1e6 * fps.
_VIDEO_BPP = {
    "h264_mp4": 0.10,
    "h265_mp4": 0.05,
    "prores_422_mov": 1.10,
    "prores_4444_mov": 1.80,
    "dnxhr_mov": 0.90,
    "vp9_webm": 0.08,
    "ffv1_mkv": 1.50,
    "v210_mov": 2.40,
}


def estimate_size_bytes(
    settings: ExportSettings,
    source_w: int,
    source_h: int,
    source_fps: float,
) -> int:
    """Best-effort size estimate in bytes.

    Used only for the dialog's "~340 MB" label — not load-bearing.
    Falls back to ``0`` on any computation error rather than raising
    (a missing estimate shouldn't break the dialog).
    """
    try:
        w = settings.width if settings.width is not None else source_w
        h = settings.height if settings.height is not None else source_h
        fps = settings.fps if settings.fps is not None else source_fps
        if w <= 0 or h <= 0 or fps <= 0:
            return 0
        n_frames = settings.total_frames
        if n_frames <= 0:
            return 0
        if settings.is_image_sequence:
            bpp = _IMAGE_BPP_ESTIMATE.get(settings.format_key, 2.0)
            return int(n_frames * w * h * bpp)
        # Video
        bpp = _VIDEO_BPP.get(settings.format_key, 0.10)
        bitrate_bits_per_s = bpp * (w * h / 1e6) * fps * 1e6
        duration_s = n_frames / fps
        return int(bitrate_bits_per_s * duration_s / 8.0)
    except (ZeroDivisionError, ValueError, AttributeError):
        return 0


def format_bytes(n: int) -> str:
    """Human-readable byte count: 340 MB / 1.2 GB / 14 KB."""
    if n <= 0:
        return "—"
    units = ("B", "KB", "MB", "GB", "TB")
    f = float(n)
    for unit in units:
        if f < 1024 or unit == units[-1]:
            return f"{f:.0f} {unit}" if f >= 10 or unit == "B" else f"{f:.1f} {unit}"
        f /= 1024
    return f"{f:.0f} {units[-1]}"
