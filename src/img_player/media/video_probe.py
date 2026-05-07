"""Probe a video file's container/streams via PyAV.

Intentionally I/O-cheap: opens the container, reads stream metadata,
closes. Produces a :class:`VideoMetadata` snapshot the rest of the
player can use without keeping the file handle open.

This module is deliberately tolerant of unusual containers — PyAV
exposes whatever FFmpeg understands, and we surface what we have
(``None`` for missing fields) rather than refusing to load. Strict
validation happens at the decoder layer where it matters for
playback correctness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

log = logging.getLogger(__name__)

# Containers Flick recognises as video on the drag-and-drop / file-open
# paths. Routing logic in ``app.py`` / ``scan_handler.py`` checks the
# extension first; the actual decode is PyAV/FFmpeg's call. Keep this
# list narrow on purpose — exotic containers (.mxf, .ts, .webm) work
# through PyAV but we'd rather opt them in once tested than surprise
# users with half-broken formats.
VIDEO_EXTENSIONS: frozenset[str] = frozenset({
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".avi",
})


def is_video_file(path: Path | str) -> bool:
    """Cheap extension check used by the drop-zone / scanner routers.

    Does NOT open the file — pure string match. The probe step is
    where actual format validation happens.
    """
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


@dataclass(frozen=True)
class VideoMetadata:
    """Immutable snapshot of a video file's pertinent stream info.

    All fields except ``path`` and ``has_video`` may be ``None`` when
    the container doesn't expose them (rare in practice but not
    impossible — some legacy AVIs lack a clean duration, and
    audio-less mp4s have no audio fields). The decoder validates
    what it actually needs at open-time.

    ``frame_count`` is from the container's ``nb_frames`` field — it
    can be missing or wrong for VFR / streamed sources. The decoder
    falls back to duration × fps when it needs an integer count.
    """

    path: Path
    has_video: bool
    width: int | None
    height: int | None
    # Native average FPS as a Fraction (e.g. 24000/1001 for NTSC). We
    # keep the rational form so the time-axis math stays exact — float
    # rounding on long sequences is what creates a/v drift.
    fps: Fraction | None
    duration_seconds: float | None
    frame_count: int | None
    pixel_format: str | None
    video_codec: str | None
    # Color metadata — used by the future OCIO input-transform picker.
    # FFmpeg's enum strings: 'bt709', 'smpte170m', 'bt2020nc', etc.
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    color_range: str | None  # 'tv' (limited) or 'pc' (full), or None
    # Audio fields — absent when the container has no audio stream.
    has_audio: bool
    audio_codec: str | None
    audio_sample_rate: int | None
    audio_channels: int | None


def probe_video(path: Path | str) -> VideoMetadata:
    """Open ``path`` with PyAV, read stream metadata, close.

    Raises :class:`FileNotFoundError` if the path doesn't exist, and
    :class:`ValueError` if PyAV can't open it (corrupt / unsupported).
    Other PyAV errors propagate — callers in the routing layer should
    catch broad ``Exception`` and surface a friendly message.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    # Imported here so ``import img_player.media`` doesn't pull PyAV
    # at module-load time — keeps test collection fast and lets
    # non-video code paths run without the dep loaded into memory.
    import av  # type: ignore[import-untyped]

    try:
        # ``metadata_errors='replace'`` keeps QuickTime ``.mov`` files
        # readable when they carry non-ASCII bytes (e.g. ``é``) in
        # tags that PyAV would otherwise UnicodeDecodeError on with
        # the default UTF-8-strict mode. The replacement char ends up
        # in the tag string but we never display tags so the loss is
        # invisible to the user — and the alternative is the file
        # refusing to open at all.
        container = av.open(str(p), metadata_errors="replace")
    except av.error.InvalidDataError as exc:  # type: ignore[attr-defined]
        raise ValueError(f"Cannot open video: {p}") from exc

    try:
        video_streams = [s for s in container.streams if s.type == "video"]
        audio_streams = [s for s in container.streams if s.type == "audio"]

        has_video = bool(video_streams)
        width: int | None = None
        height: int | None = None
        fps: Fraction | None = None
        pixel_format: str | None = None
        video_codec: str | None = None
        color_primaries: str | None = None
        color_transfer: str | None = None
        color_space: str | None = None
        color_range: str | None = None
        frame_count: int | None = None

        if has_video:
            v = video_streams[0]
            width = v.codec_context.width or None
            height = v.codec_context.height or None
            # ``average_rate`` is the most reliable FPS field across
            # containers; ``base_rate`` is the encoded rate and can
            # disagree on VFR sources. Both are Fraction in PyAV.
            if v.average_rate is not None:
                fps = Fraction(v.average_rate)
            elif v.base_rate is not None:
                fps = Fraction(v.base_rate)
            pixel_format = v.codec_context.pix_fmt or None
            video_codec = v.codec_context.name or None
            try:
                cc = v.codec_context
                color_primaries = (
                    cc.color_primaries.name
                    if getattr(cc, "color_primaries", None) is not None
                    else None
                )
                color_transfer = (
                    cc.color_trc.name
                    if getattr(cc, "color_trc", None) is not None
                    else None
                )
                color_space = (
                    cc.colorspace.name
                    if getattr(cc, "colorspace", None) is not None
                    else None
                )
                color_range = (
                    cc.color_range.name
                    if getattr(cc, "color_range", None) is not None
                    else None
                )
            except (AttributeError, ValueError):
                # Older PyAV versions and some rare codecs don't expose
                # these as enums. Color management still works at the
                # OCIO layer with manual input-space selection.
                pass
            if v.frames:  # ``nb_frames`` from the container header
                frame_count = int(v.frames)

        # Container-level duration in seconds (PyAV stores microseconds-ish
        # in ``container.duration`` as AV_TIME_BASE units = 1e6).
        duration_seconds: float | None = None
        if container.duration is not None:
            duration_seconds = float(container.duration) / 1_000_000.0
        elif has_video and video_streams[0].duration is not None:
            v = video_streams[0]
            duration_seconds = float(v.duration * v.time_base)

        # Backfill frame_count from duration × fps when the container
        # didn't report it (common with mp4 streamed through ffmpeg).
        if frame_count is None and duration_seconds and fps:
            frame_count = max(1, round(duration_seconds * float(fps)))

        has_audio = bool(audio_streams)
        audio_codec: str | None = None
        audio_sample_rate: int | None = None
        audio_channels: int | None = None
        if has_audio:
            a = audio_streams[0]
            audio_codec = a.codec_context.name or None
            audio_sample_rate = a.codec_context.sample_rate or None
            # ``layout.channels`` is the canonical channel count in
            # recent PyAV; fall back to the deprecated ``channels``
            # attribute for older versions.
            try:
                audio_channels = a.codec_context.layout.nb_channels
            except (AttributeError, ValueError):
                audio_channels = getattr(a.codec_context, "channels", None)

        return VideoMetadata(
            path=p,
            has_video=has_video,
            width=width,
            height=height,
            fps=fps,
            duration_seconds=duration_seconds,
            frame_count=frame_count,
            pixel_format=pixel_format,
            video_codec=video_codec,
            color_primaries=color_primaries,
            color_transfer=color_transfer,
            color_space=color_space,
            color_range=color_range,
            has_audio=has_audio,
            audio_codec=audio_codec,
            audio_sample_rate=audio_sample_rate,
            audio_channels=audio_channels,
        )
    finally:
        container.close()
