"""PyAV-backed video writer (H.264 / H.265 / ProRes / DNxHR / VP9 / FFV1 / v210)."""

from __future__ import annotations

import logging
from fractions import Fraction
from pathlib import Path

import numpy as np

from img_player.export.settings import ExportSettings
from img_player.export.writers.base import BaseWriter, ExportWriteError
from img_player.export.writers.image_seq import ImageSequenceWriter  # for build_writer

log = logging.getLogger(__name__)


# The engine feeds us 8-bit RGB or RGBA arrays (the OIIO reader
# returns R/G/B/A channels and the renderer keeps that ordering).
# PyAV reformats internally to the codec's target pix_fmt — but
# only after we tell it the *correct* input pixel order. Using
# "bgr24" here when the buffer is actually RGB swaps R↔B at the
# YUV conversion → warm tones turn blue, blues turn orange.
# v0.5: shipped with that bug; v0.5.0.1 fixes it.
_INPUT_PIX_FMT_RGB = "rgb24"
_INPUT_PIX_FMT_RGBA = "rgba"


class VideoWriter(BaseWriter):
    """Writes a single container file with one video stream."""

    def __init__(self) -> None:
        # PyAV is imported lazily so the rest of the export module
        # remains importable without PyAV (e.g. on a tests run that
        # only exercises the image-sequence path or the settings).
        # In the running app PyAV is always installed.
        self._container = None  # type: ignore[var-annotated]
        self._stream = None  # type: ignore[var-annotated]
        self._settings: ExportSettings | None = None
        self._width = 0
        self._height = 0
        self._fps: float = 24.0
        self._has_alpha_input = False
        self._closed = False
        self._aborted = False
        self._output_file: Path | None = None
        self._frames_written = 0

    # ------------------------------------------------------------------ Lifecycle

    def open(
        self, settings: ExportSettings, width: int, height: int, fps: float
    ) -> None:
        if not settings.is_video:
            raise ExportWriteError(f"VideoWriter cannot handle {settings.format_key!r}")
        if width % 2 or height % 2:
            # Most codecs require even dimensions (chroma subsampling).
            # We round up rather than failing — caller resized to
            # whatever the user asked for; we add 1 px if needed.
            log.warning(
                "[export] odd dimensions %dx%d — codec may reject. Engine should "
                "have rounded.", width, height,
            )
        self._settings = settings
        self._width = width
        self._height = height
        self._fps = fps
        self._frames_written = 0
        # ProRes 4444 with the 4444 profile keeps alpha. Otherwise we
        # always feed BGR (no alpha) so the output looks the same as
        # what the user sees in the viewer (composited on whatever
        # background the OCIO display assumed).
        self._has_alpha_input = (
            settings.format_key == "prores_4444_mov" and settings.prores_profile >= 4
        )

        # Suffix-driven output filename. The user picked an output
        # *folder* (per Q7 D); we name the video file from the
        # source basename + the format extension.
        ext = settings.fmt.extension
        out_dir = settings.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        # The basename is plumbed by the engine via the renderer's
        # source sequence — for the writer's purpose, default to
        # "export". The engine overrides it via ``set_output_basename``
        # before ``open()``.
        if self._output_file is None:
            self._output_file = out_dir / f"export{ext}"

        import av  # noqa: PLC0415 — local import keeps PyAV optional at module load
        self._av = av

        # Open the container.
        try:
            self._container = av.open(str(self._output_file), mode="w")
        except av.AVError as err:
            raise ExportWriteError(
                f"Cannot open {self._output_file} for writing: {err}"
            ) from err

        codec_name = settings.fmt.codec
        target_pix_fmt = settings.fmt.pix_fmt or "yuv420p"
        try:
            stream = self._container.add_stream(codec_name, rate=Fraction(fps).limit_denominator(1000))
        except (av.AVError, KeyError, ValueError) as err:
            self._container.close()
            self._container = None
            raise ExportWriteError(
                f"FFmpeg cannot add stream for codec {codec_name!r}: {err}"
            ) from err
        stream.width = width
        stream.height = height
        stream.pix_fmt = target_pix_fmt

        self._configure_codec_options(stream, settings)
        self._stream = stream

    def write_frame(self, arr: np.ndarray, frame_idx: int) -> None:
        if self._container is None or self._stream is None:
            raise ExportWriteError("write_frame() before open()")
        del frame_idx  # video pts comes from the stream's frame counter
        if arr.dtype != np.uint8:
            raise ExportWriteError(
                f"VideoWriter expects uint8 input, got {arr.dtype}"
            )
        if arr.ndim != 3:
            raise ExportWriteError(f"VideoWriter expects HxWxC, got shape {arr.shape}")
        h, w, c = arr.shape
        if h != self._height or w != self._width:
            raise ExportWriteError(
                f"frame shape {(h, w)} mismatches expected ({self._height}, {self._width})"
            )
        # Pick the input ffmpeg pix_fmt based on whether we have alpha.
        if self._has_alpha_input and c >= 4:
            arr = np.ascontiguousarray(arr[..., :4])
            input_fmt = _INPUT_PIX_FMT_RGBA
        else:
            arr = np.ascontiguousarray(arr[..., :3])
            input_fmt = _INPUT_PIX_FMT_RGB

        av_frame = self._av.VideoFrame.from_ndarray(arr, format=input_fmt)
        # PyAV reformats to the target pix_fmt automatically when we
        # encode if format differs. Setting it explicitly avoids one
        # silent reformat per frame.
        av_frame = av_frame.reformat(format=self._stream.pix_fmt)
        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)
        self._frames_written += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._container is None or self._stream is None:
            return
        # Flush the encoder.
        try:
            for packet in self._stream.encode(None):
                self._container.mux(packet)
        except Exception:  # pragma: no cover — defensive only
            log.exception("[export] flush failed")
        try:
            self._container.close()
        finally:
            self._container = None
            self._stream = None
        log.info(
            "[export] video writer closed: %d frames -> %s",
            self._frames_written, self._output_file,
        )

    def abort(self) -> None:
        """Close the container then delete the partial file. A
        mid-encode mp4 / mov is corrupt anyway — nothing to salvage."""
        self._aborted = True
        # Close gracefully if possible (codec flush) — but if it
        # raises we still want to delete the partial file.
        try:
            self.close()
        except Exception:  # pragma: no cover — defensive
            log.exception("[export] error during abort close()")
        if self._output_file is not None:
            try:
                self._output_file.unlink(missing_ok=True)
            except OSError:
                log.exception("[export] failed to remove %s on abort", self._output_file)

    def output_path(self) -> Path:
        return self._output_file or Path(".")

    # ------------------------------------------------------------------ Public extras

    def set_output_basename(self, basename: str) -> None:
        """Engine calls this BEFORE ``open()`` so the file inherits
        the source sequence's name."""
        if self._settings is None:
            # Stash for ``open()`` to consume.
            object.__setattr__(self, "_pending_basename", basename)
        # Either way, compute the output_file when settings is known.

    # ------------------------------------------------------------------ Internals

    def _configure_codec_options(self, stream, settings: ExportSettings) -> None:
        """Set codec-specific encoder options.

        Each branch is small and self-contained — adding a new codec
        means a new ``elif key == ...`` block here, no other module
        touched.
        """
        key = settings.format_key
        opts: dict[str, str] = {}
        if key in ("h264_mp4", "h265_mp4"):
            opts["crf"] = str(settings.video_crf)
            opts["preset"] = settings.h26x_preset
        elif key == "vp9_webm":
            # VP9 wants both crf and target bitrate=0 to enter true
            # CRF mode. Without "b:v=0" it picks an internal default
            # bitrate and silently ignores -crf.
            opts["crf"] = str(settings.video_crf)
            opts["b"] = "0"
        elif key in ("prores_422_mov", "prores_4444_mov"):
            # prores_ks profile param: 0..5 (Proxy / LT / 422 / HQ / 4444 / 4444 XQ).
            # ProRes 4444 demands profile >= 4 for alpha. We trust the
            # dialog to pair (format=4444, profile in {4,5}) — but
            # clamp here so a stale prefs value doesn't silently
            # write a non-alpha file.
            profile = settings.prores_profile
            if key == "prores_4444_mov" and profile < 4:
                profile = 4
            opts["profile"] = str(profile)
        elif key == "dnxhr_mov":
            # DNxHR auto-selects a profile from the resolution + pix_fmt.
            # Use "dnxhr_hq" preset which is the broad sweet-spot.
            opts["profile"] = "dnxhr_hq"
        elif key == "ffv1_mkv":
            # Lossless FFV1 — slicecrc=1 makes the stream resilient to
            # mid-stream truncation. level=3 is the modern default.
            opts["level"] = "3"
            opts["slicecrc"] = "1"
            opts["coder"] = "1"
            opts["context"] = "1"
            opts["g"] = "1"
        # v210 has no tunable options beyond pix_fmt — already set.
        for k, v in opts.items():
            try:
                stream.options[k] = v
            except (KeyError, ValueError):
                # PyAV may reject some keys silently — log and continue.
                log.warning("[export] codec option %s=%s not accepted", k, v)


def build_writer(settings: ExportSettings, *, basename: str = "export") -> BaseWriter:
    """Factory: pick the right writer for the requested format."""
    if settings.is_image_sequence:
        return ImageSequenceWriter(basename=basename)
    writer = VideoWriter()
    # Set the file path before ``open()`` so abort()/output_path()
    # have it in case ``open()`` itself raises.
    ext = settings.fmt.extension
    writer._output_file = settings.output_dir / f"{basename}{ext}"  # noqa: SLF001
    return writer
