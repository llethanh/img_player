"""Tests for image-sequence + video writers (v0.5.0).

The image-sequence writer is exercised end-to-end across all four
formats (PNG / JPG / EXR / TIFF) — round-trip via OIIO assertions.
The video writer is smoke-tested on H.264 + a smoke FFV1 (lossless,
deterministic enough to assert frame count exactly).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import OpenImageIO as oiio
import pytest

from img_player.export.settings import ExportSettings
from img_player.export.writers.image_seq import ImageSequenceWriter
from img_player.export.writers.video import VideoWriter, build_writer


def _ramp_uint8(width: int = 64, height: int = 48, channels: int = 4) -> np.ndarray:
    arr = np.zeros((height, width, channels), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)
    arr[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    if channels >= 3:
        arr[..., 2] = 128
    if channels >= 4:
        arr[..., 3] = 255
    return arr


# ============================================================================
# ImageSequenceWriter
# ============================================================================


class TestImageSeqWriter:
    @pytest.mark.parametrize("fmt_key", ["png", "jpg", "exr", "tiff"])
    def test_writes_three_frames(self, tmp_path: Path, fmt_key: str) -> None:
        settings = ExportSettings(
            output_dir=tmp_path,
            in_frame=1, out_frame=3,
            format_key=fmt_key,
            start_frame=1,
        )
        w = ImageSequenceWriter(basename="test")
        # Pick the dtype the renderer would have produced.
        if fmt_key in ("png", "jpg"):
            arr = _ramp_uint8(64, 48, 4)
        elif fmt_key == "tiff":
            arr = (_ramp_uint8(64, 48, 4).astype(np.uint16) * 257)
        else:  # exr
            arr = (_ramp_uint8(64, 48, 4).astype(np.float32) / 255.0).astype(np.float16)

        w.open(settings, width=64, height=48, fps=24.0)
        for i in range(3):
            w.write_frame(arr, i)
        w.close()

        files = sorted(tmp_path.glob(f"test.*{settings.fmt.extension}"))
        assert len(files) == 3
        # Spot-check the first file dims via OIIO.
        inp = oiio.ImageInput.open(str(files[0]))
        assert inp is not None
        try:
            spec = inp.spec()
            assert spec.width == 64
            assert spec.height == 48
        finally:
            inp.close()

    def test_jpg_strips_alpha(self, tmp_path: Path) -> None:
        settings = ExportSettings(
            output_dir=tmp_path, in_frame=1, out_frame=1, format_key="jpg",
        )
        w = ImageSequenceWriter(basename="test")
        w.open(settings, 64, 48, 24.0)
        w.write_frame(_ramp_uint8(64, 48, 4), 0)
        w.close()
        files = list(tmp_path.glob("*.jpg"))
        assert len(files) == 1
        inp = oiio.ImageInput.open(str(files[0]))
        assert inp is not None
        try:
            spec = inp.spec()
            assert spec.nchannels == 3  # alpha stripped
        finally:
            inp.close()

    def test_filename_padding_at_least_4(self, tmp_path: Path) -> None:
        settings = ExportSettings(
            output_dir=tmp_path,
            in_frame=1, out_frame=2,
            format_key="png",
            start_frame=1,
        )
        w = ImageSequenceWriter(basename="seq")
        w.open(settings, 16, 16, 24.0)
        w.write_frame(_ramp_uint8(16, 16, 4), 0)
        w.close()
        files = list(tmp_path.glob("*.png"))
        # 0001.png → 4-digit padding minimum
        assert any(".0001." in f.name for f in files)

    def test_filename_padding_grows_for_large_ranges(self, tmp_path: Path) -> None:
        # start_frame 99000 + total 1500 → max number 100499 → 6 digits.
        settings = ExportSettings(
            output_dir=tmp_path,
            in_frame=1, out_frame=2,
            format_key="png",
            start_frame=99_000,
        )
        w = ImageSequenceWriter(basename="seq")
        w.open(settings, 16, 16, 24.0)
        w.write_frame(_ramp_uint8(16, 16, 4), 0)
        w.close()
        files = list(tmp_path.glob("*.png"))
        assert any("99000" in f.name for f in files)
        # 5 zero-padded digits = no left zero before "99000"
        assert all(len(f.stem.split(".")[-1]) >= 5 for f in files)

    def test_abort_removes_partial_files(self, tmp_path: Path) -> None:
        settings = ExportSettings(
            output_dir=tmp_path, in_frame=1, out_frame=5, format_key="png",
        )
        w = ImageSequenceWriter(basename="aborted")
        w.open(settings, 16, 16, 24.0)
        w.write_frame(_ramp_uint8(16, 16, 4), 0)
        w.write_frame(_ramp_uint8(16, 16, 4), 1)
        # Pretend the user clicked Cancel.
        w.abort()
        files = list(tmp_path.glob("aborted.*.png"))
        assert files == []  # everything cleaned up

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        settings = ExportSettings(
            output_dir=tmp_path, in_frame=1, out_frame=1, format_key="png",
        )
        w = ImageSequenceWriter(basename="t")
        w.open(settings, 16, 16, 24.0)
        w.write_frame(_ramp_uint8(16, 16, 4), 0)
        w.close()
        w.close()  # no raise


# ============================================================================
# VideoWriter (smoke)
# ============================================================================


class TestVideoWriter:
    def test_h264_writes_a_playable_file(self, tmp_path: Path) -> None:
        # Even dimensions for H.264 chroma subsampling.
        settings = ExportSettings(
            output_dir=tmp_path,
            in_frame=1, out_frame=10,
            format_key="h264_mp4",
            video_crf=28,
            h26x_preset="ultrafast",
        )
        w = build_writer(settings, basename="h264_smoke")
        assert isinstance(w, VideoWriter)
        w.open(settings, 64, 64, 24.0)
        for i in range(10):
            w.write_frame(_ramp_uint8(64, 64, 3), i)
        w.close()
        out = w.output_path()
        assert out.exists()
        assert out.stat().st_size > 0

    def test_ffv1_lossless_writes_exact_count(self, tmp_path: Path) -> None:
        """FFV1 is lossless and deterministic — useful smoke test."""
        import av
        settings = ExportSettings(
            output_dir=tmp_path,
            in_frame=1, out_frame=5,
            format_key="ffv1_mkv",
        )
        w = build_writer(settings, basename="ffv1_smoke")
        w.open(settings, 64, 64, 24.0)
        for i in range(5):
            w.write_frame(_ramp_uint8(64, 64, 3), i)
        w.close()
        # Re-open and count frames.
        container = av.open(str(w.output_path()), mode="r")
        try:
            stream = container.streams.video[0]
            decoded = list(container.decode(stream))
            assert len(decoded) == 5
        finally:
            container.close()

    def test_rgb_color_order_preserved(self, tmp_path: Path) -> None:
        """Regression test (v0.5.0.1): a pure-red input frame must
        decode back as red, not blue. Catches the R/B-swap bug from
        labeling RGB pixels as bgr24 to PyAV.

        FFV1 is lossless so we can assert exact dominance of the red
        channel.
        """
        import av
        # 24 frames of solid red (255, 0, 0).
        red = np.zeros((64, 64, 3), dtype=np.uint8)
        red[..., 0] = 255  # R
        settings = ExportSettings(
            output_dir=tmp_path,
            in_frame=1, out_frame=10,
            format_key="ffv1_mkv",
        )
        w = build_writer(settings, basename="rgb_check")
        w.open(settings, 64, 64, 24.0)
        for i in range(10):
            w.write_frame(red, i)
        w.close()
        # Decode and inspect the first frame's average colour.
        container = av.open(str(w.output_path()), mode="r")
        try:
            stream = container.streams.video[0]
            frames = list(container.decode(stream))
            assert frames, "no frames decoded"
            decoded = frames[0].to_ndarray(format="rgb24")
            r_mean = decoded[..., 0].mean()
            g_mean = decoded[..., 1].mean()
            b_mean = decoded[..., 2].mean()
            # FFV1 RGB → YUV → RGB round-trip is bit-exact for
            # primary colours, so we assert strict dominance.
            assert r_mean > 240, f"R should dominate, got R={r_mean}"
            assert g_mean < 30, f"G should be near zero, got G={g_mean}"
            assert b_mean < 30, f"B should be near zero, got B={b_mean}"
        finally:
            container.close()

    def test_abort_removes_partial_video(self, tmp_path: Path) -> None:
        settings = ExportSettings(
            output_dir=tmp_path,
            in_frame=1, out_frame=5,
            format_key="h264_mp4",
            h26x_preset="ultrafast",
        )
        w = build_writer(settings, basename="aborted_video")
        w.open(settings, 64, 64, 24.0)
        w.write_frame(_ramp_uint8(64, 64, 3), 0)
        out = w.output_path()
        w.abort()
        assert not out.exists()


# ============================================================================
# build_writer factory
# ============================================================================


class TestBuildWriter:
    def test_image_format_routes_to_image_seq_writer(self, tmp_path: Path) -> None:
        settings = ExportSettings(output_dir=tmp_path, in_frame=1, out_frame=1, format_key="png")
        w = build_writer(settings)
        assert isinstance(w, ImageSequenceWriter)

    def test_video_format_routes_to_video_writer(self, tmp_path: Path) -> None:
        settings = ExportSettings(output_dir=tmp_path, in_frame=1, out_frame=1, format_key="h264_mp4")
        w = build_writer(settings)
        assert isinstance(w, VideoWriter)
