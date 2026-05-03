"""Tests for ``img_player.media.video_source``.

Generates short H.264 mp4s on the fly so the suite has no fixture
files to ship. Each frame is a solid colour keyed to its index, so
we can assert the decoder returns the right frame for a given time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from img_player.media.video_source import VideoSource  # noqa: E402


def _make_indexed_video(
    path: Path,
    *,
    n_frames: int = 24,
    fps: int = 24,
    width: int = 64,
    height: int = 48,
    codec: str = "h264",
) -> None:
    """Encode a short video where frame ``i`` is solid grey ``i*10``.

    With ``n_frames=24`` and grey values 0, 10, 20… the index is
    visible in any decoded frame (uint8 in [0, 230]). The H.264
    encoder is lossy, but the spread between values (10) is large
    enough that ``round(grey / 10)`` recovers the index reliably.
    """
    container = av.open(str(path), mode="w")
    stream = container.add_stream(codec, rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    # Force keyframes everywhere — short test clips, we want exact
    # seek behaviour without GOP-boundary effects masking decoder bugs.
    stream.options = {"g": "1"}

    for i in range(n_frames):
        grey = i * 10
        arr = np.full((height, width, 3), grey, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def _frame_index(arr: np.ndarray) -> int:
    """Recover the frame index from its dominant grey level."""
    return int(round(float(arr.mean()) / 10.0))


def test_open_close(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    _make_indexed_video(p, n_frames=8)
    src = VideoSource(p)
    try:
        assert src.width == 64
        assert src.height == 48
        assert float(src.fps) == 24.0
        assert src.duration_seconds > 0
    finally:
        src.close()


def test_frame_at_time_first(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    _make_indexed_video(p, n_frames=24, fps=24)
    with VideoSource(p) as src:
        arr = src.frame_at_time(0.0)
        assert arr.shape == (48, 64, 3)
        assert arr.dtype == np.uint8
        assert _frame_index(arr) == 0


def test_frame_at_time_forward_scan(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    _make_indexed_video(p, n_frames=24, fps=24)
    with VideoSource(p) as src:
        # Sample times that should land on frames 0, 6, 12, 18.
        # Add 1ms offset so we're firmly inside each frame's display
        # interval, not exactly on the boundary.
        for i in (0, 6, 12, 18):
            t = i / 24.0 + 0.001
            arr = src.frame_at_time(t)
            assert _frame_index(arr) == i, f"expected {i}, got {_frame_index(arr)}"


def test_frame_at_time_backward_seek(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    _make_indexed_video(p, n_frames=24, fps=24)
    with VideoSource(p) as src:
        # Walk forward to mid-stream, then jump back — exercises
        # the seek path.
        src.frame_at_time(15 / 24.0 + 0.001)
        arr = src.frame_at_time(3 / 24.0 + 0.001)
        assert _frame_index(arr) == 3


def test_frame_cache_hit_no_redecode(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    _make_indexed_video(p, n_frames=24, fps=24)
    with VideoSource(p) as src:
        arr1 = src.frame_at_time(7 / 24.0 + 0.001)
        # Within the same frame's display interval — should hit the
        # single-frame cache and return the SAME ndarray object.
        arr2 = src.frame_at_time(7 / 24.0 + 0.005)
        assert arr1 is arr2


def test_frame_at_time_clamps_past_end(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    _make_indexed_video(p, n_frames=12, fps=24)
    with VideoSource(p) as src:
        # 1 second in but the clip is only 0.5s long → freeze on the
        # last available frame instead of raising.
        arr = src.frame_at_time(1.0)
        assert _frame_index(arr) == 11


def test_no_video_stream_raises(tmp_path: Path) -> None:
    """Audio-only mp4 (rare but possible) — VideoSource must refuse it."""
    p = tmp_path / "audio_only.m4a"
    container = av.open(str(p), mode="w")
    astream = container.add_stream("aac", rate=48000)
    astream.layout = "stereo"
    samples = np.zeros((2, 1024), dtype=np.float32)
    aframe = av.AudioFrame.from_ndarray(samples, format="fltp", layout="stereo")
    aframe.sample_rate = 48000
    aframe.pts = 0
    for packet in astream.encode(aframe):
        container.mux(packet)
    for packet in astream.encode(None):
        container.mux(packet)
    container.close()

    with pytest.raises(ValueError, match="No video stream"):
        VideoSource(p)
