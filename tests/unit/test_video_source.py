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


class TestRamCache:
    """Pin the v1.8.2 RAM cache contract: every decoded frame goes
    into the LRU cache, re-reads hit RAM, eviction respects budget."""

    def test_cache_starts_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "v.mp4"
        _make_indexed_video(p, n_frames=8)
        with VideoSource(p, prefetch=False) as src:
            stats = src.cache_stats()
            assert stats["frames"] == 0
            assert stats["bytes"] == 0
            assert stats["budget"] > 0

    def test_first_read_populates_cache(self, tmp_path: Path) -> None:
        p = tmp_path / "v.mp4"
        _make_indexed_video(p, n_frames=8)
        with VideoSource(p, prefetch=False) as src:
            src.frame_at_time(0.0)
            stats = src.cache_stats()
            assert stats["frames"] >= 1
            assert stats["bytes"] > 0

    def test_reread_does_not_grow_cache(self, tmp_path: Path) -> None:
        p = tmp_path / "v.mp4"
        _make_indexed_video(p, n_frames=8)
        with VideoSource(p, prefetch=False) as src:
            src.frame_at_time(0.0)
            frames_after_first = src.cache_stats()["frames"]
            src.frame_at_time(0.0)
            src.frame_at_time(0.0)
            assert src.cache_stats()["frames"] == frames_after_first

    def test_zero_budget_disables_cache(self, tmp_path: Path) -> None:
        p = tmp_path / "v.mp4"
        _make_indexed_video(p, n_frames=4)
        with VideoSource(p, cache_budget_bytes=0) as src:
            src.frame_at_time(0.0)
            assert src.cache_stats()["frames"] == 0

    def test_budget_evicts_oldest(self, tmp_path: Path) -> None:
        p = tmp_path / "v.mp4"
        # 64×48 RGBA float32 = 49 152 bytes/frame (since v1.8.2 the
        # cache stores display-ready float32 to skip the cast at
        # read time). With a 110 000-byte budget the cache fits 2
        # frames + a sliver, so a 3rd read must evict the 1st (LRU).
        # ``prefetch=False`` keeps the test deterministic — the
        # background worker would race with the manual reads.
        _make_indexed_video(p, n_frames=8)
        with VideoSource(
            p, cache_budget_bytes=110_000, prefetch=False,
        ) as src:
            src.frame_at_time(0.0 / 24)
            src.frame_at_time(1.0 / 24)
            src.frame_at_time(2.0 / 24)
            stats = src.cache_stats()
            assert stats["bytes"] <= 110_000
            # Cache shouldn't have grown past the budget.
            assert stats["frames"] <= 3

    def test_close_clears_cache(self, tmp_path: Path) -> None:
        p = tmp_path / "v.mp4"
        _make_indexed_video(p, n_frames=4)
        src = VideoSource(p)
        try:
            src.frame_at_time(0.0)
            assert src.cache_stats()["frames"] >= 1
        finally:
            src.close()
        # Cache cleared on close — subsequent stats reads return 0.
        assert src.cache_stats()["frames"] == 0
        assert src.cache_stats()["bytes"] == 0


def _frame_index(arr: np.ndarray) -> int:
    """Recover the frame index from its dominant grey level.

    VideoSource returns float32 RGBA (alpha uniform 1.0) since
    v1.8.2; the alpha plane would skew the mean if averaged
    together with RGB. Pulled the values are normalised to [0,1],
    so multiply by 255 to recover the original grey level.
    """
    rgb = arr[..., :3]
    # If the array is float32 in [0, 1], scale back up. uint8
    # straight through.
    mean = float(rgb.mean())
    if np.issubdtype(arr.dtype, np.floating):
        mean *= 255.0
    return int(round(mean / 10.0))


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
        # VideoSource caches + returns float32 RGBA in [0, 1] since
        # v1.8.2 — the conversion is rolled into the cache layer so
        # cache hits return display-ready (no per-frame cast in
        # decode_at). The alpha plane is uniform 1.0 from swscale.
        assert arr.shape == (48, 64, 4)
        assert arr.dtype == np.float32
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
