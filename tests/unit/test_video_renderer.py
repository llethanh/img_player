"""Tests for ``img_player.media.video_renderer.VideoSourceManager``.

The manager is the lifecycle layer between the PyAV decoder and the
app: it lazy-opens, lazy-closes, and converts decoded RGB uint8 into
the RGBA float32 the GL viewport expects. These tests cover the
contract; the integration with ``app.py`` is exercised separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from img_player.media.video_renderer import VideoSourceManager  # noqa: E402


def _make_video(path: Path, *, n_frames: int = 24, fps: int = 24,
                width: int = 64, height: int = 48) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"g": "1"}
    arr = np.full((height, width, 3), 128, dtype=np.uint8)
    for i in range(n_frames):
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = i
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()


def test_decode_at_returns_rgba_uint8(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    _make_video(p)
    mgr = VideoSourceManager()
    try:
        rgba = mgr.decode_at("layer-1", p, 0.0)
        # v1.8.3 viewport refactor: decode_at returns uint8 RGBA;
        # the GPU normalizes on upload via GL_UNSIGNED_BYTE so the
        # main thread skips the ~16 ms cast that was capping cached
        # 60 fps playback at 30 fps.
        assert rgba.shape == (48, 64, 4)
        assert rgba.dtype == np.uint8
        assert int(rgba[:, :, 3].min()) == 255
    finally:
        mgr.shutdown()


def test_get_or_open_caches_per_layer(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    _make_video(p)
    mgr = VideoSourceManager()
    try:
        s1 = mgr.get_or_open("layer-1", p)
        s2 = mgr.get_or_open("layer-1", p)
        assert s1 is s2
        # Different layer id pointing at the same path → distinct
        # decoder so per-layer scrub state stays independent.
        s3 = mgr.get_or_open("layer-2", p)
        assert s3 is not s1
    finally:
        mgr.shutdown()


def test_close_releases_layer(tmp_path: Path) -> None:
    p = tmp_path / "v.mp4"
    _make_video(p)
    mgr = VideoSourceManager()
    mgr.get_or_open("layer-1", p)
    assert "layer-1" in mgr._sources
    mgr.close("layer-1")
    assert "layer-1" not in mgr._sources
    # Closing an unknown layer is a no-op (defensive — used at session
    # swap when the manager and the stack might temporarily disagree).
    mgr.close("does-not-exist")


def test_shutdown_closes_all(tmp_path: Path) -> None:
    p1 = tmp_path / "a.mp4"
    p2 = tmp_path / "b.mp4"
    _make_video(p1)
    _make_video(p2)
    mgr = VideoSourceManager()
    mgr.get_or_open("a", p1)
    mgr.get_or_open("b", p2)
    assert len(mgr._sources) == 2
    mgr.shutdown()
    assert len(mgr._sources) == 0


def test_cache_budget_provider_is_resolved_per_open(tmp_path: Path) -> None:
    """Pin the v1.8.3 contract: when a callable provider is passed at
    manager construction, it is re-resolved on every get_or_open so
    a Preferences-dialog tweak between layers takes effect without
    restarting the app.
    """
    p1 = tmp_path / "a.mp4"
    p2 = tmp_path / "b.mp4"
    _make_video(p1)
    _make_video(p2)

    # Mutable budget — would be backed by Preferences.video_cache_budget_gb
    # in the live app; here it's just a single-element list so the test
    # can change it between opens.
    box = [1_000_000]  # 1 MB initial

    def provider() -> int:
        return box[0]

    mgr = VideoSourceManager(cache_budget_provider=provider)
    try:
        dec1 = mgr.get_or_open("a", p1)
        # First open snapped the initial 1 MB.
        assert dec1._source._frame_cache_budget == 1_000_000  # type: ignore[attr-defined]

        # User opens Preferences, cranks budget to 5 MB, closes dialog,
        # then opens a second layer.
        box[0] = 5_000_000
        dec2 = mgr.get_or_open("b", p2)
        # Second open MUST see the new value, not stay at 1 MB.
        assert dec2._source._frame_cache_budget == 5_000_000  # type: ignore[attr-defined]

        # And the first layer keeps its original budget — already-open
        # decoders aren't retroactively resized (would be confusing
        # and require evicting on the fly).
        assert dec1._source._frame_cache_budget == 1_000_000  # type: ignore[attr-defined]
    finally:
        mgr.shutdown()
