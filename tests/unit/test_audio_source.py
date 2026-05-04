"""Tests for ``img_player.media.audio_source``.

Generates a short stereo mp4 with a sine wave on the audio track so
we can assert resampled-output shape and basic seek behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")

from img_player.media.audio_source import AudioSource  # noqa: E402


def _make_av_file(
    path: Path,
    *,
    duration_s: float = 1.0,
    audio_sr: int = 48000,
    audio_channels: int = 2,
    freq_hz: float = 440.0,
) -> None:
    """Encode a video+audio mp4 with a sine on the audio track."""
    container = av.open(str(path), mode="w")
    vstream = container.add_stream("h264", rate=24)
    vstream.width = 64
    vstream.height = 48
    vstream.pix_fmt = "yuv420p"
    vstream.options = {"g": "1"}

    astream = container.add_stream("aac", rate=audio_sr)
    astream.layout = "stereo" if audio_channels == 2 else "mono"

    n_video = int(duration_s * 24)
    arr = np.full((48, 64, 3), 128, dtype=np.uint8)
    for i in range(n_video):
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = i
        for packet in vstream.encode(frame):
            container.mux(packet)

    n_samples_total = int(duration_s * audio_sr)
    chunk = 1024
    pts = 0
    t = np.arange(n_samples_total) / audio_sr
    sine = np.sin(2 * np.pi * freq_hz * t).astype(np.float32) * 0.5
    for start in range(0, n_samples_total, chunk):
        length = min(chunk, n_samples_total - start)
        block = np.tile(sine[start:start + length], (audio_channels, 1))
        aframe = av.AudioFrame.from_ndarray(
            block, format="fltp",
            layout="stereo" if audio_channels == 2 else "mono",
        )
        aframe.sample_rate = audio_sr
        aframe.pts = pts
        pts += length
        for packet in astream.encode(aframe):
            container.mux(packet)
    for packet in astream.encode(None):
        container.mux(packet)
    for packet in vstream.encode(None):
        container.mux(packet)
    container.close()


def test_open_and_metadata(tmp_path: Path) -> None:
    p = tmp_path / "a.mp4"
    _make_av_file(p)
    src = AudioSource(p, output_sample_rate=48000, output_channels=2)
    try:
        assert src.sample_rate == 48000
        assert src.channels == 2
        assert src.native_sample_rate == 48000
    finally:
        src.close()


def test_read_returns_correct_shape(tmp_path: Path) -> None:
    p = tmp_path / "a.mp4"
    _make_av_file(p, duration_s=0.5)
    with AudioSource(p) as src:
        block = src.read(2048)
        assert block.shape[1] == 2
        assert block.dtype == np.float32
        assert block.shape[0] == 2048
        # Sine wave: values should span both signs in a 2048-sample
        # block at 440 Hz / 48 kHz.
        assert block.min() < -0.1
        assert block.max() > 0.1


def test_read_eof_returns_short_block(tmp_path: Path) -> None:
    p = tmp_path / "a.mp4"
    _make_av_file(p, duration_s=0.1)  # ~4800 samples (+/- AAC priming)
    with AudioSource(p) as src:
        # Drain — keep reading 4 k blocks until we get a short one
        # (= EOF). AAC priming makes the exact total unpredictable;
        # the contract is "eventually short / empty", not "exact count".
        total = 0
        for _ in range(20):  # generous upper bound
            block = src.read(4096)
            total += block.shape[0]
            if block.shape[0] < 4096:
                break
        else:
            pytest.fail("AudioSource never reported EOF on a 0.1 s file")
        # Subsequent read after EOF must be empty (contract for the
        # AudioOutput callback path).
        empty = src.read(4096)
        assert empty.shape[0] == 0
        assert empty.shape[1] == 2
        assert total > 0


def test_seek_resets_buffer(tmp_path: Path) -> None:
    p = tmp_path / "a.mp4"
    _make_av_file(p, duration_s=1.0)
    with AudioSource(p) as src:
        src.read(1024)
        src.seek(0.5)
        block = src.read(1024)
        # After seek the read should still produce valid samples.
        assert block.shape[0] == 1024
        assert block.shape[1] == 2


def test_no_audio_stream_raises(tmp_path: Path) -> None:
    """Video-only file (no audio track) → AudioSource refuses."""
    p = tmp_path / "v.mp4"
    container = av.open(str(p), mode="w")
    vstream = container.add_stream("h264", rate=24)
    vstream.width = 64
    vstream.height = 48
    vstream.pix_fmt = "yuv420p"
    arr = np.full((48, 64, 3), 128, dtype=np.uint8)
    for i in range(12):
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = i
        for packet in vstream.encode(frame):
            container.mux(packet)
    for packet in vstream.encode(None):
        container.mux(packet)
    container.close()

    with pytest.raises(ValueError, match="No audio stream"):
        AudioSource(p)


def test_decoded_sample_count_matches_duration(tmp_path: Path) -> None:
    """Regression: decoding 1 s of audio must yield ~48000 samples.

    Earlier the resampler was configured for ``flt`` (interleaved)
    and ``to_ndarray`` returned ``(1, samples*channels)``; a naive
    transpose then produced 2× the samples (each stereo pair read
    as two mono samples), and the ring buffer played back at half
    speed → audio pitched down an octave.
    """
    p = tmp_path / "a.mp4"
    _make_av_file(p, duration_s=1.0, audio_sr=48000, audio_channels=2)
    with AudioSource(p, output_sample_rate=48000, output_channels=2) as src:
        total = 0
        for _ in range(80):  # plenty of room for AAC priming
            block = src.read(4096)
            total += block.shape[0]
            if block.shape[0] < 4096:
                break
        # Tolerance: AAC encodes priming + flushing samples that can
        # add a few thousand. The bug doubled the count → assert we're
        # within ±10 % of the expected 48k, not 96k+.
        expected = 48000
        assert 0.9 * expected <= total <= 1.2 * expected, (
            f"got {total} samples, expected ~{expected} (bug would "
            f"give ~{2*expected})"
        )


def test_mono_to_stereo_conversion(tmp_path: Path) -> None:
    """Mono source + stereo output: both channels carry the same signal."""
    p = tmp_path / "mono.mp4"
    _make_av_file(p, audio_channels=1)
    with AudioSource(p, output_channels=2) as src:
        block = src.read(1024)
        assert block.shape == (1024, 2)
        # Mono → stereo: left == right (within float tolerance).
        assert np.allclose(block[:, 0], block[:, 1], atol=1e-6)
