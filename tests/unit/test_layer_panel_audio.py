"""Tests for the per-layer audio toggles (M / S buttons) on LayerRow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

av = pytest.importorskip("av")
PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from img_player.layers import LayerStack  # noqa: E402
from img_player.layers.models import Layer  # noqa: E402
from img_player.media.video_probe import probe_video  # noqa: E402
from img_player.sequence.models import FrameInfo, SequenceInfo  # noqa: E402
from img_player.ui.layer_panel import LayerPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _make_video(path: Path, *, with_audio: bool = True) -> None:
    """Encode a small mp4 (and optional AAC stereo). Mirrors the
    pattern used in ``test_audio_source._make_av_file`` — declare
    ALL streams up-front, then encode + mux in interleaved order,
    so PyAV's mux time-base resolution succeeds."""
    container = av.open(str(path), mode="w")
    vstream = container.add_stream("h264", rate=24)
    vstream.width = 64
    vstream.height = 48
    vstream.pix_fmt = "yuv420p"
    vstream.options = {"g": "1"}
    astream = None
    if with_audio:
        astream = container.add_stream("aac", rate=48000)
        astream.layout = "stereo"

    arr = np.full((48, 64, 3), 128, dtype=np.uint8)
    for i in range(8):
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame.pts = i
        for packet in vstream.encode(frame):
            container.mux(packet)

    if astream is not None:
        pts = 0
        for start in range(0, 4800, 1024):
            length = min(1024, 4800 - start)
            block = np.zeros((2, length), dtype=np.float32)
            aframe = av.AudioFrame.from_ndarray(
                block, format="fltp", layout="stereo",
            )
            aframe.sample_rate = 48000
            aframe.pts = pts
            pts += length
            for packet in astream.encode(aframe):
                container.mux(packet)
        for packet in astream.encode(None):
            container.mux(packet)
    for packet in vstream.encode(None):
        container.mux(packet)
    container.close()


def _image_layer(tmp_path: Path) -> Layer:
    seq = SequenceInfo(
        base_name="img.",
        extension=".exr",
        directory=tmp_path,
        padding=4,
        frames=(FrameInfo(path=tmp_path / "img.0001.exr", frame_number=1),),
    )
    return Layer.from_sequence(seq)


def _video_layer(tmp_path: Path, *, with_audio: bool = True) -> Layer:
    p = tmp_path / ("av.mp4" if with_audio else "v_silent.mp4")
    _make_video(p, with_audio=with_audio)
    return Layer.from_video(probe_video(p))


def test_audio_buttons_disabled_on_image_layer(qapp, tmp_path):
    stack = LayerStack()
    stack.add(_image_layer(tmp_path))
    panel = LayerPanel(stack)
    layer_id = stack.layers()[0].id
    row = panel._rows[layer_id]
    assert row._audio_mute_btn.isEnabled() is False
    assert row._audio_solo_btn.isEnabled() is False


def test_audio_buttons_enabled_on_video_with_audio(qapp, tmp_path):
    stack = LayerStack()
    stack.add(_video_layer(tmp_path, with_audio=True))
    panel = LayerPanel(stack)
    layer_id = stack.layers()[0].id
    row = panel._rows[layer_id]
    assert row._audio_mute_btn.isEnabled() is True
    assert row._audio_solo_btn.isEnabled() is True


def test_audio_buttons_disabled_on_silent_video(qapp, tmp_path):
    stack = LayerStack()
    stack.add(_video_layer(tmp_path, with_audio=False))
    panel = LayerPanel(stack)
    layer_id = stack.layers()[0].id
    row = panel._rows[layer_id]
    assert row._audio_mute_btn.isEnabled() is False
    assert row._audio_solo_btn.isEnabled() is False


def test_mute_button_updates_layer(qapp, tmp_path):
    stack = LayerStack()
    stack.add(_video_layer(tmp_path))
    panel = LayerPanel(stack)
    layer_id = stack.layers()[0].id
    row = panel._rows[layer_id]
    row._audio_mute_btn.setChecked(True)
    assert stack.find(layer_id).audio_mute is True
    row._audio_mute_btn.setChecked(False)
    assert stack.find(layer_id).audio_mute is False


def test_solo_is_exclusive(qapp, tmp_path):
    """Turning solo ON for layer B turns it OFF on layer A."""
    stack = LayerStack()
    stack.add(_video_layer(tmp_path), position=0)
    p2 = tmp_path / "av2.mp4"
    _make_video(p2, with_audio=True)
    stack.add(Layer.from_video(probe_video(p2)), position=0)

    panel = LayerPanel(stack)
    layers = stack.layers()
    a_id = layers[1].id
    b_id = layers[0].id
    row_a = panel._rows[a_id]
    row_b = panel._rows[b_id]

    row_a._audio_solo_btn.setChecked(True)
    assert stack.find(a_id).audio_solo is True
    assert stack.find(b_id).audio_solo is False

    row_b._audio_solo_btn.setChecked(True)
    assert stack.find(b_id).audio_solo is True
    assert stack.find(a_id).audio_solo is False  # exclusive flip


def test_external_mutation_syncs_button(qapp, tmp_path):
    """When stack.update flips audio_mute, the row button reflects it."""
    stack = LayerStack()
    stack.add(_video_layer(tmp_path))
    panel = LayerPanel(stack)
    layer_id = stack.layers()[0].id
    row = panel._rows[layer_id]
    assert row._audio_mute_btn.isChecked() is False
    stack.update(layer_id, audio_mute=True)
    # update_layer is called via layer_modified signal in the panel.
    qapp.processEvents()
    assert row._audio_mute_btn.isChecked() is True
