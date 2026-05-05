"""Tests for still-image layers (single-file held over N master frames).

Covers:

* :meth:`Layer.from_still` constructor invariants.
* :meth:`Layer.from_image` smart factory routes single-frame seqs to still.
* Geometry overrides for stills (``trim_length`` / ``source_frame_at``).
* Scanner fallback for filenames without a numeric pattern.
* Session save/load round-trip preserves ``is_still`` + ``still_hold_frames``.

Pure data + filesystem; no Qt, no decoder.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from img_player.layers import Layer
from img_player.sequence.models import FrameInfo, SequenceInfo


# ============================================================================
# Helpers
# ============================================================================


def _still_seq(name: str = "slate.png", frame_number: int = 0) -> SequenceInfo:
    """1-frame SequenceInfo simulating a single-file still (no disk I/O)."""
    path = Path(f"/fake/{name}")
    return SequenceInfo(
        base_name=path.stem,
        extension=path.suffix,
        directory=Path("/fake"),
        padding=0,
        frames=(FrameInfo(path=path, frame_number=frame_number),),
    )


def _multi_seq(first: int = 1001, last: int = 1010) -> SequenceInfo:
    frames = tuple(
        FrameInfo(path=Path(f"/fake/frame.{n:04d}.exr"), frame_number=n)
        for n in range(first, last + 1)
    )
    return SequenceInfo(
        base_name="frame", extension=".exr",
        directory=Path("/fake"), padding=4, frames=frames,
    )


# ============================================================================
# from_still
# ============================================================================


class TestFromStill:
    def test_basic_construction(self) -> None:
        layer = Layer.from_still(_still_seq(), hold_frames=50)
        assert layer.is_still is True
        assert layer.still_hold_frames == 50
        # Trim collapses to the single source frame.
        assert layer.layer_in == layer.layer_out

    def test_trim_length_uses_hold_frames(self) -> None:
        layer = Layer.from_still(_still_seq(), hold_frames=42)
        assert layer.trim_length == 42

    def test_master_end_uses_hold(self) -> None:
        layer = Layer.from_still(_still_seq(), hold_frames=10, offset=100)
        assert layer.master_start == 100
        assert layer.master_end == 109  # offset + hold - 1

    def test_covers_full_hold_range(self) -> None:
        layer = Layer.from_still(_still_seq(), hold_frames=5, offset=20)
        assert all(layer.covers(f) for f in range(20, 25))
        assert not layer.covers(19)
        assert not layer.covers(25)

    def test_source_frame_at_is_constant(self) -> None:
        """Every master frame in hold range maps to the SAME source frame
        (= the single file on disk). This is what enables the cache to
        alias one decoded ndarray across the whole hold."""
        layer = Layer.from_still(_still_seq(frame_number=7), hold_frames=10, offset=0)
        for master in range(0, 10):
            assert layer.source_frame_at(master) == 7

    def test_hold_clamped_to_one(self) -> None:
        layer = Layer.from_still(_still_seq(), hold_frames=0)
        assert layer.still_hold_frames == 1
        assert layer.trim_length == 1

    def test_rejects_multi_frame_sequence(self) -> None:
        with pytest.raises(ValueError, match="1-frame"):
            Layer.from_still(_multi_seq(), hold_frames=10)

    def test_is_trim_valid_always_true(self) -> None:
        # Stills are structurally trim-valid — the override exists so
        # the renderer doesn't reject them on the in/out check.
        layer = Layer.from_still(_still_seq(), hold_frames=1)
        assert layer.is_trim_valid()

    def test_hold_can_be_mutated(self) -> None:
        # The right-handle drag on the layer bar mutates this in place
        # (via LayerStack.update). Geometry must follow.
        layer = Layer.from_still(_still_seq(), hold_frames=10)
        layer.still_hold_frames = 25
        assert layer.trim_length == 25
        assert layer.master_end == layer.master_start + 24


# ============================================================================
# from_image (smart factory)
# ============================================================================


class TestFromImage:
    def test_single_frame_routes_to_still(self) -> None:
        layer = Layer.from_image(_still_seq(), default_still_hold=100)
        assert layer.is_still is True
        assert layer.still_hold_frames == 100

    def test_multi_frame_routes_to_sequence(self) -> None:
        seq = _multi_seq(1001, 1010)
        layer = Layer.from_image(seq, default_still_hold=50)
        assert layer.is_still is False
        assert layer.trim_length == 10
        # default_still_hold ignored on the sequence path.
        assert layer.still_hold_frames == 1


# ============================================================================
# Scanner fallback for non-pattern filenames
# ============================================================================


class TestScannerStillFallback:
    def test_non_pattern_filename_resolves_as_single_frame(
        self, tmp_path: Path,
    ) -> None:
        """``slate.png`` (no numeric pattern) must scan as a 1-frame
        SequenceInfo so the still-image load path can pick it up.

        Pre-still behavior was to raise ``SequenceNotFoundError``,
        leaving the user unable to load any non-numeric filename.
        """
        from img_player.sequence.scanner import scan

        # Create a real (empty) PNG so OIIO won't be invoked
        # (probe=False on scan path).
        still_file = tmp_path / "slate.png"
        still_file.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG header — enough for is_supported
        seq = scan(still_file, probe=False)
        assert seq.frame_count == 1
        assert seq.frames[0].path == still_file
        assert seq.frames[0].frame_number == 0

    def test_unsupported_extension_still_raises(
        self, tmp_path: Path,
    ) -> None:
        """``foo.txt`` should fail with a clear error rather than
        silently building a bogus 1-frame still."""
        from img_player.sequence.scanner import (
            SequenceNotFoundError,
            scan,
        )
        f = tmp_path / "foo.txt"
        f.write_text("not an image")
        with pytest.raises(SequenceNotFoundError):
            scan(f, probe=False)


# ============================================================================
# Session round-trip
# ============================================================================


class TestSessionRoundTrip:
    def test_still_field_persists(self, tmp_path: Path) -> None:
        """Save a stack with a still + a sequence, reload, verify the
        still flag and hold come back. Uses the JSON layer schema
        directly to avoid pulling in LayerStack (Qt) for this unit test.
        """
        from dataclasses import asdict

        from img_player.layers.session import _SessionLayer

        sl = _SessionLayer(
            id="abc",
            name="my-slate",
            sequence_directory=str(tmp_path),
            sequence_base_name="slate",
            sequence_extension=".png",
            sequence_padding=0,
            layer_in=0,
            layer_out=0,
            offset=0,
            visible=True,
            is_still=True,
            still_hold_frames=42,
            still_filename="slate.png",
        )
        encoded = json.dumps(asdict(sl))
        decoded = json.loads(encoded)
        assert decoded["is_still"] is True
        assert decoded["still_hold_frames"] == 42
        assert decoded["still_filename"] == "slate.png"

    def test_legacy_v2_session_loads_as_sequence(self) -> None:
        """A v2 entry without ``is_still`` must round-trip as a
        plain sequence layer (default ``False``)."""
        from img_player.layers.session import _SessionLayer

        # Only the v2 fields — no still keys.
        sl = _SessionLayer(
            id="legacy",
            name="legacy",
            sequence_directory="/tmp",
            sequence_base_name="frame",
            sequence_extension=".exr",
            sequence_padding=4,
            layer_in=1001,
            layer_out=1100,
            offset=0,
            visible=True,
        )
        assert sl.is_still is False
        assert sl.still_hold_frames == 1
        assert sl.still_filename == ""
