"""Tests for sequence/scanner.py — sequence detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from img_player.sequence.scanner import (
    SequenceNotFoundError,
    scan,
    scan_all,
)


def test_scan_directory_with_one_sequence(sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    assert seq.base_name == "render."
    assert seq.extension == ".png"
    assert seq.padding == 4
    assert seq.frame_count == 10
    assert seq.first_frame == 1
    assert seq.last_frame == 10
    assert seq.is_contiguous


def test_scan_from_file_finds_its_sequence(sequence_dir: Path) -> None:
    a_frame = sequence_dir / "render.0005.png"
    seq = scan(a_frame)
    assert seq.frame_count == 10


def test_scan_detects_gaps(sequence_with_gaps_dir: Path) -> None:
    seq = scan(sequence_with_gaps_dir)
    assert not seq.is_contiguous
    assert seq.missing_frames == (2, 4, 6, 7)
    assert seq.frame_count == 4


def test_scan_all_returns_multiple_sorted_by_size(mixed_sequences_dir: Path) -> None:
    sequences = scan_all(mixed_sequences_dir)
    assert len(sequences) == 2
    assert sequences[0].frame_count == 5  # the "big" one first
    assert sequences[1].frame_count == 2


def test_scan_dir_returns_largest(mixed_sequences_dir: Path) -> None:
    seq = scan(mixed_sequences_dir)
    assert seq.frame_count == 5
    assert seq.base_name == "big."


def test_scan_populates_header_info(sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    assert seq.width == 16
    assert seq.height == 16
    assert len(seq.channel_names) == 4


def test_scan_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(SequenceNotFoundError):
        scan(tmp_path / "nope")


def test_scan_file_not_a_frame_raises(tmp_path: Path) -> None:
    standalone = tmp_path / "nothing.png"
    standalone.write_bytes(b"")
    with pytest.raises(SequenceNotFoundError):
        scan(standalone)


def test_scan_empty_directory_raises(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SequenceNotFoundError):
        scan(empty)


def test_display_pattern(sequence_dir: Path) -> None:
    seq = scan(sequence_dir)
    assert seq.display_pattern() == "render.####.png"
