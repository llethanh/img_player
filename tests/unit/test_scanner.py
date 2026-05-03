"""Tests for sequence/scanner.py — sequence detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from img_player.sequence.scanner import (
    FolderGroup,
    SequenceNotFoundError,
    scan,
    scan_all,
    scan_paths,
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


# ---------------------------------------------------------------------------
# scan_paths — multi-source aggregation for the grouped picker
# ---------------------------------------------------------------------------


def test_scan_paths_groups_per_folder(
    sequence_dir: Path, mixed_sequences_dir: Path,
) -> None:
    groups = scan_paths([sequence_dir, mixed_sequences_dir])
    # One group per folder, no loose group, sorted alphabetically by name.
    assert all(isinstance(g, FolderGroup) for g in groups)
    assert all(g.folder is not None for g in groups)
    folders = [g.folder.name for g in groups]
    assert folders == sorted(folders, key=str.lower)
    # Mixed folder yields its 2 sequences (big + small); seq_dir yields 1.
    counts = {g.folder.name: len(g.sequences) for g in groups}
    assert counts[mixed_sequences_dir.name] == 2
    assert counts[sequence_dir.name] == 1


def test_scan_paths_loose_files_at_root(sequence_dir: Path) -> None:
    a_frame = sequence_dir / "render.0005.png"
    groups = scan_paths([a_frame])
    assert len(groups) == 1
    # Loose group: folder=None, single resolved sequence.
    assert groups[0].folder is None
    assert len(groups[0].sequences) == 1
    assert groups[0].sequences[0].frame_count == 10


def test_scan_paths_empty_folder_marked(tmp_path: Path) -> None:
    empty = tmp_path / "empty_folder"
    empty.mkdir()
    groups = scan_paths([empty])
    assert len(groups) == 1
    assert groups[0].folder == empty
    assert groups[0].empty is True
    assert groups[0].sequences == ()


def test_scan_paths_dedups_loose_pointing_to_same_seq(sequence_dir: Path) -> None:
    a = sequence_dir / "render.0001.png"
    b = sequence_dir / "render.0002.png"
    groups = scan_paths([a, b])
    # Both files belong to the same sequence — should produce ONE
    # entry under the loose group, not two.
    assert len(groups) == 1
    assert groups[0].folder is None
    assert len(groups[0].sequences) == 1


def test_scan_paths_mixed_loose_and_folder(
    sequence_dir: Path, mixed_sequences_dir: Path,
) -> None:
    # Drop a raw file from sequence_dir + the mixed_sequences_dir folder.
    raw = sequence_dir / "render.0001.png"
    groups = scan_paths([raw, mixed_sequences_dir])
    # Loose group always comes first, folder group(s) follow.
    assert groups[0].folder is None
    assert all(g.folder is not None for g in groups[1:])
    assert len(groups[0].sequences) == 1
