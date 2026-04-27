"""Tests for :mod:`img_player.annotate.persistence`.

Sidecar JSON: atomic save, schema versioning, graceful failures,
basename routing for cohabiting sequences in one directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from img_player.annotate.persistence import (
    SCHEMA_VERSION,
    SIDECAR_FILENAME,
    load_annotations,
    save_annotations,
    sidecar_path,
)
from img_player.annotate.store import AnnotationStore
from img_player.annotate.stroke import Stroke


def _stroke(color: str = "#FF0000", size: float = 5.0) -> Stroke:
    return Stroke(points=((0.0, 0.0), (10.0, 10.0)), color=color, size=size)


# ============================================================================
# sidecar_path
# ============================================================================


class TestSidecarPath:
    def test_sidecar_filename_is_dot_prefixed(self) -> None:
        """Hidden on Linux/macOS, less visible on Windows. Avoid
        cluttering the dossier visually."""
        assert SIDECAR_FILENAME.startswith(".")

    def test_path_lives_in_sequence_dir(self, tmp_path: Path) -> None:
        assert sidecar_path(tmp_path) == tmp_path / SIDECAR_FILENAME


# ============================================================================
# Save (round-trip + atomicity)
# ============================================================================


class TestSave:
    def test_save_creates_file_with_payload(self, tmp_path: Path) -> None:
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        path = sidecar_path(tmp_path)
        assert save_annotations(path, store, basename="render") is True
        assert path.exists()

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["schema_version"] == SCHEMA_VERSION
        assert "saved_at" in data
        assert "img_player_version" in data
        assert "render" in data["sequences"]

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        """Successful save leaves no .tmp file behind."""
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        path = sidecar_path(tmp_path)
        save_annotations(path, store, basename="render")
        assert path.exists()
        assert not (tmp_path / (SIDECAR_FILENAME + ".tmp")).exists()

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Edge case: the sequence dir doesn't exist yet (programmatic
        callers). Should create it gracefully rather than raising."""
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        path = tmp_path / "nested" / "deep" / SIDECAR_FILENAME
        assert save_annotations(path, store, basename="render") is True
        assert path.exists()

    def test_save_preserves_other_basenames_in_existing_file(
        self, tmp_path: Path
    ) -> None:
        """Two sequences sharing a dir must cohabit in one sidecar.
        Saving one must not clobber the other's annotations."""
        path = sidecar_path(tmp_path)
        # First sequence saves.
        store_a = AnnotationStore()
        store_a.add_stroke(10, _stroke(color="#FF0000"))
        save_annotations(path, store_a, basename="render")

        # Second sequence saves.
        store_b = AnnotationStore()
        store_b.add_stroke(20, _stroke(color="#00FF00"))
        save_annotations(path, store_b, basename="playblast")

        # Both basenames present.
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "render" in data["sequences"]
        assert "playblast" in data["sequences"]

    def test_save_overwrites_same_basename(self, tmp_path: Path) -> None:
        """Re-saving the same sequence replaces its previous payload
        entirely — no append. Otherwise removed strokes would
        resurrect across saves."""
        path = sidecar_path(tmp_path)
        store = AnnotationStore()
        store.add_stroke(10, _stroke(color="#FF0000"))
        save_annotations(path, store, basename="render")

        # Same store, but we removed the stroke.
        store.remove_stroke(10, 0)
        save_annotations(path, store, basename="render")

        data = json.loads(path.read_text(encoding="utf-8"))
        # Frame 10 should NOT appear — its only stroke was removed.
        assert "10" not in data["sequences"]["render"]["frames"]

    def test_save_treats_corrupt_existing_as_empty(self, tmp_path: Path) -> None:
        """If the existing sidecar is unreadable JSON, the save
        proceeds and overwrites it (we can't recover the lost
        basenames, but we don't crash either)."""
        path = sidecar_path(tmp_path)
        path.write_text("{ this is not json", encoding="utf-8")

        store = AnnotationStore()
        store.add_stroke(10, _stroke())
        assert save_annotations(path, store, basename="render") is True

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "render" in data["sequences"]


# ============================================================================
# Load
# ============================================================================


class TestLoad:
    def test_round_trip_preserves_strokes(self, tmp_path: Path) -> None:
        path = sidecar_path(tmp_path)
        store = AnnotationStore()
        s1 = _stroke(color="#FF0000")
        s2 = _stroke(color="#00FF00")
        store.add_stroke(42, s1)
        store.add_stroke(42, s2)
        store.add_stroke(87, _stroke(color="#0000FF"))
        save_annotations(path, store, basename="render")

        loaded = load_annotations(path, basename="render")
        assert loaded is not None
        assert loaded.annotated_frames() == frozenset({42, 87})
        assert loaded.strokes_at(42) == (s1, s2)

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        result = load_annotations(
            tmp_path / "does_not_exist.json", basename="render"
        )
        assert result is None

    def test_unknown_basename_returns_none(self, tmp_path: Path) -> None:
        path = sidecar_path(tmp_path)
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        save_annotations(path, store, basename="render")

        # Sequence is in the file under "render", but we ask for
        # "playblast" — the file is fine, just doesn't have what we
        # want for THIS sequence.
        assert load_annotations(path, basename="playblast") is None

    def test_malformed_json_returns_none(self, tmp_path: Path) -> None:
        path = sidecar_path(tmp_path)
        path.write_text("{ this is not json", encoding="utf-8")
        assert load_annotations(path, basename="render") is None

    def test_unknown_schema_version_returns_none(self, tmp_path: Path) -> None:
        """A future schema bump must not crash a current-version
        loader. The user gets an empty store and the file stays
        on disk for the new version to handle."""
        path = sidecar_path(tmp_path)
        path.write_text(
            json.dumps(
                {
                    "schema_version": 999,
                    "sequences": {"render": {"frames": {}}},
                }
            ),
            encoding="utf-8",
        )
        assert load_annotations(path, basename="render") is None

    def test_load_basename_isolation(self, tmp_path: Path) -> None:
        """Loading basename A returns A's strokes only — B's strokes
        in the same file don't leak."""
        path = sidecar_path(tmp_path)

        store_a = AnnotationStore()
        store_a.add_stroke(10, _stroke(color="#FF0000"))
        save_annotations(path, store_a, basename="render")

        store_b = AnnotationStore()
        store_b.add_stroke(20, _stroke(color="#00FF00"))
        save_annotations(path, store_b, basename="playblast")

        loaded_a = load_annotations(path, basename="render")
        loaded_b = load_annotations(path, basename="playblast")

        assert loaded_a is not None and loaded_b is not None
        assert loaded_a.annotated_frames() == frozenset({10})
        assert loaded_b.annotated_frames() == frozenset({20})

    def test_load_skips_malformed_stroke_inside_valid_file(
        self, tmp_path: Path
    ) -> None:
        """A single broken stroke (e.g. invalid color) in an
        otherwise-valid file is dropped, the rest loads."""
        path = sidecar_path(tmp_path)
        path.write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "saved_at": "2026-04-27T00:00:00+00:00",
                    "img_player_version": "0.3.0",
                    "sequences": {
                        "render": {
                            "frames": {
                                "42": [
                                    {
                                        "color": "#FF0000",
                                        "size": 5.0,
                                        "points": [[0, 0], [1, 1]],
                                    },
                                    {
                                        # Bad stroke — invalid color.
                                        "color": "not-a-hex",
                                        "size": 5.0,
                                        "points": [[0, 0]],
                                    },
                                ]
                            }
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        loaded = load_annotations(path, basename="render")
        assert loaded is not None
        assert len(loaded.strokes_at(42)) == 1
