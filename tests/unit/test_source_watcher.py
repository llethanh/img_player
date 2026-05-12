"""Tests for :class:`img_player.sequence.source_watcher.SourceWatcher`.

Covers the auto-reload trigger (E3):

  * ``set_watched_layers`` diffs against the live set, only the
    delta is forwarded to QFileSystemWatcher.
  * Debounce coalesces a burst of OS file events into one signal.
  * ``stop()`` releases handles and cancels pending debounces.
  * Non-existent directories are filtered out silently.

Uses pytest-qt for the event loop. ``qtbot.wait_signal`` blocks until
the watcher's ``sources_changed`` fires or times out.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import QTimer

from img_player.sequence.source_watcher import SourceWatcher


class _FakeSequence:
    def __init__(self, directory: Path) -> None:
        self.directory = directory


class _FakeLayer:
    def __init__(self, directory: Path) -> None:
        self.sequence = _FakeSequence(directory)


# ============================================================================
# set_watched_layers diffing
# ============================================================================


class TestWatchedLayers:
    def test_initial_empty(self, qtbot) -> None:
        w = SourceWatcher()
        try:
            assert w.watched_dirs() == ()
        finally:
            w.stop()

    def test_add_single_dir(self, qtbot, tmp_path: Path) -> None:
        w = SourceWatcher()
        try:
            d = tmp_path / "layerA"
            d.mkdir()
            w.set_watched_layers([_FakeLayer(d)])
            assert w.watched_dirs() == (str(d),)
        finally:
            w.stop()

    def test_duplicate_dirs_deduplicated(self, qtbot, tmp_path: Path) -> None:
        """Two layers in the same folder must register a single watch
        (the OS handle is shared)."""
        w = SourceWatcher()
        try:
            d = tmp_path / "shared"
            d.mkdir()
            w.set_watched_layers([_FakeLayer(d), _FakeLayer(d)])
            assert len(w.watched_dirs()) == 1
        finally:
            w.stop()

    def test_diff_removes_stale_dirs(self, qtbot, tmp_path: Path) -> None:
        w = SourceWatcher()
        try:
            d1 = tmp_path / "a"
            d2 = tmp_path / "b"
            d1.mkdir()
            d2.mkdir()
            w.set_watched_layers([_FakeLayer(d1), _FakeLayer(d2)])
            assert set(w.watched_dirs()) == {str(d1), str(d2)}
            # Drop the second layer
            w.set_watched_layers([_FakeLayer(d1)])
            assert w.watched_dirs() == (str(d1),)
        finally:
            w.stop()

    def test_non_existent_dir_silently_skipped(self, qtbot, tmp_path: Path) -> None:
        """A layer pointing to a deleted folder must not crash the
        watcher — it just doesn't register that path."""
        w = SourceWatcher()
        try:
            phantom = tmp_path / "does_not_exist"
            w.set_watched_layers([_FakeLayer(phantom)])
            assert w.watched_dirs() == ()
        finally:
            w.stop()

    def test_layer_without_sequence_skipped(self, qtbot) -> None:
        """Defensive against half-initialised Layer objects (e.g. a
        layer torn down mid-removal)."""
        w = SourceWatcher()
        try:
            class _BareLayer:
                sequence = None

            # Should not raise.
            w.set_watched_layers([_BareLayer()])
            assert w.watched_dirs() == ()
        finally:
            w.stop()


# ============================================================================
# Debounce behaviour
# ============================================================================


class TestDebounce:
    def test_single_change_fires_signal(self, qtbot, tmp_path: Path) -> None:
        d = tmp_path / "watched"
        d.mkdir()
        # Seed one file so the dir is non-empty (Qt sometimes ignores
        # changes on empty dirs on Windows).
        (d / "seed.txt").write_text("seed")
        w = SourceWatcher()
        try:
            w.set_watched_layers([_FakeLayer(d)])
            with qtbot.wait_signal(w.sources_changed, timeout=2000):
                # Trigger a directory change.
                (d / "frame_0001.exr").write_text("frame")
        finally:
            w.stop()

    def test_burst_coalesces_to_one_signal(self, qtbot, tmp_path: Path) -> None:
        """Re-renders write many files in a burst; the watcher must
        fire exactly once after the 200 ms debounce expires."""
        d = tmp_path / "burst"
        d.mkdir()
        (d / "seed.txt").write_text("seed")
        w = SourceWatcher()
        try:
            w.set_watched_layers([_FakeLayer(d)])

            fires = {"count": 0}

            def on_fired() -> None:
                fires["count"] += 1

            w.sources_changed.connect(on_fired)

            # Schedule a burst of writes.
            def burst() -> None:
                for i in range(10):
                    (d / f"frame_{i:04d}.exr").write_text(f"frame{i}")

            QTimer.singleShot(0, burst)

            # Wait for at least one signal then drain the event loop
            # well past the debounce window so any further fires would
            # also land.
            with qtbot.wait_signal(w.sources_changed, timeout=2000):
                pass
            # Give the debounce timer plenty of post-burst breathing
            # room (200 ms debounce + safety margin).
            qtbot.wait(500)

            assert fires["count"] == 1, (
                f"expected exactly 1 signal after burst, got {fires['count']}"
            )
        finally:
            w.stop()

    def test_no_signal_after_stop(self, qtbot, tmp_path: Path) -> None:
        d = tmp_path / "stopped"
        d.mkdir()
        (d / "seed.txt").write_text("seed")
        w = SourceWatcher()
        try:
            w.set_watched_layers([_FakeLayer(d)])
            fires = {"count": 0}
            w.sources_changed.connect(lambda: fires.__setitem__(
                "count", fires["count"] + 1
            ))
            w.stop()
            # Mutations on the (no-longer-watched) dir should be silent.
            (d / "post.txt").write_text("noise")
            qtbot.wait(500)
            assert fires["count"] == 0
        finally:
            w.stop()  # idempotent
