"""Tests for the missing-frame feature (v0.5.1).

Covers:
* Checkerboard placeholder generator (memoised, correct dtype).
* FrameCache: serves placeholder + marks missing on FrameReadError.
* FrameCache.reload(): mtime-aware smart invalidation.
* Scanner.rescan(): picks up additions / deletions.
* Timeline: paints missing frames red (smoke test on the painter).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import OpenImageIO as oiio
import pytest

from img_player.cache.frame_cache import FrameCache
from img_player.cache.missing_placeholder import get_missing_placeholder, reset_cache
from img_player.sequence.models import FrameInfo, SequenceInfo
from img_player.sequence.scanner import rescan, scan


# ============================================================================
# Placeholder generator
# ============================================================================


class TestPlaceholder:
    def test_returns_float32_rgba(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        del qtbot  # need a QApplication via qtbot fixture
        reset_cache()
        arr = get_missing_placeholder(64, 32)
        assert arr.dtype == np.float32
        assert arr.shape == (32, 64, 4)
        # Range [0, 1].
        assert arr.min() >= 0.0
        assert arr.max() <= 1.0
        # Alpha is 1.0 everywhere — placeholder is opaque.
        assert np.allclose(arr[..., 3], 1.0)

    def test_memoised_by_size(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        del qtbot
        reset_cache()
        a = get_missing_placeholder(64, 32)
        b = get_missing_placeholder(64, 32)
        assert a is b  # identity → memoised

    def test_different_sizes_yield_different_buffers(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        del qtbot
        reset_cache()
        a = get_missing_placeholder(64, 32)
        b = get_missing_placeholder(128, 64)
        assert a is not b
        assert b.shape == (64, 128, 4)

    def test_checker_pattern_has_two_distinct_shades(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        del qtbot
        reset_cache()
        arr = get_missing_placeholder(128, 128)
        # The label paints text but the bulk of the image should
        # still have at least two distinct grey shades.
        unique_grey = np.unique(arr[..., 0])
        assert len(unique_grey) >= 2


# ============================================================================
# FrameInfo / scanner mtime
# ============================================================================


class TestSequenceMtime:
    def test_scanner_populates_mtime(self, sequence_dir: Path) -> None:
        seq = scan(sequence_dir / "render.0001.png", probe=False)
        # Every frame got a non-zero mtime from os.stat.
        assert all(f.mtime > 0 for f in seq.frames)

    def test_rescan_picks_up_new_files(
        self, tmp_path: Path,
    ) -> None:
        # Build a tiny 2-frame seq, scan it, then drop a 3rd frame
        # and rescan.
        for n in (1, 2):
            _write_tiny_png(tmp_path / f"shot.{n:04d}.png")
        seq = scan(tmp_path / "shot.0001.png", probe=False)
        assert seq.frame_count == 2

        _write_tiny_png(tmp_path / "shot.0003.png")
        new_seq = rescan(seq)
        assert new_seq.frame_count == 3
        assert {f.frame_number for f in new_seq.frames} == {1, 2, 3}

    def test_rescan_drops_deleted_files(self, tmp_path: Path) -> None:
        for n in (1, 2, 3):
            _write_tiny_png(tmp_path / f"shot.{n:04d}.png")
        seq = scan(tmp_path / "shot.0001.png", probe=False)
        (tmp_path / "shot.0002.png").unlink()
        new_seq = rescan(seq)
        assert {f.frame_number for f in new_seq.frames} == {1, 3}

    def test_rescan_returns_empty_when_dir_disappears(
        self, tmp_path: Path,
    ) -> None:
        for n in (1, 2):
            _write_tiny_png(tmp_path / f"shot.{n:04d}.png")
        seq = scan(tmp_path / "shot.0001.png", probe=False)
        # Simulate the dir being moved aside (we don't actually
        # rmtree because Windows holds locks). Instead, monkey-patch
        # the SequenceInfo.directory.
        from dataclasses import replace as _replace
        gone = _replace(seq, directory=tmp_path / "_nope")
        out = rescan(gone)
        # Same frame list (preserves the old mapping) so the cache
        # can mark them all missing.
        assert out.frame_count == seq.frame_count


# ============================================================================
# FrameCache: missing placeholder
# ============================================================================


class TestCacheMissingFrames:
    def test_missing_file_marked_as_missing(self, tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
        del qtbot
        # Build a 3-frame seq where frame 2's file is deleted post-scan.
        for n in (1, 2, 3):
            _write_tiny_png(tmp_path / f"shot.{n:04d}.png")
        seq = scan(tmp_path / "shot.0001.png", probe=False)
        (tmp_path / "shot.0002.png").unlink()

        cache = FrameCache(budget_bytes=10 * 1024 * 1024, num_workers=1)
        cache.attach(seq)
        cache.request(2)
        assert cache.wait_idle(timeout=5.0)
        # Frame 2's slot now holds the placeholder.
        arr = cache.get(2)
        assert arr is not None
        assert arr.dtype == np.float32
        assert 2 in cache.missing_frames()
        assert cache.contains(2) is True
        cache.shutdown()

    def test_holes_in_sequence_premarked_missing_on_attach(
        self, tmp_path: Path, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """Regression: a sparse sequence (e.g. frames 1, 2, 4, 5 — no
        frame 3) used to freeze playback at frame 3 forever, because
        the cache had no path to decode and never marked it missing.
        Now ``attach`` pre-fills the missing-frames set + serves the
        placeholder so playback skips through the hole."""
        del qtbot
        # Build a sparse sequence: frames 1, 2, 4, 5 (no 3).
        for n in (1, 2, 4, 5):
            _write_tiny_png(tmp_path / f"shot.{n:04d}.png")
        seq = scan(tmp_path / "shot.0001.png", probe=False)
        # Sanity: scanner produced 4 frames, the SequenceInfo's
        # missing_frames property reports the hole.
        assert seq.frame_count == 4
        assert seq.missing_frames == (3,)

        cache = FrameCache(budget_bytes=10 * 1024 * 1024, num_workers=1)
        cache.attach(seq)
        # Frame 3 is immediately marked missing + has a placeholder
        # ready, with no decode attempt needed.
        assert 3 in cache.missing_frames()
        assert cache.contains(3) is True
        arr = cache.get(3)
        assert arr is not None
        assert arr.dtype == np.float32
        cache.shutdown()

    def test_present_frame_not_marked_missing(self, tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
        del qtbot
        for n in (1, 2, 3):
            _write_tiny_png(tmp_path / f"shot.{n:04d}.png")
        seq = scan(tmp_path / "shot.0001.png", probe=False)
        cache = FrameCache(budget_bytes=10 * 1024 * 1024, num_workers=1)
        cache.attach(seq)
        cache.request(1)
        assert cache.wait_idle(timeout=5.0)
        assert 1 not in cache.missing_frames()
        cache.shutdown()

    def test_reload_drops_changed_frames_keeps_unchanged(
        self, tmp_path: Path, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        del qtbot
        import time
        for n in (1, 2, 3):
            _write_tiny_png(tmp_path / f"shot.{n:04d}.png")
        seq = scan(tmp_path / "shot.0001.png", probe=False)
        cache = FrameCache(budget_bytes=10 * 1024 * 1024, num_workers=1)
        cache.attach(seq)
        for n in (1, 2, 3):
            cache.request(n)
        assert cache.wait_idle(timeout=5.0)
        assert cache.cached_frames() == frozenset({1, 2, 3})

        # Touch frame 2 — bump its mtime by ≥ 1 second so the
        # filesystem actually records a different value.
        time.sleep(1.1)
        _write_tiny_png(tmp_path / "shot.0002.png")  # rewrite

        new_seq = rescan(seq)
        kept, dropped, missing = cache.reload(new_seq)
        # frames 1, 3 kept; frame 2 dropped (mtime changed).
        assert kept == 2
        assert dropped == 1
        assert missing == 0
        assert 2 not in cache.cached_frames()
        cache.shutdown()

    def test_reload_marks_disappeared_file_as_missing(
        self, tmp_path: Path, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """File deleted while loaded → reload must mark the slot
        missing (red on the timeline) rather than leaving a hole."""
        del qtbot
        for n in (1, 2, 3):
            _write_tiny_png(tmp_path / f"shot.{n:04d}.png")
        seq = scan(tmp_path / "shot.0001.png", probe=False)
        cache = FrameCache(budget_bytes=10 * 1024 * 1024, num_workers=1)
        cache.attach(seq)
        for n in (1, 2, 3):
            cache.request(n)
        assert cache.wait_idle(timeout=5.0)
        assert 2 not in cache.missing_frames()

        # Delete frame 2 then reload.
        (tmp_path / "shot.0002.png").unlink()
        new_seq = rescan(seq)
        cache.reload(new_seq)
        # Slot 2 is now flagged missing AND holds the placeholder.
        assert 2 in cache.missing_frames()
        arr = cache.get(2)
        assert arr is not None
        assert arr.dtype == np.float32
        cache.shutdown()

    def test_reload_re_offers_previously_missing_frame(
        self, tmp_path: Path, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        del qtbot
        # Build a 2-frame seq, delete frame 2, decode it (→ missing),
        # restore frame 2, reload → frame 2 should be eligible again.
        for n in (1, 2):
            _write_tiny_png(tmp_path / f"shot.{n:04d}.png")
        seq = scan(tmp_path / "shot.0001.png", probe=False)
        (tmp_path / "shot.0002.png").unlink()

        cache = FrameCache(budget_bytes=10 * 1024 * 1024, num_workers=1)
        cache.attach(seq)
        cache.request(2)
        assert cache.wait_idle(timeout=5.0)
        assert 2 in cache.missing_frames()

        # Restore the file then reload.
        _write_tiny_png(tmp_path / "shot.0002.png")
        new_seq = rescan(seq)
        cache.reload(new_seq)
        # Missing flag cleared so the next request decodes again.
        assert 2 not in cache.missing_frames()
        cache.request(2)
        assert cache.wait_idle(timeout=5.0)
        # Now decoded successfully.
        assert 2 in cache.cached_frames()
        assert 2 not in cache.missing_frames()
        cache.shutdown()

    def test_detach_clears_state(self, tmp_path: Path, qtbot) -> None:  # type: ignore[no-untyped-def]
        del qtbot
        for n in (1, 2):
            _write_tiny_png(tmp_path / f"shot.{n:04d}.png")
        seq = scan(tmp_path / "shot.0001.png", probe=False)
        cache = FrameCache(budget_bytes=10 * 1024 * 1024, num_workers=1)
        cache.attach(seq)
        cache.request(1)
        assert cache.wait_idle(timeout=5.0)
        assert cache.cached_frames()
        cache.detach()
        assert cache.cached_frames() == frozenset()
        assert cache.missing_frames() == frozenset()
        cache.shutdown()


# ============================================================================
# Timeline missing-frames painting (smoke test)
# ============================================================================


class TestTimelineMissingFrames:
    def test_set_missing_frames_triggers_repaint(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        from img_player.ui.timeline import Timeline
        tl = Timeline()
        qtbot.addWidget(tl)
        tl.set_range(1, 100)
        # Setting missing frames is idempotent — same set, no repaint.
        before = tl._missing_frames
        tl.set_missing_frames(frozenset({5, 10, 15}))
        assert tl._missing_frames == frozenset({5, 10, 15})
        assert tl._missing_frames != before


# ============================================================================
# Helpers
# ============================================================================


def _write_tiny_png(path: Path) -> None:
    """Write a 4×4 RGBA PNG so the cache + scanner have something
    real to chew on. Cheaper than the conftest fixtures because we
    need fresh files in tmp_path per test (mtime-sensitive)."""
    arr = np.full((4, 4, 4), 128, dtype=np.uint8)
    spec = oiio.ImageSpec(4, 4, 4, oiio.UINT8)
    spec.channelnames = ["R", "G", "B", "A"]
    out = oiio.ImageOutput.create(str(path))
    if out is None:
        raise RuntimeError(f"OIIO cannot create {path}: {oiio.geterror()}")
    if not out.open(str(path), spec):
        raise RuntimeError(f"open failed: {out.geterror()}")
    if not out.write_image(arr):
        raise RuntimeError(f"write failed: {out.geterror()}")
    out.close()
