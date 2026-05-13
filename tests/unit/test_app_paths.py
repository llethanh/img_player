"""Tests for :mod:`img_player.app_paths` — canonical app-data paths
+ the v1.5.9 ``img_player`` → ``FlickPlayer`` directory migration."""

from __future__ import annotations

from pathlib import Path

import pytest

from img_player import app_paths


@pytest.fixture(autouse=True)
def _reset_migration_latch(monkeypatch: pytest.MonkeyPatch):
    """Each test wants a fresh ``_migrated_once`` latch."""
    monkeypatch.setattr(app_paths, "_migrated_once", False, raising=False)
    yield
    monkeypatch.setattr(app_paths, "_migrated_once", False, raising=False)


class TestPathConstants:
    def test_app_dir_name_is_flickplayer(self) -> None:
        assert app_paths.APP_DIR_NAME == "FlickPlayer"

    def test_legacy_name_still_img_player(self) -> None:
        assert app_paths._LEGACY_APP_DIR_NAME == "img_player"

    def test_user_prefs_dir_ends_in_canonical_name(self) -> None:
        assert app_paths.user_prefs_dir().name == "FlickPlayer"

    def test_disk_cache_dir_under_canonical_name(self) -> None:
        path = app_paths.disk_cache_default_dir()
        assert path.name == "disk_cache"
        assert path.parent.name == "FlickPlayer"

    def test_calibration_profile_under_canonical_name(self) -> None:
        path = app_paths.calibration_profile_path()
        assert path.name == "profile.json"
        assert path.parent.name == "FlickPlayer"


class TestLegacyMigration:
    """The one-shot rename ``img_player`` → ``FlickPlayer``.

    We can't easily monkeypatch ``%APPDATA%`` from Python because
    Windows resolves that env var at the C runtime level. Instead we
    test ``_migrate_one`` directly — it's the unit that contains the
    interesting logic; the outer ``migrate_legacy_dirs_once`` just
    pairs it with the appdata roots."""

    def test_renames_legacy_when_current_absent(self, tmp_path: Path) -> None:
        legacy = tmp_path / "img_player"
        current = tmp_path / "FlickPlayer"
        legacy.mkdir()
        (legacy / "marker.txt").write_text("hello", encoding="utf-8")

        renamed = app_paths._migrate_one(legacy, current)

        assert renamed is True
        assert not legacy.exists()
        assert current.is_dir()
        assert (current / "marker.txt").read_text(encoding="utf-8") == "hello"

    def test_does_not_clobber_existing_current(self, tmp_path: Path) -> None:
        legacy = tmp_path / "img_player"
        current = tmp_path / "FlickPlayer"
        legacy.mkdir()
        current.mkdir()
        (legacy / "old.txt").write_text("OLD", encoding="utf-8")
        (current / "new.txt").write_text("NEW", encoding="utf-8")

        renamed = app_paths._migrate_one(legacy, current)

        # Should bail out — both dirs left as-is, no auto-merge.
        assert renamed is False
        assert legacy.is_dir()
        assert current.is_dir()
        assert (legacy / "old.txt").read_text(encoding="utf-8") == "OLD"
        assert (current / "new.txt").read_text(encoding="utf-8") == "NEW"

    def test_noop_when_legacy_absent(self, tmp_path: Path) -> None:
        legacy = tmp_path / "img_player"
        current = tmp_path / "FlickPlayer"
        renamed = app_paths._migrate_one(legacy, current)
        assert renamed is False
        assert not current.exists()  # not created defensively either

    def test_migrate_runs_only_once(self) -> None:
        """``migrate_legacy_dirs_once`` is idempotent — second call
        is a guaranteed no-op."""
        app_paths.migrate_legacy_dirs_once()
        # Mutate the latch is internal; verify by checking that a
        # second call returns immediately (no side effect we can
        # observe directly without a temp dir, so this just checks
        # it doesn't raise).
        app_paths.migrate_legacy_dirs_once()
        # And the latch is set.
        assert app_paths._migrated_once is True
