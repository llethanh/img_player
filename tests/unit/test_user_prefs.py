"""Tests for :mod:`img_player.user_prefs` and the layered Preferences
resolution it powers (user TOML > site TOML > hardcoded)."""

from __future__ import annotations

from pathlib import Path

import pytest

from img_player import site_config as sc
from img_player import user_prefs as up


# ============================================================================
# UserPrefsStore — get / set / atomic write
# ============================================================================


class TestUserPrefsStore:
    def test_fresh_store_returns_default(self, tmp_path: Path) -> None:
        store = up.UserPrefsStore(tmp_path / "flick.toml")
        assert store.get("color.ocio_builtin_uri", "fb") == "fb"
        assert store.exists() is False

    def test_set_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "flick.toml"
        store = up.UserPrefsStore(path)
        store.set("color.ocio_builtin_uri", "ocio://x")
        assert path.is_file(), "user toml should be created on first set"
        assert store.exists() is True

    def test_set_then_get_round_trip(self, tmp_path: Path) -> None:
        store = up.UserPrefsStore(tmp_path / "flick.toml")
        store.set("color.ocio_builtin_uri", "ocio://aces-1.3")
        store.set("disk_cache.enabled", True)
        store.set("disk_cache.budget_gb", 200)
        store.set("disk_cache.path", "D:/cache")
        assert store.get("color.ocio_builtin_uri") == "ocio://aces-1.3"
        assert store.get("disk_cache.enabled") is True
        assert store.get("disk_cache.budget_gb") == 200
        assert store.get("disk_cache.path") == "D:/cache"

    def test_set_persists_across_instances(self, tmp_path: Path) -> None:
        """A fresh store pointed at the same file must see the prior
        writes — that's the whole point of file-backed prefs."""
        path = tmp_path / "flick.toml"
        s1 = up.UserPrefsStore(path)
        s1.set("color.ocio_builtin_uri", "ocio://persistent")
        s2 = up.UserPrefsStore(path)
        assert s2.get("color.ocio_builtin_uri") == "ocio://persistent"

    def test_remove_drops_key(self, tmp_path: Path) -> None:
        store = up.UserPrefsStore(tmp_path / "flick.toml")
        store.set("color.ocio_builtin_uri", "ocio://x")
        store.remove("color.ocio_builtin_uri")
        assert store.get("color.ocio_builtin_uri", "fb") == "fb"

    def test_remove_drops_empty_section(self, tmp_path: Path) -> None:
        """When the last key in a section is removed, the section
        header should not survive — keeps the file tidy."""
        path = tmp_path / "flick.toml"
        store = up.UserPrefsStore(path)
        store.set("color.ocio_builtin_uri", "ocio://x")
        store.remove("color.ocio_builtin_uri")
        # Reading back should NOT contain "[color]"
        text = path.read_text(encoding="utf-8")
        assert "[color]" not in text

    def test_atomic_write_uses_tmp_rename(self, tmp_path: Path) -> None:
        """No partial files should linger after a normal write."""
        store = up.UserPrefsStore(tmp_path / "flick.toml")
        store.set("color.ocio_builtin_uri", "ocio://x")
        # Only the target file should remain.
        leftovers = [p.name for p in tmp_path.iterdir() if p.name != "flick.toml"]
        assert leftovers == [], f"leftover temp files: {leftovers}"

    def test_set_writes_valid_toml(self, tmp_path: Path) -> None:
        """The file produced must be parseable by tomllib."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        path = tmp_path / "flick.toml"
        store = up.UserPrefsStore(path)
        store.set("color.ocio_builtin_uri", 'ocio://has"quote-and\\backslash')
        store.set("disk_cache.enabled", False)
        with path.open("rb") as fh:
            parsed = tomllib.load(fh)
        assert parsed["color"]["ocio_builtin_uri"] == 'ocio://has"quote-and\\backslash'
        assert parsed["disk_cache"]["enabled"] is False

    def test_malformed_file_yields_empty_store_with_file_left_intact(
        self, tmp_path: Path,
    ) -> None:
        path = tmp_path / "flick.toml"
        path.write_text("not = valid = toml\n", encoding="utf-8")
        store = up.UserPrefsStore(path)
        assert store.get("anything", "fb") == "fb"
        # File should still exist for the user to inspect / fix.
        assert path.is_file()


# ============================================================================
# Layered resolution through Preferences
# ============================================================================


@pytest.fixture
def isolated_qsettings(monkeypatch: pytest.MonkeyPatch, request):
    """Give each test its own QSettings scope so leaks from a previous
    test (or the real user profile) can't influence the result."""
    from PySide6.QtCore import QCoreApplication, QSettings

    suffix = request.node.name
    QCoreApplication.setOrganizationName(f"flick-test-{suffix[:30]}")
    QCoreApplication.setApplicationName(f"flick-test-{suffix[:30]}")
    QSettings().clear()
    yield
    QSettings().clear()


@pytest.fixture
def isolated_stores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Force user-prefs + site-config singletons to point at tmp_path
    so cross-test contamination is impossible.

    Skips the legacy-QSettings migration too — Preferences() in test
    context shouldn't try to read the dev's real registry values into
    our hermetic tmp store. Also forces the SITE config to a
    guaranteed-empty state so a real ``flick.toml`` a developer may
    have at the repo root doesn't bleed into the test layered
    resolution.
    """
    user_path = tmp_path / "user_flick.toml"
    monkeypatch.setattr(
        up, "_cached", up.UserPrefsStore(user_path), raising=False,
    )
    # Pre-cache an empty SiteConfig so ``site_config()`` never resolves
    # against the real filesystem during the test.
    monkeypatch.setattr(sc, "_cached", sc.SiteConfig({}), raising=False)
    import img_player.preferences as P
    # Latch migration as already done so Preferences() doesn't try
    # to read the dev's actual QSettings into our test tmp store.
    monkeypatch.setattr(P, "_legacy_migration_done", True, raising=False)
    yield {"user_path": user_path}
    monkeypatch.setattr(up, "_cached", None, raising=False)
    monkeypatch.setattr(sc, "_cached", None, raising=False)


class TestLayeredPreferences:
    def test_hardcoded_default_when_no_overrides(
        self, qtbot, isolated_qsettings, isolated_stores,
    ) -> None:
        from img_player.preferences import Preferences

        # Shipped default ACES 1.3 CG.
        assert Preferences().ocio_builtin_uri == (
            "ocio://cg-config-v2.2.0_aces-v1.3_ocio-v2.4"
        )

    def test_site_overrides_hardcoded(
        self, qtbot, isolated_qsettings, isolated_stores, tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        site_toml = tmp_path / "site.toml"
        site_toml.write_text(
            '[color]\nocio_builtin_uri = "ocio://from-site"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("FLICK_SITE_CONFIG", str(site_toml))
        sc.invalidate_cache()

        from img_player.preferences import Preferences

        assert Preferences().ocio_builtin_uri == "ocio://from-site"

    def test_user_overrides_site(
        self, qtbot, isolated_qsettings, isolated_stores, tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        site_toml = tmp_path / "site.toml"
        site_toml.write_text(
            '[color]\nocio_builtin_uri = "ocio://from-site"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("FLICK_SITE_CONFIG", str(site_toml))
        sc.invalidate_cache()

        # Write a user override.
        up._cached.set("color.ocio_builtin_uri", "ocio://from-user")

        from img_player.preferences import Preferences

        assert Preferences().ocio_builtin_uri == "ocio://from-user"

    def test_setter_writes_to_user_toml_only(
        self, qtbot, isolated_qsettings, isolated_stores,
    ) -> None:
        from img_player.preferences import Preferences

        prefs = Preferences()
        prefs.ocio_builtin_uri = "ocio://user-pick"
        # The file should exist now.
        store = up._cached
        assert store.exists()
        # And it should hold the value we just set.
        assert store.get("color.ocio_builtin_uri") == "ocio://user-pick"

    def test_disk_cache_round_trip(
        self, qtbot, isolated_qsettings, isolated_stores,
    ) -> None:
        """Sanity: every disk-cache pref round-trips through the user
        TOML — exercises ``_set_user_pref`` + ``_layered_default``
        for booleans, ints, paths, and strings in one go."""
        from img_player.preferences import Preferences

        prefs = Preferences()
        prefs.disk_cache_enabled = False
        prefs.disk_cache_budget_gb = 200
        prefs.disk_cache_compression = False
        prefs.disk_cache_path = "D:/scratch/flick"

        prefs2 = Preferences()
        assert prefs2.disk_cache_enabled is False
        assert prefs2.disk_cache_budget_gb == 200
        assert prefs2.disk_cache_compression is False
        # Compare via Path equality so the test passes on Windows where
        # ``str(Path("D:/x"))`` normalises forward slashes to backslashes.
        assert prefs2.disk_cache_path == Path("D:/scratch/flick")

    def test_unset_user_value_falls_back_to_site(
        self, qtbot, isolated_qsettings, isolated_stores, tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        site_toml = tmp_path / "site.toml"
        site_toml.write_text(
            '[disk_cache]\nbudget_gb = 99\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("FLICK_SITE_CONFIG", str(site_toml))
        sc.invalidate_cache()

        from img_player.preferences import Preferences

        # Set then unset (= remove from user TOML) → falls back to site.
        prefs = Preferences()
        prefs.disk_cache_budget_gb = 42
        assert Preferences().disk_cache_budget_gb == 42
        up._cached.remove("disk_cache.budget_gb")
        assert Preferences().disk_cache_budget_gb == 99

    def test_color_management_apply_no_op_when_no_change(
        self, qtbot, isolated_qsettings, isolated_stores, tmp_path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Regression test: opening Preferences > Color Management and
        clicking Apply without changing any field must NOT create a
        user TOML. The bug we shipped in v1.5.6..v1.5.11 called every
        setter unconditionally inside ``apply()``, materialising the
        site-config defaults into a user-level override on disk and
        making the next launch ignore subsequent site-config updates.
        """
        # Site config supplies a non-default URI so we can verify
        # the dropdown initialises to it via the layered resolution.
        site_toml = tmp_path / "site.toml"
        site_toml.write_text(
            '[color]\n'
            'ocio_builtin_uri = "ocio://studio-config-v2.2.0_aces-v1.3_ocio-v2.4"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("FLICK_SITE_CONFIG", str(site_toml))
        sc.invalidate_cache()

        from img_player.preferences import Preferences
        from img_player.ui.preferences_dialog import _ColorManagementPage

        page = _ColorManagementPage(Preferences(), on_reload=None)
        try:
            # User opened the dialog, did nothing, clicks Apply.
            assert page.apply() is False, (
                "apply() should report no-change when fields are pristine"
            )
            # User TOML must NOT have been created — the headline bug.
            assert not up._cached.exists(), (
                "Apply with no edits must not materialise the user TOML"
            )
        finally:
            page.deleteLater()
