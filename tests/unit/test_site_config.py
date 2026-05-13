"""Tests for :mod:`img_player.site_config`.

Covers the TOML parsing + dotted-key lookup, the file-resolution
order ($FLICK_SITE_CONFIG > frozen-bundle path > repo root), and the
integration with :class:`Preferences` (site value supplies the
QSettings default when the key is not set).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from img_player import site_config as sc


# ============================================================================
# SiteConfig — direct construction
# ============================================================================


class TestSiteConfigClass:
    def test_empty_get_returns_default(self) -> None:
        c = sc.SiteConfig({})
        assert c.get("any.thing", "fallback") == "fallback"
        assert c.is_empty is True

    def test_flat_lookup(self) -> None:
        c = sc.SiteConfig({"top": "val"})
        assert c.get("top") == "val"
        assert c.is_empty is False

    def test_nested_dotted_lookup(self) -> None:
        c = sc.SiteConfig({"color": {"ocio_builtin_uri": "ocio://x"}})
        assert c.get("color.ocio_builtin_uri") == "ocio://x"

    def test_missing_intermediate_returns_default(self) -> None:
        c = sc.SiteConfig({"color": {}})
        assert c.get("color.ocio_builtin_uri", "fb") == "fb"

    def test_malformed_intermediate_returns_default(self) -> None:
        # Studio writes ``[color]\nocio_builtin_uri = "x"`` then
        # later tries to look up ``color.ocio_builtin_uri.foo`` —
        # we should NOT crash on the "string isn't a dict" case.
        c = sc.SiteConfig({"color": {"ocio_builtin_uri": "x"}})
        assert c.get("color.ocio_builtin_uri.foo", "fb") == "fb"


# ============================================================================
# File resolution + parsing
# ============================================================================


@pytest.fixture(autouse=True)
def _clear_cache_around_tests():
    """Drop the site-config singleton between tests so we always
    re-resolve from scratch."""
    sc.invalidate_cache()
    yield
    sc.invalidate_cache()


class TestFileResolution:
    def test_env_var_takes_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        toml = tmp_path / "env.toml"
        toml.write_text(
            '[color]\nocio_builtin_uri = "ocio://from-env"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("FLICK_SITE_CONFIG", str(toml))
        loaded = sc.site_config()
        assert loaded.source == toml
        assert loaded.get("color.ocio_builtin_uri") == "ocio://from-env"

    def test_env_var_pointing_at_missing_file_falls_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(
            "FLICK_SITE_CONFIG", str(tmp_path / "does-not-exist.toml"),
        )
        loaded = sc.site_config()
        # No frozen mode, no repo-root toml in CI → empty.
        assert loaded.is_empty

    def test_malformed_toml_logs_and_yields_empty_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog,
    ) -> None:
        toml = tmp_path / "bad.toml"
        toml.write_text("not = valid = toml = at all\n", encoding="utf-8")
        monkeypatch.setenv("FLICK_SITE_CONFIG", str(toml))
        with caplog.at_level("WARNING"):
            loaded = sc.site_config()
        assert loaded.is_empty
        assert any("failed to parse" in r.message for r in caplog.records)

    def test_invalidate_cache_lets_next_call_reread(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        toml = tmp_path / "first.toml"
        toml.write_text('val = "first"\n', encoding="utf-8")
        monkeypatch.setenv("FLICK_SITE_CONFIG", str(toml))
        assert sc.site_config().get("val") == "first"

        # Swap the file content — without invalidate, we still see "first"
        toml.write_text('val = "second"\n', encoding="utf-8")
        assert sc.site_config().get("val") == "first", "should be cached"

        sc.invalidate_cache()
        assert sc.site_config().get("val") == "second"


# ============================================================================
# Integration with Preferences
# ============================================================================


class TestPreferencesIntegration:
    """A site config value should become the QSettings default — but
    only when the user hasn't explicitly set that key. We verify both
    halves of the contract."""

    def test_site_value_supplies_default_when_pref_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot,
    ) -> None:
        # Site config sets a non-default OCIO URI.
        toml = tmp_path / "studio.toml"
        toml.write_text(
            "[color]\n"
            'ocio_builtin_uri = "ocio://studio-config-v2.2.0_aces-v1.3_ocio-v2.4"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("FLICK_SITE_CONFIG", str(toml))
        sc.invalidate_cache()
        # Use a clean QSettings scope so no leftover user pref shadows
        # the site value.
        from PySide6.QtCore import QCoreApplication, QSettings

        QCoreApplication.setOrganizationName("flick-test-site")
        QCoreApplication.setApplicationName("flick-test-app")
        # Wipe just in case a previous test wrote something.
        QSettings().clear()

        from img_player.preferences import Preferences

        assert (
            Preferences().ocio_builtin_uri
            == "ocio://studio-config-v2.2.0_aces-v1.3_ocio-v2.4"
        )

    def test_user_override_wins_over_site_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot,
    ) -> None:
        toml = tmp_path / "studio.toml"
        toml.write_text(
            '[color]\nocio_builtin_uri = "ocio://site-pick"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("FLICK_SITE_CONFIG", str(toml))
        sc.invalidate_cache()

        from PySide6.QtCore import QCoreApplication, QSettings

        QCoreApplication.setOrganizationName("flick-test-site2")
        QCoreApplication.setApplicationName("flick-test-app2")
        QSettings().clear()

        from img_player.preferences import Preferences

        prefs = Preferences()
        # User explicit override.
        prefs.ocio_builtin_uri = "ocio://user-pick"
        # New Preferences instance should see the user value, not site.
        assert Preferences().ocio_builtin_uri == "ocio://user-pick"
