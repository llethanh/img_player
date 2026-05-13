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
        # Also force the repo-root fallback path to a sentinel that
        # doesn't exist on disk — a developer may have a real
        # ``flick.toml`` at the repo root for their own use, but it
        # mustn't sneak into the test result.
        monkeypatch.setattr(
            sc, "_SITE_FILE_NAME", "_nonexistent_under_test.toml",
        )
        loaded = sc.site_config()
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
#
# The end-to-end behaviour (user TOML > site TOML > hardcoded) is
# exercised by ``tests/unit/test_user_prefs.py::TestLayeredPreferences``
# which has the proper test fixtures for isolating the user TOML store.
# Keeping the Preferences leg out of this file avoids duplicating that
# fixture machinery here.
