"""Tests for the ephemeral-related preferences key (v0.4.1).

Level 5 of the ephemeral feature's testing strategy (spec §8.5).

We use an isolated ``QSettings`` backend per test (via the
``QCoreApplication.setOrganizationName`` / ``setApplicationName``
override pattern) so the developer's real preference store isn't
polluted.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QCoreApplication, QSettings

from img_player.preferences import Preferences


@pytest.fixture(autouse=True)
def _isolated_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> Iterator[None]:
    """Force ``QSettings`` to use a per-test ``IniFormat`` file.

    Without this, ``Preferences()`` would touch the developer's real
    registry (Windows) or preference plist (macOS / Linux), leaking
    test state into the user's environment and vice versa.
    """
    # IniFormat is fully file-backed — no registry, no system store.
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(
        QSettings.Format.IniFormat,
        QSettings.Scope.UserScope,
        str(tmp_path / "qsettings"),
    )
    # Force a unique org/app name per test so cached QSettings instances
    # in PySide6 don't get reused between fixture invocations.
    QCoreApplication.setOrganizationName(f"img_player_test_{tmp_path.name}")
    QCoreApplication.setApplicationName("img_player_test")
    yield
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)


# ============================================================================
# Default value
# ============================================================================


class TestDefault:
    def test_default_is_index_1(self) -> None:
        """No persisted value → returns 1 (the spec's "moyen" default)."""
        prefs = Preferences()
        assert prefs.ephemeral_duration_preset == 1


# ============================================================================
# Round-trip
# ============================================================================


class TestRoundTrip:
    def test_set_and_read_index_0(self) -> None:
        prefs = Preferences()
        prefs.ephemeral_duration_preset = 0
        # New instance simulates a fresh app boot reading the same QSettings.
        assert Preferences().ephemeral_duration_preset == 0

    def test_set_and_read_index_1(self) -> None:
        prefs = Preferences()
        prefs.ephemeral_duration_preset = 1
        assert Preferences().ephemeral_duration_preset == 1

    def test_set_and_read_index_2(self) -> None:
        prefs = Preferences()
        prefs.ephemeral_duration_preset = 2
        assert Preferences().ephemeral_duration_preset == 2


# ============================================================================
# Validation — out of range silently rejected
# ============================================================================


class TestValidation:
    def test_negative_index_rejected(self) -> None:
        prefs = Preferences()
        prefs.ephemeral_duration_preset = 2  # known good
        prefs.ephemeral_duration_preset = -1  # invalid
        # Should still be 2 (the previously-saved good value).
        assert Preferences().ephemeral_duration_preset == 2

    def test_too_large_index_rejected(self) -> None:
        prefs = Preferences()
        prefs.ephemeral_duration_preset = 0
        prefs.ephemeral_duration_preset = 99
        assert Preferences().ephemeral_duration_preset == 0

    def test_non_numeric_input_rejected(self) -> None:
        prefs = Preferences()
        prefs.ephemeral_duration_preset = 0
        prefs.ephemeral_duration_preset = "foo"  # type: ignore[assignment]
        assert Preferences().ephemeral_duration_preset == 0

    def test_corrupted_value_in_settings_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the stored value is somehow garbage (e.g. the user
        edited the .ini by hand and put 'banana'), the getter
        defends with the default rather than crashing."""
        prefs = Preferences()
        # Inject garbage directly into QSettings.
        prefs._s.setValue("ephemeral/duration_preset", "banana")
        assert prefs.ephemeral_duration_preset == 1
