"""Tests for :class:`img_player.ui.channel_menu.ChannelMenu`.

The menu is mostly UI but its public contract drives the cache via
``selection_changed`` → ``set_channel_selection``. Pin those signals
so future cosmetic refactors can't silently break the controller
wiring.

Historical: contact-sheet specific cases (tile checkboxes, layout
mode, "Show labels" toggle, reset button) were retired with the
contact-sheet feature in v1.2.
"""

from __future__ import annotations

import pytest

from img_player.sequence.channels import ChannelGroup, ChannelSelection
from img_player.ui.channel_menu import (
    ChannelMenu,
    _ChannelRow,
    _ProgressLabel,
)


@pytest.fixture
def groups() -> list[ChannelGroup]:
    return [
        ChannelGroup(label="RGB", channels=("R", "G", "B")),
        ChannelGroup(label="albedo", channels=("albedo.R", "albedo.G", "albedo.B")),
        ChannelGroup(label="Z", channels=("Z",)),
        ChannelGroup(label="N", channels=("N.X", "N.Y", "N.Z")),
    ]


@pytest.fixture
def menu(qtbot, groups: list[ChannelGroup]) -> ChannelMenu:
    m = ChannelMenu()
    qtbot.addWidget(m)
    m.set_groups(groups)
    return m


class TestInitialState:
    def test_first_group_is_active_by_default(self, menu: ChannelMenu) -> None:
        assert menu.active_label == "RGB"

    def test_current_selection_carries_first_group(self, menu: ChannelMenu) -> None:
        sel = menu.current_selection()
        assert sel is not None
        assert sel.active.label == "RGB"


class TestSetState:
    def test_restore_active_label(self, menu: ChannelMenu) -> None:
        menu.set_state(active="albedo")
        assert menu.active_label == "albedo"

    def test_unknown_label_silently_dropped(self, menu: ChannelMenu) -> None:
        # Saved state from a different sequence — labels that don't
        # exist now must NOT crash and must NOT change the active.
        menu.set_state(active="ghost_aov")
        assert menu.active_label == "RGB"  # unchanged, since ghost_aov absent

    def test_set_state_does_not_emit(self, menu: ChannelMenu, qtbot) -> None:
        # Restore from prefs must NOT trigger selection_changed —
        # the controller already knows the saved state.
        with qtbot.assertNotEmitted(menu.selection_changed):
            menu.set_state(active="albedo")


class TestSelectionEmissions:
    def test_radio_pick_emits_new_active(
        self, menu: ChannelMenu, qtbot,
    ) -> None:
        with qtbot.waitSignal(menu.selection_changed, timeout=200) as sig:
            menu._on_radio_picked("Z")  # type: ignore[attr-defined]
        sel = sig.args[0]
        assert isinstance(sel, ChannelSelection)
        assert sel.active.label == "Z"


class TestCycleActive:
    """``cycle_active`` powers the channel button's Up / Down arrow
    keyboard navigation. Steps through ``_groups`` with wrap-around
    and emits the same ``selection_changed`` signal as a radio click."""

    def test_cycle_down_picks_next_group(
        self, menu: ChannelMenu, qtbot,
    ) -> None:
        # Default active = "RGB" (first group). +1 → "albedo".
        with qtbot.waitSignal(menu.selection_changed, timeout=200) as sig:
            menu.cycle_active(+1)
        assert menu.active_label == "albedo"
        assert sig.args[0].active.label == "albedo"

    def test_cycle_up_picks_previous_group_with_wrap(
        self, menu: ChannelMenu, qtbot,
    ) -> None:
        # From "RGB" (idx 0), -1 wraps to the last group ("N").
        with qtbot.waitSignal(menu.selection_changed, timeout=200) as sig:
            menu.cycle_active(-1)
        assert menu.active_label == "N"
        assert sig.args[0].active.label == "N"

    def test_cycle_no_op_when_only_one_group(self, qtbot) -> None:
        m = ChannelMenu()
        qtbot.addWidget(m)
        m.set_groups([ChannelGroup(label="solo", channels=("R",))])
        with qtbot.assertNotEmitted(m.selection_changed):
            m.cycle_active(+1)

    def test_cycle_silent_when_no_groups(self, qtbot) -> None:
        m = ChannelMenu()
        qtbot.addWidget(m)
        with qtbot.assertNotEmitted(m.selection_changed):
            m.cycle_active(+1)


class TestEmpty:
    def test_empty_groups_no_selection(self, qtbot) -> None:
        m = ChannelMenu()
        qtbot.addWidget(m)
        m.set_groups([])
        assert m.current_selection() is None
        assert m.active_label == ""

    def test_set_groups_resets_active_to_first(
        self, menu: ChannelMenu,
    ) -> None:
        menu.set_state(active="albedo")
        # Loading a new sequence's groups puts the active radio on the
        # first group of the new list.
        menu.set_groups([
            ChannelGroup(label="OnlyOne", channels=("R", "G", "B")),
        ])
        assert menu.active_label == "OnlyOne"


class TestProgressNoFlicker:
    """The cache-fill bar paints on the row's surface; the label is a
    transparent child sitting on top. Before this contract was
    pinned, ``_ProgressLabel.set_fraction`` called its own
    ``update()`` independently of the row — so each timer tick
    scheduled two separate paint events that Qt could (and did)
    process out of order, flashing the label rect every tick while
    the cache was filling. The user reported "elle a tendance a
    clignoter".

    Fix: the label stores the fraction but does NOT trigger its own
    repaint. The row's ``update()`` (after writing the new fraction
    onto both itself and the label) propagates a single coherent
    paint cycle to both widgets. Pin both halves of the contract
    below."""

    def test_label_set_fraction_does_not_schedule_update(
        self, qtbot,
    ) -> None:
        label = _ProgressLabel("RGB")
        qtbot.addWidget(label)
        calls: list[None] = []
        label.update = lambda *_a, **_kw: calls.append(None)  # type: ignore[method-assign]
        label.set_fraction(0.5)
        assert calls == [], (
            "_ProgressLabel.set_fraction must not call update() — "
            "the parent row owns the paint cycle; an independent "
            "label update races with the row's bar paint and "
            "flashes the label rect each tick."
        )
        # But the fraction WAS stored — pin the data contract too.
        assert label._fraction == pytest.approx(0.5)  # noqa: SLF001

    def test_row_set_progress_updates_once_per_change(
        self, qtbot,
    ) -> None:
        # set_progress on a row should call its own update() exactly
        # once when the fraction changes — and zero times when it
        # doesn't. The label's set_fraction is invoked too so the
        # value is in sync, but it must not trigger its own update.
        row = _ChannelRow(
            ChannelGroup(label="RGB", channels=("R", "G", "B")),
        )
        qtbot.addWidget(row)
        row_calls: list[None] = []
        label_calls: list[None] = []
        row.update = lambda *_a, **_kw: row_calls.append(None)  # type: ignore[method-assign]
        row._label.update = lambda *_a, **_kw: label_calls.append(None)  # type: ignore[method-assign]
        # First tick — fraction changes (-1.0 → 0.5) → ONE row update.
        row.set_progress(50, 100)
        assert len(row_calls) == 1
        assert label_calls == []
        # Same fraction → NO additional update (guarded by the
        # ``new_fraction == self._fraction`` early-return).
        row.set_progress(50, 100)
        assert len(row_calls) == 1
        # Fraction changes again → one more update.
        row.set_progress(75, 100)
        assert len(row_calls) == 2
        assert label_calls == []
