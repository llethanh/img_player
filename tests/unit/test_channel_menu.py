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
from img_player.ui.channel_menu import ChannelMenu


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
