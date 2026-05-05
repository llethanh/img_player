"""Pure-function tests for :class:`ChannelSelection`.

Historical: this file used to cover ``ChannelSelection.tiles``,
``is_contact_sheet``, ``union_channels``, ``tile_layout`` and
``auto_grid`` — all retired with the contact-sheet feature in v1.2.
The selection now carries a single ``active`` group; the tests
that remain check that minimal contract.
"""

from __future__ import annotations

from img_player.sequence.channels import ChannelGroup, ChannelSelection


def _g(label: str, *channels: str) -> ChannelGroup:
    return ChannelGroup(label=label, channels=channels)


class TestChannelSelection:
    def test_carries_active_group(self) -> None:
        active = _g("RGB", "R", "G", "B")
        sel = ChannelSelection(active=active)
        assert sel.active is active

    def test_frozen_dataclass(self) -> None:
        # Frozen so signal slots can compare instances cheaply.
        from dataclasses import FrozenInstanceError

        sel = ChannelSelection(active=_g("RGB", "R", "G", "B"))
        try:
            sel.active = _g("Z", "Z")  # type: ignore[misc]
        except FrozenInstanceError:
            pass
        else:
            raise AssertionError("ChannelSelection should be frozen")

    def test_equality_by_active(self) -> None:
        a = ChannelSelection(active=_g("RGB", "R", "G", "B"))
        b = ChannelSelection(active=_g("RGB", "R", "G", "B"))
        c = ChannelSelection(active=_g("Z", "Z"))
        assert a == b
        assert a != c
