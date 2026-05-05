"""Popup menu for picking the active channel group.

Replaces the legacy ``QComboBox`` in the transport bar with a real
``QMenu`` so the click-outside-to-close + Esc-to-close + native
dropdown shadow all come for free. Each row is a single radio button
inside a :class:`QWidgetAction` — the user picks one of N channel
groups (RGB, albedo, normal, Z, …) and the menu emits a
:class:`ChannelSelection` with that group as the active pick.

Historical note (v1.2): an earlier version of this menu also offered
per-row checkboxes (build a contact-sheet from N tiled channels) +
a footer with layout mode and a "Show labels" toggle. The contact-
sheet feature was retired; the menu was simplified back to a flat
radio-button list.
"""

from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QRadioButton,
    QWidget,
    QWidgetAction,
)

from img_player.sequence.channels import ChannelGroup, ChannelSelection
from img_player.ui.theme import S


class _ChannelRow(QFrame):  # type: ignore[misc]
    """One line in the menu: radio (active) + label."""

    radio_picked = Signal(str)  # label of the row whose radio became checked

    def __init__(self, group: ChannelGroup, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = group.label

        self.setFrameShape(QFrame.Shape.NoFrame)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(S.SM, 2, S.SM, 2)
        layout.setSpacing(S.SM)

        self._radio = QRadioButton()
        # Radio buttons inside a QMenu need autoExclusive disabled;
        # the parent menu manages exclusivity via a single QButtonGroup
        # (otherwise two rows in the same menu fight each other when
        # a third becomes parent).
        self._radio.setAutoExclusive(False)
        self._radio.toggled.connect(self._on_radio_toggled)
        layout.addWidget(self._radio)

        self._label = QLabel(group.label)
        self._label.setMinimumWidth(120)
        self._label.setToolTip("Channels: " + ", ".join(group.channels))
        # Click-on-label is a synonym for click-on-radio: bigger
        # target, more forgiving UX, especially on touchpads.
        self._label.mousePressEvent = self._on_label_clicked  # type: ignore[method-assign]
        layout.addWidget(self._label, 1)

    # -------------------------------------------------- public API

    def set_active(self, on: bool) -> None:
        """Set the radio without retriggering the signal."""
        self._radio.blockSignals(True)
        self._radio.setChecked(on)
        self._radio.blockSignals(False)

    @property
    def radio(self) -> QRadioButton:
        return self._radio

    # -------------------------------------------------- handlers

    def _on_radio_toggled(self, checked: bool) -> None:
        if checked:
            self.radio_picked.emit(self.label)

    def _on_label_clicked(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton:
            self._radio.setChecked(True)
        super().mousePressEvent(event)


class ChannelMenu(QMenu):  # type: ignore[misc]
    """Popup menu with one radio row per channel group."""

    # Emitted whenever the active radio changes. Carries a fresh
    # :class:`ChannelSelection` with the picked group as ``active``.
    selection_changed = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._groups: tuple[ChannelGroup, ...] = ()
        self._active_label: str = ""
        # Maps label → row widget so we can drive radio state from
        # set_groups / set_state without index juggling.
        self._row_by_label: dict[str, _ChannelRow] = {}
        # All rows share one QButtonGroup so the radios are mutually
        # exclusive across the whole menu — we did set autoExclusive
        # to False on each radio, this group reinstates it globally.
        self._radio_group = QButtonGroup(self)
        self._radio_group.setExclusive(True)

        # Ensure the menu is wide enough that long layer names don't
        # get truncated. 220 is a tested sweet spot — fits "albedo"
        # comfortably without dwarfing the transport bar in the rare
        # case of just RGB.
        self.setMinimumWidth(220)

    # ------------------------------------------------------------------ Public API

    def set_groups(self, groups: Iterable[ChannelGroup]) -> None:
        """Rebuild the row list from a fresh list of groups.

        Called by the transport bar when a new sequence is loaded.
        Resets state to "first group active" — same baseline the
        legacy combo had at index 0.
        """
        self.clear()
        self._row_by_label.clear()
        # Drain the QButtonGroup — addButton() is safe to re-call but
        # the group tracks references so leaving stale ones leaks.
        for btn in list(self._radio_group.buttons()):
            self._radio_group.removeButton(btn)

        self._groups = tuple(groups)
        for group in self._groups:
            row = _ChannelRow(group, parent=self)
            row.radio_picked.connect(self._on_radio_picked)
            self._radio_group.addButton(row.radio)
            action = QWidgetAction(self)
            action.setDefaultWidget(row)
            self.addAction(action)
            self._row_by_label[group.label] = row

        if self._groups:
            self._active_label = self._groups[0].label
            self._row_by_label[self._active_label].set_active(True)
        else:
            self._active_label = ""

    def set_state(self, active: str) -> None:
        """Restore a saved active label without emitting selection_changed.

        Called from preferences round-trip on app boot. Unknown labels
        are silently dropped — the user might have switched sequences
        and the persisted active no longer applies.
        """
        if active and active in self._row_by_label:
            self._set_active_silent(active)

    def current_selection(self) -> ChannelSelection | None:
        """Read the current state out as a :class:`ChannelSelection`.

        ``None`` when no groups have been loaded yet (typically before
        the first sequence opens).
        """
        if not self._groups or not self._active_label:
            return None
        active_group = next(
            (g for g in self._groups if g.label == self._active_label),
            self._groups[0],
        )
        return ChannelSelection(active=active_group)

    @property
    def active_label(self) -> str:
        return self._active_label

    # ------------------------------------------------------------------ Internals

    def _set_active_silent(self, label: str) -> None:
        """Update the active radio without triggering selection_changed."""
        if label not in self._row_by_label:
            return
        for lbl, row in self._row_by_label.items():
            row.set_active(lbl == label)
        self._active_label = label

    def _emit_selection(self) -> None:
        sel = self.current_selection()
        if sel is not None:
            self.selection_changed.emit(sel)

    def _on_radio_picked(self, label: str) -> None:
        self._active_label = label
        self._emit_selection()
