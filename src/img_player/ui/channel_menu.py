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

from collections.abc import Callable, Iterable

from PySide6.QtCore import QRect, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
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
from img_player.ui.theme import C, S


# Same warm-cream tone used by the bottom info band's typography
# (info_band.py: ``color: #FFE5C0``). Reusing it keeps the two
# orange-on-orange readouts visually kin: cache-fill on a channel row
# now reads with the same legibility as the HUD.
_PROGRESS_TEXT_COLOR = QColor("#FFE5C0")


class _ProgressLabel(QLabel):  # type: ignore[misc]
    """QLabel that repaints its text in cream over the cache-fill area.

    The parent ``_ChannelRow`` paints a translucent orange bar behind
    this label up to ``fraction``. Default theme text colour reads
    poorly through that orange wash, so we overpaint the same text in
    cream (matching the ``InfoBand`` HUD) clipped to the filled region.
    Outside that region the standard QLabel rendering shows through —
    typography colour stays untouched until loading actually begins.
    """

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self._fraction: float = -1.0

    def set_fraction(self, fraction: float) -> None:
        """Store the new fill fraction.

        Does NOT trigger ``self.update()`` — the parent :class:`_ChannelRow`
        owns the paint cycle (it draws the orange / blue cache bars on
        its own surface and Qt composites this transparent label on
        top in the same cycle). Triggering an independent label
        update would race with the row's paint and flash the label's
        rect each tick — visible as a flicker on the channel menu
        while the cache is filling. The row's ``update()`` after
        calling this method already propagates a repaint to its
        children, so the label is redrawn coherently with the bars
        beneath it.
        """
        if fraction != self._fraction:
            self._fraction = fraction

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().paintEvent(event)
        if self._fraction <= 0:
            return
        rect = self.rect()
        fill_w = int(round(rect.width() * self._fraction))
        if fill_w <= 0:
            return
        painter = QPainter(self)
        painter.setClipRect(QRect(rect.x(), rect.y(), fill_w, rect.height()))
        painter.setPen(_PROGRESS_TEXT_COLOR)
        # Match the QLabel's own text layout (alignment + indent) so
        # the overpaint registers pixel-for-pixel with the underlying
        # default rendering — anything else would show as a faint
        # double-strike artefact.
        painter.drawText(rect, int(self.alignment()), self.text())


class _ChannelRow(QFrame):  # type: ignore[misc]
    """One line in the menu: radio (active) + label."""

    radio_picked = Signal(str)  # label of the row whose radio became checked

    def __init__(self, group: ChannelGroup, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.label = group.label
        self._original_channels = tuple(group.channels)

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

        self._label = _ProgressLabel(group.label)
        self._label.setMinimumWidth(120)
        self._label.setToolTip("Channels: " + ", ".join(group.channels))
        # Transparent label background so the row's cache-fill paint
        # (drawn in paintEvent below) shows through the text.
        self._label.setStyleSheet("background: transparent;")
        # Click-on-label is a synonym for click-on-radio: bigger
        # target, more forgiving UX, especially on touchpads.
        self._label.mousePressEvent = self._on_label_clicked  # type: ignore[method-assign]
        layout.addWidget(self._label, 1)

        # Cache-fill fractions painted as translucent bars in the
        # label's geometry — same idiom as the timeline cache bar so
        # the two readouts feel kin. ``_fraction`` is the RAM tier
        # (orange ``C.CACHE_BAR``); ``_disk_fraction`` is the on-disk
        # tier (blue ``C.CACHE_BAR_DISK``) drawn behind it. Negative
        # = "no data, paint nothing" (multi-layer / no-AOV stacks
        # stay clean).
        self._fraction: float = -1.0
        self._disk_fraction: float = -1.0

    # -------------------------------------------------- public API

    def set_active(self, on: bool) -> None:
        """Set the radio without retriggering the signal."""
        self._radio.blockSignals(True)
        self._radio.setChecked(on)
        self._radio.blockSignals(False)

    def set_progress(self, cached: int, total: int) -> None:
        """Update the row's cache-fill fraction. ``total <= 0`` resets
        to "no data" (= paints nothing). Called by the menu's polling
        loop while it's visible.

        Tooltip + label fraction are pushed in the same gated branch
        as the row's ``update()``: re-setting the tooltip and the
        child fraction on every tick (even when nothing changed) was
        wasted work, and the child ``set_fraction`` used to call its
        own ``update()`` independently — which raced with the row's
        paint and flashed the label rect every tick. See
        :meth:`_ProgressLabel.set_fraction`.
        """
        if total <= 0:
            if self._fraction != -1.0:
                self._fraction = -1.0
                self._label.set_fraction(-1.0)
                self.update()
            return
        new_fraction = max(0.0, min(1.0, cached / total))
        if new_fraction == self._fraction:
            return
        self._fraction = new_fraction
        self._label.set_fraction(new_fraction)
        self._label.setToolTip(
            f"Channels: {', '.join(self._original_channels)}\n"
            f"Cached: {cached} / {total} frames"
        )
        self.update()

    def set_disk_progress(self, disk_cached: int, total: int) -> None:
        """Update the row's on-disk fill fraction (blue tier). Polled
        at a slower cadence than :meth:`set_progress` since the disk
        scan is heavier. ``total <= 0`` resets to "no data"."""
        new_fraction = (
            -1.0 if total <= 0
            else max(0.0, min(1.0, disk_cached / total))
        )
        if new_fraction != self._disk_fraction:
            self._disk_fraction = new_fraction
            self.update()

    @property
    def radio(self) -> QRadioButton:
        return self._radio

    # -------------------------------------------------- paint

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Paint the cache-fill bars behind the label, then let Qt
        render the regular children (radio + label) on top.

        Two tiers: the on-disk fill (blue) is drawn first, the RAM
        fill (orange) overpaints its portion — so a group warm on
        disk but not promoted to RAM reads as blue, and the in-RAM
        head of the bar reads as orange. Bars are drawn over the
        label's geometry only — not the radio — so the indicator
        stays readable."""
        if self._fraction >= 0 or self._disk_fraction >= 0:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            label_geom = self._label.geometry()
            # Start the bar a few pixels before the label's first
            # character so the fill never crashes flush against the
            # text. We borrow the radio↔label gutter (``S.SM``).
            left_offset = S.SM
            full = QRectF(
                label_geom.x() - left_offset,
                label_geom.y(),
                label_geom.width() + left_offset,
                label_geom.height(),
            )

            def _draw_tier(fraction: float, fill, border) -> None:  # type: ignore[no-untyped-def]
                if fraction <= 0:
                    return
                fill_w = full.width() * fraction
                if fill_w <= 0:
                    return
                rect = QRectF(full.x(), full.y(), fill_w, full.height())
                # Translucent fill — text reads cleanly through it.
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(fill))
                painter.drawRect(rect)
                # Opaque outline tracing the fill — half-pixel inset
                # so the line lands inside the rect (sub-pixel
                # crispness at any DPI).
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(border, 1.0))
                painter.drawRect(rect.adjusted(0.5, 0.5, -0.5, -0.5))

            # Disk tier first so the orange RAM tier overpaints it.
            _draw_tier(self._disk_fraction, C.CACHE_BAR_DISK, C.CACHE_BAR_DISK_BORDER)
            _draw_tier(self._fraction, C.CACHE_BAR, C.CACHE_BAR_BORDER)
        super().paintEvent(event)

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
        # get truncated. 220 is the tested sweet spot — fits "albedo"
        # comfortably without dwarfing the transport bar in the rare
        # case of just RGB. The cache-fill bar paints inside the
        # label area so no extra horizontal budget needed.
        self.setMinimumWidth(220)

        # Optional callable that returns ``{group_label: (cached, total)}``
        # for the focused layer. Set by the host (transport / app) at
        # wire-up time; the menu polls it on a 250 ms timer while
        # visible to refresh each row's pip. Stays at ``None`` for
        # tests / standalone use of the menu.
        self._progress_provider: Callable[[], dict[str, tuple[int, int]]] | None = None
        # Disk-cache fill provider — same shape as the RAM one but
        # heavier (per-frame hash + SQLite query), so it's polled
        # once every ``_DISK_POLL_EVERY`` ticks rather than each tick.
        self._disk_progress_provider: Callable[[], dict[str, tuple[int, int]]] | None = None
        self._disk_poll_tick = 0
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(250)
        self._progress_timer.timeout.connect(self._refresh_progress)
        # Only spend cycles on the poll when the menu is actually
        # on screen — the popup hides via ``aboutToHide`` after a
        # pick or an outside click.
        self.aboutToShow.connect(self._on_about_to_show)
        self.aboutToHide.connect(self._progress_timer.stop)

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

    def set_progress_provider(
        self,
        provider: Callable[[], dict[str, tuple[int, int]]] | None,
    ) -> None:
        """Wire (or unwire) the RAM cache-fill data source. Called
        once at app startup with a closure over the cache; subsequent
        ``aboutToShow`` events poll the provider and push the
        ``(cached, total)`` pair into each row's pip."""
        self._progress_provider = provider

    def set_disk_progress_provider(
        self,
        provider: Callable[[], dict[str, tuple[int, int]]] | None,
    ) -> None:
        """Wire (or unwire) the on-disk cache-fill data source.

        Heavier than the RAM provider, so it's polled once every
        ``_DISK_POLL_EVERY`` ticks (~1.5 s) rather than each 250 ms
        tick — see :meth:`_refresh_progress`."""
        self._disk_progress_provider = provider

    def update_progress(
        self, progress: dict[str, tuple[int, int]] | None,
    ) -> None:
        """Push a fresh ``{group_label: (cached, total)}`` map into
        every row's RAM pip. Missing labels reset their pip to "no
        data"; unknown labels in the input are silently ignored."""
        if progress is None:
            for row in self._row_by_label.values():
                row.set_progress(0, 0)
            return
        for label, row in self._row_by_label.items():
            cached, total = progress.get(label, (0, 0))
            row.set_progress(cached, total)

    def update_disk_progress(
        self, progress: dict[str, tuple[int, int]] | None,
    ) -> None:
        """Push a fresh ``{group_label: (disk_cached, total)}`` map
        into every row's blue on-disk pip."""
        if progress is None:
            for row in self._row_by_label.values():
                row.set_disk_progress(0, 0)
            return
        for label, row in self._row_by_label.items():
            disk_cached, total = progress.get(label, (0, 0))
            row.set_disk_progress(disk_cached, total)

    # Poll the (heavier) disk provider once every N RAM ticks.
    _DISK_POLL_EVERY = 6  # 6 × 250 ms ≈ 1.5 s

    def _on_about_to_show(self) -> None:
        # Push a fresh tick immediately so the user doesn't see the
        # previous (stale) progress for 250 ms before the timer fires.
        # Reset the disk counter so the disk tier is refreshed at once
        # on open rather than waiting out a partial cycle.
        self._disk_poll_tick = 0
        self._refresh_progress()
        self._progress_timer.start()

    def _refresh_disk_progress(self) -> None:
        if self._disk_progress_provider is None:
            return
        try:
            data = self._disk_progress_provider()
        except Exception:
            data = None
        self.update_disk_progress(data)

    def _refresh_progress(self) -> None:
        # On-disk tier — refreshed on the first tick and then once
        # every ``_DISK_POLL_EVERY`` ticks (the disk scan is heavier
        # than the RAM one). The RAM tier below stays at full cadence.
        if self._disk_poll_tick % self._DISK_POLL_EVERY == 0:
            self._refresh_disk_progress()
        self._disk_poll_tick += 1
        if self._progress_provider is None:
            return
        try:
            data = self._progress_provider()
        except Exception:
            data = None
        self.update_progress(data)

    def cycle_active(self, delta: int) -> None:
        """Move the active radio by ``delta`` positions in the group
        list (wrap-around). Emits ``selection_changed``.

        Used by the channel button's keyboard handler so the user
        can step through channel groups with the arrow keys without
        opening the popup. ``delta=-1`` = previous group (Up arrow);
        ``delta=+1`` = next group (Down arrow).
        """
        if not self._groups:
            return
        labels = [g.label for g in self._groups]
        try:
            idx = labels.index(self._active_label)
        except ValueError:
            idx = 0
        new_label = labels[(idx + int(delta)) % len(labels)]
        if new_label == self._active_label:
            return
        self._set_active_silent(new_label)
        self._emit_selection()

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
