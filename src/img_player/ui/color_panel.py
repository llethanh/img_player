"""Color management panel: source, display, view, exposure, gamma."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from img_player.color.ocio_manager import OCIOManager
from img_player.ui.theme import H, S


class ColorPanel(QWidget):  # type: ignore[misc]
    """Emits ``color_params_changed(src, display, view, exposure, gamma)`` on any change.

    Also exposes the unmarked-EXR override controls: a "Save current as
    EXR default" button + a status row showing what's currently pinned.
    The actual pref read/write goes through ``unmarked_exr_save_requested``
    / ``unmarked_exr_clear_requested`` so the panel doesn't reach into
    :class:`Preferences` directly — keeps the panel a leaf widget that
    only knows about OCIO names and signals.
    """

    color_params_changed = Signal(str, str, str, float, float)
    # User clicked "Save current as EXR default" — the app stores the
    # current source + view as the unmarked-EXR prefs.
    unmarked_exr_save_requested = Signal(str, str)  # (source, view)
    # User clicked "Clear" on the pinned EXR default.
    unmarked_exr_clear_requested = Signal()

    def __init__(self, manager: OCIOManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._manager = manager
        self._emit_enabled = True

        self._src_combo = QComboBox()
        self._src_combo.addItems(manager.list_colorspaces())
        default_src = manager.role("scene_linear") or manager.list_colorspaces()[0]
        self._src_combo.setCurrentText(default_src)

        self._display_combo = QComboBox()
        self._display_combo.addItems(manager.list_displays())
        default_display = manager.default_display()
        self._display_combo.setCurrentText(default_display)

        self._view_combo = QComboBox()
        self._refresh_views(default_display)

        self._exposure_spin = QDoubleSpinBox()
        self._exposure_spin.setRange(-6.0, 6.0)
        self._exposure_spin.setSingleStep(0.25)
        self._exposure_spin.setValue(0.0)
        self._exposure_spin.setSuffix(" stops")

        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setRange(0.1, 4.0)
        self._gamma_spin.setSingleStep(0.05)
        self._gamma_spin.setValue(1.0)

        self._reset_btn = QPushButton("Reset exposure / gamma")
        self._reset_btn.clicked.connect(self._reset_adjustments)

        # Unmarked-EXR override row — lets the user pin the current
        # source + view as the auto-detector default for EXRs that
        # carry no colorspace tag. The status label tracks the pinned
        # pair so the user can see at a glance what's active without
        # opening a preferences dialog.
        self._unmarked_exr_status = QLabel("EXR default: industry (linear)")
        self._unmarked_exr_status.setStyleSheet(
            f"color: {H.TEXT_SECONDARY}; font-size: 11px;",
        )
        self._unmarked_exr_save_btn = QPushButton("Save as EXR default")
        self._unmarked_exr_save_btn.setToolTip(
            "Pin the current source + view as the auto-detector default "
            "for EXR files that have no colorspace tag.",
        )
        self._unmarked_exr_save_btn.clicked.connect(self._on_save_unmarked_exr)
        self._unmarked_exr_clear_btn = QPushButton("Clear")
        self._unmarked_exr_clear_btn.setToolTip(
            "Drop the pinned EXR default — auto-detection reverts to "
            "the industry-standard linear assumption.",
        )
        self._unmarked_exr_clear_btn.clicked.connect(
            self.unmarked_exr_clear_requested.emit,
        )
        self._unmarked_exr_clear_btn.setEnabled(False)

        self._src_combo.currentTextChanged.connect(self._notify)
        self._display_combo.currentTextChanged.connect(self._on_display_changed)
        self._view_combo.currentTextChanged.connect(self._notify)
        self._exposure_spin.valueChanged.connect(self._notify)
        self._gamma_spin.valueChanged.connect(self._notify)

        form = QFormLayout()
        form.addRow("Source colorspace:", self._src_combo)
        form.addRow("Display:", self._display_combo)
        form.addRow("View:", self._view_combo)
        form.addRow("Exposure:", self._exposure_spin)
        form.addRow("Gamma:", self._gamma_spin)

        group = QGroupBox("Color management")
        group.setLayout(form)

        # Two-row footer: save/clear buttons, then the status hint.
        # Buttons share a row so they read as related actions; the
        # status sits below in muted text so it never competes for
        # attention with the main combos above.
        exr_btn_row = QHBoxLayout()
        exr_btn_row.setContentsMargins(0, 0, 0, 0)
        exr_btn_row.setSpacing(S.SM)
        exr_btn_row.addWidget(self._unmarked_exr_save_btn)
        exr_btn_row.addWidget(self._unmarked_exr_clear_btn)
        exr_btn_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(S.SM, S.SM, S.SM, S.SM)
        layout.addWidget(group)
        layout.addWidget(self._reset_btn)
        layout.addLayout(exr_btn_row)
        layout.addWidget(self._unmarked_exr_status)
        layout.addStretch(1)

    # -------------------------------------------------------------- Public helpers

    def current_params(self) -> tuple[str, str, str, float, float]:
        return (
            self._src_combo.currentText(),
            self._display_combo.currentText(),
            self._view_combo.currentText(),
            float(self._exposure_spin.value()),
            float(self._gamma_spin.value()),
        )

    def available_source_colorspaces(self) -> list[str]:
        """List the entries currently in the source-colorspace combo.

        Public accessor so callers can check "is this name pickable?"
        without reaching into the private widget. Mirrors
        :meth:`available_displays` / :meth:`available_views`.
        """
        return [self._src_combo.itemText(i) for i in range(self._src_combo.count())]

    def available_displays(self) -> list[str]:
        """List the entries currently in the display combo."""
        return [self._display_combo.itemText(i) for i in range(self._display_combo.count())]

    def available_views(self) -> list[str]:
        """List the entries currently in the view combo.

        Note: the view combo's content depends on the selected display,
        so the list shown reflects the *current* display only.
        """
        return [self._view_combo.itemText(i) for i in range(self._view_combo.count())]

    def apply_state(
        self,
        *,
        source_colorspace: str | None = None,
        display: str | None = None,
        view: str | None = None,
        exposure: float | None = None,
        gamma: float | None = None,
        on_missing: "Callable[[str, str, str], None] | None" = None,
    ) -> None:
        """Apply a saved color-panel snapshot (from a session restore).

        Each field is optional — ``None`` means "leave that combo /
        spinbox alone". For combo fields, if the requested name is
        not in the current list we leave the existing pick in place
        and call ``on_missing(kind, requested, current)`` so the
        caller can log a warning. ``kind`` is one of
        ``"source"`` / ``"display"`` / ``"view"``.

        Setting via ``setCurrentText`` triggers the panel's standard
        change signals → re-emits ``color_params_changed`` →
        rebuilds the OCIO shader, exactly as if the user had clicked
        the combos manually.
        """
        if source_colorspace is not None:
            if source_colorspace in self.available_source_colorspaces():
                self._src_combo.setCurrentText(source_colorspace)
            elif on_missing is not None:
                on_missing(
                    "source", source_colorspace, self._src_combo.currentText(),
                )
        if display is not None:
            if display in self.available_displays():
                # ``setCurrentText`` fires ``_on_display_changed`` which
                # rebuilds the view list, so the next ``view`` set lands
                # in the freshly populated combo.
                self._display_combo.setCurrentText(display)
            elif on_missing is not None:
                on_missing("display", display, self._display_combo.currentText())
        if view is not None:
            if view in self.available_views():
                self._view_combo.setCurrentText(view)
            elif on_missing is not None:
                on_missing("view", view, self._view_combo.currentText())
        if exposure is not None:
            self._exposure_spin.setValue(exposure)
        if gamma is not None:
            self._gamma_spin.setValue(gamma)

    def set_source_colorspace(self, name: str) -> None:
        """Set the source colorspace without emitting — used when opening a new
        sequence where we can guess the input colorspace from metadata."""
        self._emit_enabled = False
        self._src_combo.setCurrentText(name)
        self._emit_enabled = True

    def bump_exposure(self, delta: float) -> None:
        """Nudge the exposure spinbox (used by keyboard shortcuts)."""
        self._exposure_spin.setValue(self._exposure_spin.value() + delta)

    def emit_current(self) -> None:
        """Force an emission of ``color_params_changed`` with current values."""
        self._notify()

    def reload_from_manager(self, manager: OCIOManager) -> dict[str, object]:
        """Repopulate combos against a freshly loaded OCIO config.

        Called when the user changes the OCIO config source via
        :class:`PreferencesDialog` and we hot-swap the manager. Tries
        to preserve the user's current picks (source / display / view)
        when they still exist in the new config; falls back to sensible
        defaults otherwise.

        Returns a small status dict the caller can surface in a status
        message:
          * ``source_preserved`` — was the previous source colorspace
            still present in the new config?
          * ``display_preserved`` — same question for display.
          * ``view_preserved`` — same for view.

        Signal emission is suppressed throughout the rebuild so callers
        don't see a flurry of intermediate ``color_params_changed``
        signals; one is fired at the very end with the final triple.
        """
        prev_source = self._src_combo.currentText()
        prev_display = self._display_combo.currentText()
        prev_view = self._view_combo.currentText()

        self._manager = manager
        self._emit_enabled = False
        try:
            new_colorspaces = manager.list_colorspaces()
            new_displays = manager.list_displays()

            self._src_combo.blockSignals(True)
            self._src_combo.clear()
            self._src_combo.addItems(new_colorspaces)
            self._src_combo.blockSignals(False)

            self._display_combo.blockSignals(True)
            self._display_combo.clear()
            self._display_combo.addItems(new_displays)
            self._display_combo.blockSignals(False)

            # Preserve source if still available; otherwise scene_linear
            # role; otherwise first entry.
            source_preserved = prev_source in new_colorspaces
            if source_preserved:
                self._src_combo.setCurrentText(prev_source)
            else:
                fallback_src = (
                    manager.role("scene_linear")
                    or (new_colorspaces[0] if new_colorspaces else "")
                )
                self._src_combo.setCurrentText(fallback_src)

            # Same logic for display.
            display_preserved = prev_display in new_displays
            target_display = (
                prev_display if display_preserved else manager.default_display()
            )
            self._display_combo.setCurrentText(target_display)

            # _refresh_views uses the (already swapped) self._manager.
            self._refresh_views(target_display)

            # View preserved iff display was preserved AND the view name
            # still exists for that display in the new config. Otherwise
            # _refresh_views has already picked the default for us.
            view_preserved = False
            if display_preserved:
                available_views = manager.list_views(target_display)
                if prev_view in available_views:
                    self._view_combo.setCurrentText(prev_view)
                    view_preserved = True
        finally:
            self._emit_enabled = True

        # Single emission with the final, validated triple.
        self._notify()

        return {
            "source_preserved": source_preserved,
            "display_preserved": display_preserved,
            "view_preserved": view_preserved,
        }

    def set_unmarked_exr_default(
        self, source: str | None, view: str | None,
    ) -> None:
        """Update the EXR-default status row from outside.

        The app calls this on boot (to reflect what's stored in prefs)
        and after the save / clear signals fire (to confirm the new
        state). ``None`` for either side means "no override pinned" —
        the auto-detector falls back to its industry default.
        """
        if source and view:
            self._unmarked_exr_status.setText(
                f"EXR default: {source} / {view}",
            )
            self._unmarked_exr_clear_btn.setEnabled(True)
        else:
            self._unmarked_exr_status.setText(
                "EXR default: industry (linear)",
            )
            self._unmarked_exr_clear_btn.setEnabled(False)

    # -------------------------------------------------------------- Internals

    def _reset_adjustments(self) -> None:
        self._emit_enabled = False
        self._exposure_spin.setValue(0.0)
        self._gamma_spin.setValue(1.0)
        self._emit_enabled = True
        self._notify()

    def _on_save_unmarked_exr(self) -> None:
        src = self._src_combo.currentText()
        view = self._view_combo.currentText()
        if not src or not view:
            return
        self.unmarked_exr_save_requested.emit(src, view)

    def _refresh_views(self, display: str) -> None:
        self._view_combo.blockSignals(True)
        self._view_combo.clear()
        self._view_combo.addItems(self._manager.list_views(display))
        default_view = self._manager.default_view(display)
        if default_view:
            self._view_combo.setCurrentText(default_view)
        self._view_combo.blockSignals(False)

    def _on_display_changed(self, display: str) -> None:
        self._refresh_views(display)
        self._notify()

    def _notify(self) -> None:
        if not self._emit_enabled:
            return
        src, display, view, exposure, gamma = self.current_params()
        if not (src and display and view):
            return
        self.color_params_changed.emit(src, display, view, exposure, gamma)
