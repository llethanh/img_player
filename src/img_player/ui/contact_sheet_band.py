"""Top-of-viewer band for the contact-sheet grid view.

Same UX shape as :class:`CompareBand` — a transparent strip sitting in
the menu-bar row that exposes the contact-sheet settings (Auto smart
toggle, manual cols / rows spin boxes, label overlay toggle, output
downscale picker) without forcing the user to reach for a kebab menu
or a sub-menu deep in View.

Pure UI: emits signals, doesn't own state. The owning app
(``ImgPlayerApp``) is the single source of truth — on every state
change it re-feeds the band via :meth:`set_state` so a programmatic
update (session load, keyboard shortcut, …) keeps the widget in
sync without bespoke setters per field.

Replaces the older ``View → Contact sheet settings`` sub-menu and the
``⋯`` kebab popup that hung off the transport bar's contact-sheet
button. Visibility is toggled by ``app.py`` based on
``ContactSheetState.enabled`` — band lives in the top layout
sandwiched between the same two stretches as the compare band, but
the two bands are mutually exclusive (compare and contact sheet
hijack the GL upload the same way, so the app forces one off when
the other turns on).
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFocusEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QWidget,
)

from img_player.ui.theme import F, G, H, S


class _SelectAllSpinBox(QSpinBox):  # type: ignore[misc]
    """QSpinBox that selects its entire text on every focus-in AND
    snaps back to the special "Auto" value when the user empties the
    field.

    Two UX tweaks rolled into one subclass:

    1. **Select-all on focus.** Default Qt behaviour drops the caret
       at the click position and leaves the value un-selected — to
       overwrite, the user has to triple-click or Ctrl+A first. For
       numeric fields where intent is almost always "replace the
       value", select-all-on-focus means a single click is enough to
       start typing the new number.

       Implementation: defer the ``selectAll`` via
       ``QTimer.singleShot(0, ...)``. Without the defer, Qt's own
       click-handling runs *after* ``focusInEvent`` returns and moves
       the caret to the click position, which would un-do our
       select-all. Single-shot 0 pushes the select-all past that
       handler so the selection sticks.

    2. **Empty → "Auto" snap.** When the user clears the field (Del /
       Backspace until empty), the spinbox by default keeps the
       lineEdit visually blank until focus leaves. That looks broken.
       We intercept ``textChanged`` on the lineEdit: when it becomes
       empty, we defer one event-loop tick then ``setValue(minimum)``
       — which makes the special-value-text ``"Auto"`` appear AND
       re-select it so the user can immediately type a digit that
       overwrites "Auto" without an extra Ctrl+A.

       Re-entrance is guarded so the ``setValue(0)`` we trigger
       doesn't itself re-fire ``textChanged`` → ``setValue(0)`` → … in
       an infinite loop.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Set this BEFORE wiring textChanged so the very first
        # signal (from Qt's initial lineEdit population) doesn't
        # re-enter the reset path.
        self._resetting_to_auto = False
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.textChanged.connect(self._on_text_changed)

    def focusInEvent(self, event: QFocusEvent) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        # Unconditional — the SpinBoxes only get focus from clicks /
        # tabs into the band, both of which warrant select-all.
        line_edit = self.lineEdit()
        if line_edit is not None:
            QTimer.singleShot(0, line_edit.selectAll)

    def _on_text_changed(self, text: str) -> None:
        if self._resetting_to_auto:
            return
        # Whitespace-only counts as empty too — paranoia for paste
        # gestures that drop a stray space.
        if text.strip():
            return

        def _reset() -> None:
            # Re-check the guard inside the deferred lambda: by the
            # time this fires the user may have typed a digit (e.g.
            # they pressed Backspace then immediately typed "5"
            # before the singleShot fired). In that case the
            # lineEdit isn't empty anymore and we MUST NOT reset.
            le = self.lineEdit()
            if le is None or le.text().strip():
                return
            self._resetting_to_auto = True
            try:
                # ``minimum()`` is 0 in our use-case → triggers the
                # ``specialValueText`` ("Auto") substitution Qt does
                # for the bottom-of-range value.
                self.setValue(self.minimum())
                # Re-select the freshly populated "Auto" so the
                # next typed character replaces it cleanly. Without
                # this the caret would sit at the end of "Auto"
                # and typing "5" would yield "Auto5", which the
                # validator rejects with a frustrating no-op.
                le.selectAll()
            finally:
                self._resetting_to_auto = False

        QTimer.singleShot(0, _reset)

# Output divisor presets — same set the old menu offered. (1, label)
# pairs ordered fine → coarse so the combo defaults to the most
# detailed option on top.
_DIVISOR_PRESETS: tuple[tuple[int, str], ...] = (
    (1, "Full (÷1)"),
    (2, "Half (÷2)"),
    (3, "Third (÷3)"),
    (4, "Quarter (÷4)"),
    (6, "Sixth (÷6)"),
    (8, "Eighth (÷8)"),
)

# Label-typo scale presets. ``1.0`` is the historical default (≈ 3.5 %
# of tile height). The pill background sizes itself off the rendered
# text metrics so the cartouche scales with the typo automatically.
# Ordered small → large so the combo's natural reading direction
# matches "less prominent → more prominent".
_LABEL_SIZE_PRESETS: tuple[tuple[float, str], ...] = (
    (0.75, "Small"),
    (1.0, "Medium"),
    (1.5, "Large"),
    (2.5, "Extra large"),
)

# Spin-box range. 0 is the special "Auto" value (shown as "Auto" via
# ``setSpecialValueText``); 1..16 lets the user pin a dim. 16 matches
# the old Custom-grid QInputDialog cap so we don't quietly tighten
# the upper bound on existing workflows.
_SPIN_MIN = 0
_SPIN_MAX = 16


class ContactSheetBand(QFrame):  # type: ignore[misc]
    """``[Auto smart]  Cols [□]  Rows [□]  [Show labels]  Output [▼]  ✕``."""

    # User changed the cols spin-box. Carries the int value, with ``-1``
    # meaning "auto" (= the spin-box is at 0 / displaying "Auto"). The
    # app's existing ``_on_contact_sheet_grid_changed(cols, rows)`` slot
    # takes the cols / rows pair so we re-use that contract: the band
    # always emits BOTH dims, with -1 for whichever is auto.
    grid_changed = Signal(int, int)
    # User clicked the "Auto smart" button — shortcut for "both dims
    # auto" (cols=-1, rows=-1). Kept as a separate signal so the app
    # can log / track it independently from a manual zero-out via the
    # spin-boxes; both paths land on the same state update though.
    auto_requested = Signal()
    # User toggled the "Show labels" pill.
    labels_toggled = Signal(bool)
    # User picked a different output-size divisor (1, 2, 3, 4, 6, 8).
    divisor_changed = Signal(int)
    # User picked a different label-typo scale (0.75, 1.0, 1.5, 2.5).
    # Float because the scale is continuous in principle; the band UI
    # only exposes 4 presets but a future "custom" slider could feed
    # the same signal without a contract change.
    label_size_changed = Signal(float)
    # User clicked ✕ — exit contact-sheet mode entirely.
    close_requested = Signal()

    BAND_HEIGHT = G.CTRL_BUTTON_H + 4  # 32

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("contactSheetBand")
        self.setFrameShape(QFrame.Shape.NoFrame)
        # Transparent — the band lives inside the top-of-window layout
        # alongside the menu bar (see ``MainWindow._build_menu``), so
        # it should inherit the menu-bar background rather than draw
        # its own raised panel. Mirrors :class:`CompareBand`.
        #
        # The toggleable pills (Auto smart, Show labels) use the
        # global ``btnToggle`` QSS variant — no local checked-state
        # override needed here.
        self.setStyleSheet(
            "QFrame#contactSheetBand { background: transparent; }"
            # KV labels above each input — secondary mono uppercase
            # text, brief §7. Object-named so the rule doesn't reset
            # every other QLabel in the band (none currently, but
            # future additions stay isolated).
            "QLabel#csKV {"
            f"  color: {H.T_SEC};"
            f"  font-family: {F.FAMILY_MONO};"
            f"  font-size: {F.SIZE_MONO_LABEL}px;"
            "  font-weight: 500;"
            "  padding: 0 2px;"
            "}"
        )
        self.setFixedHeight(self.BAND_HEIGHT)
        # Re-emit guard: when the app pushes new state via
        # :meth:`set_state`, the QSpinBox / QComboBox / QPushButton
        # widgets fire their ``valueChanged`` / ``activated`` /
        # ``toggled`` signals as a side effect of the setter. Without a
        # block, those would loop right back into the app slot, which
        # would push the same state again, etc. The flag is consulted
        # by every internal slot to short-circuit the emission.
        self._syncing = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(S.S_6)

        # ---- Auto smart shortcut ----
        # Checkable pill — checked = "both dims auto". Clicking it
        # forces auto on both dims; clicking again does nothing (Qt
        # would normally un-check, but for a "current state indicator"
        # un-checking is meaningless — the user has to pick a dim
        # explicitly via the spin-boxes to leave auto mode).
        # ``btnToggle`` opts into the global QSS's orange-tint
        # checked state.
        self._auto_btn = QPushButton("Auto smart")
        self._auto_btn.setObjectName("btnToggle")
        self._auto_btn.setCheckable(True)
        self._auto_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._auto_btn.setToolTip(
            "Pick the grid automatically — smart layout that maximises "
            "per-tile area given the viewer's aspect ratio. Equivalent "
            "to setting both Cols and Rows to Auto."
        )
        self._auto_btn.clicked.connect(self._on_auto_clicked)
        layout.addWidget(self._auto_btn)

        # Thin separator before the grid group.
        layout.addWidget(self._build_separator())

        # ---- Cols KV ----
        layout.addLayout(self._build_kv_group(
            "COLS",
            lambda: self._build_cols_spin(),
            attr="_cols_spin",
        ))

        # ---- Rows KV ----
        layout.addLayout(self._build_kv_group(
            "ROWS",
            lambda: self._build_rows_spin(),
            attr="_rows_spin",
        ))

        # Thin separator before the labels group.
        layout.addWidget(self._build_separator())

        # ---- Show labels ----
        self._labels_btn = QPushButton("Show labels")
        self._labels_btn.setObjectName("btnToggle")
        self._labels_btn.setCheckable(True)
        self._labels_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._labels_btn.setToolTip(
            "Overlay the layer name on each tile in a warm-amber "
            "cartouche matching the rest of the UI accent. Off by "
            "default — turn it on, then use the Size combo to dial "
            "in the typo size that fits the review."
        )
        self._labels_btn.toggled.connect(self._on_labels_toggled)
        layout.addWidget(self._labels_btn)

        # Thin separator before the size / output combos.
        layout.addWidget(self._build_separator())

        # ---- Label size combo (KV-wrapped) ----
        # Scales the auto-computed typo size on each tile. The pill
        # background follows the text metrics so the cartouche
        # scales with it automatically. Disabled while Show labels
        # is off (no point picking a size when no label is rendered)
        # — that gating is applied by ``set_state``.
        self._label_size_combo = QComboBox()
        self._label_size_combo.setFixedHeight(G.CTRL_INPUT_H)
        self._label_size_combo.setMinimumWidth(78)
        self._label_size_combo.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed,
        )
        self._label_size_combo.setToolTip(
            "Label typo size. Small / Medium / Large / Extra large "
            "are multipliers on the auto-computed font size — the "
            "pill cartouche grows / shrinks with the text."
        )
        for size, label in _LABEL_SIZE_PRESETS:
            # Store the float as combo item data so we can read it
            # back on activation without parsing the label string.
            self._label_size_combo.addItem(label, size)
        self._label_size_combo.activated.connect(self._on_label_size_activated)
        layout.addLayout(
            self._kv_wrap("SIZE", self._label_size_combo),
        )

        # ---- Output size combo (KV-wrapped) ----
        self._output_combo = QComboBox()
        self._output_combo.setFixedHeight(G.CTRL_INPUT_H)
        self._output_combo.setMinimumWidth(110)
        self._output_combo.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed,
        )
        self._output_combo.setToolTip(
            "Composite downscale divisor. ÷1 = full source res per tile "
            "(big buffer, sharp); ÷2 = quarter-pixel count (~4× faster "
            "compose + upload, sweet spot for review)."
        )
        for div, label in _DIVISOR_PRESETS:
            self._output_combo.addItem(label, div)
        self._output_combo.activated.connect(self._on_output_activated)
        layout.addLayout(
            self._kv_wrap("OUTPUT", self._output_combo),
        )

        # NB: the older ✕ close button used to live here. It was
        # removed because clicking the transport's contact-sheet
        # toggle (or Ctrl+G) already exits the mode — the ✕ was
        # redundant. The ``close_requested`` signal is kept on the
        # class so a future affordance can re-emit it, but the band
        # never fires it on its own anymore.

    # ------------------------------------------------------------------ Public API

    def set_state(
        self,
        cols: int | None,
        rows: int | None,
        show_labels: bool,
        output_divisor: int,
        label_size: float = 1.0,
    ) -> None:
        """Re-feed the band from the canonical ``ContactSheetState``.

        Called by the app whenever the state mutates (toggle on,
        grid change via signal, divisor change via signal, session
        load, …). Blocks internal emissions for the duration so the
        widget setters don't re-fire signals back into the app.

        ``cols``/``rows`` = ``None`` is treated as "Auto" (= the spin
        box shows the special value text, 0 internally). Anything
        else is clamped to the spin-box's accepted range so a
        pathological session value (e.g. cols=99) doesn't leave the
        widget in an undefined state.

        ``label_size`` is the float multiplier the render path
        applies. We pick the closest matching preset for the combo
        display; a custom value outside the preset list snaps to the
        nearest known preset so the combo never lands on an empty
        selection.
        """
        self._syncing = True
        try:
            self._cols_spin.setValue(
                0 if cols is None else max(_SPIN_MIN, min(_SPIN_MAX, int(cols))),
            )
            self._rows_spin.setValue(
                0 if rows is None else max(_SPIN_MIN, min(_SPIN_MAX, int(rows))),
            )
            self._auto_btn.setChecked(cols is None and rows is None)
            self._labels_btn.setChecked(bool(show_labels))
            # Find the closest matching divisor preset. If the state
            # carries a custom divisor outside the preset list (older
            # session / future expansion), default to "Full" so the
            # combo never lands on -1.
            idx = next(
                (
                    i for i, (div, _) in enumerate(_DIVISOR_PRESETS)
                    if div == int(output_divisor)
                ),
                0,
            )
            self._output_combo.setCurrentIndex(idx)
            # Label-size combo. Pick the preset whose factor is
            # closest to the stored value — handles both exact preset
            # matches and "session was saved with a custom slider
            # value, now bucket back onto the nearest preset".
            size_idx = min(
                range(len(_LABEL_SIZE_PRESETS)),
                key=lambda i: abs(_LABEL_SIZE_PRESETS[i][0] - float(label_size)),
            )
            self._label_size_combo.setCurrentIndex(size_idx)
            # Disable the size combo when labels are off — picking a
            # size is meaningless without a label to scale. The combo
            # stays at its last value so re-enabling Show labels
            # immediately uses the size the user last picked.
            self._label_size_combo.setEnabled(bool(show_labels))
        finally:
            self._syncing = False

    # ------------------------------------------------------------------ Internals

    def _build_separator(self) -> QWidget:
        """Thin vertical hairline between band groups.

        1 px wide × 22 px tall in BORDER_SUB — same recipe used by
        the compare band so the two bands read as part of the same
        design system.
        """
        sep = QWidget()
        sep.setFixedWidth(1)
        sep.setFixedHeight(G.CTRL_SEP_H)
        sep.setStyleSheet(f"background:{H.BORDER_SUB};")
        return sep

    def _kv_wrap(self, label: str, control: QWidget) -> QHBoxLayout:
        """Wrap a control in a horizontal KV pair (label + control).

        The label is rendered in the mono "caption" style (secondary
        text, uppercase) so the group reads as `KEY  [control]`.
        Returns a QHBoxLayout the caller adds to the main band layout.
        """
        h = QHBoxLayout()
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(S.S_4)
        kv = QLabel(label)
        kv.setObjectName("csKV")
        h.addWidget(kv)
        h.addWidget(control)
        return h

    def _build_kv_group(
        self, label: str, make_control, *, attr: str,
    ) -> QHBoxLayout:
        """Build a KV-wrapped control and assign it to ``self.<attr>``.

        Tiny helper to keep the ``__init__`` body straight-line readable
        for the cols / rows spin-box construction.
        """
        control = make_control()
        setattr(self, attr, control)
        return self._kv_wrap(label, control)

    def _build_cols_spin(self) -> QSpinBox:
        spin = self._make_spin(
            "Columns. 0 = Auto (computed from rows + layer count, or "
            "from the viewer aspect if rows is also Auto).",
        )
        spin.valueChanged.connect(self._on_cols_changed)
        return spin

    def _build_rows_spin(self) -> QSpinBox:
        spin = self._make_spin(
            "Rows. 0 = Auto (computed from cols + layer count, or "
            "from the viewer aspect if cols is also Auto).",
        )
        spin.valueChanged.connect(self._on_rows_changed)
        return spin

    def _make_spin(self, tooltip: str) -> QSpinBox:
        # Use the select-all-on-focus subclass so a single click in
        # the cols / rows field surfaces the current value already
        # selected — the user just types the new number, no extra
        # selection gesture needed.
        spin = _SelectAllSpinBox()
        spin.setRange(_SPIN_MIN, _SPIN_MAX)
        # ``setSpecialValueText`` substitutes the rendered text for the
        # *minimum* value only — at 0 the user sees "Auto", at 1..16
        # they see the digit. This is the cleanest way to fit a tri-
        # state semantics (auto / manual N) into a stock QSpinBox.
        spin.setSpecialValueText("Auto")
        # Hide the native up/down step buttons. With a 0..16 range and
        # a "type the number" UX, the steppers waste pixels AND render
        # nearly invisibly against the dark theme — users couldn't
        # tell they were there and clicked on the empty zone wondering
        # what it did. ``NoButtons`` collapses the widget to a pure
        # text editor.
        spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        spin.setFixedHeight(G.CTRL_INPUT_H)
        spin.setFixedWidth(56)
        spin.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        spin.setToolTip(tooltip)
        return spin

    def _on_auto_clicked(self) -> None:
        if self._syncing:
            return
        # Force checked back on — un-checking from the user click would
        # be meaningless (they'd land in "no dim chosen" state which is
        # the same as auto anyway). Leaving auto mode happens by typing
        # a positive value in cols or rows.
        if not self._auto_btn.isChecked():
            self._auto_btn.setChecked(True)
        self.auto_requested.emit()
        # Also emit grid_changed(-1, -1) so the app's existing slot
        # gets the same shape it would from the old menu wiring.
        self.grid_changed.emit(-1, -1)

    def _on_cols_changed(self, value: int) -> None:
        if self._syncing:
            return
        cols = -1 if value <= 0 else int(value)
        rows = -1 if self._rows_spin.value() <= 0 else int(self._rows_spin.value())
        self.grid_changed.emit(cols, rows)

    def _on_rows_changed(self, value: int) -> None:
        if self._syncing:
            return
        cols = -1 if self._cols_spin.value() <= 0 else int(self._cols_spin.value())
        rows = -1 if value <= 0 else int(value)
        self.grid_changed.emit(cols, rows)

    def _on_labels_toggled(self, checked: bool) -> None:
        if self._syncing:
            return
        self.labels_toggled.emit(bool(checked))

    def _on_output_activated(self, index: int) -> None:
        if self._syncing:
            return
        div = self._output_combo.itemData(index)
        if isinstance(div, int):
            self.divisor_changed.emit(div)

    def _on_label_size_activated(self, index: int) -> None:
        if self._syncing:
            return
        size = self._label_size_combo.itemData(index)
        if isinstance(size, (int, float)):
            self.label_size_changed.emit(float(size))
