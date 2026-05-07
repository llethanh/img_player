"""Modal dialog: pick one or more sequences out of a multi-source drop.

Two flavours
------------

* **Single-source legacy** — :meth:`SequencePickerDialog.pick` keeps
  the v1.0 behaviour: a flat list, single selection, returns one
  :class:`SequenceInfo` or ``None``. Used by the multi-sequence
  resolver for a single dropped folder so callers that don't yet
  know about groups keep working.

* **Multi-source grouped** — :meth:`SequencePickerDialog.pick_grouped`
  takes a list of :class:`FolderGroup` (one per dropped folder, plus
  an optional "loose files" group for raw files), shows them under
  bold non-selectable folder headers with checkboxes on the
  sequence rows, and returns the user's selection as a list. Empty
  folders show a greyed "[no sequence found]" entry so a misfired
  drop doesn't disappear silently.

Default check state
-------------------
* If the entire drop resolves to exactly one sequence, that sequence
  is pre-checked (= one click on Load and you're done).
* Otherwise nothing is pre-checked — the user opts into each
  sequence rather than discovering 30 layers loaded at once.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from img_player.sequence.models import SequenceInfo
from img_player.sequence.scanner import FolderGroup
from img_player.ui.theme import F, H, S

# Custom item-data role to track which list rows hold a sequence
# (carries the SequenceInfo) versus a header / placeholder (carries
# ``None``). Drives ``selected_sequences()`` and the per-row check
# semantics — headers are non-selectable and ignore checks.
_ROLE_SEQ = Qt.ItemDataRole.UserRole + 1


class SequencePickerDialog(QDialog):  # type: ignore[misc]
    """Modal sequence chooser. See module docstring for the two APIs."""

    def __init__(
        self,
        sequences: list[SequenceInfo] | None = None,
        groups: list[FolderGroup] | None = None,
        *,
        parent: QWidget | None = None,
        multi: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick a sequence")
        self.setModal(True)
        self.setMinimumWidth(560)
        self._multi = multi
        # Either a flat sequence list or a grouped list — never both.
        self._sequences: list[SequenceInfo] = list(sequences) if sequences else []
        self._groups: list[FolderGroup] = list(groups) if groups else []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(S.LG, S.LG, S.LG, S.LG)
        outer.setSpacing(S.MD)

        if multi:
            total = sum(len(g.sequences) for g in self._groups)
            if total <= 1:
                header_text = (
                    "One sequence detected in the dropped sources. "
                    "Confirm to load:"
                )
            else:
                header_text = (
                    f"{total} sequences detected in the dropped sources. "
                    f"Tick the ones you want to load:"
                )
        else:
            header_text = (
                f"This folder contains {len(self._sequences)} sequences. "
                f"Pick one to load:"
            )
        header = QLabel(header_text)
        header.setWordWrap(True)
        outer.addWidget(header)

        self._list = QListWidget()
        self._list.setFont(F.mono(F.SIZE_SM))
        self._list.setMinimumHeight(220)
        self._list.setStyleSheet(_LIST_QSS.format(accent=H.ACCENT))
        if multi:
            # Checkboxes are the entire selection model — row
            # highlight on top of them adds noise without information,
            # so disable selection in multi mode.
            self._list.setSelectionMode(
                QListWidget.SelectionMode.NoSelection,
            )
            # Click-anywhere-on-the-row toggles the checkbox: a small
            # affordance that makes the picker feel native (Qt's
            # default requires a precise hit on the indicator
            # rectangle, which is fiddly for fast users — and with
            # ``NoSelection`` it doesn't even toggle on the indicator
            # itself, which is the original bug).
            #
            # We bypass Qt's flaky native toggle entirely: an
            # eventFilter on the viewport intercepts every press on a
            # checkable row, toggles the state ourselves, and
            # consumes the event. A single source of truth, works the
            # same whether the click lands on the indicator or
            # elsewhere on the row.
            self._list.viewport().installEventFilter(self)
        outer.addWidget(self._list, 1)

        if multi:
            self._populate_groups()
        else:
            self._populate_flat()

        # --- Buttons row ----------------------------------------------
        btn_row = QHBoxLayout()
        if multi:
            self._select_all_btn = QPushButton("Select all")
            self._deselect_all_btn = QPushButton("Deselect all")
            self._select_all_btn.clicked.connect(
                lambda: self._set_all_checked(True),
            )
            self._deselect_all_btn.clicked.connect(
                lambda: self._set_all_checked(False),
            )
            btn_row.addWidget(self._select_all_btn)
            btn_row.addWidget(self._deselect_all_btn)
        btn_row.addStretch(1)

        std = (
            QDialogButtonBox.StandardButton.Cancel
            | (
                QDialogButtonBox.StandardButton.Ok if multi
                else QDialogButtonBox.StandardButton.Open
            )
        )
        buttons = QDialogButtonBox(std, parent=self)
        ok_btn = buttons.button(
            QDialogButtonBox.StandardButton.Ok if multi
            else QDialogButtonBox.StandardButton.Open
        )
        if ok_btn is not None:
            if multi:
                ok_btn.setText("Load selected")
            ok_btn.setDefault(True)
            ok_btn.setAutoDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        btn_row.addWidget(buttons)
        outer.addLayout(btn_row)

        # In flat mode, double-click = pick + accept (legacy behaviour).
        if not multi:
            self._list.itemDoubleClicked.connect(lambda _i: self.accept())
            if self._sequences:
                self._list.setCurrentRow(0)

    # ------------------------------------------------------------------ Population

    def _populate_flat(self) -> None:
        for seq in self._sequences:
            item = QListWidgetItem(_format_seq_row(seq))
            item.setToolTip(str(seq.directory / seq.display_pattern()))
            item.setData(_ROLE_SEQ, seq)
            self._list.addItem(item)

    def _populate_groups(self) -> None:
        total = sum(len(g.sequences) for g in self._groups)
        default_checked = (total == 1)
        first_group = True
        for group in self._groups:
            if not first_group:
                # Thin horizontal divider — replaces the old empty
                # spacer row. Reads as a clear visual break without
                # eating vertical space the way a blank line did.
                self._add_separator()
            first_group = False
            if group.folder is not None:
                self._add_header(group.folder.name)
            if group.empty:
                self._add_empty_marker()
                continue
            for seq in group.sequences:
                self._add_seq_item(seq, checked=default_checked)

    def _add_header(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setData(_ROLE_SEQ, None)
        flags = item.flags()
        flags &= ~Qt.ItemFlag.ItemIsSelectable
        flags &= ~Qt.ItemFlag.ItemIsUserCheckable
        item.setFlags(flags)
        item.setFont(F.ui(F.SIZE_MD, bold=True))
        item.setForeground(QBrush(QColor("#E8E8E8")))
        self._list.addItem(item)

    def _add_empty_marker(self) -> None:
        item = QListWidgetItem("    [no sequence found]")
        item.setData(_ROLE_SEQ, None)
        flags = item.flags()
        flags &= ~Qt.ItemFlag.ItemIsSelectable
        flags &= ~Qt.ItemFlag.ItemIsUserCheckable
        item.setFlags(flags)
        item.setForeground(QBrush(QColor("#6B6E74")))
        self._list.addItem(item)

    def _add_separator(self) -> None:
        """Tiny non-interactive row drawn as a horizontal rule.

        Uses Unicode box-drawing dashes (``─``) — they tile into a
        continuous line at any DPI without needing a custom item
        delegate. The row is short (4 px padding + the line itself)
        so groups sit close to each other while staying visually
        distinct.
        """
        from PySide6.QtCore import QSize
        item = QListWidgetItem("─" * 200)
        item.setData(_ROLE_SEQ, None)
        flags = item.flags()
        flags &= ~Qt.ItemFlag.ItemIsSelectable
        flags &= ~Qt.ItemFlag.ItemIsUserCheckable
        flags &= ~Qt.ItemFlag.ItemIsEnabled
        item.setFlags(flags)
        item.setForeground(QBrush(QColor("#3A3D43")))
        # Force a thin row height. The default would inherit the
        # mono font's full line — way too tall for a separator.
        item.setSizeHint(QSize(0, 6))
        self._list.addItem(item)

    def _add_seq_item(self, seq: SequenceInfo, *, checked: bool) -> None:
        item = QListWidgetItem("    " + _format_seq_row(seq))
        item.setData(_ROLE_SEQ, seq)
        item.setToolTip(str(seq.directory / seq.display_pattern()))
        flags = item.flags()
        flags |= Qt.ItemFlag.ItemIsUserCheckable
        item.setFlags(flags)
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked,
        )
        self._list.addItem(item)

    # ------------------------------------------------------------------ Helpers

    def _toggle_item(self, item: QListWidgetItem) -> None:
        """Flip the check state of a sequence row.

        Headers / separators / empty markers carry ``None`` in
        ``_ROLE_SEQ`` and are silently ignored.
        """
        if item.data(_ROLE_SEQ) is None:
            return
        new = (
            Qt.CheckState.Unchecked
            if item.checkState() == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        item.setCheckState(new)

    # ------------------------------------------------------------------ Event filter

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        """Single point of truth for click-to-toggle in multi mode.

        Catches every left-button press on a checkable row, toggles
        the state ourselves, and **consumes the event** so Qt's
        native (and unreliable under ``NoSelection``) handler
        doesn't get a second say. Clicks on headers / separators
        fall through untouched.
        """
        if (
            obj is self._list.viewport()
            and event.type() == QEvent.Type.MouseButtonPress
            and event.button() == Qt.MouseButton.LeftButton
        ):
            # QMouseEvent on PySide6 — ``position()`` returns a QPointF
            # (Qt 6 API); fall back to ``pos()`` for safety on older
            # bindings.
            pos = (
                event.position().toPoint()
                if hasattr(event, "position") else event.pos()
            )
            item = self._list.itemAt(pos)
            if (
                item is not None
                and bool(item.flags() & Qt.ItemFlag.ItemIsUserCheckable)
            ):
                self._toggle_item(item)
                return True  # eat the event — we own the toggle
        return super().eventFilter(obj, event)

    def _set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None:
                continue
            seq = item.data(_ROLE_SEQ)
            if seq is None:
                continue
            item.setCheckState(state)

    def selected_sequence(self) -> SequenceInfo | None:
        """Legacy single-select API — returns the highlighted row's seq."""
        row = self._list.currentRow()
        item = self._list.item(row) if row >= 0 else None
        if item is None:
            return None
        return item.data(_ROLE_SEQ)

    def selected_sequences(self) -> list[SequenceInfo]:
        """Multi-select API — returns every checked sequence in display order."""
        out: list[SequenceInfo] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None:
                continue
            seq = item.data(_ROLE_SEQ)
            if seq is None:
                continue
            if item.checkState() == Qt.CheckState.Checked:
                out.append(seq)
        return out

    # ------------------------------------------------------------------ Convenience

    @classmethod
    def pick(
        cls,
        sequences: list[SequenceInfo],
        parent: QWidget | None = None,
    ) -> SequenceInfo | None:
        """Single-select flow (legacy). Returns the chosen seq or ``None``."""
        dlg = cls(sequences=sequences, parent=parent, multi=False)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        return dlg.selected_sequence()

    @classmethod
    def pick_grouped(
        cls,
        groups: list[FolderGroup],
        parent: QWidget | None = None,
    ) -> list[SequenceInfo]:
        """Multi-select flow. Returns every checked seq, ``[]`` on cancel."""
        dlg = cls(groups=groups, parent=parent, multi=True)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return []
        return dlg.selected_sequences()


# ---------------------------------------------------------------- Formatting

def _format_seq_row(seq: SequenceInfo) -> str:
    """One-line summary: pattern + range + resolution."""
    pattern = seq.display_pattern()
    rng = f"[{seq.first_frame}-{seq.last_frame}]"
    if seq.width and seq.height:
        res = f"{seq.width}×{seq.height}"
    else:
        res = f"{seq.frame_count}f"
    return f"{pattern:<32}  {rng:<14}  {res}"


_LIST_QSS = """
QListWidget {{
    background: #14161A;
    border: 1px solid #2A2D33;
    border-radius: 4px;
    outline: 0;
}}
QListWidget::item {{
    padding: 4px 10px;
    color: #D8D8D8;
}}
QListWidget::item:hover {{
    background: #1F2228;
}}
QListWidget::item:selected {{
    background: {accent};
    color: #0A0A0A;
}}
QListWidget::item:selected:!active {{
    background: {accent};
    color: #0A0A0A;
}}
/* Custom check indicator. Qt's default 13 px square gets lost next
   to a 14 px mono font and looks half-disabled — bump the box to
   18 px and give it loud two-state styling.

   The check glyph itself is drawn as an inline SVG data URI so the
   dialog stays self-contained (no extra resource file to keep in
   sync). %23 is the URL-encoded ``#`` so we can write the colour
   inline. */
QListWidget::indicator {{
    width: 18px;
    height: 18px;
    margin-right: 6px;
}}
QListWidget::indicator:unchecked {{
    border: 2px solid #6B7079;
    background: #14161A;
    border-radius: 3px;
}}
QListWidget::indicator:unchecked:hover {{
    border-color: {accent};
    background: #1A1D22;
}}
QListWidget::indicator:checked {{
    background: {accent};
    border: 2px solid {accent};
    border-radius: 3px;
    image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><path d='M3 8.5 L6.5 12 L13 4.5' fill='none' stroke='%23000000' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/></svg>");
}}
"""
