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

from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QFont, QFontMetrics
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
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
# Raw column payload — drives :class:`_SeqRowDelegate` which paints
# each column at a fixed x. Value layout:
# ``tuple[tuple[text:str, color:str, bold:bool, x:int], ...]``.
# ``x`` is computed once by :meth:`SequencePickerDialog._apply_column_layout`
# from the longest text in each column so the dialog can shrink to
# fit its content while keeping vertical alignment row-to-row.
_ROLE_COLS = Qt.ItemDataRole.UserRole + 2
# Pre-layout column data — same shape minus the ``x`` field. Held
# separately so :meth:`_apply_column_layout` can rebuild the laid-out
# tuple after measuring widths without losing the source text.
_ROLE_COLS_DATA = Qt.ItemDataRole.UserRole + 3

# Indent applied to all columns for grouped rows so they sit visually
# under their folder header.
_COL_INDENT = 18
# Minimum horizontal gap between two adjacent columns. Wide enough
# that "1920×1080" and "complete" never read as one token, narrow
# enough that the dialog stays tight on short patterns.
_COL_MIN_GAP = 24

# Foreground colour for the non-badge columns — kept aligned with the
# QSS-defined default item text colour so the row reads identical to
# the headers / unstyled items.
_COL_DEFAULT_FG = "#D8D8D8"


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
        # Floor only — the actual width is computed from the longest
        # row's content via :meth:`_apply_column_layout` / :meth:`_resize_for_content`.
        self.setMinimumWidth(420)
        # Floor the dialog at a height that comfortably shows the
        # header + ~6 rows + buttons row. ``_resize_for_content``
        # below grows it from there based on actual row count, capped
        # at 80 % of the screen — so this is the "minimum useful"
        # floor, not the typical opening size.
        self.setMinimumHeight(260)
        self._multi = multi
        # Either a flat sequence list or a grouped list — never both.
        self._sequences: list[SequenceInfo] = list(sequences) if sequences else []
        self._groups: list[FolderGroup] = list(groups) if groups else []

        outer = QVBoxLayout(self)
        # Tighter chrome — the previous LG / MD pairing left a
        # noticeable empty band above and below the list. Pull the
        # margins to MD and the inter-widget spacing to SM so the
        # header, list, and button row sit close together; the
        # vertical real estate goes to the content the user actually
        # scans (the sequence rows).
        outer.setContentsMargins(S.MD, S.MD, S.MD, S.MD)
        outer.setSpacing(S.SM)

        if multi:
            total = sum(len(g.sequences) for g in self._groups)
            if total <= 1:
                header_text = "1 sequence detected — confirm to load:"
            else:
                header_text = (
                    f"{total} sequences detected — tick the ones to load:"
                )
        else:
            header_text = (
                f"{len(self._sequences)} sequences in this folder — pick one:"
            )
        header = QLabel(header_text)
        header.setWordWrap(True)
        outer.addWidget(header)

        self._list = QListWidget()
        self._list.setFont(F.mono(F.SIZE_SM))
        # HTML-aware delegate for sequence rows — lets us paint the
        # trailing "complete" / "N missing" badge in its own colour
        # while leaving the rest of the row in the default foreground.
        # Headers / separators carry no HTML payload so the delegate
        # falls back to the default plain-text path for them.
        self._list.setItemDelegate(_SeqRowDelegate(self._list))
        # Internal list floor — gives the rows enough vertical
        # breathing room even on a single-sequence drop without the
        # whole dialog feeling oversized. The dialog's own floor
        # (260 px) covers the header + buttons chrome.
        self._list.setMinimumHeight(180)
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

        # Measure every row's column data, derive per-column max
        # widths, and bake the final x offsets into each item. Also
        # returns the total content width so the dialog can grow to
        # fit the longest row (the user dropped a sequence with a
        # 60-char filename, the dialog widens; a short pattern keeps
        # it compact).
        self._content_width = self._apply_column_layout(indent=multi)

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
            self._select_complete_btn = QPushButton(
                "Select all completed sequences",
            )
            self._select_complete_btn.setToolTip(
                "Tick only multi-frame sequences with no missing "
                "frames (skips single-frame stills)",
            )
            self._select_complete_btn.clicked.connect(
                self._select_only_complete,
            )
            btn_row.addWidget(self._select_all_btn)
            btn_row.addWidget(self._select_complete_btn)
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

        # Size the dialog to its actual content: a multi-source drop
        # with 30 sequences shouldn't open the same height as a
        # single-folder drop with 4. Caps at ~80 % of the screen
        # so we never produce a dialog taller than the display.
        self._resize_for_content()

    def _resize_for_content(self) -> None:
        """Compute an initial dialog height that fits the populated
        rows, capped at 80 % of the screen.

        The default Qt sizeHint for a ``QListWidget`` doesn't grow
        with row count past its viewport height, so a 30-sequence
        drop opens the same as a 3-sequence drop and the user has
        to resize manually every time. Walk the rows, sum their
        ``sizeHintForRow`` values, add the dialog chrome (header
        text + buttons + margins ~140 px), then clamp to the screen
        rectangle so the dialog never exceeds the display."""
        row_count = self._list.count()
        if row_count <= 0:
            return
        # Per-row height: trust ``sizeHintForRow`` (correctly accounts
        # for the mono font + checkbox indicator) and fall back to a
        # conservative 20 px per row when the widget hasn't been laid
        # out yet (= can return 0 / -1 pre-show). 20 matches the
        # tightened ``padding: 2px 10px`` in the list QSS.
        per_row = max(20, self._list.sizeHintForRow(0))
        list_h = per_row * row_count + 8  # +8 for inner scrollarea padding
        # Dialog chrome: header label + buttons row + outer margins +
        # inter-widget spacings. ~90 px after the LG→MD margin pull
        # and the MD→SM spacing pull (2024-05 design pass).
        chrome_h = 90
        target_h = list_h + chrome_h
        # Cap to 80 % of the available screen so we never open a
        # dialog taller than the user's monitor (4 K + tall picker
        # dropdowns can add up).
        screen = self.screen()
        if screen is not None:
            avail = screen.availableGeometry().height()
            target_h = min(target_h, int(avail * 0.80))
        # Floor at the minimum height so a one-sequence drop still
        # opens at the readable baseline rather than collapsing.
        target_h = max(target_h, self.minimumHeight())
        # Width: shrink-wrap to the widest row's content + dialog
        # chrome (outer margins, list border, checkbox indicator,
        # QSS padding, scrollbar). The 80 % cap mirrors the height
        # logic so we never produce a window wider than the screen.
        content_w = getattr(self, "_content_width", 0)
        # Chrome breakdown for a single row: indicator (~28) + QSS
        # padding-left (10) + QSS padding-right (10) + list border
        # (2) + scrollbar gutter (16) + outer margins (S.MD * 2).
        chrome_w = 28 + 10 + 10 + 2 + 16 + (2 * S.MD)
        target_w = content_w + chrome_w
        screen = self.screen()
        if screen is not None:
            avail_w = screen.availableGeometry().width()
            target_w = min(target_w, int(avail_w * 0.80))
        target_w = max(target_w, self.minimumWidth())
        self.resize(int(target_w), int(target_h))

    def _apply_column_layout(self, *, indent: bool) -> int:
        """Measure column-data widths and bake the final x offsets.

        Walks every row carrying ``_ROLE_COLS_DATA``, computes the
        widest text per column index using the appropriate font
        metrics (bold vs regular), then writes back to
        :data:`_ROLE_COLS` with each column at
        ``x = sum_of_prev_widths + N * _COL_MIN_GAP + indent``. The
        result is "compact column layout": each column is exactly as
        wide as its longest cell, with a uniform readable gap between
        them.

        Returns the total content width (used by
        :meth:`_resize_for_content` to size the dialog).
        """
        regular = self._list.font()
        bold = QFont(regular)
        bold.setBold(True)
        fm_regular = QFontMetrics(regular)
        fm_bold = QFontMetrics(bold)

        # Collect every (col_index, text, bold) so we can compute the
        # per-column max width before laying anything out.
        max_w = [0, 0, 0, 0, 0]
        rows: list[tuple[QListWidgetItem, tuple]] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None:
                continue
            data = item.data(_ROLE_COLS_DATA)
            if not data:
                continue
            rows.append((item, data))
            for j, (text, _color, is_bold) in enumerate(data):
                fm = fm_bold if is_bold else fm_regular
                w = fm.horizontalAdvance(text)
                if w > max_w[j]:
                    max_w[j] = w

        if not rows:
            return 0

        # Compute cumulative x for each column. Indent applies to the
        # first column only (visual indent under the folder header);
        # subsequent columns chain off the previous max width + gap.
        shift = _COL_INDENT if indent else 0
        xs = [shift]
        for j in range(4):
            xs.append(xs[-1] + max_w[j] + _COL_MIN_GAP)

        for item, data in rows:
            baked = tuple(
                (text, color, is_bold, xs[j])
                for j, (text, color, is_bold) in enumerate(data)
            )
            item.setData(_ROLE_COLS, baked)

        # Total content width = last column's x + its own width.
        return xs[-1] + max_w[4]

    # ------------------------------------------------------------------ Population

    def _populate_flat(self) -> None:
        for seq in self._sequences:
            item = QListWidgetItem(_format_seq_row(seq))
            item.setToolTip(_seq_tooltip(seq))
            item.setData(_ROLE_SEQ, seq)
            item.setData(_ROLE_COLS_DATA, _seq_columns_data(seq))
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
        item.setSizeHint(QSize(0, 4))
        self._list.addItem(item)

    def _add_seq_item(self, seq: SequenceInfo, *, checked: bool) -> None:
        item = QListWidgetItem("    " + _format_seq_row(seq))
        item.setData(_ROLE_SEQ, seq)
        item.setData(_ROLE_COLS_DATA, _seq_columns_data(seq))
        item.setToolTip(_seq_tooltip(seq))
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

    def _select_only_complete(self) -> None:
        """Tick every multi-frame, contiguous sequence; untick the
        rest (incomplete sequences AND single-frame stills).

        The intent is "grab the actual sequences ready for review"
        — single-frame items in a VFX drop are usually slates,
        thumbnails, or reference images, not the renders the user
        came to look at, so they shouldn't be lumped in with the
        complete sequences.
        """
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item is None:
                continue
            seq = item.data(_ROLE_SEQ)
            if seq is None:
                continue
            is_completed_sequence = (
                seq.is_contiguous and seq.frame_count > 1
            )
            item.setCheckState(
                Qt.CheckState.Checked if is_completed_sequence
                else Qt.CheckState.Unchecked,
            )

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
    """One-line plain-text summary: pattern + range + resolution + status.

    Used as the accessibility / fallback ``text`` payload on each
    QListWidgetItem; the on-screen rendering goes through
    :func:`_seq_columns` instead (rendered by the delegate).
    Plain version is kept so screen readers and copy-paste land on
    something useful, but it isn't what the dialog displays.
    """
    pattern = seq.display_pattern()
    rng = f"[{seq.first_frame}-{seq.last_frame}]"
    w, h = _resolve_dims(seq)
    res = f"{w}×{h}" if w and h else "—"
    total = f"{seq.last_frame - seq.first_frame + 1}f"
    status = _seq_status_text(seq)
    return f"{pattern:<32}  {rng:<14}  {res:<12}  {total:<8}  {status}"


def _seq_status_text(seq: SequenceInfo) -> str:
    """Plain-text status badge — used by the plain ``text`` payload
    (delegate falls back to it when no HTML is set, screen readers see
    it, etc.). HTML version below mirrors the wording."""
    if seq.is_contiguous:
        return "complete"
    n = len(seq.missing_frames)
    return f"{n} missing frame" + ("s" if n != 1 else "")


def _seq_columns_data(
    seq: SequenceInfo,
) -> tuple[tuple[str, str, bool], ...]:
    """Raw column data for :class:`_SeqRowDelegate`.

    Returns the five columns — pattern / range / resolution / total
    frames / status — as ``(text, color, bold)`` tuples without an x
    coordinate. :meth:`SequencePickerDialog._apply_column_layout`
    measures the widest text per column index across every row, then
    bakes the final x offsets into :data:`_ROLE_COLS`.

    "Total" is the expected length of the range
    (``last - first + 1``), not :attr:`SequenceInfo.frame_count` —
    so an incomplete sequence with 120 present frames over a 139-frame
    range still reports "139f" alongside the "19 missing frames"
    badge, which is the count the user wants to see ("what should the
    sequence be?").
    """
    pattern = seq.display_pattern()
    rng = f"[{seq.first_frame}-{seq.last_frame}]"
    w, h = _resolve_dims(seq)
    res = f"{w}×{h}" if w and h else "—"
    total = seq.last_frame - seq.first_frame + 1
    total_text = f"{total}f"
    if seq.is_contiguous:
        badge_text = "complete"
        badge_color = "#7FCC7F"
        badge_bold = False
    else:
        n = len(seq.missing_frames)
        word = "frame" if n == 1 else "frames"
        badge_text = f"{n} missing {word}"
        badge_color = "#E06B6B"
        badge_bold = True
    return (
        (pattern, _COL_DEFAULT_FG, False),
        (rng, _COL_DEFAULT_FG, False),
        (res, _COL_DEFAULT_FG, False),
        (total_text, _COL_DEFAULT_FG, False),
        (badge_text, badge_color, badge_bold),
    )


def _resolve_dims(seq: SequenceInfo) -> tuple[int | None, int | None]:
    """Return ``(width, height)`` for the sequence, falling back to a
    fresh probe of the first frame when the scanner didn't fill them.

    The scanner can leave dims empty when the OIIO probe failed at
    scan time (slow Drive fetch, unusual PNG variant, etc.). Re-trying
    here is cheap (one header read) and usually succeeds the second
    time round — the file is by now warm in the OS cache, or the
    user-initiated picker open gave Drive enough time to materialise
    the bytes.
    """
    if seq.width and seq.height:
        return seq.width, seq.height
    if not seq.frames:
        return None, None
    try:
        from img_player.io.reader import read_header
        spec = read_header(seq.frames[0].path)
        return spec.width, spec.height
    except Exception:
        return None, None


def _seq_tooltip(seq: SequenceInfo) -> str:
    """Tooltip: full file pattern path + status detail when incomplete."""
    base = str(seq.directory / seq.display_pattern())
    if seq.is_contiguous:
        return base
    missing = seq.missing_frames
    # Show the first few gap frames so the user can spot the holes
    # without leaving the picker. Truncate at ten so a sequence with
    # 500 missing frames doesn't produce a tooltip wall.
    preview = ", ".join(str(f) for f in missing[:10])
    if len(missing) > 10:
        preview += ", …"
    return f"{base}\nMissing frames: {preview}"


class _SeqRowDelegate(QStyledItemDelegate):  # type: ignore[misc]
    """Paints each sequence row as a set of fixed-x text columns so
    the picker's pattern / range / resolution / status badges line up
    vertically across every row.

    Reads the column payload from :data:`_ROLE_COLS`. Rows without one
    (folder headers, separators, empty markers) fall through to the
    default plain-text path so the delegate stays a pure enhancement.

    Background, selection and the checkbox indicator are drawn by the
    style (via ``CE_ItemViewItem`` with the text blanked out) so the
    QSS in :data:`_LIST_QSS` keeps owning the visual chrome — we only
    paint the row's text region ourselves.
    """

    def paint(self, painter, option, index) -> None:  # type: ignore[no-untyped-def]
        cols = index.data(_ROLE_COLS)
        if not cols:
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        widget = opt.widget
        style = (
            widget.style() if widget is not None else QApplication.style()
        )
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem, opt, painter, widget,
        )
        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText, opt, widget,
        )

        painter.save()
        # Baseline y: centre of the row, plus half the font ascent so
        # the glyphs sit on the optical centre rather than crawling
        # to the top. Matches how QStyle paints the checkbox indicator
        # so columns and checkbox line up.
        fm = painter.fontMetrics()
        baseline_y = (
            text_rect.top()
            + (text_rect.height() + fm.ascent() - fm.descent()) // 2
        )
        bold_font = QFont(opt.font)
        bold_font.setBold(True)
        for text, color, bold, x in cols:
            painter.setFont(bold_font if bold else opt.font)
            painter.setPen(QColor(color))
            painter.drawText(text_rect.left() + x, baseline_y, text)
        painter.restore()

    def sizeHint(self, option, index) -> QSize:  # type: ignore[no-untyped-def]
        return super().sizeHint(option, index)


_LIST_QSS = """
QListWidget {{
    background: #14161A;
    border: 1px solid #2A2D33;
    border-radius: 4px;
    outline: 0;
}}
QListWidget::item {{
    padding: 2px 10px;
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
