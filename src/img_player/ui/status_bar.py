"""Custom status bar widget — brief §10.

Replaces the legacy ``QStatusBar`` + three inline ``QLabel`` block built
in :meth:`MainWindow._build_status_bar`. The brief specifies a three-
zone layout with distinct typographic conventions per zone:

* **Left zone** — coloured dot + session message in Inter Medium. The
  dot signals the loaded-project health (green = OK, amber = warn,
  red = error, grey = no project).
* **Middle zone** — "FOCUS" mono-uppercase tag pill + focused layer
  name in JetBrains Mono. Quiets down to empty when no layer is
  focused.
* **Right zone** — three perf KVs (cache N/T · fps · RAM used/budget ·
  free). Rendered as rich-HTML from
  :func:`img_player.ui.status_format.format_perf_html`; the widget
  hosts the formatted blob in a ``QLabel`` configured for
  ``RichText``.

The widget exposes the same public API as the legacy inline labels so
``MainWindow.set_status`` / ``set_selected_layers`` and the existing
``status_right.setText(html)`` calls in ``app._refresh_status`` keep
working without a refactor on the call sites.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStatusBar,
    QWidget,
)

from img_player.ui.theme import F, G, H, S


# Width of the status-dot disc. Sized so a 6 px disc sits comfortably
# above the typographic baseline of the adjacent label without
# crowding it.
_DOT_SIZE = 6


class _StatusDot(QWidget):  # type: ignore[misc]
    """Tiny circular dot rendered via ``QPainter``.

    Set the colour via :meth:`set_color` — pass ``None`` to hide the
    dot entirely (the widget keeps its layout footprint so the
    adjacent label doesn't shift horizontally when the dot goes away).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color: str | None = None
        # Fixed footprint so the dot acts like an inline glyph in the
        # surrounding row layout.
        self.setFixedSize(_DOT_SIZE + 4, _DOT_SIZE + 4)

    def set_color(self, color: str | None) -> None:
        if color == self._color:
            return
        self._color = color
        self.update()

    def paintEvent(self, _event) -> None:  # type: ignore[no-untyped-def, override]
        if self._color is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        from PySide6.QtGui import QBrush, QColor
        painter.setBrush(QBrush(QColor(self._color)))
        # Centre the disc inside the widget's box.
        cx = self.width() // 2
        cy = self.height() // 2
        r = _DOT_SIZE // 2
        painter.drawEllipse(cx - r, cy - r, _DOT_SIZE, _DOT_SIZE)


class StatusBar(QStatusBar):  # type: ignore[misc]
    """Three-zone status bar.

    Inherits from :class:`QStatusBar` so :meth:`QMainWindow.setStatusBar`
    accepts it as a drop-in replacement for the default status bar
    that ``QMainWindow`` builds itself. We rebuild the visual
    content from scratch though — the default ``QStatusBar``'s
    item-management API (``addWidget`` / ``showMessage`` / etc.) is
    bypassed in favour of a custom three-column layout hosted on an
    internal ``QWidget``.

    Public API (matches the legacy inline labels' contract so the
    existing call sites in :class:`MainWindow` /
    :class:`ImgPlayerApp` keep working):

    * :meth:`set_status` — set the left-zone message text.
    * :meth:`set_session_dot` — set the left dot colour (hex string
      or ``None`` to hide). Defaults to grey at construction.
    * :meth:`set_selected_layers` — set the middle-zone focused-
      layer text. Empty string hides the FOCUS tag + name.
    * :meth:`set_perf_html` — accept the rich-text HTML produced by
      :func:`status_format.format_perf_html` and render it in the
      right zone.

    Attribute aliases ``status_left`` / ``status_selection`` /
    ``status_right`` are also exposed so any caller that historically
    reached for them via ``self._window.status_right.setText(...)``
    keeps compiling. Internally these all point at the new
    sub-widgets.
    """

    HEIGHT = G.CTRL_BUTTON_H - 2  # 26 px per brief §10

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusBar")
        self.setFixedHeight(self.HEIGHT)
        # ``QStatusBar`` ships with a size grip in the bottom-right
        # corner and a tiny inner margin around its items area; both
        # would compete with our custom layout. Disable.
        self.setSizeGripEnabled(False)
        self.setContentsMargins(0, 0, 0, 0)
        # The container background is the deepest surface in the
        # palette — the brief calls for BG_DEEP here to seat the
        # status bar visually at the very bottom of the stack.
        # ``WA_StyledBackground`` is required for the QSS background
        # rule to actually paint on a bare ``QWidget`` subclass.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(
            f"QStatusBar#statusBar {{"
            f"  background-color: {H.BG_DEEP};"
            f"  border-top: 1px solid {H.BORDER_SUB};"
            f"}}"
            # Disable the QStatusBar item-frame border (rendered around
            # each ``addWidget`` child by default) so our embedded
            # container reads as a single continuous surface.
            f"QStatusBar#statusBar::item {{ border: none; }}"
        )

        # Inner container hosts the actual 3-column layout. We add it
        # to the QStatusBar via ``addPermanentWidget(stretch=1)`` so it
        # spans the full width and doesn't get bumped around by any
        # transient ``showMessage`` calls (which the legacy code did
        # use occasionally — keeping that path harmless).
        inner = QWidget(self)
        inner.setObjectName("statusBarInner")
        inner.setStyleSheet(
            "QWidget#statusBarInner { background: transparent; }"
        )
        layout = QHBoxLayout(inner)
        # Brief §10: padding-x 14, gap 24 between zones.
        layout.setContentsMargins(S.S_14, 0, S.S_14, 0)
        layout.setSpacing(S.S_24)
        self.addPermanentWidget(inner, 1)

        # ---- Left zone --------------------------------------------------
        # Dot + session message. The dot is created hidden (no colour)
        # and lit up by ``set_session_dot`` once the app reports
        # session state.
        left_row = QHBoxLayout()
        left_row.setContentsMargins(0, 0, 0, 0)
        left_row.setSpacing(S.S_8)
        self._dot = _StatusDot(self)
        self.status_left = QLabel(
            "Ready — drop a sequence (folder or file) to start."
        )
        self.status_left.setStyleSheet(
            f"color: {H.T_SEC}; "
            f"font-family: {F.FAMILY_UI}; "
            f"font-size: {F.SIZE_BODY_SMALL}px; "
            f"font-weight: 500;"
        )
        left_row.addWidget(self._dot)
        left_row.addWidget(self.status_left)
        left_container = QWidget(self)
        left_container.setLayout(left_row)
        left_container.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(left_container, 0)

        # ---- Middle zone (FOCUS pill + layer name) ----------------------
        mid_row = QHBoxLayout()
        mid_row.setContentsMargins(0, 0, 0, 0)
        mid_row.setSpacing(S.S_8)
        # FOCUS tag — mono uppercase wide-tracked. The brief specifies
        # 9 px JetBrains Mono with 0.18em letter-spacing; QSS letter-
        # spacing isn't supported per-px on Qt < 6.5, so we approximate
        # by uppercasing the text and applying a soft accent border.
        self._focus_tag = QLabel("FOCUS")
        self._focus_tag.setStyleSheet(
            f"color: {H.ACC_BRIGHT};"
            f"background-color: {H.ACC_TINT_10};"
            f"border: 1px solid {H.ACC_BORDER_ON};"
            f"border-radius: {G.RADIUS_SM}px;"
            f"padding: 1px 5px;"
            f"font-family: {F.FAMILY_MONO};"
            f"font-size: {F.SIZE_MONO_CAPS}px;"
            f"font-weight: 500;"
        )
        self._focus_tag.setVisible(False)
        self.status_selection = QLabel("")
        self.status_selection.setStyleSheet(
            f"color: {H.T_PRI};"
            f"font-family: {F.FAMILY_MONO};"
            f"font-size: {F.SIZE_MONO_LABEL}px;"
            f"font-weight: 500;"
        )
        self.status_selection.setToolTip(
            "Selected layer(s) — click rows in the layer panel to "
            "single them out."
        )
        # Truncate with ellipsis instead of growing the bar when the
        # name is very long. The right zone has its own fixed-ish
        # footprint, so leaving the middle zone with ``Preferred``
        # policy keeps the layout calm under long names.
        self.status_selection.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred,
        )
        mid_row.addWidget(self._focus_tag)
        mid_row.addWidget(self.status_selection)
        mid_container = QWidget(self)
        mid_container.setLayout(mid_row)
        mid_container.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred,
        )
        layout.addWidget(mid_container, 1)

        # ---- Right zone (perf metrics) ---------------------------------
        self.status_right = QLabel()
        self.status_right.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self.status_right.setTextFormat(Qt.TextFormat.RichText)
        # Same monospace font the legacy implementation used, so the
        # existing ``format_perf_html`` HTML lays out identically.
        self.status_right.setFont(F.mono(F.SIZE_MONO_LABEL))
        self.status_right.setStyleSheet(f"color: {H.T_SEC};")
        layout.addWidget(self.status_right, 0)

    # ------------------------------------------------------------------ Public API

    def set_status(self, text: str) -> None:
        """Set the left-zone contextual message.

        Same contract as the legacy
        :meth:`MainWindow.set_status` — accepts any plain-text string.
        Used by load progress messages, in/out feedback,
        "No sequence loaded — File → Open" hints, etc.
        """
        self.status_left.setText(text)

    def set_session_dot(self, color: str | None) -> None:
        """Set the colour of the left-zone status dot.

        Pass a hex string (e.g. ``H.RUNNING``, ``H.ACC``,
        ``H.DANGER``) or ``None`` to hide the dot. The widget
        keeps its layout footprint when the dot is hidden, so the
        message text stays at a constant x position.
        """
        self._dot.set_color(color)

    def set_selected_layers(self, text: str) -> None:
        """Set the middle-zone focused-layer name text.

        Empty string hides the FOCUS tag entirely (so the middle
        zone collapses cleanly when nothing is focused — no
        leftover "FOCUS" pill pointing at nothing).
        """
        text = (text or "").strip()
        self.status_selection.setText(text)
        # Show / hide the FOCUS tag in sync with the text presence.
        # Layout reserves the space either way so the column doesn't
        # jump when toggling.
        self._focus_tag.setVisible(bool(text))

    def set_perf_html(self, html: str) -> None:
        """Set the right-zone perf metrics blob.

        Receives the rich-text HTML produced by
        :func:`img_player.ui.status_format.format_perf_html`. The
        widget renders it in the right ``QLabel`` configured for
        ``Qt.TextFormat.RichText``.
        """
        self.status_right.setText(html)
