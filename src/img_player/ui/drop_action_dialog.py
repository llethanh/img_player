"""Modal asking the user how to handle a path drop / Open when layers exist.

Three outcomes:

* **Add as layer** — append a new top-of-stack layer; existing
  layers stay loaded.
* **Replace** — clear the stack and load this path as the only
  layer (= legacy single-sequence flow).
* **Cancel** — abort the load.

A "Remember for this session" checkbox lets the user lock in their
choice so subsequent drops don't bring the dialog back up. The
remembered action lives on :class:`ImgPlayerApp` as
``_drop_action_remember`` and resets to ``None`` whenever the
stack drops to zero layers (= a fresh New / closed-empty session).
"""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from img_player.ui.theme import F, S


DropAction = Literal["add", "replace", "cancel"]


class DropActionDialog(QDialog):  # type: ignore[misc]
    """Three-button modal: Add / Replace / Cancel."""

    def __init__(self, path: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add or Replace?")
        self.setModal(True)
        self.setMinimumWidth(420)

        self._action: DropAction = "cancel"
        self._remember_checked: bool = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(S.LG, S.LG, S.LG, S.LG)
        outer.setSpacing(S.MD)

        msg = QLabel(
            f"You already have layers loaded.\n"
            f"What should be done with <b>{path}</b>?"
        )
        msg.setWordWrap(True)
        msg.setTextFormat(Qt.TextFormat.RichText)
        msg.setFont(F.ui(F.SIZE_SM))
        outer.addWidget(msg)

        # Remember checkbox sits above the buttons so it reads as a
        # modifier on the upcoming choice rather than a separate
        # action.
        self._remember = QCheckBox("Remember this choice for the rest of the session")
        self._remember.setFont(F.ui(F.SIZE_XS))
        outer.addWidget(self._remember)

        # Button row — Add is highlighted (the most common intent
        # once the user has multiple layers in mind), Replace is the
        # legacy default, Cancel is on the left for keyboard-Escape
        # discoverability.
        row = QHBoxLayout()
        row.setSpacing(S.SM)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel)
        row.addWidget(cancel_btn)

        row.addStretch(1)

        replace_btn = QPushButton("Replace")
        replace_btn.setToolTip(
            "Drop the current layers and load this path as a fresh single layer."
        )
        replace_btn.clicked.connect(self._on_replace)
        row.addWidget(replace_btn)

        add_btn = QPushButton("Add as layer")
        add_btn.setDefault(True)
        add_btn.setAutoDefault(True)
        add_btn.setToolTip(
            "Append this path as a new top-of-stack layer. Existing layers stay loaded."
        )
        add_btn.clicked.connect(self._on_add)
        row.addWidget(add_btn)

        outer.addLayout(row)

    # ------------------------------------------------------------------ Public

    @property
    def action(self) -> DropAction:
        return self._action

    @property
    def remember(self) -> bool:
        return self._remember_checked

    @classmethod
    def ask(
        cls, path: str, parent: QWidget | None = None,
    ) -> tuple[DropAction, bool]:
        """Convenience wrapper: returns (action, remember_flag)."""
        dlg = cls(path, parent=parent)
        dlg.exec()
        return dlg.action, dlg.remember

    # ------------------------------------------------------------------ Slots

    def _on_cancel(self) -> None:
        self._action = "cancel"
        self._remember_checked = self._remember.isChecked()
        self.reject()

    def _on_replace(self) -> None:
        self._action = "replace"
        self._remember_checked = self._remember.isChecked()
        self.accept()

    def _on_add(self) -> None:
        self._action = "add"
        self._remember_checked = self._remember.isChecked()
        self.accept()
