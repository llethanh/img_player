"""Channel panel: show which channels are available in the current sequence.

For V1 this is display-only information; channel selection is scheduled for
V2 along with comparison / scopes.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout, QWidget


class ChannelPanel(QWidget):  # type: ignore[misc]
    """A tiny read-only panel listing the current sequence's channels."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._label = QLabel("—")
        self._label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        inner = QVBoxLayout()
        inner.addWidget(self._label)
        group = QGroupBox("Channels")
        group.setLayout(inner)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(group)
        layout.addStretch(1)

    def set_channels(self, channels: tuple[str, ...]) -> None:
        if not channels:
            self._label.setText("(no channel metadata)")
            return
        self._label.setText(", ".join(channels))
