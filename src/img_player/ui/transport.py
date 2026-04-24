"""Transport bar: play/pause, stop, prev/next frame, first/last frame."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QStyle, QWidget

if TYPE_CHECKING:
    from img_player.player.state import PlaybackState


class TransportBar(QWidget):  # type: ignore[misc]
    """Emits high-level intents — the actual playback logic lives in the controller."""

    play_toggled = Signal()
    stop_clicked = Signal()
    step_clicked = Signal(int)  # +1 or -1
    jump_to_ends = Signal(int)  # -1 = first frame, +1 = last

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        style = self.style()

        self._first_btn = _icon_button(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward),
            "Go to first frame (Home)",
        )
        self._prev_btn = _icon_button(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward), "Previous frame (Left)"
        )
        self._play_btn = _icon_button(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay), "Play / Pause (Space)"
        )
        self._stop_btn = _icon_button(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaStop), "Stop"
        )
        self._next_btn = _icon_button(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward), "Next frame (Right)"
        )
        self._last_btn = _icon_button(
            style.standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward), "Go to last frame (End)"
        )

        self._first_btn.clicked.connect(lambda: self.jump_to_ends.emit(-1))
        self._prev_btn.clicked.connect(lambda: self.step_clicked.emit(-1))
        self._play_btn.clicked.connect(self.play_toggled.emit)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        self._next_btn.clicked.connect(lambda: self.step_clicked.emit(1))
        self._last_btn.clicked.connect(lambda: self.jump_to_ends.emit(1))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)
        layout.addStretch(1)
        for btn in (
            self._first_btn,
            self._prev_btn,
            self._play_btn,
            self._stop_btn,
            self._next_btn,
            self._last_btn,
        ):
            layout.addWidget(btn)
        layout.addStretch(1)

    def update_from_state(self, state: PlaybackState) -> None:
        """Reflect whether we're currently playing (play icon <-> pause icon)."""
        style = self.style()
        icon = style.standardIcon(
            QStyle.StandardPixmap.SP_MediaPause
            if state.is_playing
            else QStyle.StandardPixmap.SP_MediaPlay
        )
        self._play_btn.setIcon(icon)


def _icon_button(icon: QIcon, tooltip: str) -> QPushButton:
    btn = QPushButton()
    btn.setIcon(icon)
    btn.setIconSize(QSize(22, 22))
    btn.setFixedSize(34, 30)
    btn.setToolTip(tooltip)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return btn
