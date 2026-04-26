"""Transport bar: I/O markers, loop mode, playback controls, FPS."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from img_player.player.state import LoopMode
from img_player.ui.frame_display import DisplayMode, FrameDisplay
from img_player.ui.icons import make_icon
from img_player.ui.theme import G, H, S

if TYPE_CHECKING:
    from img_player.player.state import PlaybackState


_LOOP_CYCLE = [LoopMode.LOOP, LoopMode.ONCE, LoopMode.PING_PONG]
_LOOP_LABELS = {
    LoopMode.LOOP:      ("↻", "Loop (play → first frame at the end)"),
    LoopMode.ONCE:      ("→", "Play once (stop at the end)"),
    LoopMode.PING_PONG: ("⇌", "Ping-pong (reverse at the end)"),
}


class TransportBar(QWidget):  # type: ignore[misc]
    """Emits high-level intents — the controller applies the logic."""

    play_toggled     = Signal()
    stop_clicked     = Signal()
    step_clicked     = Signal(int)   # +1 or -1
    jump_to_ends     = Signal(int)   # -1 = first frame, +1 = last
    fps_changed      = Signal(float)
    mark_in_clicked  = Signal()
    mark_out_clicked = Signal()
    clear_in_out_clicked = Signal()
    loop_mode_requested  = Signal(object)  # LoopMode
    # Reverse-play button (in addition to the existing "J" shortcut).
    # The transport doesn't know about playback state — it just signals
    # "the user wants to play in this direction" and the controller
    # decides whether to start, flip, or pause accordingly.
    reverse_play_clicked  = Signal()
    # User typed a frame / timecode in the FrameDisplay and pressed
    # Enter. Carries the absolute frame index.
    frame_seek_requested  = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(G.TRANSPORT_H)

        self._loop_mode = LoopMode.LOOP

        # --- In/Out markers -------------------------------------------------
        self._mark_in_btn  = _text_button(" I ", "Mark IN at current frame (I)")
        self._mark_out_btn = _text_button(" O ", "Mark OUT at current frame (O)")
        self._clear_io_btn = _text_button("⌫",  "Clear IN/OUT range (Shift+R)")

        self._mark_in_btn.clicked.connect(self.mark_in_clicked.emit)
        self._mark_out_btn.clicked.connect(self.mark_out_clicked.emit)
        self._clear_io_btn.clicked.connect(self.clear_in_out_clicked.emit)

        # --- Loop mode ------------------------------------------------------
        self._loop_btn = _text_button("↻", "Loop mode (click to cycle)")
        self._loop_btn.clicked.connect(self._cycle_loop_mode)

        # --- Playback controls ---------------------------------------------
        # All transport buttons use our custom SVG icon set so they
        # match ui_mockup.html. Both play buttons are in the warm
        # accent (orange); the others are TEXT_PRIMARY (white-ish) for
        # visual hierarchy.
        self._first_btn = _icon_button(make_icon("first"), "Go to first frame (Home)")
        self._prev_btn  = _icon_button(make_icon("prev"),  "Previous frame (Left)")
        self._reverse_play_btn = _icon_button(
            make_icon("play_reverse", color=H.ACCENT),
            "Play in reverse (J)",
        )
        self._play_btn  = _icon_button(
            make_icon("play", color=H.ACCENT), "Play / Pause forward (Space / L)"
        )
        self._stop_btn  = _icon_button(make_icon("stop"),  "Stop")
        self._next_btn  = _icon_button(make_icon("next"),  "Next frame (Right)")
        self._last_btn  = _icon_button(make_icon("last"),  "Go to last frame (End)")

        # --- Frame / timecode display -------------------------------------
        # Sits between the navigation half and the playback half so the
        # current frame stays at the centre of attention, matching the
        # Nuke-like layout the user asked for.
        self._frame_display = FrameDisplay(self)
        self._frame_display.frame_seek_requested.connect(self.frame_seek_requested.emit)

        self._first_btn.clicked.connect(lambda: self.jump_to_ends.emit(-1))
        self._prev_btn.clicked.connect(lambda: self.step_clicked.emit(-1))
        self._reverse_play_btn.clicked.connect(self.reverse_play_clicked.emit)
        self._play_btn.clicked.connect(self.play_toggled.emit)
        self._stop_btn.clicked.connect(self.stop_clicked.emit)
        self._next_btn.clicked.connect(lambda: self.step_clicked.emit(1))
        self._last_btn.clicked.connect(lambda: self.jump_to_ends.emit(1))

        # --- FPS ------------------------------------------------------------
        self._fps_combo = QComboBox()
        self._fps_combo.setEditable(True)
        self._fps_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for rate in ("23.976", "24", "25", "29.97", "30", "48", "50", "59.94", "60"):
            self._fps_combo.addItem(rate)
        self._fps_combo.setCurrentText("24")
        self._fps_combo.setFixedWidth(72)
        self._fps_combo.setFixedHeight(G.INPUT_H)
        self._fps_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._fps_combo.setToolTip("Playback rate (fps)")
        self._fps_combo.currentTextChanged.connect(self._on_fps_text)

        # --- Layout ---------------------------------------------------------
        layout = QHBoxLayout(self)
        layout.setContentsMargins(S.MD, S.SM, S.MD, S.SM)
        layout.setSpacing(S.SM)
        layout.addStretch(1)

        layout.addWidget(self._mark_in_btn)
        layout.addWidget(self._mark_out_btn)
        layout.addWidget(self._clear_io_btn)
        layout.addWidget(_separator())
        layout.addWidget(self._loop_btn)
        layout.addWidget(_separator())

        # Layout order — navigation on the left, playback in the
        # middle (anchored around the FrameDisplay so the current
        # frame is the visual centre of the bar), navigation on the
        # right. Reverse-play sits to the right of the FrameDisplay
        # exactly as the user asked, mirroring the forward-play btn
        # that sits one further out.
        layout.addWidget(self._first_btn)
        layout.addWidget(self._prev_btn)
        layout.addWidget(self._frame_display)
        layout.addWidget(self._reverse_play_btn)
        layout.addWidget(self._play_btn)
        layout.addWidget(self._stop_btn)
        layout.addWidget(self._next_btn)
        layout.addWidget(self._last_btn)

        layout.addWidget(_separator())
        fps_label = QLabel("FPS")
        fps_label.setFixedWidth(24)
        layout.addWidget(fps_label)
        layout.addWidget(self._fps_combo)
        layout.addStretch(1)

        self._refresh_loop_button()

    # ------------------------------------------------------------------ Public

    def update_from_state(self, state: PlaybackState) -> None:
        # Swap the play button's icon between the two states. We keep
        # the warm ACCENT colour on "play" (encourages the click) and
        # use the neutral TEXT_PRIMARY on "pause" (calmer, less
        # attention-grabbing while playback is in progress).
        # The forward play button shows pause only when the controller
        # is *playing forward*; while playing backward the pause icon
        # belongs to the reverse button instead.
        playing_fwd = state.is_playing and state.direction >= 0
        playing_rev = state.is_playing and state.direction < 0
        if playing_fwd:
            self._play_btn.setIcon(make_icon("pause"))
        else:
            self._play_btn.setIcon(make_icon("play", color=H.ACCENT))
        if playing_rev:
            self._reverse_play_btn.setIcon(make_icon("pause"))
        else:
            self._reverse_play_btn.setIcon(make_icon("play_reverse", color=H.ACCENT))

        # Push the current frame into the editable display.
        self._frame_display.set_frame(state.current_frame)
        self._frame_display.set_fps(state.fps)

        if state.loop_mode != self._loop_mode:
            self._loop_mode = state.loop_mode
            self._refresh_loop_button()

        current_fps = self._parse_fps(self._fps_combo.currentText())
        if current_fps is None or abs(current_fps - state.fps) > 1e-3:
            self._fps_combo.blockSignals(True)
            self._fps_combo.setCurrentText(self._format_fps(state.fps))
            self._fps_combo.blockSignals(False)

    def set_display_mode(self, mode: DisplayMode) -> None:
        """Propagate the global frame/timecode toggle (View menu) to
        the FrameDisplay so it stays in sync with the timeline."""
        self._frame_display.set_display_mode(mode)

    # ------------------------------------------------------------------ Internals

    def _cycle_loop_mode(self) -> None:
        try:
            idx = _LOOP_CYCLE.index(self._loop_mode)
        except ValueError:
            idx = 0
        self._loop_mode = _LOOP_CYCLE[(idx + 1) % len(_LOOP_CYCLE)]
        self._refresh_loop_button()
        self.loop_mode_requested.emit(self._loop_mode)

    def _refresh_loop_button(self) -> None:
        label, tooltip = _LOOP_LABELS[self._loop_mode]
        self._loop_btn.setText(label)
        self._loop_btn.setToolTip(tooltip)

    def _on_fps_text(self, text: str) -> None:
        fps = self._parse_fps(text)
        if fps is not None:
            self.fps_changed.emit(fps)

    @staticmethod
    def _parse_fps(text: str) -> float | None:
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            return None
        if value <= 0 or value > 240:
            return None
        return value

    @staticmethod
    def _format_fps(fps: float) -> str:
        if abs(fps - round(fps)) < 1e-3:
            return f"{round(fps)}"
        return f"{fps:.3f}".rstrip("0").rstrip(".")


def _icon_button(icon: QIcon, tooltip: str) -> QPushButton:
    btn = QPushButton()
    btn.setIcon(icon)
    btn.setIconSize(QSize(G.ICON_SIZE, G.ICON_SIZE))
    btn.setFixedSize(G.BTN_TRANSPORT_W, G.BTN_TRANSPORT_H)
    btn.setToolTip(tooltip)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return btn


def _text_button(label: str, tooltip: str) -> QPushButton:
    btn = QPushButton(label)
    btn.setFixedSize(G.BTN_TEXT_W, G.BTN_TRANSPORT_H)
    btn.setToolTip(tooltip)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return btn


def _separator() -> QWidget:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setFixedWidth(1)
    line.setFixedHeight(18)
    line.setStyleSheet(f"background-color: {H.BORDER_DEFAULT};")
    return line
