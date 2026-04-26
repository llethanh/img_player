"""Transport bar: I/O markers, loop mode, playback controls, FPS."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from img_player.player.state import LoopMode
from img_player.sequence.channels import group_channels
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

    # play_toggled is reserved for the Space / K shortcut on MainWindow:
    # plain "toggle play/pause without touching direction". The two
    # *direction* buttons of the transport bar use forward_play_clicked
    # / reverse_play_clicked instead because clicking them is a
    # statement of intent ("I want to go this way"), not a toggle.
    play_toggled     = Signal()
    forward_play_clicked = Signal()
    reverse_play_clicked = Signal()
    step_clicked     = Signal(int)   # +1 or -1
    jump_to_ends     = Signal(int)   # -1 = first frame, +1 = last
    fps_changed      = Signal(float)
    mark_in_clicked  = Signal()
    mark_out_clicked = Signal()
    clear_in_out_clicked = Signal()
    loop_mode_requested  = Signal(object)  # LoopMode
    # User typed a frame / timecode in the FrameDisplay and pressed
    # Enter. Carries the absolute frame index.
    frame_seek_requested  = Signal(int)
    # Multichannel EXR: user picked a different channel to display.
    # Carries either ``None`` (= "RGB" composite, default) or a list
    # like ``["Z"]`` / ``["N.X"]`` for a single channel readout.
    channels_requested = Signal(object)
    # Zoom — either ``None`` for fit-to-window, or a float factor
    # (1.0 = 100 %, 0.5 = 50 %, 2.0 = 200 %).
    zoom_requested = Signal(object)
    # Per-channel show/hide. Carries a 4-tuple of bools:
    # (R, G, B, A) where True = visible, False = masked. The viewer
    # multiplies the corresponding channel by 0 in the shader — so
    # toggling is free runtime cost and does not invalidate the
    # frame cache.
    channel_mask_changed = Signal(tuple)

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
        # Layout order: navigation outward, then reverse_play left of
        # the frame display, forward_play right of it. The frame
        # display itself sits at the visual centre between the two
        # play buttons — matches the Nuke-style transport.
        self._first_btn = _icon_button(make_icon("first"), "Go to first frame (Home)")
        self._prev_btn  = _icon_button(make_icon("prev"),  "Previous frame (Left)")
        self._reverse_play_btn = _icon_button(
            make_icon("play_reverse", color=H.ACCENT),
            "Play in reverse (J)",
        )
        self._play_btn  = _icon_button(
            make_icon("play", color=H.ACCENT),
            "Play forward (L)",
        )
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
        # Forward-play *button* is intent-bearing (= play in this
        # direction) — not a plain toggle. The plain toggle stays
        # bound to the Space / K shortcuts in MainWindow via the
        # play_toggled signal.
        self._play_btn.clicked.connect(self.forward_play_clicked.emit)
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

        # --- Channel selector ----------------------------------------------
        # Multichannel EXR support: pick which channel(s) to display.
        # "RGB" is the composite default; the others appear once a
        # sequence is loaded (populated via ``set_available_channels``).
        # ``_channel_groups`` maps a UI label to the OIIO channel list
        # to load — built by ``set_available_channels`` from the
        # grouped channel list.
        self._channel_groups: dict[str, tuple[str, ...]] = {}
        self._channel_combo = QComboBox()
        self._channel_combo.setFixedWidth(96)
        self._channel_combo.setFixedHeight(G.INPUT_H)
        self._channel_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._channel_combo.setToolTip("Channel to display")
        self._channel_combo.addItem("RGB")  # always the first entry
        self._channel_combo.currentIndexChanged.connect(self._on_channel_changed)

        # --- RGBA mute toggles ---------------------------------------------
        # Four small checkable buttons — the viewer's fragment shader
        # multiplies each component by the matching mask, so toggling
        # is essentially free at runtime and doesn't invalidate the
        # frame cache the way the channel-selector does.
        self._channel_btns: dict[str, QPushButton] = {}
        for letter, tooltip in (
            ("R", "Show / hide red channel"),
            ("G", "Show / hide green channel"),
            ("B", "Show / hide blue channel"),
            ("A", "Show / hide alpha channel"),
        ):
            btn = _channel_toggle_button(letter, tooltip)
            btn.toggled.connect(self._emit_channel_mask)
            self._channel_btns[letter] = btn

        # --- Zoom selector -------------------------------------------------
        # The combo is *editable* solely so we can call setText() with
        # arbitrary values like "127%" coming from the wheel — but the
        # internal QLineEdit is read-only so the user can't type. We
        # also intercept mouse clicks on the line edit to open the
        # dropdown (Qt's default behaviour for editable combos is to
        # focus the line edit, *not* show the popup, which left the
        # dropdown unreachable in our config).
        self._zoom_combo = QComboBox()
        self._zoom_combo.setEditable(True)
        self._zoom_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        line_edit = self._zoom_combo.lineEdit()
        if line_edit is not None:
            line_edit.setReadOnly(True)
            line_edit.setCursor(Qt.CursorShape.PointingHandCursor)
            # The event filter routes a click on the line edit's area
            # to combo.showPopup(). See _ZoomLineEditClickFilter below.
            self._zoom_click_filter = _ZoomLineEditClickFilter(self._zoom_combo)
            line_edit.installEventFilter(self._zoom_click_filter)
        # Presets ordered top → bottom from largest zoom to smallest;
        # Fit pinned at the very top so the user always finds it
        # without scrolling.
        for label in ("Fit", "200%", "150%", "100%", "50%", "25%", "15%", "10%"):
            self._zoom_combo.addItem(label)
        self._zoom_combo.setCurrentText("Fit")
        self._zoom_combo.setFixedWidth(78)
        self._zoom_combo.setFixedHeight(G.INPUT_H)
        self._zoom_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._zoom_combo.setToolTip(
            "Zoom level — pick a preset, or wheel-zoom in the viewer for arbitrary values"
        )
        # Listen to *activated* (= user pick from dropdown), not
        # currentTextChanged (which fires every time we setText() from
        # the wheel and would feedback-loop).
        self._zoom_combo.activated.connect(self._on_zoom_picked)

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

        # Layout order — navigation outward, the two direction-aware
        # play buttons hugging the FrameDisplay at the visual centre.
        # No stop button here; Stop the action is still available via
        # the controller API but not as a transport widget — pause +
        # seek-to-IN-frame is the natural way to "stop" and the user
        # asked to free up the bar.
        layout.addWidget(self._first_btn)
        layout.addWidget(self._prev_btn)
        layout.addWidget(self._reverse_play_btn)
        layout.addWidget(self._frame_display)
        layout.addWidget(self._play_btn)
        layout.addWidget(self._next_btn)
        layout.addWidget(self._last_btn)

        layout.addWidget(_separator())
        fps_label = QLabel("FPS")
        fps_label.setFixedWidth(24)
        layout.addWidget(fps_label)
        layout.addWidget(self._fps_combo)

        layout.addWidget(_separator())
        channel_label = QLabel("CH")
        channel_label.setFixedWidth(20)
        layout.addWidget(channel_label)
        layout.addWidget(self._channel_combo)

        # The four RGBA mute toggles sit right after the channel combo,
        # because they're conceptually about the *same* thing (what
        # of the loaded data ends up on screen) — just at a finer
        # grain.
        for letter in ("R", "G", "B", "A"):
            layout.addWidget(self._channel_btns[letter])

        layout.addWidget(_separator())
        zoom_label = QLabel("Zoom")
        zoom_label.setFixedWidth(34)
        layout.addWidget(zoom_label)
        layout.addWidget(self._zoom_combo)
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

    def set_available_channels(self, channels: tuple[str, ...]) -> None:
        """Replace the channel-selector content with grouped channels.

        Layers like ``albedo.R``/``.G``/``.B`` collapse into a single
        ``"albedo"`` entry that loads the three channels as an RGB
        composite — same convention as Nuke's channel selector.
        Single-component channels (``Z``, ``volume_Z``,
        ``normal.X``…) keep their own entry. The first entry is
        always the beauty ``"RGB"``/``"RGBA"`` from the root.

        The mapping ``label → list[channel]`` is stored on the widget
        so :meth:`_on_channel_changed` knows which channels to ask
        the cache for.
        """
        groups = group_channels(channels) if channels else []
        # Cache the mapping label → channels so the dropdown handler
        # can look up the right OIIO channel list when the user
        # picks an item.
        self._channel_groups: dict[str, tuple[str, ...]] = {
            g.label: g.channels for g in groups
        }

        self._channel_combo.blockSignals(True)
        self._channel_combo.clear()
        if not groups:
            # No header info yet — at least show RGB so the combo
            # isn't blank. ``None`` → reader's default (R/G/B/A).
            self._channel_combo.addItem("RGB")
        else:
            for g in groups:
                self._channel_combo.addItem(g.label)
        self._channel_combo.setCurrentIndex(0)
        self._channel_combo.blockSignals(False)

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

    def _emit_channel_mask(self) -> None:
        """Bundle the four RGBA toggle states and emit
        ``channel_mask_changed`` with a (R, G, B, A) bool tuple."""
        mask = tuple(
            self._channel_btns[letter].isChecked() for letter in ("R", "G", "B", "A")
        )
        self.channel_mask_changed.emit(mask)

    def _on_channel_changed(self, index: int) -> None:
        """Translate the combo selection into a ``channels_requested``
        signal.

        Index 0 (the beauty pass) emits ``None`` so the cache reverts
        to the reader's default (R/G/B/A). Any other index looks up
        the cached label → channels mapping built by
        :meth:`set_available_channels` and emits the matching list.
        """
        if index <= 0:
            self.channels_requested.emit(None)
            return
        label = self._channel_combo.itemText(index)
        channels = self._channel_groups.get(label)
        if channels is None:
            # Defensive: if for some reason the mapping is out of sync,
            # treat the label itself as a single channel name. Better
            # than swallowing the click silently.
            channels = (label,)
        self.channels_requested.emit(list(channels))

    def _on_zoom_picked(self, index: int) -> None:
        """User picked a preset from the dropdown.

        Hooked to ``activated`` (not ``currentTextChanged``) so we
        only fire on a real user interaction — calls to
        ``set_zoom_display`` from the wheel back-channel don't
        re-emit and ping-pong.
        """
        text = self._zoom_combo.itemText(index).strip().lower().rstrip("%")
        if text in ("", "fit"):
            self.zoom_requested.emit(None)
            return
        try:
            percent = float(text)
        except ValueError:
            return
        self.zoom_requested.emit(percent / 100.0)

    def set_zoom_display(self, factor: object) -> None:
        """Reflect a zoom value in the combo without re-emitting.

        Called from the wiring code when the *wheel* (not the combo)
        changed the zoom — keeps the combo in sync without bouncing
        a second ``zoom_requested`` back through the same path.
        """
        self._zoom_combo.blockSignals(True)
        if factor is None:
            self._zoom_combo.setCurrentText("Fit")
        else:
            try:
                pct = round(float(factor) * 100)
                self._zoom_combo.setCurrentText(f"{pct}%")
            except (TypeError, ValueError):
                pass
        self._zoom_combo.blockSignals(False)

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


# Per-letter colours used by the RGBA channel-mute toggles. Matches
# the convention every VFX viewer follows: red letter for R, green
# for G, blue for B, neutral grey for A. Keeps the buttons readable
# at a glance even when several are off.
_CHANNEL_BTN_COLORS = {
    "R": "#E0606E",  # warm red, not too saturated
    "G": "#6CC275",  # mid green
    "B": "#5E9DD8",  # mid blue
    "A": H.TEXT_SECONDARY,
}


def _channel_toggle_button(letter: str, tooltip: str) -> QPushButton:
    btn = QPushButton(letter)
    btn.setFixedSize(22, G.BTN_TRANSPORT_H)
    btn.setCheckable(True)
    btn.setChecked(True)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    btn.setToolTip(tooltip)
    color = _CHANNEL_BTN_COLORS[letter]
    # Inline QSS: enabled = letter colour, disabled / muted = dim grey.
    # The button still uses the global QPushButton border / background
    # so it sits visually with the rest of the transport bar.
    btn.setStyleSheet(
        "QPushButton {"
        f"  color: {color};"
        f"  font-weight: 600;"
        f"  padding: 0;"
        "}"
        "QPushButton:!checked {"
        f"  color: {H.TEXT_DISABLED};"
        f"  background: {H.BG_RAISED};"
        "}"
    )
    return btn


class _ZoomLineEditClickFilter(QObject):
    """Make a click on the zoom QLineEdit open the combo's dropdown.

    By default an editable QComboBox treats a click on its line edit
    as a focus-and-edit gesture. We've made the line edit read-only
    (so the user can't type), but Qt still doesn't open the popup —
    it just focuses the field. This filter intercepts the mouse
    press and forwards it to ``combo.showPopup()``. Filter is
    attached to the line edit, not the combo itself, because that's
    where the click lands.
    """

    def __init__(self, combo: QComboBox) -> None:
        super().__init__(combo)
        self._combo = combo

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: D401, N802
        if event.type() == QEvent.Type.MouseButtonPress and isinstance(event, QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                self._combo.showPopup()
                return True  # consume — don't move focus to the line edit
        return super().eventFilter(watched, event)
