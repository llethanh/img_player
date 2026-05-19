"""Transport bar: I/O markers, loop mode, playback controls, FPS."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from PySide6.QtCore import QEvent, QObject, QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QActionGroup,
    QColor,
    QIcon,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPixmap,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
)

from img_player.player.state import LoopMode
from img_player.sequence.channels import (
    ChannelGroup,
    ChannelSelection,
    group_channels,
)
from img_player.ui.channel_menu import ChannelMenu
from img_player.ui.frame_display import DisplayMode, FrameDisplay
from img_player.ui.icons import make_icon
from img_player.ui.theme import F, G, H, S

if TYPE_CHECKING:
    from img_player.player.state import PlaybackState


_LOOP_CYCLE = [LoopMode.LOOP, LoopMode.ONCE, LoopMode.PING_PONG]
# Native emoji glyphs for the three loop modes — same style as the
# annotation toolbar's ✏️ 🧽 📌 (colorful OS-rendered emojis rather
# than monochrome text symbols).
_LOOP_LABELS = {
    LoopMode.LOOP:      ("loop", "Loop (play → first frame at the end)"),
    LoopMode.ONCE:      ("step-fwd", "Play once (stop at the end)"),
    LoopMode.PING_PONG: ("swap-arrows", "Ping-pong (reverse at the end)"),
}


class _ChannelToolButton(QToolButton):  # type: ignore[misc]
    """``QToolButton`` whose Up / Down arrows step through the
    channel groups while the button has keyboard focus.

    Lets the user pick a channel by clicking the button (= focus
    grabbed via ``ClickFocus``), then nudge the active radio
    without re-opening the menu. Focus is preserved across menu
    close via the parent's ``aboutToHide`` wiring so the arrows
    keep working until the user clicks somewhere else.

    Also paints a translucent orange cache-fill bar in its own
    button face when ``set_active_progress`` is called — same
    visual idiom as the timeline cache bar and the per-row bars
    in the open menu, so the closed dropdown still tells the user
    "active channel is N % cached" at a glance.
    """

    def __init__(self, menu: ChannelMenu, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._menu_ref = menu
        # Negative = "no data, paint nothing" — keeps the button
        # clean for sequences where alt prefetch doesn't apply
        # (multi-layer, no AOVs, etc.).
        self._active_progress: float = -1.0

    def set_active_progress(self, fraction: float) -> None:
        """Update the cache-fill fraction painted under the button
        face. Pass any negative value to hide the bar."""
        if fraction < 0:
            new_value = -1.0
        else:
            new_value = max(0.0, min(1.0, float(fraction)))
        if new_value == self._active_progress:
            return
        self._active_progress = new_value
        self.update()

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        key = event.key()
        if key == Qt.Key.Key_Up:
            self._menu_ref.cycle_active(-1)
            event.accept()
            return
        if key == Qt.Key.Key_Down:
            self._menu_ref.cycle_active(+1)
            event.accept()
            return
        super().keyPressEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """Render the standard button first, then overlay the
        cache-fill bar so the text underneath stays readable
        through the translucent fill."""
        super().paintEvent(event)
        if self._active_progress < 0:
            return
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QBrush, QPainter, QPen
        from img_player.ui.theme import C
        # 2-px inset so the fill doesn't hug the button border —
        # gives the orange a frame of breathing room and keeps the
        # rounded-corner radius of the button intact.
        rect = QRectF(self.rect()).adjusted(2.0, 2.0, -2.0, -2.0)
        fill_w = rect.width() * self._active_progress
        if fill_w <= 0:
            return
        fill_rect = QRectF(rect.x(), rect.y(), fill_w, rect.height())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(C.CACHE_BAR))
        painter.drawRect(fill_rect)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(C.CACHE_BAR_BORDER, 1.0))
        painter.drawRect(fill_rect.adjusted(0.5, 0.5, -0.5, -0.5))


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
    # Channel selection (active group). Emitted whenever the user
    # picks a different radio in the channel menu — the cache uses
    # the carried :class:`ChannelSelection` to invalidate the right
    # decode plan.
    channel_selection_changed = Signal(object)
    # Zoom — either ``None`` for fit-to-window, or a float factor
    # (1.0 = 100 %, 0.5 = 50 %, 2.0 = 200 %).
    zoom_requested = Signal(object)
    # Per-channel show/hide. Carries a 4-tuple of bools:
    # (R, G, B, A) where True = visible, False = masked. The viewer
    # multiplies the corresponding channel by 0 in the shader — so
    # toggling is free runtime cost and does not invalidate the
    # frame cache.
    channel_mask_changed = Signal(tuple)
    # Transparency-background pick. Carries an int in 0..3:
    #   0 = checker (default), 1 = black, 2 = mid-grey, 3 = white.
    # Pure GL-uniform change — no cache invalidation, no re-decode.
    transparency_bg_mode_changed = Signal(int)
    # Annotation transport buttons (slice 4): the user can toggle the
    # toolbar visibility, and jump to the previous / next annotated
    # frame. App.py decides what "previous / next" means by reading
    # the AnnotationStore, and disables the buttons when there's
    # nothing on either side.
    annotation_toggle_clicked = Signal()
    annotation_prev_clicked = Signal()
    annotation_next_clicked = Signal()
    # Show / hide annotations during playback. Carries the new bool
    # state. Checked = visible while playing; unchecked = hidden so
    # the user gets a clean review pass without strokes flickering
    # over moving content. Mirrored on the ``A`` keyboard shortcut.
    annotation_show_during_play_toggled = Signal(bool)
    fullscreen_clicked = Signal()
    # Master audio (popup volume slider). Carries a float in
    # [0.0, 1.0] (linear gain, sliderValue / 100). Wired to
    # ``AudioOutput.set_master_gain`` in app.py. Mute is implicit:
    # the gain falls to zero when the slider sits at the bottom,
    # which silences the output naturally in the audio callback —
    # no separate mute toggle.
    master_volume_changed = Signal(float)
    # Export button (v0.5.0) — opens the export dialog. Disabled
    # until the app calls ``set_export_enabled(True)`` (which the
    # app does after a sequence loads).
    export_clicked = Signal()
    # Compare-mode toggle (v1.2). Carries no payload — the receiver
    # checks the button's ``isChecked()`` state via the public API.
    compare_toggled = Signal()
    # Contact-sheet toggle (v1.5.14). Same shape as compare — the
    # main click area flips on/off; the small arrow on the right
    # opens a popup with the grid / divisor / labels presets.
    contact_sheet_toggled = Signal()
    # Reload button (v0.5.1) — smart re-scan of the source folder,
    # keeping cached frames whose mtime hasn't changed.
    reload_clicked = Signal()
    # Alt-channel background prefetch kill-switch. Carries the new
    # paused state (``True`` = paused, ``False`` = resumed). Sits
    # right next to the channel selector since "pause channel cache"
    # is a verb on the same object the user is reading. App.py wires
    # this to ``PlayerController.set_alt_channel_paused``.
    channel_cache_pause_toggled = Signal(bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(G.TRANSPORT_H)

        self._loop_mode = LoopMode.LOOP

        # --- In/Out markers -------------------------------------------------
        # Brief §5 + §11.1: line-art monochrome SVG icons (mark-in /
        # mark-out / clear-in-out) replace the legacy emoji glyphs
        # (🚩 🏁 🧹) which rendered as colour emojis or empty squares
        # depending on the system font stack.
        self._mark_in_btn  = _icon_button(
            make_icon("mark-in"), "Mark IN at current frame (I)",
        )
        self._mark_out_btn = _icon_button(
            make_icon("mark-out"), "Mark OUT at current frame (O)",
        )
        self._clear_io_btn = _icon_button(
            make_icon("clear-in-out"), "Clear IN/OUT range (Shift+R)",
        )

        self._mark_in_btn.clicked.connect(self.mark_in_clicked.emit)
        self._mark_out_btn.clicked.connect(self.mark_out_clicked.emit)
        self._clear_io_btn.clicked.connect(self.clear_in_out_clicked.emit)

        # --- Loop mode ------------------------------------------------------
        # Brief §5: line-art loop icon (SVG) instead of the 🔁 emoji.
        # The button is checkable so the global ``btnIcon:checked`` QSS
        # rule paints the active state in ACC_BRIGHT — same orange
        # active treatment as the rest of the toggle buttons.
        # ``_refresh_loop_button`` swaps the icon as the user cycles
        # through LOOP / ONCE / PING_PONG modes.
        self._loop_btn = _icon_button(
            make_icon("loop"), "Loop mode (click to cycle)",
        )
        self._loop_btn.setCheckable(True)
        self._loop_btn.setChecked(True)  # default state = LOOP
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
        # Use the new brief icon set for the step / skip glyphs;
        # ``first`` / ``last`` are kept as backwards-compatible aliases
        # but the brief-named ``skip-start`` / ``skip-end`` carry the
        # updated geometry, and ``step-back`` / ``step-fwd`` replace
        # the legacy ``prev`` / ``next`` chevrons.
        self._first_btn = _icon_button(make_icon("skip-start"), "Go to first frame (Home)")
        self._prev_btn  = _icon_button(make_icon("step-back"),  "Previous frame (Left)")
        self._reverse_play_btn = _icon_button(
            make_icon("play_reverse", color="#1A1206"),
            "Play in reverse (J)",
        )
        self._play_btn  = _icon_button(
            make_icon("play", color="#1A1206"),
            "Play forward (L)",
        )
        self._next_btn  = _icon_button(make_icon("step-fwd"),   "Next frame (Right)")
        self._last_btn  = _icon_button(make_icon("skip-end"),   "Go to last frame (End)")

        # Play / reverse-play buttons get the wider 38 px ``btnPrimary``
        # footprint and the orange gradient. We overwrite the size set
        # by ``_icon_button`` (which uses CTRL_ICON_W = 30) so the
        # primary key visually outweighs the surrounding nav buttons.
        for play_btn in (self._reverse_play_btn, self._play_btn):
            play_btn.setObjectName("btnPrimary")
            play_btn.setFixedSize(G.CTRL_PRIMARY_W, G.CTRL_BUTTON_H)

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

        # --- Annotation controls (slice 4) ---------------------------------
        # Sit between the playback nav (last_btn) and FPS — a separate
        # logical group, exposed via three signals (toggle / prev /
        # next). The toggle button is checkable and reflects whether
        # the toolbar is currently visible.
        #
        # Brief §5 + §11: SVG line-art icons (cache-prev / pen /
        # cache-next / ann-hide) replace the legacy emoji glyphs
        # (⏮️ ✏️ ⏭️ 👁/🚫). The annotation_prev / next semantically
        # double as "previous / next annotated frame" navigation —
        # the cache-prev / cache-next glyphs (double chevrons) read
        # cleanly as "jump to neighbouring annotated frame" too.
        self._annotation_prev_btn = _icon_button(
            make_icon("cache-prev"), "Frame annotée précédente ([)",
        )
        # Pen — THE annotation-tool entry point. Painted in the warm
        # amber accent (ACC_BRIGHT) at rest so it visually reads as
        # "the annotation button" alongside the other monochrome
        # transport icons. When toggled ON (toolbar visible), the
        # global ``QPushButton:checked`` rule additionally tints the
        # background so the active state remains unambiguous.
        self._annotation_toggle_btn = _icon_button(
            make_icon("pen", color=H.ACC_BRIGHT),
            "Afficher / masquer la toolbar d'annotation (D)",
        )
        self._annotation_toggle_btn.setCheckable(True)
        self._annotation_next_btn = _icon_button(
            make_icon("cache-next"), "Frame annotée suivante (])",
        )
        # Show / hide annotations DURING playback. Checkable so the
        # user can lock it in either state. Default = checked
        # (annotations visible during play) to match the legacy
        # behaviour. Same toggle the ``A`` keyboard shortcut drives.
        # The ``btnIcon:checked`` global QSS rule paints the active
        # state in ACC_BRIGHT — no inline stylesheet needed anymore.
        # On toggle we keep the same icon (the slashed circle reads
        # as "hide" in both states); the checked state's background
        # tint signals "currently visible".
        self._annotation_show_play_btn = _icon_button(
            make_icon("ann-hide"),
            "Afficher les annotations pendant la lecture (A)",
        )
        self._annotation_show_play_btn.setCheckable(True)
        self._annotation_show_play_btn.setChecked(True)
        self._annotation_prev_btn.clicked.connect(self.annotation_prev_clicked.emit)
        self._annotation_toggle_btn.clicked.connect(
            self.annotation_toggle_clicked.emit
        )
        self._annotation_next_btn.clicked.connect(self.annotation_next_clicked.emit)
        self._annotation_show_play_btn.toggled.connect(
            self.annotation_show_during_play_toggled.emit
        )
        # Disabled by default — App.py enables them once the store has
        # annotated frames on either side of the current playhead.
        self._annotation_prev_btn.setEnabled(False)
        self._annotation_next_btn.setEnabled(False)

        # --- Export button (v0.5.0) -----------------------------------
        # 💾 floppy-disk emoji is the universal "save / export" cue.
        # Brief §8 + §11.3: SVG line-art glyphs replace the legacy
        # emoji buttons (💾 / 🔄) and the previous custom-QSS compare /
        # contact-sheet buttons. All four lean on the global #btnIcon
        # / #btnToggle QSS variants from theme.build_stylesheet() so
        # no ad-hoc border/hover stylesheets remain — the active
        # (checked) state automatically picks up ACC_BRIGHT background
        # tint + border.

        # --- Export button (SVG save) ---------------------------------
        # Disabled until a sequence is loaded — the app flips it on
        # via ``set_export_enabled``.
        self._export_btn = _icon_button(
            make_icon("save"),
            "Export sequence to image seq or video (Ctrl+Shift+E)",
        )
        self._export_btn.clicked.connect(self.export_clicked.emit)
        self._export_btn.setEnabled(False)

        # --- Reload button (SVG refresh) ------------------------------
        # Smart re-scan — keeps cached frames whose mtime is unchanged,
        # drops the rest, picks up files that were added to the source
        # folder while the app was running. Disabled until a sequence
        # is loaded.
        self._reload_btn = _icon_button(
            make_icon("refresh"),
            "Reload cache — re-scan source folder (Ctrl+R)",
        )
        self._reload_btn.clicked.connect(self.reload_clicked.emit)
        self._reload_btn.setEnabled(False)

        # --- Compare toggle (SVG ab-toggle) ---------------------------
        # The A/B compare entry point in the top-right toolbar. Painted
        # in the warm accent (ACC_BRIGHT) at rest so it reads as
        # "review mode" alongside the contact-sheet sibling — same
        # visual language as the pen button on the transport side.
        # Checkable: stays "down" + active-tinted while compare mode
        # is active (via global btnToggle:checked QSS). Disabled until
        # the layer stack reaches at least two layers.
        self._compare_btn = _icon_button(
            make_icon("ab-toggle", color=H.ACC_BRIGHT),
            "Compare two layers (W)",
        )
        self._compare_btn.setObjectName("btnToggle")
        self._compare_btn.setCheckable(True)
        self._compare_btn.clicked.connect(self.compare_toggled.emit)
        self._compare_btn.setEnabled(False)

        # --- Contact sheet toggle (SVG contact-sheet) -----------------
        # Sibling to the compare toggle. Same orange-icon + btnToggle
        # treatment so the two review-mode buttons read as a group.
        # The legacy ``⋯`` kebab menu next to it (settings popup)
        # remains here for backwards-compat with main_window.py
        # references but is NOT added to any visible layout — those
        # settings live in the ContactSheetBand toolbar that appears
        # above the viewer when the mode is on.
        from PySide6.QtWidgets import QMenu  # noqa: PLC0415 — local to this section
        self._contact_sheet_btn = _icon_button(
            make_icon("contact-sheet", color=H.ACC_BRIGHT),
            "Contact sheet — all layers tiled in a grid (Ctrl+G)",
        )
        self._contact_sheet_btn.setObjectName("btnToggle")
        self._contact_sheet_btn.setCheckable(True)
        self._contact_sheet_btn.clicked.connect(
            self.contact_sheet_toggled.emit,
        )

        # Settings popup: tiny QToolButton with the "options" glyph,
        # InstantPopup so a single click opens the menu. App-side
        # rebuilds the menu on ``aboutToShow``.
        self._contact_sheet_menu_btn = QToolButton(self)
        self._contact_sheet_menu_btn.setText("⋯")
        self._contact_sheet_menu_btn.setToolTip(
            "Contact sheet settings (grid, output size, labels)"
        )
        self._contact_sheet_menu_btn.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup,
        )
        self._contact_sheet_menu = QMenu(self._contact_sheet_menu_btn)
        self._contact_sheet_menu_btn.setMenu(self._contact_sheet_menu)
        # Compact width — only needs to fit the "⋯" glyph plus a
        # small padding so it doesn't compete with the icon button
        # for visual weight.
        self._contact_sheet_menu_btn.setFixedHeight(G.BTN_TRANSPORT_H)
        self._contact_sheet_menu_btn.setFixedWidth(20)
        # Match the toolbar's default transport buttons visually:
        # thin grey outline that lights up to accent on hover, no
        # chevron decoration (we replaced that with the centred
        # "⋯" so the button reads as a kebab menu). User feedback
        # was that an orange outline at rest made the cadre look
        # permanently "active".
        _border_d = f"1px solid {H.BORDER_DEFAULT}"
        _border_h = f"1px solid {H.ACCENT}"
        self._contact_sheet_menu_btn.setStyleSheet(
            f"QToolButton {{"
            f"  background-color: {H.BG_SURFACE};"
            f"  color: {H.TEXT_PRIMARY};"
            f"  border: {_border_d};"
            f"  border-radius: 3px;"
            f"  font-size: 14px;"
            f"  padding: 0;"
            f"}}"
            f"QToolButton:hover {{"
            f"  background-color: {H.BG_HOVER};"
            f"  border: {_border_h};"
            f"}}"
            f"QToolButton::menu-indicator {{ image: none; width: 0; }}"
        )

        # --- FPS ------------------------------------------------------------
        # Plain editable line — no dropdown of presets. The user
        # types whatever rate they want (24, 23.976, 60, …). A
        # double validator keeps the field numeric; on Enter / focus
        # loss the new value fires ``fps_changed``.
        from PySide6.QtGui import QDoubleValidator
        self._fps_combo = QLineEdit()
        self._fps_combo.setText("24")
        self._fps_combo.setFixedWidth(40)
        self._fps_combo.setFixedHeight(G.CTRL_INPUT_H)
        self._fps_combo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fps_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._fps_combo.setToolTip("Playback rate (fps) — type a value, Enter to apply")
        # Brief §5 readout style: dark BG_BASE field, hairline border,
        # monospace digits. Overrides the global QLineEdit rule which
        # paints BG_SURFACE (too bright next to the gradient play key).
        self._fps_combo.setStyleSheet(
            "QLineEdit {"
            f"  background:{H.BG_BASE};"
            f"  border:1px solid {H.BORDER_DEF};"
            f"  border-radius:{G.RADIUS_MD}px;"
            f"  color:{H.T_PRI};"
            f"  font-family:{F.FAMILY_MONO};"
            f"  font-size:{F.SIZE_MONO_CODE}px;"
            "  font-weight:500;"
            "}"
            "QLineEdit:focus {"
            f"  border:1px solid {H.ACC};"
            f"  color:{H.ACC_BRIGHT};"
            "}"
        )
        validator = QDoubleValidator(0.1, 1000.0, 3, self._fps_combo)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self._fps_combo.setValidator(validator)
        # Commit on Enter or when the user clicks elsewhere.
        self._fps_combo.editingFinished.connect(
            lambda: self._on_fps_text(self._fps_combo.text()),
        )

        # --- Channel selector ----------------------------------------------
        # Multichannel EXR support: a QToolButton that opens the
        # :class:`ChannelMenu` popup. The button label is the active
        # group name (e.g. "RGB" / "albedo" / "Z"). Up / Down arrows
        # cycle through groups while the button has focus — see
        # :class:`_ChannelToolButton`.
        self._channel_menu = ChannelMenu(self)
        self._channel_menu.selection_changed.connect(self._on_channel_selection_changed)
        self._channel_button = _ChannelToolButton(self._channel_menu)
        self._channel_button.setFixedHeight(G.INPUT_H)
        self._channel_button.setMinimumWidth(96)
        self._channel_button.setToolTip(
            "Channel to display — click to open the menu, then use "
            "Up / Down arrows to step through groups while focused"
        )
        self._channel_button.setText("RGB")
        # InstantPopup (vs MenuButtonPopup) — the whole button area
        # opens the menu, no mini arrow split. Cohérent with the loop
        # button's visual: one bordered button = one click action.
        self._channel_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        # ClickFocus so the button can grab keyboard focus on a click,
        # which lets ``_ChannelToolButton.keyPressEvent`` see Up/Down
        # afterwards. ``aboutToHide`` re-grabs focus when the menu
        # closes via Close button so arrows still work without
        # re-clicking the button.
        self._channel_button.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        # Visual cue when focused — the orange ACCENT outline mirrors
        # the rest of the warm UI palette and makes "arrows will
        # navigate channels" obvious.
        self._channel_button.setStyleSheet(
            "QToolButton:focus { "
            f"  border: 1px solid {H.ACCENT}; "
            "}"
        )
        self._channel_button.setMenu(self._channel_menu)
        # Re-claim focus when the menu closes via the footer Close
        # button, so the user can keep stepping with arrows after
        # a popup interaction. Click-outside still moves focus to
        # whatever was clicked (= the user's explicit intent).
        self._channel_menu.aboutToHide.connect(
            lambda: self._channel_button.setFocus(Qt.FocusReason.OtherFocusReason),
        )

        # Track the current ChannelSelection so the button label can
        # be derived without re-querying the menu.
        self._current_selection: ChannelSelection | None = None

        # --- Pause / Resume channel-cache prefetch ----------------------
        # Small status-toggle right next to the channel selector. The
        # user may want to silence the alt-channel background prefetch
        # (disk I/O + decode + RAM churn for channels they're not
        # actively switching between). The button face is a status
        # indicator, not an action label:
        #   ▶ in green = prefetch is currently RUNNING (click to stop)
        #   ■ in red   = prefetch is currently STOPPED (click to resume)
        # Same "what's happening right now" idiom as the cache-fill
        # bar — the user reads state at a glance, the click flips it.
        self._channel_cache_paused: bool = False
        self._channel_cache_pause_btn = QPushButton()
        self._channel_cache_pause_btn.setFixedHeight(G.INPUT_H)
        # Wider than the square transport buttons so the glyph has
        # breathing room — the red square in particular reads cramped
        # when squeezed against the borders.
        self._channel_cache_pause_btn.setFixedWidth(36)
        self._channel_cache_pause_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._channel_cache_pause_btn.clicked.connect(
            self._on_channel_cache_pause_clicked,
        )
        # Paint the initial glyph + tooltip for the default
        # (= running) state.
        self._refresh_channel_cache_pause_button()

        # --- RGBA channel mode selector ------------------------------------
        # One button replaces the old four-toggle row. Left-click cycles
        # RGB → R → G → B → A → RGB. The dropdown arrow opens a menu
        # for a direct pick. The fragment shader still consumes the
        # same (R,G,B,A) uniform — see ``_emit_channel_mask`` for the
        # mode→mask mapping. Cache stays valid: this is a uniform-only
        # change just like the old 4-button row.
        self._channel_mode: str = "RGB"
        self._channel_mode_button = _ChannelModeButton()
        self._channel_mode_button.setFixedHeight(G.BTN_TRANSPORT_H)
        # Fixed width so the left half doesn't shrink when the mode
        # is a single letter (R / G / B / A) vs the wider ``RGB``
        # label. Sized to fit ``RGB`` + arrow + comfortable padding
        # while still reading as one of the small transport
        # controls.
        self._channel_mode_button.setFixedWidth(72)
        self._channel_mode_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._channel_mode_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextOnly,
        )
        self._channel_mode_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup,
        )
        self._channel_mode_button.setToolTip(
            "Channel view — click to cycle RGB → R → G → B → A, "
            "or use the dropdown to pick directly. "
            "A solos the alpha channel as grayscale "
            "(white = opaque, black = transparent).",
        )
        self._channel_mode_menu = self._build_channel_mode_menu()
        self._channel_mode_button.setMenu(self._channel_mode_menu)
        self._channel_mode_button.clicked.connect(self._cycle_channel_mode)
        self._apply_channel_mode_style()

        # --- Transparency background picker ---------------------------
        # Same widget shape as the channel-mode button next door:
        # a ``_ChannelModeButton`` in ``MenuButtonPopup`` mode where
        # the main click cycles through the four BG modes and the
        # right-side arrow opens the dropdown menu. Pure GL uniform
        # change either way, no cache invalidation. The button face
        # is an icon-only square swatch (checker pattern, black,
        # grey, white) so the current mode reads at a glance — the
        # text-based "Blk" label was lost on the dark menu chrome.
        self._current_bg_mode: int = 0
        self._bg_button = _ChannelModeButton()
        self._bg_button.setFixedHeight(G.BTN_TRANSPORT_H)
        # 22 (icon) + 20 (arrow) + slack. Earlier 44 was too tight:
        # Qt's MenuButtonPopup zone reserves a bit more than the
        # declared ``menu-button width`` for the arrow chrome, so
        # the arrow ended up drawing into the icon's right edge.
        # 52 keeps them visually separated while staying compact
        # enough that the ``BG :`` label sits right next to the
        # swatch.
        self._bg_button.setFixedWidth(52)
        self._bg_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._bg_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonIconOnly,
        )
        self._bg_button.setIconSize(QSize(22, 22))
        self._bg_button.setPopupMode(
            QToolButton.ToolButtonPopupMode.MenuButtonPopup,
        )
        self._bg_button.setToolTip(
            "Background for transparent pixels — click to cycle "
            "Checker → Black → Mid grey → White, or use the dropdown "
            "to pick directly.",
        )
        self._bg_menu = self._build_bg_menu()
        self._bg_button.setMenu(self._bg_menu)
        self._bg_button.clicked.connect(self._cycle_bg_mode)
        self._apply_bg_button_style()

        # NB: T + αS toggles moved to each :class:`LayerRow` since
        # they reflect *per-layer* state. Keeping them here would
        # silently target a "focused layer" that's invisible from
        # the transport's perspective — confusing UX.

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
        # Wide enough to fit "200%" (4 chars) plus the dropdown arrow
        # without truncation. The previous 52 px was clipping values
        # like "200%" and "Fit ▼".
        self._zoom_combo.setFixedWidth(70)
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
        # NB: ``_reverse_play_btn`` is intentionally NOT added to the
        # visible layout — brief §5 specifies a single play key, so
        # the reverse-play affordance lives only on the J keyboard
        # shortcut now. The button object itself stays alive (parented
        # to ``self`` below) so existing signal wiring +
        # ``set_playback_enabled`` calls keep working unchanged.
        self._reverse_play_btn.setParent(self)
        self._reverse_play_btn.hide()
        layout.addWidget(self._play_btn)
        layout.addWidget(self._next_btn)
        layout.addWidget(self._last_btn)
        # NB: ``self._frame_display`` is constructed in __init__ but
        # NOT added to this layout — it's reparented into the master
        # timeline panel's left gutter (next to the timeline scrubber)
        # by ``MainWindow``. The transport bar keeps the wiring
        # (``update_from_state``, ``set_frame_immediate``, …) so the
        # signal contract is unchanged.

        layout.addWidget(_separator())
        layout.addWidget(self._annotation_prev_btn)
        layout.addWidget(self._annotation_toggle_btn)
        layout.addWidget(self._annotation_next_btn)
        layout.addWidget(self._annotation_show_play_btn)

        layout.addWidget(_separator())
        fps_label = QLabel("FPS")
        fps_label.setFixedWidth(24)
        # Brief §5 mono caption — secondary text, JetBrains Mono with
        # a hint of letter-spacing so the three-letter label reads as
        # a key, not as body text.
        fps_label.setStyleSheet(
            f"color:{H.T_SEC};"
            "letter-spacing:0.08em;"
            f"font-family:{F.FAMILY_MONO};"
            f"font-size:{F.SIZE_MONO_LABEL}px;"
        )
        layout.addWidget(fps_label)
        layout.addWidget(self._fps_combo)

        # NB: ``_reload_btn``, ``_export_btn``, ``_channel_button``,
        # the RGBA mute toggles and ``_zoom_combo`` are constructed
        # in __init__ but NOT added to this layout — they live in
        # the menu bar's top-right corner widget (built in
        # ``MainWindow._build_menu``). Same reparent pattern as
        # ``_frame_display`` above. Keeps the global-state controls
        # at the very top of the window and frees the transport bar
        # for playback-only tools.

        # Master audio: a compact speaker icon. Click toggles a
        # vertical-slider popup that floats above the button and
        # auto-closes on outside click (``Qt.WindowType.Popup``).
        # Volume == 0 swaps the icon to the muted glyph; there's no
        # separate mute button (slider zero IS the mute).
        layout.addWidget(_separator())
        self._volume_value = 100  # 0-100, mirrors the popup slider
        self._volume_popup: _VolumePopup | None = None
        # Monotonic timestamp of the last popup hide. ``Qt.Popup``
        # dismisses on any outside click — but the same click also
        # reaches the volume button right after and would re-open
        # the popup. We swallow the re-open inside a short guard
        # window so the button reads as a clean toggle. See
        # ``_show_volume_popup``.
        self._volume_popup_closed_at: float = 0.0
        self._volume_btn = QPushButton()
        self._volume_btn.setObjectName("btnIcon")
        self._volume_btn.setFixedSize(G.CTRL_ICON_W, G.CTRL_BUTTON_H)
        # Use the brief's ``audio`` SVG glyph at rest. The mute swap
        # is driven by ``_refresh_volume_button`` below; we initialise
        # to the un-muted icon here.
        self._volume_btn.setIcon(make_icon("audio", color=H.T_PRI))
        self._volume_btn.setIconSize(QSize(16, 16))
        self._volume_btn.setToolTip("Volume")
        self._volume_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._volume_btn.clicked.connect(self._show_volume_popup)
        layout.addWidget(self._volume_btn)

        # Fullscreen toggle, sitting last on the right — that's the
        # corner reviewers reach for instinctively (YouTube / VLC
        # convention). Click cycles in / out of fullscreen, the
        # icon swaps between "expand" and "contract" arrows in
        # ``set_fullscreen_state``.
        layout.addWidget(_separator())
        self._fullscreen_btn = QPushButton()
        self._fullscreen_btn.setObjectName("btnIcon")
        self._fullscreen_btn.setFixedSize(G.CTRL_ICON_W, G.CTRL_BUTTON_H)
        self._fullscreen_btn.setIcon(make_icon("fullscreen"))
        self._fullscreen_btn.setIconSize(QSize(16, 16))
        self._fullscreen_btn.setToolTip("Fullscreen (F)")
        self._fullscreen_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._fullscreen_btn.clicked.connect(self.fullscreen_clicked.emit)
        layout.addWidget(self._fullscreen_btn)

        layout.addStretch(1)

        self._refresh_loop_button()

    def set_fullscreen_state(self, on: bool) -> None:
        """Swap the fullscreen button's icon to reflect the current
        mode. Called from ``MainWindow`` when fullscreen toggles.

        The brief introduces a single ``fullscreen`` icon (vs the
        legacy enter/exit pair); we still use the legacy
        ``fullscreen_exit`` for the active state since the new icon
        set has no contract glyph yet. Resting state uses the brief's
        ``fullscreen``.
        """
        self._fullscreen_btn.setIcon(
            make_icon("fullscreen_exit" if on else "fullscreen")
        )
        self._fullscreen_btn.setToolTip(
            "Exit fullscreen (F / Esc)" if on else "Fullscreen (F)"
        )

    # ------------------------------------------------------------------ Public

    @property
    def frame_display(self) -> FrameDisplay:
        """Public accessor for the frame readout widget. Used by
        :class:`MainWindow` to reparent it into the master timeline
        panel's left gutter (and back to a floating fullscreen bar
        when the user toggles fullscreen mode)."""
        return self._frame_display

    @property
    def reload_button(self) -> QPushButton:
        return self._reload_btn

    @property
    def export_button(self) -> QPushButton:
        return self._export_btn

    @property
    def compare_button(self) -> QPushButton:
        return self._compare_btn

    @property
    def contact_sheet_button(self) -> QPushButton:
        return self._contact_sheet_btn

    @property
    def contact_sheet_menu_button(self) -> "QToolButton":
        """The tiny "…" button sitting next to the contact-sheet
        toggle. Hosts the settings menu (grid, output size, labels)."""
        return self._contact_sheet_menu_btn

    @property
    def contact_sheet_menu(self) -> "QMenu":
        """The QMenu attached to ``contact_sheet_menu_button``. The
        app populates it lazily via ``aboutToShow`` with the current
        grid / divisor / labels presets."""
        return self._contact_sheet_menu

    def set_contact_sheet_checked(self, on: bool) -> None:
        """Sync the contact-sheet toggle's checked state from
        outside — used when the user enters / exits via the View
        menu or the Ctrl+G shortcut."""
        self._contact_sheet_btn.blockSignals(True)
        self._contact_sheet_btn.setChecked(bool(on))
        self._contact_sheet_btn.blockSignals(False)

    def set_playback_enabled(self, enabled: bool) -> None:
        """Grey out the forward / reverse play buttons.

        Used by contact-sheet mode where each tile owns its own
        playback offset and a global master-clock play makes no
        sense (the user scrubs per-tile instead). Step / first /
        last navigation stays enabled — stepping the master frame
        still shifts every tile in lockstep, which is useful.

        The same ``:disabled`` QSS rule used by
        :meth:`set_compare_enabled` keeps the disabled buttons
        legible; the icon desaturation comes from Qt's Disabled
        pixmap mode.
        """
        self._play_btn.setEnabled(bool(enabled))
        self._reverse_play_btn.setEnabled(bool(enabled))

    def set_compare_enabled(self, enabled: bool) -> None:
        """Grey out the compare button when the layer stack has < 2
        entries — there's nothing to compare against then. The local
        ``:disabled`` QSS rule uses ``BORDER_DEFAULT`` (not the dimmer
        ``BORDER_SUBTLE`` which would collapse into ``QMenuBar``'s own
        ``border-bottom``) so all 4 edges of the disabled button stay
        legible. The desaturated icon is provided by Qt's Disabled
        pixmap mode (see ``make_icon``)."""
        self._compare_btn.setEnabled(bool(enabled))

    def set_compare_checked(self, on: bool) -> None:
        """Sync the compare button's checked state from outside —
        used when the user enters / exits compare via keyboard
        shortcut or from the band's ✕ button."""
        self._compare_btn.blockSignals(True)
        self._compare_btn.setChecked(bool(on))
        self._compare_btn.blockSignals(False)

    @property
    def channel_button(self) -> QToolButton:
        return self._channel_button

    @property
    def channel_cache_pause_button(self) -> QPushButton:
        """Toggle button that pauses / resumes the alt-channel
        background prefetch. Exposed so :class:`MainWindow` can
        insert it into the top-right buttons toolbar right next to
        the channel selector."""
        return self._channel_cache_pause_btn

    def set_channel_cache_paused(self, paused: bool) -> None:
        """Reflect the controller's alt-channel paused state on the
        button (label / tooltip). Does NOT re-emit
        ``channel_cache_pause_toggled`` — this is a one-way push
        used by ``app.py`` to keep the UI in sync with programmatic
        state changes."""
        new_value = bool(paused)
        if new_value == self._channel_cache_paused:
            return
        self._channel_cache_paused = new_value
        self._refresh_channel_cache_pause_button()

    def _refresh_channel_cache_pause_button(self) -> None:
        """Repaint the button face + tooltip + colour for the current
        paused state. The glyph is a status indicator (what's
        happening right now), not an action label:

        * **green triangle (▶)** = prefetch is RUNNING. Click to stop.
        * **red square (■)** = prefetch is STOPPED. Click to resume.

        Colours come through inline QSS ``color:`` so they win over
        the global QPushButton stylesheet without us having to ship
        a custom paintEvent.
        """
        if self._channel_cache_paused:
            glyph = "■"
            colour = "#E54B4B"  # warm red, matches stop-state semantics
            tooltip = (
                "Channel cache prefetch is STOPPED — click to resume "
                "loading alt-channel snapshots into RAM."
            )
        else:
            glyph = "▶"
            colour = "#3FB950"  # GitHub-green, clear "running" semantic
            tooltip = (
                "Channel cache prefetch is RUNNING — click to stop "
                "loading alt-channel snapshots into RAM."
            )
        self._channel_cache_pause_btn.setText(glyph)
        self._channel_cache_pause_btn.setToolTip(tooltip)
        self._channel_cache_pause_btn.setStyleSheet(
            "QPushButton { "
            "  font-size: 12pt; "
            "  font-weight: bold; "
            "  padding: 0; "
            f"  color: {colour}; "
            "}"
        )

    def _on_channel_cache_pause_clicked(self) -> None:
        """Flip the paused state and emit the toggle signal so
        the controller can act on it."""
        self._channel_cache_paused = not self._channel_cache_paused
        self._refresh_channel_cache_pause_button()
        self.channel_cache_pause_toggled.emit(self._channel_cache_paused)

    @property
    def channel_menu(self) -> ChannelMenu:
        """The popup menu hanging off the channel button. Exposed so
        the app can wire the cache-fill progress provider on it
        without reaching through ``_channel_menu``."""
        return self._channel_menu

    @property
    def channel_mode_button(self) -> QToolButton:
        """Single button that replaces the old R/G/B/A toggle row.
        Main click cycles RGB→R→G→B→A; the dropdown arrow exposes
        the same picks as an explicit menu."""
        return self._channel_mode_button

    @property
    def bg_button(self) -> QToolButton:
        """Background-picker for transparent pixels (checker / black /
        grey / white). Sits next to the R/G/B/A mute toggles."""
        return self._bg_button

    @property
    def zoom_combo(self) -> QComboBox:
        return self._zoom_combo

    # ----- Master audio (popup volume slider) ----------------------

    def _refresh_volume_icon(self) -> None:
        """Icon swap based purely on the current slider value: zero
        reads as muted, any positive value reads as audible. No
        second source of truth — mute is implicit in the slider
        position. Swaps between the brief's ``audio`` and
        ``audio-mute`` glyphs."""
        muted = self._volume_value == 0
        self._volume_btn.setIcon(
            make_icon("audio-mute" if muted else "audio", color=H.T_PRI)
        )
        self._volume_btn.setToolTip(
            "Volume — muted (slider at 0)" if muted else "Volume",
        )

    def _show_volume_popup(self) -> None:
        """Toggle the vertical-slider popup above the volume button.

        Lazily creates the popup on first click + reuses the same
        instance across opens so the slider's connections survive.
        Positioned bottom-anchored to the button (the transport bar
        sits at the window's bottom, so up is the only space).

        Toggle logic: ``Qt.Popup`` dismisses on any outside click
        — including the click on the volume button itself. That
        same click then reaches the button's ``clicked`` signal
        and would re-open the popup. We swallow the re-open inside
        a ~250 ms guard window after the popup hides, so the
        second-click-while-open path actually leaves the popup
        closed (= proper toggle).
        """
        import time

        # Guard window: if the popup was just dismissed by this
        # same click, treat it as the "close" half of a toggle and
        # stay closed.
        if (time.monotonic() - self._volume_popup_closed_at) < 0.25:
            return

        if self._volume_popup is None:
            self._volume_popup = _VolumePopup(self._volume_value, self)
            self._volume_popup.value_changed.connect(self._on_volume_changed)
            self._volume_popup.closed.connect(self._on_volume_popup_closed)
        else:
            # Keep the popup state in sync with any prefs / programmatic
            # change that happened while it was closed.
            self._volume_popup.set_value(self._volume_value)

        btn_top_left = self._volume_btn.mapToGlobal(self._volume_btn.rect().topLeft())
        popup_size = self._volume_popup.sizeHint()
        x = btn_top_left.x() + (self._volume_btn.width() - popup_size.width()) // 2
        y = btn_top_left.y() - popup_size.height() - 4
        self._volume_popup.move(x, y)
        self._volume_popup.show()

    def _on_volume_popup_closed(self) -> None:
        """Stamp the time the popup auto-dismissed. Read by
        ``_show_volume_popup`` to suppress the re-open that would
        otherwise follow when the dismissing click also lands on
        the volume button."""
        import time
        self._volume_popup_closed_at = time.monotonic()

    def _on_volume_changed(self, value: int) -> None:
        # Popup slider → cached value + icon refresh + gain emit.
        # The downstream audio callback skips the multiply at unity,
        # so we don't waste cycles on the default state.
        self._volume_value = max(0, min(100, int(value)))
        self._refresh_volume_icon()
        self.master_volume_changed.emit(self._volume_value / 100.0)

    def set_master_volume(self, gain: float) -> None:
        """Sync the slider state from outside (prefs restore). Doesn't
        emit ``master_volume_changed`` — the app already set the
        audio output's gain when reading prefs, re-emitting would
        just round-trip the value."""
        try:
            v = int(round(max(0.0, min(1.0, float(gain))) * 100))
        except (TypeError, ValueError):
            return
        if v == self._volume_value:
            return
        self._volume_value = v
        if self._volume_popup is not None:
            self._volume_popup.set_value(v)
        self._refresh_volume_icon()

    def set_transparency_bg_mode(self, mode: int) -> None:
        """Sync the BG picker's checked entry from outside — used when
        prefs restore the user's last pick on boot. Doesn't re-emit
        ``transparency_bg_mode_changed`` (would loop the GL setter)."""
        m = int(mode)
        if not 0 <= m <= 3:
            m = 0
        if m == self._current_bg_mode:
            return
        self._current_bg_mode = m
        self._refresh_bg_menu_check()
        # Repaint the button face so the swatch icon matches the new
        # mode. Without this the icon stuck on whatever was set in
        # ``__init__`` (always checker), giving the user a mismatch
        # between the BG indicator and the actual viewport background.
        self._apply_bg_button_style()

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
        # ``btnPrimary`` paints the play key with an orange gradient,
        # so the glyph itself reads better in dark ink (#1A1206) — matches
        # the contrast brief calls for. Pause uses the same dark ink so
        # the button reads as one continuous accent surface in either
        # state.
        if playing_fwd:
            self._play_btn.setIcon(make_icon("pause", color="#1A1206"))
        else:
            self._play_btn.setIcon(make_icon("play", color="#1A1206"))
        if playing_rev:
            self._reverse_play_btn.setIcon(make_icon("pause", color="#1A1206"))
        else:
            self._reverse_play_btn.setIcon(make_icon("play_reverse", color="#1A1206"))

        # Push the current frame into the editable display.
        self._frame_display.set_frame(state.current_frame)
        self._frame_display.set_fps(state.fps)

        if state.loop_mode != self._loop_mode:
            self._loop_mode = state.loop_mode
            self._refresh_loop_button()

        current_fps = self._parse_fps(self._fps_combo.text())
        if current_fps is None or abs(current_fps - state.fps) > 1e-3:
            self._fps_combo.blockSignals(True)
            self._fps_combo.setText(self._format_fps(state.fps))
            self._fps_combo.blockSignals(False)

    def set_display_mode(self, mode: DisplayMode) -> None:
        """Propagate the global frame/timecode toggle (View menu) to
        the FrameDisplay so it stays in sync with the timeline."""
        self._frame_display.set_display_mode(mode)

    def set_annotation_nav_enabled(self, prev_avail: bool, next_avail: bool) -> None:
        """Grey out the prev / next annotation buttons when there's
        nothing on either side of the current playhead.

        Driven by ``App`` from the AnnotationStore's
        ``annotated_frames_changed`` signal and from frame changes.
        Tooltip stays informative either way; tooltip text is only
        cosmetic so we don't bother swapping it.
        """
        self._annotation_prev_btn.setEnabled(prev_avail)
        self._annotation_next_btn.setEnabled(next_avail)

    def set_export_enabled(self, enabled: bool) -> None:
        """Enable / disable the 💾 export button.

        The app flips this on after a sequence successfully loads;
        before that there's nothing to export and clicking the
        button would just produce a confusing no-op.
        """
        self._export_btn.setEnabled(bool(enabled))

    def set_reload_enabled(self, enabled: bool) -> None:
        """Enable / disable the 🔄 reload button (same gating as
        Export — needs a loaded sequence to mean anything)."""
        self._reload_btn.setEnabled(bool(enabled))

    def is_annotation_toggle_active(self) -> bool:
        """Read the ✏ button's current checked state — used by the
        fullscreen bar to mirror the toolbar's visibility on its
        own annotation toggle."""
        return bool(self._annotation_toggle_btn.isChecked())

    def set_annotation_toggle_active(self, active: bool) -> None:
        """Reflect the toolbar's visibility on the ✏ button.

        ``True`` when the toolbar is shown — the button is checked,
        the user has visual confirmation that pressing D again will
        hide it. ``False`` resets to the unchecked state.
        """
        if self._annotation_toggle_btn.isChecked() == active:
            return
        self._annotation_toggle_btn.blockSignals(True)
        try:
            self._annotation_toggle_btn.setChecked(active)
        finally:
            self._annotation_toggle_btn.blockSignals(False)

    def set_annotation_show_during_play(self, active: bool) -> None:
        """Sync the 👁 "show during play" button without re-emitting.

        Called from the app when the ``A`` shortcut flips the store's
        ``show_during_playback`` flag — keeps the visual state in
        lockstep without round-tripping the signal back through
        ``_toggle_show_annotations_during_play``."""
        if self._annotation_show_play_btn.isChecked() == active:
            return
        self._annotation_show_play_btn.blockSignals(True)
        try:
            self._annotation_show_play_btn.setChecked(active)
            # The button now uses the SVG ``ann-hide`` icon and relies
            # on the global ``btnIcon:checked`` QSS rule to paint the
            # active state in ACC_BRIGHT — no manual glyph swap to do
            # here anymore (the legacy emoji 👁/🚫 text path is gone).
        finally:
            self._annotation_show_play_btn.blockSignals(False)

    def set_frame_immediate(self, frame: int) -> None:
        """Update the frame readout *now*, ahead of the controller.

        ``update_from_state`` only fires after the controller's seek
        completes — and the seek itself is debounced ~20 ms by
        ``app.py`` to coalesce rapid scrubs. That makes the readout
        feel laggy: the timeline cursor jumps under the mouse but the
        number above it limps behind. This entry point lets the scrub
        handler push the *requested* frame straight into the display
        for a snappy feel; the eventual ``state_changed`` will refresh
        it again with whatever frame the controller actually settled
        on (typically the same — at most a one-frame correction when
        the request was clamped to the in/out range).
        """
        self._frame_display.set_frame(frame)

    def set_available_channels(self, channels: tuple[str, ...]) -> None:
        """Replace the channel-menu content with grouped channels.

        Layers like ``albedo.R``/``.G``/``.B`` collapse into a single
        ``"albedo"`` entry that loads the three channels as an RGB
        composite — same convention as Nuke's channel selector.
        Single-component channels (``Z``, ``volume_Z``,
        ``normal.X``…) keep their own entry. The first entry is
        always the beauty ``"RGB"``/``"RGBA"`` from the root.

        Resets the menu to "first group active, no tiles" — the same
        fresh-sequence baseline the legacy combo had at index 0. The
        selection then re-emits via :meth:`_on_channel_selection_changed`
        so the controller picks up the new active channel.
        """
        groups = group_channels(channels) if channels else []
        if not groups:
            # No header info yet — at least show RGB so the menu
            # isn't blank.
            groups = [ChannelGroup(label="RGB", channels=("R", "G", "B"))]
        self._channel_menu.set_groups(groups)
        # Force-emit the initial selection so the controller switches
        # to the new sequence's beauty pass without the user having
        # to click anything.
        sel = self._channel_menu.current_selection()
        if sel is not None:
            self._on_channel_selection_changed(sel)

    def restore_channel_state(self, active: str) -> None:
        """Reapply a saved ChannelMenu state on app boot (called from
        :class:`Preferences` round-trip in ``app.py``).
        """
        self._channel_menu.set_state(active)
        sel = self._channel_menu.current_selection()
        if sel is not None:
            self._on_channel_selection_changed(sel)

    def clear_channels(self) -> None:
        """Drop every entry from the channel menu and reset the
        button label. Called from File → New (Ctrl+N) so the
        previous sequence's channel groups don't linger as stale
        menu rows / a stale button caption when no sequence is
        loaded.

        Different from ``set_available_channels(())`` which keeps a
        fallback "RGB" stub so the menu is never visually empty
        mid-load — here we *do* want it visually empty because
        there's nothing loaded.
        """
        self._current_selection = None
        self._channel_menu.set_groups([])
        self._channel_button.setText("")
        self._channel_button.setToolTip("")
        # Drop the cache-fill bar from the button face.
        self._channel_button.set_active_progress(-1.0)

    def channel_menu_state(self) -> str:
        """Return the menu's current active-channel label for
        persistence."""
        return self._channel_menu.active_label

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
        icon_name, tooltip = _LOOP_LABELS[self._loop_mode]
        # Re-paint the button's icon with the new mode's glyph. Loop
        # in active state (= any of the 3 cycle positions) — the
        # global QSS paints the checked state in ACC_BRIGHT, no
        # explicit colour needed.
        self._loop_btn.setIcon(make_icon(icon_name))
        self._loop_btn.setToolTip(tooltip)

    def _on_fps_text(self, text: str) -> None:
        fps = self._parse_fps(text)
        if fps is not None:
            self.fps_changed.emit(fps)

    # ---- Transparency background picker ------------------------------

    # Modes for the BG button — value, short label (for the closed
    # button face), long label (for the menu), and tint colour used
    # by the menu item. ``Checker`` keeps the accent-orange so it
    # reads as "the special pattern mode" rather than just a colour.
    _BG_MODES: ClassVar[tuple[tuple[int, str, str, str], ...]] = (
        (0, "Chk", "Checker (default)", H.ACCENT),
        (1, "Blk", "Black",             "#1A1A1A"),
        (2, "Gry", "Mid grey",          "#888888"),
        (3, "Wht", "White",             "#F0F0F0"),
    )

    def _build_bg_menu(self) -> QMenu:
        """Build the BG dropdown: each row is a painted colour swatch
        (or a checker chip for the default mode), wrapped in a
        :class:`QWidgetAction`. The swatch *is* the label — no text —
        which side-steps the "Black on black menu chrome" readability
        problem and makes the menu scannable at a glance."""
        menu = QMenu(self)
        self._bg_items: dict[int, _SwatchMenuItem] = {}
        for mode, _short, _long_label, tint in self._BG_MODES:
            is_checker = mode == 0
            # For the checker entry we pass ``None`` as the colour —
            # the swatch paints the pattern instead. Other entries
            # use the canonical mode colour straight.
            color: str | None = None if is_checker else tint
            item = _SwatchMenuItem(mode, color, checker=is_checker)
            item.set_checked(mode == self._current_bg_mode)
            item.clicked.connect(self._on_bg_menu_item_clicked)
            wa = QWidgetAction(menu)
            wa.setDefaultWidget(item)
            menu.addAction(wa)
            self._bg_items[mode] = item
        return menu

    def _cycle_bg_mode(self) -> None:
        """Step through Checker → Black → Grey → White → Checker.
        Bound to the main click of the BG toolbutton (the arrow on
        the right edge opens the menu instead)."""
        order = [m[0] for m in self._BG_MODES]
        idx = order.index(self._current_bg_mode)
        nxt = order[(idx + 1) % len(order)]
        self._on_bg_picked(nxt)

    def _on_bg_menu_item_clicked(self, mode: int) -> None:
        self._on_bg_picked(mode)
        if self._bg_menu.isVisible():
            self._bg_menu.close()

    def _on_bg_picked(self, mode: int) -> None:
        if int(mode) == self._current_bg_mode:
            return
        self._current_bg_mode = int(mode)
        self._refresh_bg_menu_check()
        self._apply_bg_button_style()
        self.transparency_bg_mode_changed.emit(int(mode))

    def _refresh_bg_menu_check(self) -> None:
        for mode, item in self._bg_items.items():
            item.set_checked(mode == self._current_bg_mode)

    def _apply_bg_button_style(self) -> None:
        """Refresh the BG toolbutton icon to match the current mode.

        We paint a square swatch (checker pattern for mode 0, flat
        colour otherwise) into a QPixmap and feed it through
        :meth:`QToolButton.setIcon`. Mode names like ``Blk`` were
        unreadable on the dark menu-bar chrome — the swatch sidesteps
        the contrast problem entirely.
        """
        color = next(m[3] for m in self._BG_MODES if m[0] == self._current_bg_mode)
        is_checker = self._current_bg_mode == 0
        self._bg_button.setIcon(
            _swatch_icon(color, checker=is_checker, size=22),
        )
        self._bg_button.setStyleSheet(
            "QToolButton {"
            "  padding: 0;"
            "}"
            # The swatch icon already has its own outline so the
            # two halves read as separate without an extra divider —
            # drop ``border-left`` to ditch the little vertical bar.
            "QToolButton::menu-button {"
            "  width: 20px;"
            "  border: none;"
            "  margin: 0;"
            "  padding: 0;"
            "}"
            "QToolButton::menu-arrow {"
            "  width: 10px;"
            "  height: 10px;"
            "}",
        )

    # --- Channel mode (RGB / R / G / B / A) -----------------------
    # The mode names map to a (R, G, B, A) shader mask. ``A`` zeroes
    # alpha which trips the shader's ``uChannelMask.a < 0.5`` branch
    # — that's how solo-alpha becomes grayscale (white = opaque,
    # black = transparent), mirroring Nuke / RV's convention.
    _CHANNEL_MODE_MASKS: ClassVar[dict[str, tuple[bool, bool, bool, bool]]] = {
        "RGB": (True, True, True, True),
        "R":   (True, False, False, True),
        "G":   (False, True, False, True),
        "B":   (False, False, True, True),
        "A":   (False, False, False, False),
    }
    _CHANNEL_MODE_ORDER: ClassVar[tuple[str, ...]] = ("RGB", "R", "G", "B", "A")

    def _build_channel_mode_menu(self) -> QMenu:
        """Dropdown menu hanging off the channel-mode button.

        Each row is a :class:`_ChannelMenuItem` widget wrapped in a
        :class:`QWidgetAction` so we can colour the R / G / B letters
        in their channel tint (which a plain ``QAction`` can't do —
        Qt offers no honoured per-item text-colour stylesheet on
        ``QMenu::item``). The widget owns its own check / hover
        painting; the menu just hosts them.
        """
        menu = QMenu(self)
        self._channel_mode_items: dict[str, _ChannelMenuItem] = {}
        for mode in self._CHANNEL_MODE_ORDER:
            color = _CHANNEL_BTN_COLORS.get(mode, H.TEXT_PRIMARY)
            item = _ChannelMenuItem(mode, color)
            item.set_checked(mode == self._channel_mode)
            # Click → pick the mode + close the menu (the widget
            # doesn't bubble a ``triggered`` up to the QMenu the
            # way a QAction would, so we close it explicitly).
            item.clicked.connect(self._on_channel_menu_item_clicked)
            wa = QWidgetAction(menu)
            wa.setDefaultWidget(item)
            menu.addAction(wa)
            self._channel_mode_items[mode] = item
        return menu

    def _on_channel_menu_item_clicked(self, mode: str) -> None:
        self._set_channel_mode(mode)
        if self._channel_mode_menu.isVisible():
            self._channel_mode_menu.close()

    def _cycle_channel_mode(self) -> None:
        """Step through RGB → R → G → B → A → RGB. Bound to the main
        click of the channel-mode toolbutton (the dropdown arrow
        opens the menu instead)."""
        idx = self._CHANNEL_MODE_ORDER.index(self._channel_mode)
        nxt = self._CHANNEL_MODE_ORDER[(idx + 1) % len(self._CHANNEL_MODE_ORDER)]
        self._set_channel_mode(nxt)

    def _set_channel_mode(self, mode: str) -> None:
        if mode not in self._CHANNEL_MODE_MASKS:
            return
        if mode == self._channel_mode:
            # Still refresh the menu check state — defensive in case
            # we ever get out of sync with the action checks.
            self._refresh_channel_mode_menu_check()
            return
        self._channel_mode = mode
        self._refresh_channel_mode_menu_check()
        self._apply_channel_mode_style()
        self.channel_mask_changed.emit(self._CHANNEL_MODE_MASKS[mode])

    def _refresh_channel_mode_menu_check(self) -> None:
        for mode, item in self._channel_mode_items.items():
            item.set_checked(mode == self._channel_mode)

    def _apply_channel_mode_style(self) -> None:
        """Refresh the toolbutton label + tinted text colour to match
        the current mode. RGB is a neutral white; R/G/B/A use the
        same per-channel tints we used on the old 4-button row."""
        self._channel_mode_button.setText(self._channel_mode)
        color = _CHANNEL_BTN_COLORS.get(
            self._channel_mode, H.TEXT_PRIMARY,
        )
        # Inline QSS: text colour follows the current channel, the
        # rest stays in line with the global QToolButton chrome.
        # The text-area + menu-button geometry mirrors the fixed
        # width set in ``__init__`` (72 px total): 52 px label half
        # with the text auto-centered by Qt, 20 px arrow half,
        # separated by a clear divider.
        self._channel_mode_button.setStyleSheet(
            "QToolButton {"
            f"  color: {color};"
            "  font-weight: 600;"
            "  padding: 0;"
            "  text-align: center;"
            "}"
            # No inner divider — the wider arrow zone is enough to
            # tell the two halves apart, and a hairline next to the
            # tinted letter reads as visual clutter.
            "QToolButton::menu-button {"
            "  width: 20px;"
            "  border: none;"
            "  margin: 0;"
            "  padding: 0;"
            "}"
            "QToolButton::menu-arrow {"
            "  width: 10px;"
            "  height: 10px;"
            "}",
        )

    def _on_channel_selection_changed(self, selection: ChannelSelection) -> None:
        """Forward a fresh selection from the channel menu.

        Updates the button label to summarise the state ("RGB",
        "albedo", "RGB +2", …) and emits
        :attr:`channel_selection_changed` for the controller.
        """
        self._current_selection = selection
        self._refresh_channel_button_label()
        self.channel_selection_changed.emit(selection)

    def _refresh_channel_button_label(self) -> None:
        sel = self._current_selection
        if sel is None:
            self._channel_button.setText("RGB")
            return
        self._channel_button.setText(sel.active.label)

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
    # Opt into the global ``#btnIcon`` QSS variant — same 30×28
    # footprint as the rest of the transport-bar icon buttons, with
    # the brief's hover / pressed treatment baked in. We keep the
    # explicit ``setFixedSize`` below because the QSS uses min/max
    # width only; the height needs a setFixedHeight to defeat any
    # parent layout that might try to stretch the button vertically.
    btn.setObjectName("btnIcon")
    btn.setIcon(icon)
    btn.setIconSize(QSize(G.ICON_SIZE, G.ICON_SIZE))
    btn.setFixedSize(G.CTRL_ICON_W, G.CTRL_BUTTON_H)
    btn.setToolTip(tooltip)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    return btn


def _text_button(label: str, tooltip: str) -> QPushButton:
    btn = QPushButton(label)
    # Square buttons matching the icon-button footprint
    # (BTN_TRANSPORT_W × BTN_TRANSPORT_H). Font size is dialled
    # down to 11 pt — slightly smaller than 13 pt but with room
    # to breathe inside the square so the emoji is not clipped
    # left/right. Trade-off favoured "square + uncut" per user
    # feedback (v0.5.2).
    btn.setFixedSize(G.BTN_TRANSPORT_W, G.BTN_TRANSPORT_H)
    btn.setToolTip(tooltip)
    btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
    # Global QSS ``QPushButton {…}`` overrides ``btn.setFont()`` —
    # push font-size through an inline stylesheet so it wins.
    btn.setStyleSheet("QPushButton { font-size: 11pt; padding: 0; }")
    return btn


def _separator() -> QWidget:
    """Vertical separator with breathing room on each side.

    Wraps the 1 px line in a container with horizontal padding so
    button groups around the separator don't crowd it. The layout's
    own ``spacing(S.SM)`` adds 4 px more on top of the padding here,
    giving each group a clearly framed rest area.
    """
    container = QWidget()
    container.setFixedHeight(22)
    h = QHBoxLayout(container)
    h.setContentsMargins(6, 0, 6, 0)
    h.setSpacing(0)
    line = QFrame()
    line.setFrameShape(QFrame.Shape.VLine)
    line.setFrameShadow(QFrame.Shadow.Plain)
    line.setFixedWidth(1)
    line.setFixedHeight(18)
    line.setStyleSheet(f"background-color: {H.BORDER_SUB};")
    h.addWidget(line)
    return container


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



class _ChannelMenuItem(QWidget):
    """Single row of the channel-mode dropdown menu.

    ``QAction`` text colour can't be overridden per-item with stylesheet
    selectors that Qt actually honours on ``QMenu::item``. So we ship
    each row as a ``QWidget`` wrapped in a :class:`QWidgetAction`: a
    check-mark slot on the left, then the mode letter coloured to
    match its channel (red / green / blue / grey, white for ``RGB``).
    Hover and click feedback are painted with inline QSS — small
    enough to live next to the parent class without its own module.
    """

    clicked = Signal(str)  # the mode name (RGB / R / G / B / A)

    _HOVER_BG = "rgba(255,255,255,0.08)"
    _CHECKED_BG = "rgba(255,255,255,0.05)"

    def __init__(self, mode: str, color: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._mode = mode
        self._checked = False
        self._hover = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 22, 5)
        layout.setSpacing(8)

        self._check = QLabel("")
        # Reserve the check slot's width so the label column stays
        # vertically aligned across all five rows whether they're
        # ticked or not.
        self._check.setFixedWidth(14)
        self._check.setStyleSheet(f"color: {H.ACCENT}; font-weight: 700;")

        self._label = QLabel(mode)
        self._label.setStyleSheet(f"color: {color}; font-weight: 600;")

        layout.addWidget(self._check)
        layout.addWidget(self._label, 1)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # Tracking + Hover so enter/leave events fire reliably even
        # over the child labels — Qt forwards them to the parent
        # QWidget when both are mouse-tracking.
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._refresh_bg()

    def set_checked(self, on: bool) -> None:
        if on == self._checked:
            return
        self._checked = on
        self._check.setText("✓" if on else "")
        self._refresh_bg()

    def _refresh_bg(self) -> None:
        if self._hover:
            bg = self._HOVER_BG
        elif self._checked:
            bg = self._CHECKED_BG
        else:
            bg = "transparent"
        self.setStyleSheet(f"_ChannelMenuItem {{ background: {bg}; }}")

    def enterEvent(self, event: QEvent) -> None:  # noqa: D401, N802
        self._hover = True
        self._refresh_bg()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:  # noqa: D401, N802
        self._hover = False
        self._refresh_bg()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: D401, N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit(self._mode)
        super().mouseReleaseEvent(event)


def _swatch_icon(
    color: str | None,
    *,
    checker: bool = False,
    size: int = 22,
) -> QIcon:
    """Paint a square swatch (flat colour or checker pattern) into a
    :class:`QPixmap` and wrap it in a :class:`QIcon`. Used to feed the
    BG toolbutton's icon so the closed face mirrors the menu chips.

    Logic mirrors :meth:`_SwatchPaint.paintEvent` — kept duplicated
    rather than refactored because the widget path needs to repaint
    on resize / theme change, whereas this baked pixmap path only
    fires when the mode actually changes.
    """
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    try:
        r = pm.rect().adjusted(0, 0, -1, -1)
        if checker:
            tile = 4
            cols = (r.width() // tile) + 1
            rows = (r.height() // tile) + 1
            painter.fillRect(r, QColor("#3D3D3F"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#5A5A5C"))
            for j in range(rows):
                for i in range(cols):
                    if (i + j) % 2 == 0:
                        continue
                    painter.drawRect(r.x() + i * tile, r.y() + j * tile, tile, tile)
        else:
            painter.fillRect(r, QColor(color or "#888888"))
        # Same quiet outline as the widget swatch so Black stays
        # visible against the toolbutton's dark hover state.
        painter.setPen(QColor(255, 255, 255, 64))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(r)
    finally:
        painter.end()
    return QIcon(pm)


class _VolumePopup(QWidget):
    """Vertical-slider popup that floats above the transport's volume
    button. Implemented as a top-level widget with the
    :attr:`Qt.WindowType.Popup` flag so a click outside its bounds
    closes it automatically — the standard "transient overlay"
    pattern Qt uses for combobox dropdowns.

    Carries the same 0-100 integer range as the previous inline
    slider; the parent ``TransportBar`` converts to linear gain.
    """

    value_changed = Signal(int)
    # Emitted whenever the popup transitions to hidden, regardless
    # of cause (outside click via ``Qt.Popup``, explicit ``hide()``,
    # parent destruction, etc.). The parent ``TransportBar`` uses
    # this timestamp to suppress the toggle bounce — see
    # ``_show_volume_popup``.
    closed = Signal()

    def __init__(self, value: int, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.Popup)
        # Pop the chrome on top of the menu-bar without bleeding the
        # parent's QSS into the popup. Auto-fill so the dark theme
        # stays opaque (otherwise the desktop bleeds through).
        self.setAutoFillBackground(True)
        self.setStyleSheet(
            f"_VolumePopup {{ "
            f"  background: {H.BG_RAISED}; "
            f"  border: 1px solid {H.BORDER_DEFAULT}; "
            f"  border-radius: 4px; "
            f"}}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._readout = QLabel(str(int(value)))
        self._readout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._readout.setStyleSheet(
            f"color: {H.TEXT_SECONDARY}; font-size: 10px;",
        )

        self._slider = QSlider(Qt.Orientation.Vertical)
        self._slider.setRange(0, 100)
        self._slider.setValue(int(value))
        self._slider.setFixedHeight(140)
        self._slider.setMinimumWidth(24)
        self._slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Accent-orange chrome so the slider reads as belonging to
        # the Flick palette rather than the OS default. We paint:
        #   - groove (track): muted grey + the "filled" portion in
        #     accent dim (= the orange the user has dialled in so far);
        #   - sub-page below the handle in the lit accent so the
        #     filled portion of the track is unmistakable;
        #   - add-page above in a darker neutral so the unfilled
        #     portion still reads as a track and not background;
        #   - handle in bright accent on hover for click affordance.
        self._slider.setStyleSheet(
            f"""
            QSlider::groove:vertical {{
                background: {H.BORDER_DEFAULT};
                width: 4px;
                border-radius: 2px;
            }}
            QSlider::sub-page:vertical {{
                background: {H.BORDER_DEFAULT};
                width: 4px;
                border-radius: 2px;
            }}
            QSlider::add-page:vertical {{
                background: {H.ACCENT};
                width: 4px;
                border-radius: 2px;
            }}
            QSlider::handle:vertical {{
                background: {H.ACCENT};
                border: 1px solid {H.ACCENT_BRIGHT};
                height: 12px;
                width: 14px;
                margin: 0 -5px;
                border-radius: 3px;
            }}
            QSlider::handle:vertical:hover {{
                background: {H.ACCENT_BRIGHT};
            }}
            """
        )
        self._slider.valueChanged.connect(self._on_slider)

        layout.addWidget(self._readout)
        layout.addWidget(self._slider, alignment=Qt.AlignmentFlag.AlignHCenter)

    def _on_slider(self, value: int) -> None:
        v = int(value)
        self._readout.setText(str(v))
        self.value_changed.emit(v)

    def value(self) -> int:
        return int(self._slider.value())

    def set_value(self, value: int) -> None:
        """Sync the slider from outside (boot-time prefs restore or
        external setter). Blocks signals so a programmatic set
        doesn't loop back through ``value_changed``."""
        v = int(value)
        if v == self._slider.value():
            self._readout.setText(str(v))
            return
        blocked = self._slider.blockSignals(True)
        try:
            self._slider.setValue(v)
        finally:
            self._slider.blockSignals(blocked)
        self._readout.setText(str(v))

    def hideEvent(self, event: QEvent) -> None:  # noqa: D401, N802
        """Notify the parent when the popup goes hidden so it can
        timestamp the close. Without this signal there's no
        reliable hook for "Qt.Popup just dismissed me" — we'd have
        to poll ``isVisible()`` from the button click handler."""
        super().hideEvent(event)
        self.closed.emit()


class _SwatchPaint(QWidget):
    """Small painted preview chip: either a flat colour fill or a
    VFX-style 2-tone checker pattern. Used by the BG dropdown so the
    swatch itself is the label — no text needed (which is what was
    failing for "Black" on the dark menu chrome anyway).
    """

    # Checker palette mirrors the runtime shader (mid-grey + lighter
    # grey on the dark transparency backdrop): readable enough on
    # the menu's #2x background without screaming for attention.
    _CHECKER_A = QColor("#3D3D3F")
    _CHECKER_B = QColor("#5A5A5C")
    _CHECKER_TILE = 5  # px

    def __init__(
        self,
        color: str | None,
        *,
        checker: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._color = QColor(color) if color else QColor("#888888")
        self._checker = checker

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: D401, N802
        p = QPainter(self)
        try:
            r = self.rect().adjusted(0, 0, -1, -1)
            if self._checker:
                # Tile the rect with alternating squares.
                size = self._CHECKER_TILE
                cols = (r.width() // size) + 1
                rows = (r.height() // size) + 1
                p.fillRect(r, self._CHECKER_A)
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(self._CHECKER_B)
                for j in range(rows):
                    for i in range(cols):
                        if (i + j) % 2 == 0:
                            continue
                        x = r.x() + i * size
                        y = r.y() + j * size
                        p.drawRect(x, y, size, size)
            else:
                p.fillRect(r, self._color)
            # Thin outline so "Black" doesn't vanish on the dark
            # menu background. Same border colour as the menu chrome
            # divider used elsewhere — quiet but always present.
            p.setPen(QColor(255, 255, 255, 64))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRect(r)
        finally:
            p.end()


class _SwatchMenuItem(QWidget):
    """Same row layout as :class:`_ChannelMenuItem` but the label is
    replaced by a painted colour swatch (or a checker chip). Lets the
    BG dropdown show the actual mode preview instead of a text name —
    much faster to scan, and side-steps the "Black on black"
    contrast issue.
    """

    clicked = Signal(int)  # carries the mode value

    _HOVER_BG = "rgba(255,255,255,0.08)"
    _CHECKED_BG = "rgba(255,255,255,0.05)"

    def __init__(
        self,
        mode: int,
        color: str | None,
        *,
        checker: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._checked = False
        self._hover = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 22, 5)
        layout.setSpacing(8)

        self._check = QLabel("")
        self._check.setFixedWidth(14)
        self._check.setStyleSheet(f"color: {H.ACCENT}; font-weight: 700;")

        self._swatch = _SwatchPaint(color, checker=checker)
        # Square chip — same shape language as the closed button face.
        self._swatch.setFixedSize(26, 26)

        layout.addWidget(self._check)
        layout.addWidget(self._swatch)
        # Pull the right edge in so the square chip doesn't sit in a
        # giant empty space; the ``addStretch`` keeps the column
        # left-aligned across rows.
        layout.addStretch(1)

        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self._refresh_bg()

    def set_checked(self, on: bool) -> None:
        if on == self._checked:
            return
        self._checked = on
        self._check.setText("✓" if on else "")
        self._refresh_bg()

    def _refresh_bg(self) -> None:
        if self._hover:
            bg = self._HOVER_BG
        elif self._checked:
            bg = self._CHECKED_BG
        else:
            bg = "transparent"
        self.setStyleSheet(f"_SwatchMenuItem {{ background: {bg}; }}")

    def enterEvent(self, event: QEvent) -> None:  # noqa: D401, N802
        self._hover = True
        self._refresh_bg()
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:  # noqa: D401, N802
        self._hover = False
        self._refresh_bg()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: D401, N802
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(
            event.position().toPoint()
        ):
            self.clicked.emit(self._mode)
        super().mouseReleaseEvent(event)


class _ChannelModeButton(QToolButton):
    """``QToolButton`` whose right-click also opens the dropdown menu.

    The default ``MenuButtonPopup`` mode only fires the menu when the
    user clicks the small triangle on the right edge. That target is
    fine for mice but easy to miss; making right-click on the button
    body open the menu too gives a second, less-fiddly path. The
    main left-click stays bound to ``clicked`` so the cycle logic
    still runs.
    """

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: D401, N802
        if event.button() == Qt.MouseButton.RightButton and self.menu() is not None:
            # Show the menu at the bottom-left of the button so it
            # drops down predictably, the way the dropdown arrow
            # would have done.
            self.menu().popup(self.mapToGlobal(self.rect().bottomLeft()))
            event.accept()
            return
        super().mousePressEvent(event)


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
