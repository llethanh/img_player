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
    QLineEdit,
    QPushButton,
    QToolButton,
    QWidget,
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
from img_player.ui.theme import G, H, S

if TYPE_CHECKING:
    from img_player.player.state import PlaybackState


_LOOP_CYCLE = [LoopMode.LOOP, LoopMode.ONCE, LoopMode.PING_PONG]
# Native emoji glyphs for the three loop modes — same style as the
# annotation toolbar's ✏️ 🧽 📌 (colorful OS-rendered emojis rather
# than monochrome text symbols).
_LOOP_LABELS = {
    LoopMode.LOOP:      ("🔁", "Loop (play → first frame at the end)"),
    LoopMode.ONCE:      ("▶️", "Play once (stop at the end)"),
    LoopMode.PING_PONG: ("🏓", "Ping-pong (reverse at the end)"),
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
    # Full channel selection (active + tiles + layout mode). The
    # checkable channel menu emits this whenever the user toggles a
    # radio or checkbox, so the controller can switch between single
    # and contact-sheet modes without having to interpret raw channel
    # lists. Carries a :class:`ChannelSelection`.
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
    # Export button (v0.5.0) — opens the export dialog. Disabled
    # until the app calls ``set_export_enabled(True)`` (which the
    # app does after a sequence loads).
    export_clicked = Signal()
    # Compare-mode toggle (v1.2). Carries no payload — the receiver
    # checks the button's ``isChecked()`` state via the public API.
    compare_toggled = Signal()
    # Reload button (v0.5.1) — smart re-scan of the source folder,
    # keeping cached frames whose mtime hasn't changed.
    reload_clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(G.TRANSPORT_H)

        self._loop_mode = LoopMode.LOOP

        # --- In/Out markers -------------------------------------------------
        # Same emoji-as-label style as the annotation toolbar
        # (✏️ 🧽 📌). Native OS rendering = colourful glyphs that
        # signal start / end / clean-up at a glance.
        self._mark_in_btn  = _text_button("🚩", "Mark IN at current frame (I)")
        self._mark_out_btn = _text_button("🏁", "Mark OUT at current frame (O)")
        self._clear_io_btn = _text_button("🧹", "Clear IN/OUT range (Shift+R)")

        self._mark_in_btn.clicked.connect(self.mark_in_clicked.emit)
        self._mark_out_btn.clicked.connect(self.mark_out_clicked.emit)
        self._clear_io_btn.clicked.connect(self.clear_in_out_clicked.emit)

        # --- Loop mode ------------------------------------------------------
        # Initial label reflects the default LOOP mode; the click
        # handler swaps it through ``_LOOP_LABELS`` as the user
        # cycles modes.
        self._loop_btn = _text_button("🔁", "Loop mode (click to cycle)")
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

        # --- Annotation controls (slice 4) ---------------------------------
        # Sit between the playback nav (last_btn) and FPS — a separate
        # logical group, exposed via three signals (toggle / prev /
        # next). The toggle button is checkable and reflects whether
        # the toolbar is currently visible.
        # ⏮️ / ⏭️ — track-skip emojis with U+FE0F variation selector for
        # consistent emoji-presentation across systems (the bare
        # codepoints have Emoji_Presentation=Yes by default, but the
        # selector is belt-and-braces against any future renderer
        # that reads them as text). ✏️ is U+270F LOWER RIGHT PENCIL,
        # which is text-by-default — the selector is REQUIRED here to
        # get the colorful pencil glyph the user picked from the
        # original toolbar mockup.
        self._annotation_prev_btn = _text_button(
            "⏮️", "Frame annotée précédente ([)"
        )
        self._annotation_toggle_btn = _text_button(
            "✏️", "Afficher / masquer la toolbar d'annotation (D)"
        )
        self._annotation_toggle_btn.setCheckable(True)
        self._annotation_next_btn = _text_button(
            "⏭️", "Frame annotée suivante (])"
        )
        # 👁 — show / hide annotations DURING playback. Checkable so
        # the user can lock it in either state. Default = checked
        # (annotations visible during play) to match the legacy
        # behaviour. Same toggle the ``A`` keyboard shortcut drives.
        # Inline stylesheet so checked vs unchecked is unambiguously
        # visible — without it the colour-emoji rendered by the OS
        # looks identical in both states (font-color CSS doesn't
        # touch coloured emoji glyphs). We tint the BACKGROUND
        # instead and add a ``◌`` strike-overlay via the glyph swap
        # so the difference reads at a glance.
        self._annotation_show_play_btn = _text_button(
            "👁", "Afficher les annotations pendant la lecture (A)"
        )
        self._annotation_show_play_btn.setCheckable(True)
        self._annotation_show_play_btn.setChecked(True)
        self._annotation_show_play_btn.setStyleSheet(
            "QPushButton {"
            "  font-size: 11pt;"
            "  padding: 0;"
            "}"
            "QPushButton:checked {"
            f"  background: {H.ACCENT_DIM};"
            f"  border: 1px solid {H.ACCENT};"
            f"  border-radius: {G.RADIUS_SM}px;"
            "}"
            "QPushButton:!checked {"
            f"  background: {H.BG_RAISED};"
            f"  border: 1px solid {H.BORDER_DEFAULT};"
            f"  border-radius: {G.RADIUS_SM}px;"
            "}"
        )
        # Glyph swap on toggle: open eye when ON, crossed eye when
        # OFF. Belt-and-braces with the background tint so even users
        # whose theme washes out the accent colour still see the
        # state change immediately.
        self._annotation_show_play_btn.toggled.connect(
            lambda on: self._annotation_show_play_btn.setText("👁" if on else "🚫")
        )
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
        # Disabled until a sequence is loaded — the app flips it on
        # via ``set_export_enabled``.
        self._export_btn = _text_button(
            "💾", "Export sequence to image seq or video (Ctrl+Shift+E)"
        )
        self._export_btn.clicked.connect(self.export_clicked.emit)
        self._export_btn.setEnabled(False)

        # --- Reload button (v0.5.1) -----------------------------------
        # 🔄 = smart re-scan. Keeps cached frames whose mtime is
        # unchanged, drops the rest, picks up files that were added
        # to the source folder while the app was running. Disabled
        # until a sequence is loaded.
        self._reload_btn = _text_button(
            "🔄", "Reload cache — re-scan source folder (Ctrl+R)"
        )
        self._reload_btn.clicked.connect(self.reload_clicked.emit)
        self._reload_btn.setEnabled(False)

        # --- Compare toggle (v1.2) ------------------------------------
        # Split-view icon (rectangle, vertical seam, A | B labels) in
        # the warm accent so it stands out on the menu-bar row as a
        # mode toggle rather than a generic transport control.
        # Checkable: stays "down" while compare mode is active.
        # Disabled until the stack has at least two layers — there's
        # nothing meaningful to compare against otherwise.
        self._compare_btn = _icon_button(
            make_icon(
                "compare",
                color=H.ACCENT,
                size=22,
                disabled_color=H.TEXT_DISABLED,
            ),
            "Compare two layers (W)",
        )
        # Render the icon larger inside its button than the
        # default G.ICON_SIZE (18) — this mark fills the button
        # less visually than the simpler arrow / disk glyphs and
        # was reading as "tiny" by comparison. 22 px gives it the
        # same on-screen weight as the other buttons.
        self._compare_btn.setIconSize(QSize(22, 22))
        self._compare_btn.setCheckable(True)
        self._compare_btn.clicked.connect(self.compare_toggled.emit)
        # Use ``border`` shorthand (not bare ``border-color``) so Qt
        # draws all 4 sides at the same width / style — bare
        # ``border-color`` left the bottom edge unrendered in some
        # paint paths. Add ``background-color`` so the rule isn't
        # ambiguous about the surface bg either; padding stays from
        # the global cascade.
        # Pin every property explicitly. The minute one is left to
        # the global cascade Qt's QSS engine seems to drop the
        # bottom edge on this checkable QPushButton (probably a
        # native-style merge artefact specific to the corner-widget
        # context). Including ``min-width`` / ``min-height`` /
        # ``padding`` / ``border-radius`` alongside the explicit
        # per-side borders has been the only reliable way to keep
        # all four edges painted.
        _border = f"1px solid {H.ACCENT_DIM}"
        _border_h = f"1px solid {H.ACCENT}"
        _border_chk = f"1px solid {H.ACCENT_BRIGHT}"
        # Disabled state — desaturated dim border + muted bg, so the
        # button reads "not actionable yet" when the layer stack only
        # has 0 or 1 entry. Uses a neutral border (not BORDER_SUBTLE
        # which collapses into the QMenuBar's bottom rule) so the
        # 4 edges stay visible. Icon is auto-greyed by Qt's Disabled
        # pixmap mode (handled in icons.make_icon).
        _border_dis = f"1px solid {H.BORDER_DEFAULT}"
        self._compare_btn.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {H.BG_SURFACE};"
            f"  color: {H.TEXT_PRIMARY};"
            f"  border-top: {_border};"
            f"  border-bottom: {_border};"
            f"  border-left: {_border};"
            f"  border-right: {_border};"
            f"  border-radius: 3px;"
            f"  padding: 0;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {H.BG_HOVER};"
            f"  border-top: {_border_h};"
            f"  border-bottom: {_border_h};"
            f"  border-left: {_border_h};"
            f"  border-right: {_border_h};"
            f"}}"
            f"QPushButton:checked {{"
            f"  background-color: {H.BG_SELECT};"
            f"  border-top: {_border_chk};"
            f"  border-bottom: {_border_chk};"
            f"  border-left: {_border_chk};"
            f"  border-right: {_border_chk};"
            f"}}"
            f"QPushButton:disabled {{"
            f"  background-color: {H.BG_BASE};"
            f"  color: {H.TEXT_DISABLED};"
            f"  border-top: {_border_dis};"
            f"  border-bottom: {_border_dis};"
            f"  border-left: {_border_dis};"
            f"  border-right: {_border_dis};"
            f"}}"
        )
        # Disabled by default — no sequence loaded means there's
        # nothing to compare. The app's ``_refresh_after_stack_change``
        # flips it on once the layer stack reaches at least 2 layers.
        # The :disabled QSS rule above paints a desaturated frame
        # using ``BORDER_DEFAULT`` (instead of BORDER_SUBTLE) so the
        # button still reads "not actionable yet" without the bottom
        # edge collapsing into a neighbouring border.
        self._compare_btn.setEnabled(False)

        # --- FPS ------------------------------------------------------------
        # Plain editable line — no dropdown of presets. The user
        # types whatever rate they want (24, 23.976, 60, …). A
        # double validator keeps the field numeric; on Enter / focus
        # loss the new value fires ``fps_changed``.
        from PySide6.QtGui import QDoubleValidator
        self._fps_combo = QLineEdit()
        self._fps_combo.setText("24")
        self._fps_combo.setFixedWidth(40)
        self._fps_combo.setFixedHeight(G.INPUT_H)
        self._fps_combo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fps_combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self._fps_combo.setToolTip("Playback rate (fps) — type a value, Enter to apply")
        validator = QDoubleValidator(0.1, 1000.0, 3, self._fps_combo)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self._fps_combo.setValidator(validator)
        # Commit on Enter or when the user clicks elsewhere.
        self._fps_combo.editingFinished.connect(
            lambda: self._on_fps_text(self._fps_combo.text()),
        )

        # --- Channel selector ----------------------------------------------
        # Multichannel EXR + contact-sheet support: a QToolButton that
        # opens the :class:`ChannelMenu` popup. The button label
        # summarises the current selection at a glance:
        #   * "RGB"             → single mode on RGB
        #   * "albedo"          → single mode on albedo
        #   * "RGB +2"          → contact sheet, RGB + 2 other tiles
        # The popup itself owns the per-row radio + checkbox state.
        self._channel_button = QToolButton()
        self._channel_button.setFixedHeight(G.INPUT_H)
        self._channel_button.setMinimumWidth(96)
        self._channel_button.setToolTip(
            "Channel to display — click to open the channel menu "
            "(check multiple to enable contact-sheet)"
        )
        self._channel_button.setText("RGB")
        # InstantPopup (vs MenuButtonPopup) — the whole button area
        # opens the menu, no mini arrow split. Cohérent with the loop
        # button's visual: one bordered button = one click action.
        self._channel_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self._channel_button.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._channel_menu = ChannelMenu(self._channel_button)
        self._channel_menu.selection_changed.connect(self._on_channel_selection_changed)
        self._channel_button.setMenu(self._channel_menu)

        # Track the current ChannelSelection so the button label can
        # be derived without re-querying the menu.
        self._current_selection: ChannelSelection | None = None

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
        layout.addWidget(self._reverse_play_btn)
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

        # Fullscreen toggle, sitting last on the right — that's the
        # corner reviewers reach for instinctively (YouTube / VLC
        # convention). Click cycles in / out of fullscreen, the
        # icon swaps between "expand" and "contract" arrows in
        # ``set_fullscreen_state``.
        layout.addWidget(_separator())
        self._fullscreen_btn = QPushButton()
        self._fullscreen_btn.setFixedSize(G.BTN_TRANSPORT_W, G.BTN_TRANSPORT_H)
        self._fullscreen_btn.setIcon(make_icon("fullscreen_enter"))
        self._fullscreen_btn.setIconSize(QSize(16, 16))
        self._fullscreen_btn.setToolTip("Fullscreen (F)")
        self._fullscreen_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._fullscreen_btn.clicked.connect(self.fullscreen_clicked.emit)
        layout.addWidget(self._fullscreen_btn)

        layout.addStretch(1)

        self._refresh_loop_button()

    def set_fullscreen_state(self, on: bool) -> None:
        """Swap the fullscreen button's icon to reflect the current
        mode. Called from ``MainWindow`` when fullscreen toggles."""
        self._fullscreen_btn.setIcon(
            make_icon("fullscreen_exit" if on else "fullscreen_enter")
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
    def channel_mute_buttons(self) -> dict[str, QPushButton]:
        return self._channel_btns

    @property
    def zoom_combo(self) -> QComboBox:
        return self._zoom_combo

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
            # The ``toggled`` signal would normally drive the text
            # swap; with signals blocked we have to do it manually
            # so the glyph still matches the new state.
            self._annotation_show_play_btn.setText("👁" if active else "🚫")
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
    btn.setIcon(icon)
    btn.setIconSize(QSize(G.ICON_SIZE, G.ICON_SIZE))
    btn.setFixedSize(G.BTN_TRANSPORT_W, G.BTN_TRANSPORT_H)
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
    line.setStyleSheet(f"background-color: {H.BORDER_DEFAULT};")
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
