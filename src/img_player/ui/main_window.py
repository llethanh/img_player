"""MainWindow: assembles the viewer, transport, timeline and side panels.

Signals from controls are routed to the :class:`PlayerController` and
:class:`GLViewport` by the app module; this widget only owns the UI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QIcon,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from img_player.color.ocio_manager import OCIOManager
from img_player.comment.store import CommentStore
from img_player.ui.color_panel import ColorPanel
from img_player.ui.comment_panel import CommentPanel
from img_player.ui.compare_band import CompareBand
from img_player.ui.contact_sheet_band import ContactSheetBand
from img_player.ui.icons import make_icon
from img_player.ui.theme import F, G, H, S
from img_player.ui.timeline import Timeline
from img_player.ui.transport import TransportBar
from img_player.ui.viewer_widget import ViewerWidget

if TYPE_CHECKING:
    from img_player.sequence.models import SequenceInfo

log = logging.getLogger(__name__)


class _SidePanelDock(QFrame):  # type: ignore[misc]
    """Drop-in replacement for the side ``QDockWidget`` instances.

    The annotation toolbar's docking code calls ``setWidget(w)`` /
    ``widget()`` on its dock host (the QDockWidget API). We used to
    pass a real QDockWidget; that made the annotation column span
    the full central-area height (down to the timeline / transport
    panels at the bottom), which the user noticed as a visual
    inconsistency. Lifting the host into the top row of the central
    layout keeps the annotation column flanking *only* the viewer.

    Only the small subset of the QDockWidget API the toolbar actually
    uses is reproduced here: a single child widget with replace
    semantics. Floating / drag-to-other-edge are gone but the
    annotation toolbar already had a "float" mode of its own
    (re-parents itself onto the viewer) that covers the same need.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._widget: QWidget | None = None

    def setWidget(self, widget: QWidget | None) -> None:
        if self._widget is not None:
            self.layout().removeWidget(self._widget)
            self._widget.setParent(None)
            self._widget = None
        if widget is not None:
            widget.setParent(self)
            self.layout().addWidget(widget)
            self._widget = widget

    def widget(self) -> QWidget | None:
        return self._widget


class MainWindow(QMainWindow):  # type: ignore[misc]
    """Top-level window wiring all the UI pieces together."""

    # All "open" / "add layer" signals carry a list of paths so a
    # single drop can mix multiple folders or loose files. Single-path
    # entry points (Open menu, Recent menu, programmatic boot) wrap
    # their path in a one-element list.
    open_requested = Signal(list)
    export_requested = Signal()  # File → Export… (v0.5.0)
    save_frame_requested = Signal()  # File → Save Frame As… (v1.2)
    # Compare-mode shortcuts (v1.2). W toggles the overlay; Ctrl+W
    # permutes A and B in the band's dropdowns.
    compare_toggle_requested = Signal()
    compare_swap_layers_requested = Signal()
    # Burnin overlay — View menu / Ctrl+B toggle and active-template pick.
    # ``burnin_toggle_requested`` carries the new bool (True = show).
    # ``burnin_template_requested`` carries the slug picked from the
    # "Active burnin template" submenu (``"default"`` etc).
    burnin_toggle_requested = Signal(bool)
    burnin_template_requested = Signal(str)
    # Open the burnin template editor dialog. The App owns the dialog
    # instance (single-instance) and connects the editor's
    # ``template_applied`` back through ``burnin_template_requested``.
    burnin_editor_requested = Signal()
    new_sequence_requested = Signal()      # File → New (Ctrl+N) — clear the loaded sequence
    add_layer_requested = Signal(list)     # File → Add layer… (v1.0)
                                           #   carries the picked paths
    save_session_requested = Signal(Path)  # File → Save session… (v1.0)
    open_session_requested = Signal(Path)  # File → Open session… (v1.0)
    reload_sequence_requested = Signal()   # Reload cache (Ctrl+R / button)
    # Force-reload — ignores mtime, drops every cached frame and
    # replays decode from scratch. Useful when files were overwritten
    # without an mtime bump (cp -p, restore from backup, sync still
    # propagating) or when the user just wants nuclear-option
    # confidence in the cache state.
    force_reload_sequence_requested = Signal()
    # Image → Clear cache… — wipe BOTH the RAM master cache and the
    # persistent disk cache (after user confirmation). Distinct from
    # force-reload: force-reload re-decodes the current sequence
    # right away; clear-cache also drops every persistent blob so
    # the next session restarts from the source files. The app-side
    # handler shows the QMessageBox + does the actual ``clear()``
    # calls so the UI layer stays free of policy.
    clear_cache_requested = Signal()
    # View → Contact sheet — toggle the multi-layer grid view.
    # Routes through the app singleton which owns the state +
    # decoder; the menu QAction's checked state is the source of
    # truth for the UI side and is kept in sync on every state
    # change via :meth:`set_contact_sheet_enabled`.
    contact_sheet_toggle_requested = Signal()
    # Re-emitted by the settings band when the user picks a new grid
    # or toggles the labels. ``cols`` / ``rows`` are -1 when set to
    # auto (the band has dedicated "Auto" entries in its combos);
    # the app translates ``-1`` to ``None`` for the state.
    contact_sheet_grid_changed = Signal(int, int)  # cols, rows
    contact_sheet_labels_toggled = Signal(bool)
    # Output downscale divisor: 1 (full res), 2 (half on each axis),
    # 3, 4, … The app's ``set_contact_sheet_output_divisor`` clamps
    # to a positive integer so the signal can carry any int safely.
    contact_sheet_divisor_changed = Signal(int)
    # Edit menu — same chained handlers as the Ctrl+Z / Ctrl+Shift+Z
    # QShortcuts (annotation first, layer-stack fallback). Routing
    # via signals keeps the App in charge of priority logic; the
    # menu's QAction is just the delivery mechanism.
    undo_requested = Signal()
    redo_requested = Signal()
    play_toggled = Signal()
    # Channel selection (single active group). Carries a
    # :class:`ChannelSelection`. Bridged from
    # ``TransportBar.channel_selection_changed`` so ``app.py`` can hook
    # the rich selection without reaching into the transport widget.
    channel_selection_changed = Signal(object)
    channel_mask_changed = Signal(tuple)  # (R, G, B, A) bools
    # User-controlled toggle on the alt-channel background prefetch
    # (the "⏸ / ▶" button right next to the channel selector). Carries
    # the new paused state. Bridged from
    # ``TransportBar.channel_cache_pause_toggled``.
    channel_cache_pause_toggled = Signal(bool)
    # User picked a transparency background. Carries an int 0..3
    # (0 = checker, 1 = black, 2 = mid-grey, 3 = white). Forwarded
    # from :class:`TransportBar`.
    transparency_bg_mode_changed = Signal(int)
    # Master volume slider (popup) in the transport bar. Forwarded
    # from :class:`TransportBar` so the app can push the gain to
    # ``AudioOutput`` + persist via Preferences. Mute is implicit:
    # slider at zero silences the output through the same gain
    # multiply — no separate mute toggle.
    master_volume_changed = Signal(float)  # 0.0-1.0 linear gain
    zoom_requested = Signal(object)       # float | None ; None = fit
    step_clicked = Signal(int)  # +1 / -1
    jump_to_ends = Signal(int)  # -1 first, +1 last
    frame_requested = Signal(int)
    # Scrub gesture lifecycle — forwarded from the Timeline so the
    # app can toggle video decoders into fast-seek mode for the
    # duration of the drag.
    scrub_started = Signal()
    scrub_finished = Signal()
    exposure_step = Signal(float)  # +/- keyboard adjustment
    fps_changed = Signal(float)
    direction_play_requested = Signal(int)  # +1 forward, -1 reverse (J/L)
    mark_in_requested = Signal()  # set in-point at current frame (I)
    mark_out_requested = Signal()  # set out-point at current frame (O)
    # Ctrl-click on the timeline drags an explicit in/out frame —
    # forwarded straight from Timeline.set_in_at_requested /
    # set_out_at_requested. Different signal from mark_in/out
    # because it carries a frame number rather than "use the
    # current playhead".
    set_in_at_requested = Signal(int)
    set_out_at_requested = Signal(int)
    clear_in_out_requested = Signal()  # reset in/out range (Shift+R)
    loop_mode_requested = Signal(object)  # LoopMode

    def __init__(
        self,
        ocio_manager: OCIOManager,
        comment_store: CommentStore,
        layer_stack=None,  # LayerStack | None — soft-typed to keep main_window Qt-only
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Flick Player")
        self.resize(1280, 720)
        # Per-zone drops are wired in :class:`ViewerWidget` and
        # :class:`MasterTimelinePanel`; the main window itself does
        # not accept drops anymore.

        # Path of the currently-loaded / last-saved session file. Set
        # by ``set_current_session_path`` from the App after a
        # successful Open / Save / Save As. Drives the Ctrl+S vs
        # Ctrl+Shift+S routing — Save (Ctrl+S) writes to this path
        # silently, falling back to Save As if it's None.
        self._current_session_path: Path | None = None

        # Fullscreen mode state. Populated lazily on first toggle —
        # the floating bottom bar is built once and reused.
        self._fullscreen: bool = False
        self._fs_bar: QWidget | None = None
        self._fs_exit_btn: QPushButton | None = None
        self._fs_hide_timer: QTimer | None = None
        # Cursor-position poller. We poll instead of using an event
        # filter because the viewer's GL child widget consumes mouse
        # events before they reach the parent — installing the filter
        # on every descendant is brittle. A 50 ms QTimer reading
        # ``QCursor.pos()`` covers the case bullet-proofly with
        # negligible cost.
        self._fs_cursor_timer: QTimer | None = None

        # Widgets
        self._viewer = ViewerWidget(self)
        self._transport = TransportBar(self)
        self._timeline = Timeline(self)
        self._color_panel = ColorPanel(ocio_manager, self)
        # The Channels tab was retired (channel info already lives in
        # the transport bar's combo + the four R/G/B/A mute toggles).
        # Replaced by a Comments tab — review-tool comment thread
        # attached to the current frame, see docs.
        self._comment_panel = CommentPanel(comment_store, self)
        # Multi-layer panel (v1.0) — sits between the timeline and
        # the transport bar. Always visible (Q10/A); collapses to its
        # header for vertical-space savings. Constructed only when
        # the app passes a stack reference, so existing single-layer
        # tests / programmatic callers can keep building MainWindow
        # without a stack.
        #
        # When a stack is provided we wrap the timeline + layer panel
        # in a :class:`MasterTimelinePanel` so the two widgets share
        # the exact same horizontal axis by layout construction (=
        # PDPlayer-style "one block" master timeline). Without a
        # stack, the bare ``Timeline`` is still added on its own — the
        # legacy single-sequence path keeps working.
        self._layer_panel = None
        self._master_timeline_panel = None
        if layer_stack is not None:
            from img_player.ui.layer_panel import (
                LayerPanel,
                MasterTimelinePanel,
            )
            self._layer_panel = LayerPanel(layer_stack, self)
            # The frame readout used to sit in the centre of the
            # transport bar; it migrated into the master timeline's
            # left gutter so the formerly-empty 122 px column gains
            # purpose. The widget itself is still owned by
            # ``TransportBar`` (which keeps the wiring); we just
            # reparent it visually.
            #
            # Pair the readout with a tiny "TC" toggle button so the
            # user can flip between frame numbers and timecode without
            # opening the View menu (or remembering Ctrl+T). The
            # button is wired to ``_show_tc_act`` later in
            # ``_build_menu`` since the action doesn't exist yet at
            # this point in __init__.
            from PySide6.QtWidgets import QToolButton
            # Shared QSS for the small "pill" toggles in the timeline
            # gutter. Two buttons use it (TC + info-band) so factor
            # the style string once.
            _pill_qss = (
                "QToolButton {"
                "  font-size: 9pt;"
                "  font-weight: 600;"
                f"  color: {H.TEXT_SECONDARY};"
                f"  background: {H.BG_RAISED};"
                f"  border: 1px solid {H.BORDER_DEFAULT};"
                f"  border-radius: {G.RADIUS_SM}px;"
                "  padding: 1px 4px;"
                "}"
                "QToolButton:checked {"
                f"  color: #FFFFFF;"
                f"  background: {H.ACCENT_DIM};"
                f"  border: 1px solid {H.ACCENT};"
                "}"
            )
            self._tc_toggle_btn = QToolButton(self)
            self._tc_toggle_btn.setText("TC")
            self._tc_toggle_btn.setCheckable(True)
            self._tc_toggle_btn.setToolTip(
                "Toggle timecode / frame number display (Ctrl+T)"
            )
            self._tc_toggle_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._tc_toggle_btn.setStyleSheet(_pill_qss)

            # Info-strip pill — wired to the same QAction created later
            # in ``_build_menu`` so menu / shortcut / pill stay in sync.
            self._info_band_btn = QToolButton(self)
            self._info_band_btn.setText("ⓘ")
            self._info_band_btn.setCheckable(True)
            self._info_band_btn.setChecked(True)
            self._info_band_btn.setToolTip(
                "Toggle the display info strip (Ctrl+I)"
            )
            self._info_band_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._info_band_btn.setStyleSheet(_pill_qss)
            # Compose [TC] [ⓘ] + [frame readout] into a single widget
            # so the gutter centres them as one unit.
            fd_wrapper = QWidget(self)
            fd_lay = QHBoxLayout(fd_wrapper)
            fd_lay.setContentsMargins(0, 0, 0, 0)
            fd_lay.setSpacing(4)
            fd_lay.addWidget(self._tc_toggle_btn)
            fd_lay.addWidget(self._info_band_btn)
            fd_lay.addWidget(self._transport.frame_display)
            self._master_timeline_panel = MasterTimelinePanel(
                self._timeline,
                self._layer_panel,
                frame_display=fd_wrapper,
                parent=self,
            )

        # Side panel (Color + Comments tabs). Used to live as a real
        # QDockWidget on the right side of the QMainWindow, which
        # made it span the FULL height of the central area —
        # including the bottom panels (master timeline + transport).
        # Reviewer feedback: the panel should only flank the viewer,
        # not push the timeline / transport area sideways. So it's
        # now a plain QFrame nested in the central widget's top row,
        # sitting beside the viewer. We lose the drag-to-float
        # gesture but the panel is meant to be docked anyway.
        self._side_tabs = QTabWidget()
        self._side_tabs.addTab(self._color_panel, "Color")
        self._side_tabs.addTab(self._comment_panel, "Comments")
        self._side_dock = QFrame(self)
        self._side_dock.setObjectName("side_dock")
        self._side_dock.setFrameShape(QFrame.Shape.NoFrame)
        self._side_dock.setFixedWidth(280)
        side_layout = QVBoxLayout(self._side_dock)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.setSpacing(0)
        side_layout.addWidget(self._side_tabs)
        # NB: image-dimensions readout used to live here, then
        # migrated to a top-right corner overlay on the viewer, then
        # finally to the bottom info band where it now sits along
        # with fps, layer frame and timeline frame readouts.

        # Central: top row [viewer | side panel] (only the display
        # area gets the side panel beside it), then the header info
        # strip (slim cartouche just under the viewport, where the
        # legacy InfoBand used to sit), then the master-timeline
        # composite + transport below spanning the full width.
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(S.SM, S.SM, S.SM, S.SM)
        layout.setSpacing(S.SM)
        top_row = QHBoxLayout()
        top_row.setContentsMargins(0, 0, 0, 0)
        top_row.setSpacing(S.SM)
        # NB: the annotation_dock is constructed below this layout's
        # construction call but inserted at index 0 once it exists —
        # putting it after creation here keeps the imports / order
        # readable. Pre-insert order: [viewer (stretch=1)]
        # [side_dock] ; final order after annotation insert:
        # [annotation_dock] [viewer] [side_dock].
        top_row.addWidget(self._viewer, stretch=1)
        top_row.addWidget(self._side_dock)
        self._top_row = top_row
        layout.addLayout(top_row, stretch=1)
        # Header info strip lives INSIDE the viewer now (floating
        # overlay pinned to the bottom edge) — see
        # ``ViewerWidget._reposition_header_strip``. Exposed here as
        # ``self._header_strip`` for the ``update_sequence_info`` /
        # ``app._refresh_header_strip_frames`` call sites that reach
        # for it by attribute name.
        self._header_strip = self._viewer.header_strip
        if self._master_timeline_panel is not None:
            layout.addWidget(self._master_timeline_panel)
        else:
            layout.addWidget(self._timeline)
        layout.addWidget(self._transport)
        self.setCentralWidget(central)

        # Full-window "OPEN SESSION" overlay. Shown only during a
        # drag-over carrying a ``.session`` file — the per-zone
        # REPLACE / ADD overlays don't apply to project files
        # (those need to wipe the entire stack, the spatial drop
        # disambiguation has nothing to choose between). Lives as
        # a child of ``self`` so it can absolute-position over
        # everything including the menu bar.
        from img_player.ui.drop_zone import (
            SESSION_ACCENT,
            DropOverlay,
            get_default_coordinator,
        )
        self._session_drop_overlay = DropOverlay(
            "OPEN SESSION", SESSION_ACCENT, self,
        )
        get_default_coordinator().register_session_overlay(
            self._session_drop_overlay,
        )

        # Annotation toolbar placeholder. Used to be a real QDockWidget
        # which made it span the full central-area height; reviewer
        # feedback: the toolbar should only flank the display area
        # (consistent with the right-hand panel). Now a plain QFrame
        # with a ``setWidget`` / ``widget`` compat shim so the
        # AnnotationToolbar's existing dock-mode code keeps working
        # without changes. Inserted as the first widget in the top
        # row of the central layout (= left of the viewer); the
        # bottom panels (master timeline + transport) span full
        # width and are unaffected.
        self._annotation_dock = _SidePanelDock(self)
        self._annotation_dock.setObjectName("annotation_dock")
        self._annotation_dock.hide()  # shown only when toolbar is in dock mode
        self._top_row.insertWidget(0, self._annotation_dock)

        # Callback hook fired from closeEvent before the window
        # actually closes. Set by ``app.py`` to prompt the user
        # about saving annotations.
        self._before_close_callback: Callable[[], bool] | None = None

        self._build_menu()
        self._install_shortcuts()
        self._wire_internal()
        self._build_status_bar()

    # --------------------------------------------------------------- Status bar

    def _build_status_bar(self) -> None:
        """Three-zone status bar — brief §10.

        Replaced in the 2026-Q2 redesign with a dedicated
        :class:`img_player.ui.status_bar.StatusBar` widget. The
        attribute aliases ``self.status_left`` / ``self.status_selection``
        / ``self.status_right`` are mirrored from the widget so any
        legacy caller that imported them by name (e.g.
        ``app._refresh_status``'s direct ``status_right.setText``)
        keeps working.
        """
        from img_player.ui.status_bar import StatusBar  # noqa: PLC0415

        sb = StatusBar(self)
        self.setStatusBar(sb)
        # Backward-compat aliases — old code reaches for these by name.
        self.status_left = sb.status_left
        self.status_selection = sb.status_selection
        self.status_right = sb.status_right
        # Keep a typed handle so the new helpers (set_session_dot,
        # set_perf_html) can be reached without a sub-widget walk.
        self._status_bar_widget = sb
        # Status-bar gear button → existing Preferences dialog entry
        # point. Same code path as File → Open Preferences… and
        # Ctrl+,. Adds a permanent visible discovery hook.
        sb.preferences_clicked.connect(self._open_preferences)

    def set_selected_layers(self, text: str) -> None:
        """Update the middle status-bar widget — the focused layer
        name. Empty string hides the FOCUS pill + name (no leftover
        "—" when nothing's focused)."""
        self._status_bar_widget.set_selected_layers(text)

    # --------------------------------------------------------------- Accessors

    @property
    def viewer(self) -> ViewerWidget:
        return self._viewer

    @property
    def compare_band(self) -> CompareBand:
        """Compare-mode toolbar — lives in the menu-bar's right corner
        widget so it shares its row with File/Edit/View instead of
        cropping the viewport. Hidden by default."""
        return self._compare_band

    def set_compare_band_visible(self, on: bool) -> None:
        """Toggle the compare band's visibility.

        The band lives between two stretches in the top layout. We
        also flip the right stretch's weight so the band centres
        between menus and buttons when visible, and the buttons sit
        flush right when the band is hidden.
        """
        on = bool(on)
        self._compare_band.setVisible(on)
        # Right stretch weight: 1 when EITHER band is on (equal split
        # with the left stretch → band centred), 0 when both are off
        # (left stretch absorbs the slack → buttons flush right).
        any_band_on = on or self._contact_sheet_band.isVisible()
        self._top_layout.setStretch(
            self._top_stretch_right_idx, 1 if any_band_on else 0,
        )

    @property
    def contact_sheet_band(self) -> ContactSheetBand:
        """Contact-sheet settings band — exposed so the app can wire
        its signals and push state updates. Hidden by default; the
        app shows it when contact-sheet mode turns on."""
        return self._contact_sheet_band

    def set_contact_sheet_band_visible(self, on: bool) -> None:
        """Toggle the contact-sheet band's visibility.

        Same stretch-flip pattern as :meth:`set_compare_band_visible`
        — when either band is visible the right stretch takes weight
        1 so the band centres; when both are hidden it drops back to
        0 so the buttons toolbar sits flush right.
        """
        on = bool(on)
        self._contact_sheet_band.setVisible(on)
        any_band_on = on or self._compare_band.isVisible()
        self._top_layout.setStretch(
            self._top_stretch_right_idx, 1 if any_band_on else 0,
        )

    @property
    def color_panel(self) -> ColorPanel:
        return self._color_panel

    @property
    def annotation_dock(self) -> _SidePanelDock:
        """Placeholder host the AnnotationToolbar reparents into when
        the user picks dock mode. Provides a ``setWidget`` /
        ``widget`` shim compatible with the QDockWidget API the
        toolbar was originally written against."""
        return self._annotation_dock

    @property
    def comment_panel(self) -> CommentPanel:
        """The Comments tab in the right-hand side dock — exposed
        so the app can update its current frame as the playhead
        moves."""
        return self._comment_panel

    def side_tab_index(self) -> int:
        """Currently selected tab in the right-side dock. Persisted
        across sessions via :class:`Preferences.side_tab_index`."""
        return self._side_tabs.currentIndex()

    def set_side_tab_index(self, index: int) -> None:
        """Restore the previously-selected side-tab. Clamped against
        the actual tab count so a future redesign that removes a tab
        doesn't crash an old preference value."""
        tab_count = self._side_tabs.count()
        if tab_count == 0:
            return
        clamped = max(0, min(int(index), tab_count - 1))
        self._side_tabs.setCurrentIndex(clamped)

    def display_timecode(self) -> bool:
        """Current state of the View → Show timecode toggle."""
        return self._show_tc_act.isChecked()

    def set_display_timecode(self, enabled: bool) -> None:
        """Restore the timecode-display preference. Drives both the
        QAction's checked state and the dependent widgets (timeline
        + transport's frame display)."""
        self._show_tc_act.setChecked(bool(enabled))
        # The QAction's ``triggered`` signal fires on user click, not
        # on programmatic ``setChecked`` — call the slot directly so
        # the timeline + transport pick up the mode.
        self._on_toggle_timecode(bool(enabled))

    def set_contact_sheet_enabled(self, enabled: bool) -> None:
        """Sync the View → Contact sheet checkmark to ``enabled``.

        Used at boot (preferences restore) and after the app's
        :meth:`toggle_contact_sheet` flips the state — keeps the
        menu in lockstep with the truth without retriggering the
        toggle signal.
        """
        if self._contact_sheet_act.isChecked() == bool(enabled):
            return
        self._contact_sheet_act.blockSignals(True)
        try:
            self._contact_sheet_act.setChecked(bool(enabled))
        finally:
            self._contact_sheet_act.blockSignals(False)

    # Mirror of the state on the app singleton — set by the app so
    # the band can rebuild itself with the right state. The window
    # doesn't need to know about ``ContactSheetState`` — just the
    # fields it forwards to the band.
    _contact_sheet_grid: tuple[int | None, int | None] = (None, None)
    _contact_sheet_labels: bool = False
    _contact_sheet_divisor: int = 1
    _contact_sheet_label_size: float = 1.0

    def set_contact_sheet_grid_state(
        self,
        cols: int | None,
        rows: int | None,
        show_labels: bool,
        output_divisor: int = 1,
        label_size: float = 1.0,
    ) -> None:
        """Sync the contact-sheet band's widgets from the app side.

        Pass ``None`` for cols / rows to mark "auto". ``label_size``
        is a float multiplier (1.0 = historical default); the band
        snaps it to the closest preset in the Size combo.
        """
        self._contact_sheet_grid = (cols, rows)
        self._contact_sheet_labels = bool(show_labels)
        self._contact_sheet_divisor = max(1, int(output_divisor))
        self._contact_sheet_label_size = max(0.4, min(float(label_size), 4.0))
        # Push to the band so its spin-boxes / pills / combos reflect
        # reality. The band's ``set_state`` blocks internal emissions
        # so this call is safe even when the state change came FROM
        # the band itself (no signal loop).
        self._contact_sheet_band.set_state(
            cols,
            rows,
            self._contact_sheet_labels,
            self._contact_sheet_divisor,
            self._contact_sheet_label_size,
        )

    @property
    def transport(self) -> TransportBar:
        return self._transport

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    # --------------------------------------------------------------- Public updates

    def update_sequence_info(self, sequence: SequenceInfo) -> None:
        """Refresh the title bar, timeline range, and the
        transport bar's channel selector.

        Loading a new sequence wipes any per-sequence UI state that
        would otherwise leak across (the cache-bar runs from the
        *previous* sequence, channel labels, etc.). The 200 ms refresh
        timer would eventually catch up — but a noticeable flash of
        stale cache marks is exactly what the user reported, so we
        reset eagerly here.
        """
        self.setWindowTitle(f"Flick Player — {sequence.display_pattern()}")
        self._timeline.set_range(sequence.first_frame, sequence.last_frame)
        # Header info strip — populate the per-sequence cells and
        # surface the strip itself. Frame-cell updates fire later from
        # the playhead's frame_changed signal via :meth:`update_frame_position`.
        self._header_strip.set_sequence_name(sequence.display_pattern())
        self._header_strip.set_resolution(sequence.width, sequence.height)
        # FPS isn't on SequenceInfo — it lives on the controller. The
        # app picks it up via :meth:`update_fps`; we leave the cell
        # showing — until that call lands.
        layer_total = sequence.last_frame - sequence.first_frame + 1
        self._header_strip.set_layer_position(sequence.first_frame, layer_total)
        self._header_strip.set_frame_position(sequence.first_frame, layer_total)
        self._header_strip.set_visible_for_sequence(True)
        # A sequence is loaded → enable the File → Export… action +
        # the 💾 transport bar button + Reload.
        if hasattr(self, "_export_act"):
            self._export_act.setEnabled(True)
        if hasattr(self, "_save_frame_act"):
            self._save_frame_act.setEnabled(True)
        if hasattr(self, "_reload_act"):
            self._reload_act.setEnabled(True)
        if hasattr(self, "_force_reload_act"):
            self._force_reload_act.setEnabled(True)
        if hasattr(self, "_add_layer_act"):
            self._add_layer_act.setEnabled(True)
        if hasattr(self, "_save_session_act"):
            self._save_session_act.setEnabled(True)
        if hasattr(self, "_save_session_as_act"):
            self._save_session_as_act.setEnabled(True)
        self._transport.set_export_enabled(True)
        self._transport.set_reload_enabled(True)
        # Re-detect colorspace button — needs footage to act on.
        # Enabled here alongside the rest of the sequence-gated
        # actions; disabled symmetrically by the New / detach paths.
        self._color_panel.set_redetect_enabled(True)
        # Clear the cache bar so we don't briefly show the old run
        # rectangles re-mapped onto the new range. The next
        # _refresh_cache_bar tick (~200 ms) re-populates with the
        # actually-cached frames of the new sequence.
        self._timeline.set_cached_frames(frozenset())
        self._timeline.set_missing_frames(frozenset())
        # Populate the channel-selector combo so the user can pick
        # individual channels (Z, normals, AOVs, …).
        self._transport.set_available_channels(sequence.channel_names)

    def set_status(self, message: str) -> None:
        """Set the contextual message on the *left* side of the status bar.

        Kept as a method for backwards compat with all the existing call
        sites (``Loaded …``, ``In point set to frame …``, etc.).
        """
        self._status_bar_widget.set_status(message)

    # --------------------------------------------------------------- Menu / shortcuts

    def _build_menu(self) -> None:
        # Recent-paths callback provider. The app module installs real
        # callbacks via install_recent_provider(); we seed empty defaults
        # so the menu can render before that happens.
        self._recent_paths_provider: Callable[[], list[Path]] = lambda: []
        self._clear_recent_callback: Callable[[], None] = lambda: None
        # Same shape but for ``.session`` files. Installed by
        # ``install_recent_session_provider``.
        self._recent_sessions_provider: Callable[[], list[Path]] = lambda: []
        self._clear_recent_sessions_callback: Callable[[], None] = lambda: None

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        # File → New (Ctrl+N): clear the currently loaded sequence
        # without resetting the rest of the UI (color panel, toolbar
        # mode, FPS, etc.). Useful before opening a different
        # sequence — keeps the user's tooling state intact.
        new_act = QAction("&New", self)
        new_act.setShortcut(QKeySequence("Ctrl+N"))
        new_act.triggered.connect(self.new_sequence_requested.emit)
        file_menu.addAction(new_act)

        open_act = QAction("&Open…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_action)
        file_menu.addAction(open_act)

        # Open Recent lives directly under Open so the two-step
        # "ouvrir → re-ouvrir un truc connu" path is visually contiguous,
        # mirroring the "Open Session… → Open Recent Session" pair
        # further down. (Used to sit way further down, just before the
        # Export separator — hard to find.)
        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._recent_menu.aboutToShow.connect(self._refresh_recent_menu)
        # Pre-populate so the submenu is never empty on first open.
        self._refresh_recent_menu()

        # File → Add layer… (v1.0). Pops the same folder picker as
        # Open, but the result is *added* to the LayerStack as a
        # new top layer instead of replacing the current sequence.
        # Disabled until at least one sequence is loaded — adding
        # a "first" layer goes through Open / drag-drop.
        self._add_layer_act = QAction("&Add layer…", self)
        self._add_layer_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self._add_layer_act.setEnabled(False)
        self._add_layer_act.triggered.connect(self._on_add_layer_action)
        file_menu.addAction(self._add_layer_act)

        # File → Open session… / Save session… (v1.0 phase 6b).
        # Persist the full multi-layer state to a ``.session`` JSON
        # file so the user can come back to the same review setup
        # without redoing every drop / trim / channel choice.
        file_menu.addSeparator()
        open_session_act = QAction("Open Sess&ion…", self)
        # Ctrl+L (Load) — paired with Ctrl+S below. Ctrl+O is already
        # taken by sequence open and Ctrl+Shift+O by Add Layer, so we
        # use a distinct mnemonic for the session-level entry point.
        open_session_act.setShortcut(QKeySequence("Ctrl+L"))
        open_session_act.triggered.connect(self._on_open_session_action)
        file_menu.addAction(open_session_act)
        # Open Recent submenu specifically for sessions — same shape
        # as the sequence-side "Open Recent" but driven by a
        # different QSettings key so the two lists don't blend.
        self._recent_sessions_menu = file_menu.addMenu("Open Recent S&ession")
        self._recent_sessions_menu.aboutToShow.connect(
            self._refresh_recent_sessions_menu,
        )
        self._refresh_recent_sessions_menu()
        self._save_session_act = QAction("Save Session", self)
        # Ctrl+S — silent overwrite of the current ``.session`` file
        # when one is known (= a session was opened or save-as'd
        # earlier in the session). Falls back to the file picker
        # via ``Save Session As`` when no current path is set, so the
        # user still gets a graceful path on first save. Gated by
        # the sequence-loaded flag — Qt skips disabled QActions, so
        # Ctrl+S on an empty player is a silent no-op.
        self._save_session_act.setShortcut(QKeySequence("Ctrl+S"))
        self._save_session_act.setEnabled(False)
        self._save_session_act.triggered.connect(self._on_save_session_action)
        file_menu.addAction(self._save_session_act)
        self._save_session_as_act = QAction("Save Session &As…", self)
        # Ctrl+Shift+S — always opens the file picker, regardless of
        # whether a current session path exists. Used to fork the
        # session into a new file or to give a name to a never-yet-
        # saved working state.
        self._save_session_as_act.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self._save_session_as_act.setEnabled(False)
        self._save_session_as_act.triggered.connect(
            self._on_save_session_as_action,
        )
        file_menu.addAction(self._save_session_as_act)

        # Reload (Ctrl+R): smart re-scan of the source folder.
        # Keeps cached frames whose mtime is unchanged, drops the
        # rest, and surfaces any newly-arrived / removed files.
        # Disabled until a sequence is loaded — same gating as
        # Export.
        #
        # NB: lives in the Image menu only (re-added below). Used to
        # also sit under File but that was duplication — Image is the
        # natural home since reload is about the frames, not file I/O
        # bookkeeping. The QAction itself is still created here so the
        # Ctrl+R / Ctrl+Shift+R shortcuts stay app-wide active and so
        # the existing toolbar / status hooks (`hasattr(self,
        # "_reload_act")`) keep working unchanged.
        self._reload_act = QAction("&Reload (smart)", self)
        self._reload_act.setToolTip(
            "Re-scan the source folder and re-decode only frames "
            "whose mtime changed on disk."
        )
        self._reload_act.setShortcut(QKeySequence("Ctrl+R"))
        self._reload_act.setEnabled(False)
        self._reload_act.triggered.connect(self.reload_sequence_requested.emit)

        # Reload (force) (Ctrl+Shift+R): drop every cached frame and
        # re-decode from scratch, ignoring mtime. Same enable-gating
        # as the smart reload. Also Image-menu-only — see comment
        # above.
        self._force_reload_act = QAction("Reload (&force)", self)
        self._force_reload_act.setToolTip(
            "Drop every cached frame and re-decode from scratch, "
            "regardless of mtime — for the rare case where files "
            "were overwritten without an mtime bump."
        )
        self._force_reload_act.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self._force_reload_act.setEnabled(False)
        self._force_reload_act.triggered.connect(
            self.force_reload_sequence_requested.emit,
        )

        file_menu.addSeparator()
        # Export action (v0.5.0). Disabled until a sequence is loaded —
        # the app re-enables it after a successful open.
        self._export_act = QAction("&Export…", self)
        self._export_act.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self._export_act.setEnabled(False)
        self._export_act.triggered.connect(self.export_requested.emit)
        file_menu.addAction(self._export_act)

        # Save Frame As… (v1.2) — quick WYSIWYG snapshot of the
        # current viewer with optional annotations / overlay. Lives
        # next to Export so the two related "produce a file" actions
        # are visually grouped.  Ctrl+Alt+S keeps the muscle memory
        # close to Save (Ctrl+S) without colliding with Save Session
        # As (Ctrl+Shift+S).
        self._save_frame_act = QAction("Save &Frame As…", self)
        self._save_frame_act.setShortcut(QKeySequence("Ctrl+Alt+S"))
        self._save_frame_act.setEnabled(False)
        self._save_frame_act.triggered.connect(self.save_frame_requested.emit)
        file_menu.addAction(self._save_frame_act)

        # File → Preferences… (Ctrl+,) — application-wide settings
        # dialog. Currently hosts the OCIO Color Management section;
        # designed as a sidebar+stack layout so future preferences
        # categories (Playback, Cache, …) drop in without restructuring.
        file_menu.addSeparator()
        self._preferences_act = QAction("Open &Preferences…", self)
        self._preferences_act.setShortcut(QKeySequence("Ctrl+,"))
        self._preferences_act.triggered.connect(self._open_preferences)
        file_menu.addAction(self._preferences_act)

        # NB: explicit Quit action removed — the OS-default close
        # button (X) and Alt+F4 / Cmd+Q already cover the gesture, and
        # the in-app Ctrl+Q shortcut was unreliable on Windows (Qt's
        # ``StandardKey.Quit`` resolves to nothing there). Keeping the
        # action only added a dead menu entry.

        # --- Edit menu : Undo / Redo ----------------------------------
        # Routes through ``undo_requested`` / ``redo_requested`` so
        # the app's chained handler (annotations first, layer stack
        # fallback) stays the single source of truth. The shortcuts
        # match the QShortcut bindings in ``app.py``; Qt resolves the
        # collision deterministically (action wins inside the menu's
        # context, the QShortcut catches keystrokes outside menus —
        # together they cover every focus situation).
        edit_menu = menu_bar.addMenu("&Edit")
        self._undo_act = QAction("&Undo", self)
        self._undo_act.setShortcut(QKeySequence.StandardKey.Undo)
        self._undo_act.triggered.connect(self.undo_requested.emit)
        edit_menu.addAction(self._undo_act)

        self._redo_act = QAction("&Redo", self)
        self._redo_act.setShortcut(QKeySequence.StandardKey.Redo)
        self._redo_act.triggered.connect(self.redo_requested.emit)
        edit_menu.addAction(self._redo_act)

        # --- Image menu : actions that operate on the loaded image ----
        # Currently the two reload variants (smart + force). Same
        # QActions as the File menu duplicates so the shortcuts
        # (Ctrl+R / Ctrl+Shift+R) stay single-binding and the gating
        # state is shared automatically. Discoverable from the new
        # menu without taking the user out of the File menu's
        # file-level operations.
        image_menu = menu_bar.addMenu("&Image")
        image_menu.addAction(self._reload_act)
        image_menu.addAction(self._force_reload_act)

        # Image → Clear cache… (Ctrl+Alt+Shift+R). Wipes the RAM
        # master cache AND the persistent disk cache after a
        # confirmation dialog (the app-side handler owns the dialog
        # so the UI layer doesn't import QMessageBox here). Enabled
        # unconditionally — even without a loaded sequence the disk
        # cache may still hold blobs from previous sessions, and
        # clearing them is a legitimate maintenance gesture.
        image_menu.addSeparator()
        self._clear_cache_act = QAction("&Clear cache…", self)
        self._clear_cache_act.setToolTip(
            "Wipe both the in-RAM frame cache and the persistent disk "
            "cache. The next playback will re-decode from the source "
            "files."
        )
        self._clear_cache_act.setShortcut(QKeySequence("Ctrl+Alt+Shift+R"))
        self._clear_cache_act.triggered.connect(
            self.clear_cache_requested.emit,
        )
        image_menu.addAction(self._clear_cache_act)

        # --- View menu : timeline display mode ----------------------------
        view_menu = menu_bar.addMenu("&View")
        self._show_tc_act = QAction("Show &timecode", self, checkable=True)
        self._show_tc_act.setShortcut(QKeySequence("Ctrl+T"))
        self._show_tc_act.triggered.connect(self._on_toggle_timecode)
        view_menu.addAction(self._show_tc_act)

        # Header info strip — the orange cartouche over the viewer
        # (sequence name / resolution / fps / Layer / Frame). On by
        # default; user toggles with Ctrl+I, the View menu entry, or
        # the ⓘ pill in the timeline gutter. Same triple-bind pattern
        # as the TC pill below.
        self._show_info_band_act = QAction(
            "Show &info strip", self, checkable=True,
        )
        self._show_info_band_act.setShortcut(QKeySequence("Ctrl+I"))
        self._show_info_band_act.setChecked(True)
        self._show_info_band_act.toggled.connect(
            self._viewer.header_strip.set_user_visible,
        )
        view_menu.addAction(self._show_info_band_act)
        ib_btn = getattr(self, "_info_band_btn", None)
        if ib_btn is not None:
            ib_btn.clicked.connect(self._show_info_band_act.trigger)

            def _sync_ib_btn(on: bool) -> None:
                if ib_btn.isChecked() != on:
                    ib_btn.blockSignals(True)
                    try:
                        ib_btn.setChecked(on)
                    finally:
                        ib_btn.blockSignals(False)

            self._show_info_band_act.toggled.connect(_sync_ib_btn)

        # Burnins — Ctrl+B toggles the overlay; the submenu picks the
        # active template among the shipped builtins. Both go through
        # signals so the App stays in charge of state + persistence.
        view_menu.addSeparator()
        self._show_burnins_act = QAction(
            "Show &burnins", self, checkable=True,
        )
        self._show_burnins_act.setShortcut(QKeySequence("Ctrl+B"))
        self._show_burnins_act.setChecked(False)   # App overrides from prefs at wire time
        self._show_burnins_act.toggled.connect(self.burnin_toggle_requested.emit)
        view_menu.addAction(self._show_burnins_act)

        # Active template submenu — builtins shown at boot; user
        # templates appear after the editor saves them (the App calls
        # :meth:`refresh_burnin_template_menu` to repopulate). The
        # action group is exclusive so only one slug is checked at a
        # time.
        from PySide6.QtGui import QActionGroup  # noqa: PLC0415
        self._burnin_template_menu = view_menu.addMenu("Active burnin &template")
        self._burnin_template_group = QActionGroup(self)
        self._burnin_template_group.setExclusive(True)
        self._burnin_template_actions: dict[str, QAction] = {}
        # Initial population — the App can call
        # :meth:`refresh_burnin_template_menu` later to add user
        # templates the editor has saved.
        self.refresh_burnin_template_menu(
            ["default"],
            "default",
        )

        # "Edit burnins…" — opens the template editor dialog.
        edit_act = QAction("&Edit burnins…", self)
        edit_act.triggered.connect(self.burnin_editor_requested.emit)
        view_menu.addAction(edit_act)

        # The "TC" pill next to the frame readout (built earlier in
        # ``__init__``) drives the same action. Click → trigger the
        # action; the action's toggled signal then mirrors back to
        # the button (with signals blocked to avoid the menu /
        # shortcut path double-firing ``_on_toggle_timecode``).
        tc_btn = getattr(self, "_tc_toggle_btn", None)
        if tc_btn is not None:
            tc_btn.clicked.connect(self._show_tc_act.trigger)

            def _sync_tc_btn(on: bool) -> None:
                if tc_btn.isChecked() != on:
                    tc_btn.blockSignals(True)
                    try:
                        tc_btn.setChecked(on)
                    finally:
                        tc_btn.blockSignals(False)

            self._show_tc_act.toggled.connect(_sync_tc_btn)

        # View → Contact &sheet (Ctrl+G) — multi-layer grid view.
        # All visible layers tile into a contact-sheet composite,
        # each re-aligned to the same "frame 0" regardless of its
        # timeline offset. Useful for reviewing N versions side by
        # side without dragging layer offsets around.
        view_menu.addSeparator()
        self._contact_sheet_act = QAction(
            "Contact &sheet", self, checkable=True,
        )
        self._contact_sheet_act.setShortcut(QKeySequence("Ctrl+G"))
        self._contact_sheet_act.setToolTip(
            "Toggle multi-layer grid view. Every visible layer becomes "
            "a tile, re-aligned so they all 'start at frame 0' "
            "regardless of timeline offset."
        )
        self._contact_sheet_act.triggered.connect(
            self.contact_sheet_toggle_requested.emit,
        )
        view_menu.addAction(self._contact_sheet_act)
        # Grid / labels / output-size now live in the dedicated
        # ContactSheetBand toolbar (`self._contact_sheet_band`) that
        # appears above the viewer when the mode is on — same UX as
        # the compare band. The older ``View → Contact sheet settings``
        # sub-menu was removed because the band is always visible
        # while the mode is on and is much easier to discover than a
        # nested menu.

        # NB: T / αS used to live here as global View menu toggles.
        # They moved to per-row buttons in the layer panel once the
        # underlying state became per-layer; the menu duplicate would
        # silently target a "focused layer" with no visual cue
        # showing which layer it edits, which proved confusing.

        # --- Burger button → toggle the right-hand dock ------------------
        # Lives in the menu bar's top-right corner so it's always
        # reachable, even when the dock is hidden (otherwise users
        # would have no way to bring it back).
        # NB: we use QToolButton rather than QPushButton on purpose —
        # the global QSS sets `QPushButton { min-height: 28px }` which
        # makes a button taller than the 26 px menubar and clips it
        # off-screen. QToolButton has its own independent QSS scope.
        burger = QToolButton(self)
        # Painted SVG icon: three perfectly-spaced bars. We previously
        # rendered the Unicode glyph U+2630 (☰) as text, but its
        # spacing varies wildly across system fonts (the glyph is
        # designed for CJK contexts, not UI hamburgers) — users
        # noticed the bars looked uneven. SVG primitives give us
        # pixel-exact control.
        #
        # We build a fresh QIcon with two pixmaps so the colour
        # tracks the hover state — Qt swaps to ``Active`` mode
        # automatically when the cursor enters a QAbstractButton,
        # which is the gentler equivalent of the ``:hover`` colour
        # we used to apply to the text glyph.
        burger_size = 20
        burger_icon = QIcon()
        burger_icon.addPixmap(
            make_icon("menu", color=H.ACCENT, size=burger_size).pixmap(
                burger_size, burger_size,
            ),
            QIcon.Mode.Normal,
        )
        burger_icon.addPixmap(
            make_icon("menu", color=H.ACCENT_BRIGHT, size=burger_size).pixmap(
                burger_size, burger_size,
            ),
            QIcon.Mode.Active,
        )
        burger.setIcon(burger_icon)
        burger.setIconSize(QSize(burger_size, burger_size))
        burger.setAutoRaise(True)
        burger.setFixedSize(40, 32)
        burger.setCursor(Qt.CursorShape.PointingHandCursor)
        burger.setToolTip("Show / hide the side panels")
        # The hover background gives a rounded affordance patch.
        # Color of the icon itself is baked into the SVG; QToolButton
        # mode handles the hover swap.
        burger.setStyleSheet(
            "QToolButton {"
            f"  background: transparent;"
            f"  border: none;"
            f"  padding: 0;"
            "}"
            "QToolButton:hover {"
            f"  background: {H.BG_HOVER};"
            f"  border-radius: {G.RADIUS_SM}px;"
            "}"
        )
        burger.clicked.connect(self._toggle_side_dock)

        self._burger_btn = burger

        # --- Help menu ----------------------------------------------------
        # Added before the right-toolbar wiring below so the menu bar
        # has its full set of menus before we measure / install it.
        help_menu = menu_bar.addMenu("&Help")
        shortcuts_act = QAction("&Keyboard shortcuts…", self)
        shortcuts_act.setShortcut(QKeySequence("F1"))
        shortcuts_act.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_act)
        help_menu.addSeparator()
        about_act = QAction("&About Flick Player", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

        # --- Top bar layout ----------------------------------------------
        # [QMenuBar | <stretch L> | compare_band | <stretch R> | buttons]
        #
        # Three flex zones:
        # * Left stretch — fills the gap between Help and the centred
        #   compare band.
        # * Right stretch — fills the gap between the band and the
        #   buttons toolbar (reload / export / channel / zoom / …).
        #   Its weight is toggled in ``set_compare_band_visible``: 1
        #   when compare is on (= the band centres), 0 when off (=
        #   the buttons sit flush against the right edge).
        # Standard QHBoxLayout — no manual width math, no QMenuBar
        # corner-widget tricks. Combos with Preferred policy squeeze
        # naturally when the window narrows, after the stretches have
        # collapsed to zero.
        buttons_toolbar = QWidget(self)
        buttons_toolbar.setStyleSheet("background: transparent;")
        buttons_layout = QHBoxLayout(buttons_toolbar)
        buttons_layout.setContentsMargins(0, 0, S.MD, 0)
        buttons_layout.setSpacing(S.SM)
        # Compare sits LEFT of reload — review-mode action first,
        # then file-level commands (reload, export). Contact sheet
        # is the sibling review-mode toggle next to compare; the two
        # are mutually exclusive (both hijack the GL upload) so the
        # app greys out the inactive one when its sibling is on.
        buttons_layout.addWidget(self._transport.compare_button)
        buttons_layout.addWidget(self._transport.contact_sheet_button)
        # NB: the older ``…`` kebab button next to the contact-sheet
        # toggle is no longer added to the layout — settings live in
        # the ContactSheetBand toolbar (`self._contact_sheet_band`)
        # that appears above the viewer when the mode is on. The
        # QToolButton itself still exists on ``self._transport`` (so
        # the property accessor stays valid) but is unparented from
        # any visible layout.
        buttons_layout.addWidget(self._transport.reload_button)
        buttons_layout.addWidget(self._transport.export_button)
        # Channel selector + RGBA mode selector, grouped tight.
        # The mode button (RGB / R / G / B / A) replaces the old four
        # toggle row: click to cycle, dropdown arrow for direct pick.
        # A thin vertical line marks the boundary between the channel
        # *group* selector (RGB / albedo / normal …) and the channel
        # *mode* picker (RGB / R / G / B / A) — same group of pixels,
        # different question.
        buttons_layout.addWidget(self._transport.channel_button)
        # Pause / resume the alt-channel background prefetch — sits
        # immediately to the right of the channel selector since
        # "pause channels" is a verb on the same noun. Click toggles
        # ⏸ ↔ ▶; app.py forwards the signal to the player controller.
        buttons_layout.addWidget(self._transport.channel_cache_pause_button)
        ch_sep = QFrame(self)
        ch_sep.setFrameShape(QFrame.Shape.VLine)
        ch_sep.setFrameShadow(QFrame.Shadow.Plain)
        ch_sep.setStyleSheet(
            f"color: {H.BORDER_DEFAULT}; background: {H.BORDER_DEFAULT};",
        )
        ch_sep.setFixedWidth(1)
        ch_sep.setFixedHeight(G.BTN_TRANSPORT_H - 6)
        buttons_layout.addWidget(ch_sep)
        buttons_layout.addWidget(self._transport.channel_mode_button)
        # Transparency-background picker — sits next to the RGBA
        # toggles since they're all "what does the viewer paint?"
        # controls. Popup menu gives discoverable per-mode picks.
        # ``BG :`` prefix label mirrors the ``Zoom`` / ``FPS`` pairings
        # below so the button's identity is obvious at a glance — and
        # leaves the button face free to show the active mode as a
        # painted swatch. The label + button live in their own tight
        # sub-layout (no spacing) so they read as one control rather
        # than two widgets ``S.SM`` apart.
        bg_box = QWidget(self)
        bg_box.setStyleSheet("background: transparent;")
        bg_inner = QHBoxLayout(bg_box)
        # Left margin pushes the group away from the RGB channel
        # button on its left, ``setSpacing(2)`` glues the label to
        # the button so ``BG :`` reads as the prefix of the swatch
        # rather than as a free-standing piece of chrome.
        bg_inner.setContentsMargins(8, 0, 0, 0)
        bg_inner.setSpacing(0)
        bg_label = QLabel("BG :")
        bg_label.setStyleSheet(f"color: {H.TEXT_SECONDARY};")
        bg_inner.addWidget(bg_label)
        bg_inner.addWidget(self._transport.bg_button)
        # Vertical hairline between the channel-mode picker (R/G/B/A
        # selector — about the *data* being read) and the BG group
        # (about the *background* drawn under transparent pixels).
        # Same chrome as the channel-vs-mode separator above so the
        # menu bar reads as three grouped clusters.
        bgsep = QFrame(self)
        bgsep.setFrameShape(QFrame.Shape.VLine)
        bgsep.setFrameShadow(QFrame.Shadow.Plain)
        bgsep.setStyleSheet(
            f"color: {H.BORDER_DEFAULT}; background: {H.BORDER_DEFAULT};",
        )
        bgsep.setFixedWidth(1)
        bgsep.setFixedHeight(G.BTN_TRANSPORT_H - 6)
        buttons_layout.addWidget(bgsep)
        buttons_layout.addWidget(bg_box)
        # Zoom selector — preceded by a small "Zoom" label, mirroring
        # the "FPS" label/field pairing in the transport bar.
        zoom_label = QLabel("Zoom")
        zoom_label.setStyleSheet(f"color: {H.TEXT_SECONDARY};")
        buttons_layout.addWidget(zoom_label)
        buttons_layout.addWidget(self._transport.zoom_combo)
        # Burger last (closest to the right edge — easiest mouse
        # target).
        buttons_layout.addWidget(burger)

        # Compare band — its own widget, sandwiched between two
        # stretches in the top layout so it centres horizontally
        # between the menus and the buttons toolbar when visible.
        self._compare_band = CompareBand(self)
        self._compare_band.setVisible(False)

        # Contact-sheet band — same shape / placement as the compare
        # band, mutually exclusive with it (both hijack the GL upload
        # path, so the app forces one off when the other turns on).
        # Replaces the older transport-bar ``⋯`` kebab popup and the
        # View → Contact sheet settings sub-menu — settings now live
        # right next to the on/off toggle.
        self._contact_sheet_band = ContactSheetBand(self)
        self._contact_sheet_band.setVisible(False)

        top_bar = QWidget(self)
        top_bar.setStyleSheet("background: transparent;")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        top_layout.addWidget(menu_bar, 0)
        top_layout.addStretch(1)
        top_layout.addWidget(self._compare_band, 0)
        top_layout.addWidget(self._contact_sheet_band, 0)
        # Right stretch starts at 0 so the buttons toolbar sits flush
        # right while both bands are hidden. ``set_compare_band_visible``
        # / ``set_contact_sheet_band_visible`` flip it to 1 when their
        # band turns on, which centres the band between the two equal-
        # weight stretches.
        top_layout.addStretch(0)
        top_layout.addWidget(buttons_toolbar, 0)
        # Cache the layout + the right-stretch index so the visibility
        # toggle can update its weight without rebuilding anything.
        self._top_layout = top_layout
        # Layout indices: 0 = menu_bar, 1 = stretch_L, 2 = compare_band,
        # 3 = contact_sheet_band, 4 = stretch_R, 5 = buttons_toolbar.
        self._top_stretch_right_idx = 4
        # Replace the auto-installed menu bar slot with the composite.
        # ``setMenuWidget`` takes ownership; QMainWindow positions it
        # at the top of the window where the menu bar usually lives.
        self.setMenuWidget(top_bar)

    def _refresh_image_size_label(self) -> None:
        """Push image dimensions to the header info strip."""
        w, h = self._viewer.gl.image_size()
        self._viewer.header_strip.set_resolution(w, h)

    def set_ocio_reload_callback(self, cb) -> None:  # type: ignore[no-untyped-def]
        """Register the App's hot-reload entry point.

        Called by ``App`` after both ``App`` and ``MainWindow`` are
        constructed. The callback is invoked by
        :class:`PreferencesDialog` when the user applies a new OCIO
        config, allowing a hot-swap with no restart. ``None`` falls
        the dialog back to the legacy "Restart required" message.
        """
        self._ocio_reload_cb = cb

    def set_disk_cache_handle(self, disk_cache) -> None:  # type: ignore[no-untyped-def]
        """Register the live :class:`DiskCache` instance.

        Same pattern as :meth:`set_ocio_reload_callback`: ``App``
        passes the live handle here once construction is done, and
        :class:`PreferencesDialog` forwards it to the disk-cache page
        so the "clear / usage" actions can hit the running cache
        rather than just edit prefs for the next boot.
        """
        self._disk_cache_handle = disk_cache

    def _open_preferences(self) -> None:
        """File → Open Preferences… — application-wide settings dialog.

        Opens **non-modal** so the user keeps interacting with the
        viewer (scrub, layer toggles, etc.) while watching the live
        Disk-cache stats refresh. A single instance is reused: a
        second trigger raises/activates the existing window instead
        of stacking a new one. We hold the reference on ``self`` to
        keep the dialog alive (otherwise Python GC would close it
        immediately after :meth:`show`).
        """
        from PySide6.QtCore import Qt

        from img_player.preferences import Preferences
        from img_player.ui.preferences_dialog import PreferencesDialog

        existing = getattr(self, "_preferences_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return

        on_reload = getattr(self, "_ocio_reload_cb", None)
        disk_cache = getattr(self, "_disk_cache_handle", None)
        dialog = PreferencesDialog(
            Preferences(),
            on_reload=on_reload,
            disk_cache=disk_cache,
            parent=self,
        )
        # Non-modal: don't block the main window. WA_DeleteOnClose
        # plus the destroyed-signal handler keeps ``_preferences_dialog``
        # in sync so the next open re-creates a fresh dialog.
        dialog.setModal(False)
        dialog.setWindowModality(Qt.WindowModality.NonModal)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda _=None: setattr(self, "_preferences_dialog", None))
        self._preferences_dialog = dialog
        dialog.show()

    def _on_toggle_timecode(self, checked: bool) -> None:
        mode = "tc" if checked else "frames"
        # Both the timeline tick labels *and* the transport's
        # FrameDisplay follow the same toggle so the user can't end
        # up with mismatched units between the two readouts.
        self._timeline.set_display_mode(mode)
        self._transport.set_display_mode(mode)

    def _toggle_side_dock(self) -> None:
        """Show / hide the right-hand Color/Channels dock.

        Reclaims the whole window width for the viewer when the user
        wants more screen real estate. The burger button stays in the
        menu bar regardless, so the dock can always be brought back.
        """
        self._side_dock.setVisible(not self._side_dock.isVisible())

    def _show_shortcuts(self) -> None:
        """Open the shortcuts reference as a non-modal floating panel.

        Keeping it modeless lets the user actually try the shortcuts
        while reading the list — they can drag the timeline, toggle
        the pen, etc. without closing the help first. Reopening the
        menu while the panel is already up just raises the existing
        instance instead of stacking duplicates.
        """
        from img_player.ui.shortcuts_dialog import ShortcutsDialog

        existing = getattr(self, "_shortcuts_dialog", None)
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        # Hold the reference on ``self`` so Python doesn't collect
        # the dialog after this method returns (``show()`` is
        # non-blocking, unlike ``exec()``).
        dlg = ShortcutsDialog(self)
        dlg.setModal(False)
        dlg.setWindowFlag(Qt.WindowType.Tool, True)
        dlg.show()
        self._shortcuts_dialog = dlg

    def install_recent_provider(
        self,
        provider: Callable[[], list[Path]],
        clear_callback: Callable[[], None],
    ) -> None:
        """Let the app inject the functions that list / clear recent paths."""
        self._recent_paths_provider = provider
        self._clear_recent_callback = clear_callback
        self._refresh_recent_menu()

    # NB: ``_build_contact_sheet_settings_menu`` used to live here and
    # rebuilt the View → Contact sheet settings sub-menu on each
    # ``aboutToShow``. It was removed when the settings moved to a
    # dedicated :class:`ContactSheetBand` toolbar (see
    # ``self._contact_sheet_band``) that appears above the viewer
    # while contact-sheet mode is on — same UX as the compare band.
    # The QSignals (`contact_sheet_grid_changed`, `_labels_toggled`,
    # `_divisor_changed`) are unchanged; only the widget that emits
    # them has been swapped.

    def _refresh_recent_menu(self) -> None:
        self._recent_menu.clear()
        paths = list(self._recent_paths_provider())
        if not paths:
            empty = QAction("(no recent sequences)", self)
            empty.setEnabled(False)
            self._recent_menu.addAction(empty)
            return
        for path in paths:
            act = QAction(self._shorten_path_label(path), self)
            act.setToolTip(str(path))
            act.triggered.connect(lambda _=False, p=path: self.open_requested.emit([p]))
            self._recent_menu.addAction(act)
        self._recent_menu.addSeparator()
        clear = QAction("Clear list", self)
        clear.triggered.connect(self._on_clear_recent_clicked)
        self._recent_menu.addAction(clear)

    def _on_clear_recent_clicked(self) -> None:
        self._clear_recent_callback()
        self._refresh_recent_menu()

    # --- Recent sessions (mirror of the sequence-side helpers) -----

    def install_recent_session_provider(
        self,
        provider: Callable[[], list[Path]],
        clear_callback: Callable[[], None],
    ) -> None:
        """Same contract as :meth:`install_recent_provider` but for
        the ``.session`` recent list. Kept as a separate hook so the
        app can wire each list to its own preferences key without
        risk of cross-pollination."""
        self._recent_sessions_provider = provider
        self._clear_recent_sessions_callback = clear_callback
        self._refresh_recent_sessions_menu()

    def _refresh_recent_sessions_menu(self) -> None:
        self._recent_sessions_menu.clear()
        paths = list(self._recent_sessions_provider())
        if not paths:
            empty = QAction("(no recent sessions)", self)
            empty.setEnabled(False)
            self._recent_sessions_menu.addAction(empty)
            return
        for path in paths:
            act = QAction(self._shorten_path_label(path), self)
            act.setToolTip(str(path))
            # Recent click → open the session file. Routes through
            # the same signal as the File → Open Session… dialog.
            act.triggered.connect(
                lambda _=False, p=path: self.open_session_requested.emit(p),
            )
            self._recent_sessions_menu.addAction(act)
        self._recent_sessions_menu.addSeparator()
        clear = QAction("Clear list", self)
        clear.triggered.connect(self._on_clear_recent_sessions_clicked)
        self._recent_sessions_menu.addAction(clear)

    def _on_clear_recent_sessions_clicked(self) -> None:
        self._clear_recent_sessions_callback()
        self._refresh_recent_sessions_menu()

    @staticmethod
    def _shorten_path_label(path: Path, max_len: int = 60) -> str:
        s = str(path)
        if len(s) <= max_len:
            return s
        # Keep the beginning + end, shrink the middle.
        head = s[:20]
        tail = s[-(max_len - 23) :]
        return f"{head}…{tail}"

    def _install_shortcuts(self) -> None:
        # Classic VFX shuttle: J reverse, K pause, L forward
        QShortcut(
            QKeySequence(Qt.Key.Key_J),
            self,
            activated=lambda: self.direction_play_requested.emit(-1),
        )
        QShortcut(QKeySequence(Qt.Key.Key_K), self, activated=self.play_toggled.emit)
        QShortcut(
            QKeySequence(Qt.Key.Key_L),
            self,
            activated=lambda: self.direction_play_requested.emit(1),
        )
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.play_toggled.emit)

        # Frame stepping
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self.step_clicked.emit(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self.step_clicked.emit(1))
        QShortcut(QKeySequence("Shift+Left"), self, activated=lambda: self.step_clicked.emit(-10))
        QShortcut(QKeySequence("Shift+Right"), self, activated=lambda: self.step_clicked.emit(10))
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, activated=lambda: self.jump_to_ends.emit(-1))
        QShortcut(QKeySequence(Qt.Key.Key_End), self, activated=lambda: self.jump_to_ends.emit(1))

        # Compare mode (v1.2). W toggles on/off; Ctrl+W swaps A↔B in
        # the band's dropdowns. The arrow-key seam nudges live on
        # the viewer's keyPressEvent so they only fire when the
        # viewer area has focus (= avoid colliding with the
        # frame-stepping arrows above when the user is using
        # transport).
        QShortcut(
            QKeySequence(Qt.Key.Key_W), self,
            activated=self.compare_toggle_requested.emit,
        )
        QShortcut(
            QKeySequence("Ctrl+W"), self,
            activated=self.compare_swap_layers_requested.emit,
        )

        # In / out points
        QShortcut(QKeySequence(Qt.Key.Key_I), self, activated=self.mark_in_requested.emit)
        QShortcut(QKeySequence(Qt.Key.Key_O), self, activated=self.mark_out_requested.emit)
        QShortcut(QKeySequence("Shift+R"), self, activated=self.clear_in_out_requested.emit)

        # Exposure nudges — these map to the color panel's spin box
        QShortcut(
            QKeySequence(Qt.Key.Key_Plus), self, activated=lambda: self.exposure_step.emit(0.25)
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Minus), self, activated=lambda: self.exposure_step.emit(-0.25)
        )
        QShortcut(
            QKeySequence(Qt.Key.Key_Equal), self, activated=lambda: self.exposure_step.emit(0.25)
        )

        # Fullscreen — F to toggle, Esc to exit (Esc only meaningful
        # while *in* fullscreen; the handler short-circuits otherwise
        # so it doesn't steal the key from other widgets).
        QShortcut(QKeySequence(Qt.Key.Key_F), self, activated=self.toggle_fullscreen)
        QShortcut(
            QKeySequence(Qt.Key.Key_Escape), self,
            activated=lambda: self.exit_fullscreen() if self._fullscreen else None,
        )

    def _wire_internal(self) -> None:
        # play_toggled is reserved for direction-agnostic toggles
        # (Space / K shortcuts). The direction-aware play buttons of
        # the transport bar route through direction_play_requested
        # so the controller can decide between start / flip / pause.
        self._transport.play_toggled.connect(self.play_toggled.emit)
        self._transport.forward_play_clicked.connect(
            lambda: self.direction_play_requested.emit(1)
        )
        self._transport.reverse_play_clicked.connect(
            lambda: self.direction_play_requested.emit(-1)
        )
        self._transport.step_clicked.connect(self.step_clicked.emit)
        self._transport.jump_to_ends.connect(self.jump_to_ends.emit)
        self._transport.fps_changed.connect(self.fps_changed.emit)
        self._transport.mark_in_clicked.connect(self.mark_in_requested.emit)
        self._transport.mark_out_clicked.connect(self.mark_out_requested.emit)
        self._transport.clear_in_out_clicked.connect(self.clear_in_out_requested.emit)
        self._transport.loop_mode_requested.connect(self.loop_mode_requested.emit)
        self._transport.fullscreen_clicked.connect(self.toggle_fullscreen)
        # Frame display: typing a frame number / TC and pressing Enter
        # asks the controller to seek there.
        self._transport.frame_seek_requested.connect(self.frame_requested.emit)
        self._transport.channel_selection_changed.connect(
            self.channel_selection_changed.emit
        )
        self._transport.channel_mask_changed.connect(self.channel_mask_changed.emit)
        self._transport.channel_cache_pause_toggled.connect(
            self.channel_cache_pause_toggled.emit,
        )
        self._transport.transparency_bg_mode_changed.connect(
            self.transparency_bg_mode_changed.emit,
        )
        self._transport.master_volume_changed.connect(
            self.master_volume_changed.emit,
        )
        # Zoom: combo → viewport (forward), wheel → combo (back-channel
        # so the displayed value follows the wheel without us
        # re-emitting and ping-ponging).
        self._transport.zoom_requested.connect(self.zoom_requested.emit)
        self._viewer.gl.zoom_changed.connect(self._transport.set_zoom_display)
        self._timeline.frame_requested.connect(self.frame_requested.emit)
        self._timeline.scrub_started.connect(self.scrub_started.emit)
        self._timeline.scrub_finished.connect(self.scrub_finished.emit)
        # Ctrl-click drag → forward the in/out frame request.
        self._timeline.set_in_at_requested.connect(self.set_in_at_requested.emit)
        self._timeline.set_out_at_requested.connect(self.set_out_at_requested.emit)
        # Drag-scrub inside the image viewport routes through the same
        # frame_requested → app._on_scrub_requested pipeline as the
        # timeline scrubber. From the controller's point of view the
        # two sources are indistinguishable.
        self._viewer.gl.frame_requested.connect(self.frame_requested.emit)
        self._viewer.gl.scrub_started.connect(self.scrub_started.emit)
        self._viewer.gl.scrub_finished.connect(self.scrub_finished.emit)
        # Image dimensions readout in the menu corner — refreshed on
        # every transform change (which fires on size change AND on
        # zoom/pan; the dedicated update is idempotent so zoom-only
        # fires are essentially free).
        self._viewer.gl.transform_changed.connect(self._refresh_image_size_label)
        self._refresh_image_size_label()
        # Per-zone drag-and-drop. Drops on the viewer go through the
        # standard "Open" pipeline (replace), drops on the master
        # timeline composite append a new layer.
        self._viewer.replace_requested.connect(self.open_requested.emit)
        if self._master_timeline_panel is not None:
            self._master_timeline_panel.add_layer_requested.connect(
                self.add_layer_requested.emit
            )

    # --------------------------------------------------------------- Menu handlers

    def _on_open_action(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open an image or a frame of a sequence",
            "",
            "Images (*.exr *.dpx *.tif *.tiff *.png *.jpg *.jpeg *.tga);;All files (*.*)",
        )
        if path_str:
            self.open_requested.emit([Path(path_str)])

    def _on_add_layer_action(self) -> None:
        """File → Add layer… opens a folder picker; the chosen folder
        is added to the LayerStack as a new top layer."""
        path_str = QFileDialog.getExistingDirectory(
            self,
            "Pick a folder to add as a new layer",
            "",
        )
        if path_str:
            self.add_layer_requested.emit([Path(path_str)])

    def _on_open_session_action(self) -> None:
        """File → Open session…"""
        path_str, _ = QFileDialog.getOpenFileName(
            self, "Open a saved session", "",
            "Session files (*.session);;All files (*.*)",
        )
        if path_str:
            self.open_session_requested.emit(Path(path_str))

    def _on_save_session_action(self) -> None:
        """File → Save Session (Ctrl+S).

        Silent overwrite of the current session file. When no current
        path is known (= the user hasn't opened or save-as'd yet)
        we transparently delegate to Save As so first save still
        produces a usable file rather than crashing or silently
        no-op'ing.
        """
        if self._current_session_path is not None:
            self.save_session_requested.emit(self._current_session_path)
            return
        # No current path → behave like Save As.
        self._on_save_session_as_action()

    def _on_save_session_as_action(self) -> None:
        """File → Save Session As… (Ctrl+Shift+S).

        Always opens the file picker. The chosen path is emitted via
        ``save_session_requested``; the App handler is responsible
        for calling :meth:`set_current_session_path` after a
        successful write so subsequent Ctrl+S targets the same file.
        """
        # Pre-fill the dialog with the current path when known so
        # "Save As" defaults to the same folder + a sensible filename
        # the user can edit (= classic "duplicate this project"
        # workflow).
        initial = (
            str(self._current_session_path)
            if self._current_session_path is not None
            else ""
        )
        path_str, _ = QFileDialog.getSaveFileName(
            self, "Save the current session", initial,
            "Session files (*.session)",
        )
        if path_str:
            self.save_session_requested.emit(Path(path_str))

    def set_current_session_path(self, path: Path | None) -> None:
        """Update the path Ctrl+S targets.

        Called by the App after a successful Open Session, Save
        Session, or Save As — and with ``None`` after File → New so
        a stale pointer doesn't get silently overwritten on the
        next Ctrl+S. Also reflects the file name in the window
        title so the user always knows which session they're
        working in.
        """
        self._current_session_path = path
        if path is not None:
            self.setWindowTitle(f"Flick Player — {path.name}")
        else:
            self.setWindowTitle("Flick Player")

    def _silently_check_burnin_slug(self, slug: str) -> None:
        """Make ``slug`` the single checked entry in the burnin
        template submenu **without** re-emitting
        ``burnin_template_requested`` (which would boomerang back to
        the App and rewrite prefs).

        Key insight: QActionGroup's exclusivity is driven by the
        action's ``changed`` signal (its private
        ``_q_actionChanged`` slot), NOT by ``triggered``. And
        ``setChecked(b)`` emits ``toggled`` + ``changed`` but never
        ``triggered``. Since our slug-pick lambda is wired to
        ``triggered`` only, calling ``setChecked(True)`` plainly
        does two useful things at once:

        * Fires ``changed`` → QActionGroup updates its internal
          ``current`` pointer to the new target AND auto-unchecks
          the previously-current sibling. Future user clicks then
          uncheck the right thing.
        * Leaves ``triggered`` silent → our lambda doesn't fire,
          no boomerang re-emit.

        The earlier "wrap in ``blockSignals``" was over-defensive
        and actively broke exclusivity by hiding ``changed`` from
        the group too. Result was the two-tick bug:
        ``default`` silently checked at boot, group's
        ``current`` stuck at ``None``, then the user's later click
        on a new preset added a second tick without unchecking
        the first.

        We still defensively uncheck stale siblings up front (with
        their own signals blocked — no need to ping the group
        twice per call) in case a prior bad state left more than
        one row ticked.
        """
        actions = getattr(self, "_burnin_template_actions", {}) or {}
        target = actions.get(slug)
        # Uncheck any stale siblings. Block signals on these so
        # the group's per-action ``changed`` handler doesn't run
        # repeatedly — we only want it to fire once, on the
        # final setChecked(True) below, to point ``current`` at
        # the new target.
        for other_slug, other_act in actions.items():
            if other_slug == slug:
                continue
            if other_act.isChecked():
                other_act.blockSignals(True)
                try:
                    other_act.setChecked(False)
                finally:
                    other_act.blockSignals(False)
        if target is not None and not target.isChecked():
            # NO blockSignals here — we WANT ``changed`` to fire so
            # QActionGroup updates its ``current`` tracker. See
            # docstring above. ``triggered`` does not fire from
            # ``setChecked``, so our lambda stays quiet.
            target.setChecked(True)

    def refresh_burnin_template_menu(
        self, slugs: list[str], active_slug: str,
    ) -> None:
        """Repopulate the View → Active burnin template submenu from a
        fresh slugs list. Called by the App after the editor saves /
        deletes a user template so the menu reflects what's on disk.

        Preserves the action-group exclusivity and re-checks the
        ``active_slug`` entry without re-emitting the
        ``burnin_template_requested`` signal."""
        if not hasattr(self, "_burnin_template_menu"):
            return
        # Tear down previous actions.
        for act in list(self._burnin_template_actions.values()):
            self._burnin_template_group.removeAction(act)
            self._burnin_template_menu.removeAction(act)
        self._burnin_template_actions.clear()
        # Rebuild.
        for slug in slugs:
            label = slug.replace("_", " ")
            act = QAction(label, self, checkable=True)
            act.setData(slug)
            self._burnin_template_group.addAction(act)
            self._burnin_template_menu.addAction(act)
            self._burnin_template_actions[slug] = act
            act.triggered.connect(
                lambda _checked=False, s=slug: (
                    self.burnin_template_requested.emit(s)
                ),
            )
        # All new actions are unchecked; flip the active one (and
        # only the active one) via the silent-set helper. We can't
        # rely on the action group's exclusivity here because the
        # caller doesn't want to re-emit ``burnin_template_requested``
        # — see :meth:`_silently_check_burnin_slug` for details.
        self._silently_check_burnin_slug(active_slug)

    def set_burnin_menu_state(self, enabled: bool, slug: str) -> None:
        """Sync the View → Show burnins toggle + Active template
        radios from the App's authoritative state. Called once after
        the app loads the prefs at boot so the menu reflects what the
        overlay is actually painting, without re-firing the toggle /
        pick signals (which would write the same value back to prefs).
        """
        if hasattr(self, "_show_burnins_act"):
            self._show_burnins_act.blockSignals(True)
            try:
                self._show_burnins_act.setChecked(bool(enabled))
            finally:
                self._show_burnins_act.blockSignals(False)
        # Use the silent helper rather than ``blockSignals +
        # setChecked`` — see :meth:`_silently_check_burnin_slug` for
        # why blocking signals on a checkable action defeats
        # QActionGroup's exclusivity and leaves two rows ticked.
        self._silently_check_burnin_slug(slug)

    def _show_about(self) -> None:
        # Pull the version from the package's ``__version__`` rather
        # than hard-coding it here, so a release bump in
        # ``pyproject.toml`` + ``__init__.py`` is the single source of
        # truth — same string the splash and the ``--version`` CLI
        # already use.
        from img_player import __version__  # noqa: PLC0415
        QMessageBox.about(
            self,
            "About Flick Player",
            (
                f"<b>Flick Player</b> &mdash; v{__version__}<br>"
                "VFX-grade image sequence player.<br><br>"
                "OCIO color management, async RAM cache, OpenGL viewport."
            ),
        )

    # --------------------------------------------------------------- Drag & drop

    # NB: drag-and-drop is now handled per-zone (viewer = "Replace",
    # master timeline = "Add to layers"). Each zone shows its own
    # overlay during drag-over. We deliberately don't accept drops
    # at MainWindow level anymore — a global handler used to swallow
    # the drop and route through a modal "Add / Replace / Cancel"
    # dialog, which the per-zone scheme replaces.

    # --------------------------------------------------------------- Fullscreen

    # Pixel distance from the bottom edge that activates the
    # auto-show floating bar. Anything closer than this triggers
    # show; anything further triggers a queued hide.
    _FS_HOT_ZONE_PX = 80
    # Delay before hiding once the cursor leaves the hot zone — keeps
    # the bar from flickering when the user briefly moves up to grab
    # the timeline scrubber.
    _FS_HIDE_DELAY_MS = 400

    def toggle_fullscreen(self) -> None:
        """Flip in or out of fullscreen mode."""
        if self._fullscreen:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()

    def enter_fullscreen(self) -> None:
        """Switch to fullscreen: hide every chrome panel, show the
        viewer maximised, and prepare a floating bottom bar that
        auto-reveals on cursor proximity."""
        if self._fullscreen:
            return
        self._fullscreen = True

        # Hide every chrome surface — the viewer is the only thing
        # the user sees outside the auto-show bar. ``menuWidget()``
        # rather than ``menuBar()``: we replaced the menu slot with a
        # custom QWidget (top_bar built in ``_build_menu``); calling
        # ``menuBar()`` here would auto-create a fresh empty QMenuBar
        # AND install it in the slot, evicting top_bar permanently.
        self.menuWidget().hide()
        self._transport.hide()
        if self._master_timeline_panel is not None:
            self._master_timeline_panel.hide()
        self._side_dock.hide()
        # Remember the annotation dock's pre-fullscreen visibility so
        # ``exit_fullscreen`` can restore it. Without this snapshot
        # the dock stays hidden after the fullscreen round-trip,
        # silently dropping the user's annotation toolbar.
        self._fs_annotation_dock_was_visible = self._annotation_dock.isVisible()
        self._annotation_dock.hide()
        if self.statusBar() is not None:
            self.statusBar().hide()

        self._build_fullscreen_bar_if_needed()
        # Switch the timeline into its stripped-down rendering: just
        # a track + playhead, no ticks / cache bar / annotation
        # markers. Reviewer feedback ("on préfèrera une time line
        # épurée et simple" in fullscreen) — the chrome reads as
        # noise on top of full-frame images.
        self._timeline.set_minimal_mode(True)
        # Reparent the frame display + timeline into the floating bar
        # so the user keeps the frame readout next to the scrubber in
        # fullscreen (it normally lives in the master panel's left
        # gutter, which is hidden along with the rest of the chrome).
        # Order: [frame_display] [timeline (stretch=1)] [exit_btn].
        if self._master_timeline_panel is not None:
            frame_display = self._transport.frame_display
            self._fs_bar_layout.insertWidget(0, frame_display)
            self._fs_bar_layout.insertWidget(1, self._timeline, 1)
        self._fs_bar.hide()  # will appear on cursor proximity
        self._position_fs_bar()

        # Start polling the cursor position. Polling instead of event
        # filtering because the GL child widget swallows mouse events
        # before they reach a filter on its parent — leaving the bar
        # stuck closed even when the user is right at the bottom.
        if self._fs_cursor_timer is None:
            self._fs_cursor_timer = QTimer(self)
            self._fs_cursor_timer.setInterval(50)
            self._fs_cursor_timer.timeout.connect(self._fs_poll_cursor)
        self._fs_cursor_timer.start()

        self.showFullScreen()
        self._transport.set_fullscreen_state(True)

    def exit_fullscreen(self) -> None:
        """Restore the normal window chrome."""
        if not self._fullscreen:
            return
        self._fullscreen = False

        # Reparent the timeline + frame display back into the master
        # panel. The composite exposes ``_axis_gutter_layout`` for the
        # frame display and the timeline goes back into the axis row
        # at the end (after the gutter widget).
        if self._master_timeline_panel is not None and self._timeline is not None:
            mtp = self._master_timeline_panel
            mtp._axis_row.layout().addWidget(self._timeline, 1)
            # Restore the frame display inside the gutter, sandwiched
            # between two stretches (centred — same as initial build).
            frame_display = self._transport.frame_display
            mtp._axis_gutter_layout.insertWidget(1, frame_display)
            self._master_timeline_panel.show()

        # ``menuWidget()`` rather than ``menuBar()`` for the same
        # reason as in ``enter_fullscreen`` — we set top_bar via
        # ``setMenuWidget``, and ``menuBar()`` would replace it.
        self.menuWidget().show()
        self._transport.show()
        self._side_dock.show()
        # Restore the annotation dock if it was visible before
        # going fullscreen. The snapshot is taken in
        # ``enter_fullscreen``; defaults to False on the very first
        # exit of an already-fullscreen window (= ``False`` matches
        # the dock's own startup default).
        if getattr(self, "_fs_annotation_dock_was_visible", False):
            self._annotation_dock.show()
        if self.statusBar() is not None:
            self.statusBar().show()

        if self._fs_bar is not None:
            self._fs_bar.hide()
        if self._fs_cursor_timer is not None:
            self._fs_cursor_timer.stop()
        if self._fs_hide_timer is not None:
            self._fs_hide_timer.stop()

        # Restore the timeline to its full-featured rendering.
        self._timeline.set_minimal_mode(False)

        self.showNormal()
        self._transport.set_fullscreen_state(False)

    def _fs_poll_cursor(self) -> None:
        """Poll ``QCursor.pos()`` every 50 ms while in fullscreen and
        show/hide the bottom bar based on cursor proximity to the
        bottom edge. Polling instead of event-filtering bypasses the
        GL viewport child swallowing mouse moves."""
        if not self._fullscreen or self._fs_bar is None:
            return
        from PySide6.QtGui import QCursor
        global_pos = QCursor.pos()
        local_pos = self.mapFromGlobal(global_pos)
        # Only react when the cursor is actually over the window.
        if not self.rect().contains(local_pos):
            return
        distance_from_bottom = self.height() - local_pos.y()
        if distance_from_bottom <= self._FS_HOT_ZONE_PX:
            if not self._fs_bar.isVisible():
                self._position_fs_bar()
                self._fs_bar.show()
            self._fs_hide_timer.stop()
        else:
            if self._fs_bar.isVisible() and not self._fs_hide_timer.isActive():
                self._fs_hide_timer.start()

    def _build_fullscreen_bar_if_needed(self) -> None:
        if self._fs_bar is not None:
            return
        bar = QFrame(self)
        bar.setObjectName("fullscreen_bar")
        bar.setStyleSheet(
            "QFrame#fullscreen_bar {"
            "  background: rgba(14, 15, 18, 220);"
            "  border-top: 1px solid rgba(255, 255, 255, 30);"
            "}"
        )
        bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        bar.setFixedHeight(64)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(S.SM, S.SM, S.SM, S.SM)
        layout.setSpacing(S.SM)

        # The timeline is reparented in here on ``enter_fullscreen``
        # at index 0 with stretch=1 so it fills the bar's width. The
        # annotation toggle + exit button below sit to its right at
        # fixed size.

        # Annotation toggle — same effect as the transport bar's
        # ✏️ button (route through the existing
        # ``annotation_toggle_clicked`` signal so app.py's wiring
        # toggles the toolbar visibility uniformly). Checkable so
        # the down-state mirrors the toolbar's open / closed state.
        annot_btn = QPushButton("✏️")
        annot_btn.setFixedSize(G.BTN_TRANSPORT_W, G.BTN_TRANSPORT_H)
        annot_btn.setToolTip("Afficher / masquer la toolbar d'annotation (D)")
        annot_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        annot_btn.setCheckable(True)
        annot_btn.clicked.connect(self._transport.annotation_toggle_clicked.emit)
        layout.addWidget(annot_btn)

        exit_btn = QPushButton()
        exit_btn.setFixedSize(G.BTN_TRANSPORT_W, G.BTN_TRANSPORT_H)
        exit_btn.setIcon(make_icon("fullscreen_exit"))
        exit_btn.setIconSize(QSize(16, 16))
        exit_btn.setToolTip("Exit fullscreen (F / Esc)")
        exit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        exit_btn.clicked.connect(self.exit_fullscreen)
        layout.addWidget(exit_btn)

        self._fs_bar = bar
        self._fs_bar_layout = layout
        self._fs_annot_btn = annot_btn
        self._fs_exit_btn = exit_btn
        # Sync the fs annotation button to the toolbar's current
        # visibility on entry — without this the button's checked
        # state would lag behind reality (= shows "off" while the
        # toolbar is still visible from before fullscreen entry).
        self._sync_fs_annotation_button()

        # Hide-with-delay timer: armed when the cursor leaves the hot
        # zone, fires after ``_FS_HIDE_DELAY_MS``, hides the bar.
        self._fs_hide_timer = QTimer(self)
        self._fs_hide_timer.setSingleShot(True)
        self._fs_hide_timer.setInterval(self._FS_HIDE_DELAY_MS)
        self._fs_hide_timer.timeout.connect(
            lambda: self._fs_bar and self._fs_bar.hide(),
        )

    def set_fs_annotation_toggle_active(self, active: bool) -> None:
        """Sync the fullscreen bar's annotation toggle checked state.

        Called by ``app._toggle_toolbar_visible`` so the fs button
        and the transport's ✏ button stay in lock-step regardless
        of whether the user toggled via D, the transport, the fs
        bar, or the toolbar's own ✕ close button. No-op when the
        fs bar hasn't been built yet (= we've never entered
        fullscreen this session).
        """
        btn = getattr(self, "_fs_annot_btn", None)
        if btn is None:
            return
        if btn.isChecked() == bool(active):
            return
        btn.blockSignals(True)
        try:
            btn.setChecked(bool(active))
        finally:
            btn.blockSignals(False)

    def _sync_fs_annotation_button(self) -> None:
        """Match the fs annotation button to the live toolbar state.
        Used on fs-bar build so the initial check reflects reality."""
        # Late import to avoid a circular import via app.py.
        # ``_annotation_toolbar`` doesn't live on MainWindow; the
        # transport's button mirrors it though, so we read from there.
        transport = getattr(self, "_transport", None)
        btn = getattr(self, "_fs_annot_btn", None)
        if transport is None or btn is None:
            return
        try:
            active = transport.is_annotation_toggle_active()
        except Exception:
            active = False
        btn.blockSignals(True)
        try:
            btn.setChecked(bool(active))
        finally:
            btn.blockSignals(False)

    def _position_fs_bar(self) -> None:
        """Resize the floating bar to span the window's bottom."""
        if self._fs_bar is None:
            return
        h = self._fs_bar.height()
        self._fs_bar.setGeometry(0, self.height() - h, self.width(), h)
        self._fs_bar.raise_()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        # Keep the floating bar pinned to the bottom on window resize
        # in fullscreen (Qt fires ``resize`` when entering / leaving
        # fullscreen too, so this also covers the initial sizing).
        if self._fullscreen:
            self._position_fs_bar()
        # Full-window session-drop overlay tracks the window rect so
        # ``OPEN SESSION`` always covers the entire surface during a
        # drag-over. Cheap — only sets geometry when the overlay is
        # currently visible.
        if (
            getattr(self, "_session_drop_overlay", None) is not None
            and self._session_drop_overlay.isVisible()
        ):
            self._session_drop_overlay.setGeometry(self.rect())
        # The compare band lives in a sibling toolbar inside a plain
        # QHBoxLayout (built in ``_build_menu``), so no manual reflow
        # is needed on resize — Qt handles the squeeze automatically.

    def closeEvent(self, event: QCloseEvent) -> None:
        # The app can register a callback that runs before the window
        # actually closes — used for the annotation save prompt.
        # Returning ``False`` from the callback cancels the close
        # (e.g. user clicked "Annuler" in the save dialog).
        if self._before_close_callback is not None:
            try:
                allow_close = bool(self._before_close_callback())
            except Exception:  # pragma: no cover — defensive
                log.exception("MainWindow before-close callback raised")
                allow_close = True
            if not allow_close:
                event.ignore()
                return
        log.info("MainWindow closing")
        super().closeEvent(event)

    def set_before_close_callback(
        self, callback: Callable[[], bool] | None
    ) -> None:
        """Register a function called from :meth:`closeEvent` before
        the window actually closes. The callback returns ``True`` to
        allow the close, ``False`` to cancel it (the close event is
        ``ignore()``-d and the window stays open).

        Used by ``app.py`` to prompt the user about saving annotations
        if the in-memory state is dirty.
        """
        self._before_close_callback = callback
