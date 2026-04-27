"""MainWindow: assembles the viewer, transport, timeline and side panels.

Signals from controls are routed to the :class:`PlayerController` and
:class:`GLViewport` by the app module; this widget only owns the UI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
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
from img_player.ui.icons import make_icon
from img_player.ui.theme import F, G, H, S
from img_player.ui.timeline import Timeline
from img_player.ui.transport import TransportBar
from img_player.ui.viewer_widget import ViewerWidget

if TYPE_CHECKING:
    from img_player.sequence.models import SequenceInfo

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):  # type: ignore[misc]
    """Top-level window wiring all the UI pieces together."""

    open_requested = Signal(Path)
    play_toggled = Signal()
    channels_requested = Signal(object)   # list[str] | None
    channel_mask_changed = Signal(tuple)  # (R, G, B, A) bools
    zoom_requested = Signal(object)       # float | None ; None = fit
    step_clicked = Signal(int)  # +1 / -1
    jump_to_ends = Signal(int)  # -1 first, +1 last
    frame_requested = Signal(int)
    exposure_step = Signal(float)  # +/- keyboard adjustment
    fps_changed = Signal(float)
    direction_play_requested = Signal(int)  # +1 forward, -1 reverse (J/L)
    mark_in_requested = Signal()  # set in-point at current frame (I)
    mark_out_requested = Signal()  # set out-point at current frame (O)
    clear_in_out_requested = Signal()  # reset in/out range (Shift+R)
    loop_mode_requested = Signal(object)  # LoopMode

    def __init__(
        self,
        ocio_manager: OCIOManager,
        comment_store: CommentStore,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("img_player")
        self.resize(1280, 720)
        self.setAcceptDrops(True)

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

        # Central: viewer on top, then timeline + transport stacked at the bottom
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(S.SM, S.SM, S.SM, S.SM)
        layout.setSpacing(S.SM)
        layout.addWidget(self._viewer, stretch=1)
        layout.addWidget(self._timeline)
        layout.addWidget(self._transport)
        self.setCentralWidget(central)

        # Right-hand dock holds Color + Comments as tabs.
        self._side_tabs = QTabWidget()
        self._side_tabs.addTab(self._color_panel, "Color")
        self._side_tabs.addTab(self._comment_panel, "Comments")
        # Stored on self so the burger button can toggle it.
        # objectName is mandatory for QMainWindow.saveState() /
        # restoreState() to round-trip the dock layout in QSettings —
        # without it Qt logs a warning and silently drops the state.
        self._side_dock = QDockWidget("Panels", self)
        self._side_dock.setObjectName("side_dock")
        self._side_dock.setWidget(self._side_tabs)
        self._side_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._side_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._side_dock)

        # Empty dock placeholder for the annotation toolbar's "dock"
        # mode. It starts hidden — `App` populates it with the
        # AnnotationToolbar widget when the user picks dock mode. By
        # being a real QDockWidget with an objectName, it participates
        # in saveState / restoreState so its position stays put across
        # sessions.
        self._annotation_dock = QDockWidget("Annotations", self)
        self._annotation_dock.setObjectName("annotation_dock")
        self._annotation_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._annotation_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        # Anchored on the LEFT side of the window — keeps the right
        # side free for the Color / Channels / future Comment panels,
        # and matches the user's preferred review-tool layout (drawing
        # tools on the left like Photoshop / Procreate).
        self.addDockWidget(
            Qt.DockWidgetArea.LeftDockWidgetArea, self._annotation_dock
        )
        self._annotation_dock.hide()  # only shown when toolbar is in dock mode

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
        """Two-block status bar: contextual message left, perf indicators right.

        Replaces the legacy single ``showMessage()`` line so we can render
        coloured dots (rich text) on the right while keeping a plain text
        message on the left. ``set_status()`` keeps its old contract for
        existing callers — it just routes to the left label now.
        """
        bar = self.statusBar()

        self.status_left = QLabel("Ready — drop a sequence (folder or file) to start.")
        self.status_left.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.status_left.setStyleSheet(
            f"color: {H.TEXT_SECONDARY}; font-size: {F.SIZE_XS}px;"
        )

        self.status_right = QLabel()
        self.status_right.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        self.status_right.setTextFormat(Qt.TextFormat.RichText)
        self.status_right.setFont(F.mono(F.SIZE_XS))

        bar.addWidget(self.status_left, 1)        # stretch fills the gap
        bar.addPermanentWidget(self.status_right) # right-anchored

    # --------------------------------------------------------------- Accessors

    @property
    def viewer(self) -> ViewerWidget:
        return self._viewer

    @property
    def color_panel(self) -> ColorPanel:
        return self._color_panel

    @property
    def annotation_dock(self) -> QDockWidget:
        """The empty placeholder dock the AnnotationToolbar reparents
        into when the user picks dock mode. Owned by MainWindow so its
        position participates in saveState / restoreState."""
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
        self.setWindowTitle(f"img_player — {sequence.display_pattern()}")
        self._timeline.set_range(sequence.first_frame, sequence.last_frame)
        # Clear the cache bar so we don't briefly show the old run
        # rectangles re-mapped onto the new range. The next
        # _refresh_cache_bar tick (~200 ms) re-populates with the
        # actually-cached frames of the new sequence.
        self._timeline.set_cached_frames(frozenset())
        # Populate the channel-selector combo so the user can pick
        # individual channels (Z, normals, AOVs, …).
        self._transport.set_available_channels(sequence.channel_names)

    def set_status(self, message: str) -> None:
        """Set the contextual message on the *left* side of the status bar.

        Kept as a method for backwards compat with all the existing call
        sites (``Loaded …``, ``In point set to frame …``, etc.).
        """
        self.status_left.setText(message)

    # --------------------------------------------------------------- Menu / shortcuts

    def _build_menu(self) -> None:
        # Recent-paths callback provider. The app module installs real
        # callbacks via install_recent_provider(); we seed empty defaults
        # so the menu can render before that happens.
        self._recent_paths_provider: Callable[[], list[Path]] = lambda: []
        self._clear_recent_callback: Callable[[], None] = lambda: None

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_act = QAction("&Open…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_action)
        file_menu.addAction(open_act)

        self._recent_menu = file_menu.addMenu("Open &Recent")
        self._recent_menu.aboutToShow.connect(self._refresh_recent_menu)
        # Pre-populate so the submenu is never empty on first open.
        self._refresh_recent_menu()

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # --- View menu : timeline display mode ----------------------------
        view_menu = menu_bar.addMenu("&View")
        self._show_tc_act = QAction("Show &timecode", self, checkable=True)
        self._show_tc_act.setShortcut(QKeySequence("Ctrl+T"))
        self._show_tc_act.triggered.connect(self._on_toggle_timecode)
        view_menu.addAction(self._show_tc_act)

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
        burger_size = 18
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
        burger.setFixedSize(36, 24)
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

        # The burger is wrapped in a transparent widget so we can give
        # it a tiny right margin — `setCornerWidget` would otherwise
        # paste it flush against the window edge.
        burger_wrapper = QWidget(self)
        burger_wrapper.setStyleSheet("background: transparent;")
        wrap_layout = QHBoxLayout(burger_wrapper)
        wrap_layout.setContentsMargins(0, 0, S.MD, 0)  # right padding
        wrap_layout.setSpacing(0)
        wrap_layout.addWidget(burger)
        menu_bar.setCornerWidget(burger_wrapper, Qt.Corner.TopRightCorner)
        self._burger_btn = burger

        # --- Help menu ----------------------------------------------------
        help_menu = menu_bar.addMenu("&Help")
        shortcuts_act = QAction("&Keyboard shortcuts…", self)
        shortcuts_act.setShortcut(QKeySequence("F1"))
        shortcuts_act.triggered.connect(self._show_shortcuts)
        help_menu.addAction(shortcuts_act)
        help_menu.addSeparator()
        about_act = QAction("&About img_player", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

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
        from img_player.ui.shortcuts_dialog import ShortcutsDialog

        dlg = ShortcutsDialog(self)
        dlg.exec()

    def install_recent_provider(
        self,
        provider: Callable[[], list[Path]],
        clear_callback: Callable[[], None],
    ) -> None:
        """Let the app inject the functions that list / clear recent paths."""
        self._recent_paths_provider = provider
        self._clear_recent_callback = clear_callback
        self._refresh_recent_menu()

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
            act.triggered.connect(lambda _=False, p=path: self.open_requested.emit(p))
            self._recent_menu.addAction(act)
        self._recent_menu.addSeparator()
        clear = QAction("Clear list", self)
        clear.triggered.connect(self._on_clear_recent_clicked)
        self._recent_menu.addAction(clear)

    def _on_clear_recent_clicked(self) -> None:
        self._clear_recent_callback()
        self._refresh_recent_menu()

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
        # Frame display: typing a frame number / TC and pressing Enter
        # asks the controller to seek there.
        self._transport.frame_seek_requested.connect(self.frame_requested.emit)
        self._transport.channels_requested.connect(self.channels_requested.emit)
        self._transport.channel_mask_changed.connect(self.channel_mask_changed.emit)
        # Zoom: combo → viewport (forward), wheel → combo (back-channel
        # so the displayed value follows the wheel without us
        # re-emitting and ping-ponging).
        self._transport.zoom_requested.connect(self.zoom_requested.emit)
        self._viewer.gl.zoom_changed.connect(self._transport.set_zoom_display)
        self._timeline.frame_requested.connect(self.frame_requested.emit)
        # Drag-scrub inside the image viewport routes through the same
        # frame_requested → app._on_scrub_requested pipeline as the
        # timeline scrubber. From the controller's point of view the
        # two sources are indistinguishable.
        self._viewer.gl.frame_requested.connect(self.frame_requested.emit)

    # --------------------------------------------------------------- Menu handlers

    def _on_open_action(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open an image or a frame of a sequence",
            "",
            "Images (*.exr *.dpx *.tif *.tiff *.png *.jpg *.jpeg *.tga);;All files (*.*)",
        )
        if path_str:
            self.open_requested.emit(Path(path_str))

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "About img_player",
            (
                "<b>img_player</b><br>"
                "VFX-grade image sequence player.<br><br>"
                "OCIO color management, async RAM cache, OpenGL viewport."
            ),
        )

    # --------------------------------------------------------------- Drag & drop

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            return
        local = urls[0].toLocalFile()
        if not local:
            event.ignore()
            return
        event.acceptProposedAction()
        self.open_requested.emit(Path(local))

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
