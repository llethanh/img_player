"""MainWindow: assembles the viewer, transport, timeline and side panels.

Signals from controls are routed to the :class:`PlayerController` and
:class:`GLViewport` by the app module; this widget only owns the UI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QAction,
    QCloseEvent,
    QDragEnterEvent,
    QDropEvent,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
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
from img_player.ui.channel_panel import ChannelPanel
from img_player.ui.color_panel import ColorPanel
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

    def __init__(self, ocio_manager: OCIOManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("img_player")
        self.resize(1280, 720)
        self.setAcceptDrops(True)

        # Widgets
        self._viewer = ViewerWidget(self)
        self._transport = TransportBar(self)
        self._timeline = Timeline(self)
        self._color_panel = ColorPanel(ocio_manager, self)
        self._channel_panel = ChannelPanel(self)

        # Central: viewer on top, then timeline + transport stacked at the bottom
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(S.SM, S.SM, S.SM, S.SM)
        layout.setSpacing(S.SM)
        layout.addWidget(self._viewer, stretch=1)
        layout.addWidget(self._timeline)
        layout.addWidget(self._transport)
        self.setCentralWidget(central)

        # Right-hand dock holds Color + Channels as tabs
        self._side_tabs = QTabWidget()
        self._side_tabs.addTab(self._color_panel, "Color")
        self._side_tabs.addTab(self._channel_panel, "Channels")
        # Stored on self so the burger button can toggle it.
        self._side_dock = QDockWidget("Panels", self)
        self._side_dock.setWidget(self._side_tabs)
        self._side_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._side_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._side_dock)

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
    def channel_panel(self) -> ChannelPanel:
        return self._channel_panel

    @property
    def transport(self) -> TransportBar:
        return self._transport

    @property
    def timeline(self) -> Timeline:
        return self._timeline

    # --------------------------------------------------------------- Public updates

    def update_sequence_info(self, sequence: SequenceInfo) -> None:
        """Refresh the title bar, timeline range, and channel panel."""
        self.setWindowTitle(f"img_player — {sequence.display_pattern()}")
        self._timeline.set_range(sequence.first_frame, sequence.last_frame)
        self._channel_panel.set_channels(sequence.channel_names)

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
        burger.setText("☰")
        burger.setAutoRaise(True)
        burger.setFixedSize(36, 24)
        burger.setCursor(Qt.CursorShape.PointingHandCursor)
        burger.setToolTip("Show / hide the side panels")
        burger.setStyleSheet(
            "QToolButton {"
            f"  background: transparent;"
            f"  color: {H.TEXT_SECONDARY};"
            f"  border: none;"
            f"  padding: 0;"
            "  font-size: 16px;"   # bigger than UI default — the glyph needs presence
            "}"
            "QToolButton:hover {"
            f"  color: {H.TEXT_PRIMARY};"
            f"  background: {H.BG_HOVER};"
            f"  border-radius: {G.RADIUS_SM}px;"
            "}"
        )
        burger.clicked.connect(self._toggle_side_dock)
        menu_bar.setCornerWidget(burger, Qt.Corner.TopRightCorner)
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
        log.info("MainWindow closing")
        super().closeEvent(event)
