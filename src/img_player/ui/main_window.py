"""MainWindow: assembles the viewer, transport, timeline and side panels.

Signals from controls are routed to the :class:`PlayerController` and
:class:`GLViewport` by the app module; this widget only owns the UI.
"""

from __future__ import annotations

import logging
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
    QMainWindow,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from img_player.color.ocio_manager import OCIOManager
from img_player.ui.channel_panel import ChannelPanel
from img_player.ui.color_panel import ColorPanel
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
    stop_clicked = Signal()
    step_clicked = Signal(int)  # +1 / -1
    jump_to_ends = Signal(int)  # -1 first, +1 last
    frame_requested = Signal(int)
    exposure_step = Signal(float)  # +/- keyboard adjustment
    fps_changed = Signal(float)

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
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self._viewer, stretch=1)
        layout.addWidget(self._timeline)
        layout.addWidget(self._transport)
        self.setCentralWidget(central)

        # Right-hand dock holds Color + Channels as tabs
        self._side_tabs = QTabWidget()
        self._side_tabs.addTab(self._color_panel, "Color")
        self._side_tabs.addTab(self._channel_panel, "Channels")
        dock = QDockWidget("Panels", self)
        dock.setWidget(self._side_tabs)
        dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

        self._build_menu()
        self._install_shortcuts()
        self._wire_internal()

        self.statusBar().showMessage("Ready — drop a sequence (folder or file) to start.")

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
        self.statusBar().showMessage(message)

    # --------------------------------------------------------------- Menu / shortcuts

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("&File")

        open_act = QAction("&Open…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_action)
        file_menu.addAction(open_act)

        file_menu.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        help_menu = menu_bar.addMenu("&Help")
        about_act = QAction("&About img_player", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _install_shortcuts(self) -> None:
        # Playback
        QShortcut(QKeySequence(Qt.Key.Key_Space), self, activated=self.play_toggled.emit)
        QShortcut(QKeySequence(Qt.Key.Key_K), self, activated=self.play_toggled.emit)

        # Frame stepping
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, activated=lambda: self.step_clicked.emit(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, activated=lambda: self.step_clicked.emit(1))
        QShortcut(QKeySequence("Shift+Left"), self, activated=lambda: self.step_clicked.emit(-10))
        QShortcut(QKeySequence("Shift+Right"), self, activated=lambda: self.step_clicked.emit(10))
        QShortcut(QKeySequence(Qt.Key.Key_Home), self, activated=lambda: self.jump_to_ends.emit(-1))
        QShortcut(QKeySequence(Qt.Key.Key_End), self, activated=lambda: self.jump_to_ends.emit(1))

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
        self._transport.play_toggled.connect(self.play_toggled.emit)
        self._transport.stop_clicked.connect(self.stop_clicked.emit)
        self._transport.step_clicked.connect(self.step_clicked.emit)
        self._transport.jump_to_ends.connect(self.jump_to_ends.emit)
        self._transport.fps_changed.connect(self.fps_changed.emit)
        self._timeline.frame_requested.connect(self.frame_requested.emit)

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
