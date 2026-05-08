"""Application-wide Preferences dialog (File → Preferences…).

Two-pane layout — sidebar of categories on the left, stacked editor
panels on the right — modelled on Qt Creator / VS Code so adding a new
section later (Playback, Cache, Annotation…) is just a question of
appending one widget. Each section widget is responsible for reading
its own values from :class:`Preferences` on construction and writing
them back when ``apply()`` is called.

For now there is a single section: **Color Management** (OCIO config
source). The rest of the app's preferences still live in inline UIs
(color panel, channel menu, etc.) and will migrate here over time as
they grow into "settings" rather than "in-context controls".
"""

from __future__ import annotations

import logging
import os

import PyOpenColorIO as ocio
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from img_player.preferences import Preferences

log = logging.getLogger(__name__)


class _ColorManagementPage(QWidget):
    """OCIO config source picker.

    Three modes:
      * **Default** — force the OCIO library's built-in ACES config,
        ignoring ``$OCIO``. Good first-run baseline.
      * **Environment** — honour ``$OCIO`` (legacy / studio default).
      * **Custom** — load a ``.ocio`` file picked by the user.

    Changes are flagged with a "Restart required" banner because the
    GPU shader, color panel and cached processors all bind to the
    config that was active at boot — hot-swapping would mean
    invalidating every one of them.
    """

    def __init__(self, prefs: Preferences, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._prefs = prefs
        self._initial_mode = prefs.ocio_config_mode
        self._initial_path = prefs.ocio_config_path or ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Color Management")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Choose how Flick resolves the OpenColorIO configuration "
            "used by the viewer. Custom configs are typical in studio "
            "pipelines (e.g. a project-specific .ocio shipped with the "
            "show)."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #9aa0a6;")
        layout.addWidget(subtitle)

        # ---- Mode radio group ---------------------------------------
        self._mode_group = QButtonGroup(self)
        self._radio_default = QRadioButton("Default (built-in ACES config)")
        self._radio_env = QRadioButton("From $OCIO environment variable")
        self._radio_custom = QRadioButton("Custom config file…")
        for i, btn in enumerate((self._radio_default, self._radio_env, self._radio_custom)):
            self._mode_group.addButton(btn, i)
            layout.addWidget(btn)

        # ---- Path picker (only meaningful in custom mode) -----------
        path_row = QHBoxLayout()
        path_row.setContentsMargins(24, 0, 0, 0)
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to a .ocio config file")
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(self._browse_btn)
        layout.addLayout(path_row)

        # ---- Active config readout ----------------------------------
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2a2a2a;")
        layout.addWidget(sep)

        self._status = QLabel(self._describe_active_config())
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #9aa0a6; font-size: 11px;")
        layout.addWidget(self._status)

        # ---- Restart-required banner --------------------------------
        self._restart_banner = QLabel(
            "Restart Flick to apply changes to the OCIO configuration."
        )
        self._restart_banner.setStyleSheet(
            "background: #4a3a1f; color: #f5c878; padding: 8px; border-radius: 4px;"
        )
        self._restart_banner.setVisible(False)
        layout.addWidget(self._restart_banner)

        layout.addStretch(1)

        # ---- Initial state + signals --------------------------------
        self._radio_default.setChecked(self._initial_mode == "default")
        self._radio_env.setChecked(self._initial_mode == "env")
        self._radio_custom.setChecked(self._initial_mode == "custom")
        self._path_edit.setText(self._initial_path)
        self._sync_path_enabled()

        self._mode_group.idToggled.connect(self._on_mode_changed)
        self._path_edit.textChanged.connect(self._update_dirty_state)

    # ---------------------------------------------------------------- API

    def apply(self) -> bool:
        """Persist values; return True if a restart is needed."""
        new_mode = self._current_mode()
        new_path = self._path_edit.text().strip() or None
        dirty = (new_mode != self._initial_mode) or (
            (new_path or "") != (self._initial_path or "")
        )
        self._prefs.ocio_config_mode = new_mode
        self._prefs.ocio_config_path = new_path
        # Update baseline so a second Apply in the same session doesn't
        # re-trigger the restart banner unnecessarily.
        self._initial_mode = new_mode
        self._initial_path = new_path or ""
        self._restart_banner.setVisible(False)
        return dirty

    # ---------------------------------------------------------------- Internals

    def _current_mode(self) -> str:
        if self._radio_custom.isChecked():
            return "custom"
        if self._radio_env.isChecked():
            return "env"
        return "default"

    def _on_mode_changed(self, _id: int, _checked: bool) -> None:
        self._sync_path_enabled()
        self._update_dirty_state()

    def _sync_path_enabled(self) -> None:
        custom = self._radio_custom.isChecked()
        self._path_edit.setEnabled(custom)
        self._browse_btn.setEnabled(custom)

    def _update_dirty_state(self) -> None:
        new_mode = self._current_mode()
        new_path = self._path_edit.text().strip()
        dirty = (new_mode != self._initial_mode) or (new_path != self._initial_path)
        self._restart_banner.setVisible(dirty)

    def _on_browse(self) -> None:
        start_dir = self._path_edit.text().strip() or os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select OCIO config",
            start_dir,
            "OCIO config (*.ocio);;All files (*.*)",
        )
        if path:
            self._path_edit.setText(path)

    def _describe_active_config(self) -> str:
        """Show what's currently loaded — useful for debugging studio
        setups where the user isn't sure which config Flick picked up.
        Always reflects the *boot-time* config, not pending edits."""
        try:
            cfg = self._load_current_config_snapshot()
            name = cfg.getName() or "(unnamed)"
            cs_count = sum(1 for _ in cfg.getColorSpaces())
            displays = list(cfg.getDisplays())
            return (
                f"Active config: {name} — {cs_count} colorspaces, "
                f"{len(displays)} display(s)."
            )
        except Exception as err:  # pragma: no cover — defensive
            log.debug("Could not describe active OCIO config: %s", err)
            return "Active config: (unable to read)"

    def _load_current_config_snapshot(self) -> ocio.Config:
        """Mirror :meth:`OCIOManager._resolve_default` against the
        *saved* prefs so the readout matches the next-boot resolution.
        Done locally to avoid round-tripping through a fresh
        ``OCIOManager`` (which does its own logging on failure)."""
        from img_player.color.ocio_manager import DEFAULT_BUILTIN_URI

        mode = self._initial_mode
        path = self._initial_path
        if mode == "custom" and path:
            try:
                return ocio.Config.CreateFromFile(path)
            except ocio.Exception:
                return ocio.Config.CreateFromBuiltinConfig(DEFAULT_BUILTIN_URI)
        if mode == "env":
            env = os.environ.get("OCIO")
            if env:
                try:
                    return ocio.Config.CreateFromFile(env)
                except ocio.Exception:
                    pass
        return ocio.Config.CreateFromBuiltinConfig(DEFAULT_BUILTIN_URI)


class PreferencesDialog(QDialog):
    """Modal preferences dialog with a category sidebar.

    Add a new section by:
      1. Creating a ``QWidget`` subclass with an ``apply()`` method
      2. Calling ``self._add_section("Title", widget)`` in ``__init__``
    """

    def __init__(self, prefs: Preferences, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.resize(720, 520)

        self._prefs = prefs
        self._pages: list[QWidget] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(180)
        self._sidebar.setStyleSheet(
            "QListWidget { background: #1c1c1c; border: none; padding: 8px 0; }"
            "QListWidget::item { padding: 8px 16px; }"
            "QListWidget::item:selected { background: #2d2d2d; color: #fff; }"
        )
        self._stack = QStackedWidget()

        body.addWidget(self._sidebar)
        body.addWidget(self._stack, 1)
        root.addLayout(body, 1)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Apply
        )
        button_row = QHBoxLayout()
        button_row.setContentsMargins(12, 8, 12, 12)
        button_row.addWidget(buttons)
        root.addLayout(button_row)

        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._on_apply)

        # Sections
        self._add_section("Color Management", _ColorManagementPage(prefs, self))

        self._sidebar.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._sidebar.setCurrentRow(0)

    # ---------------------------------------------------------------- API

    def _add_section(self, label: str, widget: QWidget) -> None:
        item = QListWidgetItem(label)
        item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self._sidebar.addItem(item)
        self._stack.addWidget(widget)
        self._pages.append(widget)

    # ---------------------------------------------------------------- Slots

    def _apply_all(self) -> bool:
        any_dirty = False
        for page in self._pages:
            apply = getattr(page, "apply", None)
            if callable(apply):
                if apply():
                    any_dirty = True
        return any_dirty

    def _on_apply(self) -> None:
        self._apply_all()

    def _on_ok(self) -> None:
        self._apply_all()
        self.accept()
