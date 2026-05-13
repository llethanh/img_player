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
from collections.abc import Callable
from pathlib import Path

import PyOpenColorIO as ocio
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from img_player.preferences import Preferences

log = logging.getLogger(__name__)


class _GeneralPage(QWidget):
    """Application-wide info that doesn't belong to a specific feature
    section.

    Today it surfaces the canonical on-disk locations Flick uses
    (user preferences, disk cache root, log file, site config) plus
    an "Open folder" shortcut so the user / studio admin doesn't
    have to remember ``%APPDATA%`` vs ``%LOCALAPPDATA%``. The page
    has no editable widgets — every value here is determined at boot
    by :mod:`img_player.app_paths` and the resolved site config, so
    there's no ``apply()`` method.
    """

    def __init__(
        self,
        prefs: Preferences,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._prefs = prefs

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("General")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Where Flick stores your preferences and runtime data. "
            "The user preferences file (flick.toml) is plain text — "
            "you can edit it by hand, copy it between machines, or "
            "commit it to source control."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #9aa0a6;")
        layout.addWidget(subtitle)

        # ---- Path rows --------------------------------------------------
        # Built lazily from app_paths so the labels reflect the
        # post-migration locations (``FlickPlayer\`` rather than the
        # legacy ``img_player\``).
        from img_player.app_paths import (
            calibration_profile_path,
            disk_cache_default_dir,
            log_dir,
            user_prefs_dir,
        )
        from img_player.site_config import site_config

        user_dir = user_prefs_dir()
        self._add_path_row(
            layout,
            "User preferences",
            user_dir / "flick.toml",
            open_target=user_dir,
            tooltip=(
                "Your preference overrides — color management, disk "
                "cache settings, anything you change via this dialog. "
                "Hand-editable. Backed up by copying the folder."
            ),
        )

        site_path = site_config().source
        self._add_path_row(
            layout,
            "Site configuration",
            site_path if site_path else "(none — using built-in defaults)",
            open_target=site_path.parent if site_path else None,
            tooltip=(
                "Studio-wide defaults applied to every user on this "
                "install. Drop a flick.toml next to FlickPlayer.exe "
                "(or point $FLICK_SITE_CONFIG at one) to activate."
            ),
        )

        cache_dir = self._prefs.disk_cache_path or disk_cache_default_dir()
        self._add_path_row(
            layout,
            "Disk cache",
            cache_dir,
            open_target=cache_dir if cache_dir.is_dir() else cache_dir.parent,
            tooltip=(
                "Where decoded frames evicted from RAM are persisted "
                "between sessions."
            ),
        )

        logs = log_dir()
        self._add_path_row(
            layout,
            "Log file",
            logs / "flick.log",
            open_target=logs,
            tooltip=(
                "Flick's running log. Attach this when filing a bug."
            ),
        )

        calib = calibration_profile_path()
        self._add_path_row(
            layout,
            "Performance profile",
            calib,
            open_target=calib.parent,
            tooltip=(
                "Auto-tuned hardware profile (worker count, cache "
                "size, OIIO threads). Delete to force a re-detect on "
                "next launch."
            ),
        )

        layout.addStretch(1)

    # ------------------------------------------------------------------ Helpers

    def _add_path_row(  # type: ignore[no-untyped-def]
        self,
        layout: QVBoxLayout,
        label: str,
        path: object,
        *,
        open_target: object | None = None,
        tooltip: str = "",
    ) -> None:
        """Emit one ``Label · path · [Open]`` row.

        ``path`` is shown for context; ``open_target`` is what the
        Open button reveals in the file manager. They can be the
        same (a directory) or different (e.g. a file path shown,
        its parent dir opened) — Explorer doesn't usefully open a
        single file, so we always open the containing dir.
        """
        row = QHBoxLayout()
        name = QLabel(f"<b>{label}:</b>")
        name.setStyleSheet("color: #ccc;")
        name.setMinimumWidth(140)
        value = QLabel(str(path))
        value.setStyleSheet("color: #9aa0a6; font-size: 11px;")
        value.setWordWrap(True)
        value.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        row.addWidget(name)
        row.addWidget(value, 1)
        if open_target is not None:
            from pathlib import Path

            target_path = Path(str(open_target))
            btn = QPushButton("Open")
            btn.setMaximumWidth(80)
            if tooltip:
                btn.setToolTip(tooltip)
            btn.clicked.connect(
                lambda _checked=False, p=target_path: self._open_folder(p),
            )
            row.addWidget(btn)
        layout.addLayout(row)

    def _open_folder(self, target):  # type: ignore[no-untyped-def]
        """Reveal ``target`` in the platform file manager.

        Creates the directory first if it doesn't exist yet — a fresh
        install hasn't necessarily written to every location. Dispatches
        via ``QDesktopServices.openUrl`` (Explorer on Windows, Finder
        on macOS, configured handler on Linux).
        """
        from pathlib import Path

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        target = Path(target)
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            log.warning("Could not create %s (%s)", target, err)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))


class _ColorManagementPage(QWidget):
    """OCIO config source picker.

    Three modes:
      * **Default** — force the OCIO library's built-in ACES config,
        ignoring ``$OCIO``. Good first-run baseline.
      * **Environment** — honour ``$OCIO`` (legacy / studio default).
      * **Custom** — load a ``.ocio`` file picked by the user.

    When an ``on_reload`` callback is provided, applying changes
    triggers a hot-reload (no restart needed) — the callback returns
    a status dict the page summarises in a transient banner. Without
    a callback the page falls back to the legacy "Restart required"
    banner.
    """

    def __init__(
        self,
        prefs: Preferences,
        on_reload: Callable[[], dict[str, object]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._prefs = prefs
        self._on_reload = on_reload
        self._initial_mode = prefs.ocio_config_mode
        self._initial_path = prefs.ocio_config_path or ""
        self._initial_builtin = prefs.ocio_builtin_uri

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
        # Each mode's nested controls (the built-in dropdown for
        # "Default", the path picker for "Custom") are added IMMEDIATELY
        # after their owning radio so the visual hierarchy mirrors the
        # logical one — picking a radio + tweaking the control right
        # below it reads naturally.
        self._mode_group = QButtonGroup(self)
        self._radio_default = QRadioButton("Default (built-in ACES config)")
        self._radio_env = QRadioButton("From $OCIO environment variable")
        self._radio_custom = QRadioButton("Custom config file…")
        self._mode_group.addButton(self._radio_default, 0)
        self._mode_group.addButton(self._radio_env, 1)
        self._mode_group.addButton(self._radio_custom, 2)

        # Default radio + its built-in dropdown.
        layout.addWidget(self._radio_default)

        # ---- Built-in config picker (under the Default radio) -------
        # Dropdown lists every config bundled with OCIO. Default is the
        # ACES 1.3 CG config to match Nuke / Maya / OpenRV — ACES 2.0
        # gives noticeably different tone-mapping that surprises users
        # coming from a 1.x pipeline.
        builtin_row = QHBoxLayout()
        builtin_row.setContentsMargins(24, 0, 0, 0)
        builtin_label = QLabel("Config:")
        builtin_label.setStyleSheet("color: #9aa0a6;")
        self._builtin_combo = QComboBox()
        self._builtin_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents,
        )
        builtin_row.addWidget(builtin_label)
        builtin_row.addWidget(self._builtin_combo, 1)
        layout.addLayout(builtin_row)
        # Populate. ``list_builtin_configs`` returns ordered entries
        # straight from OCIO; we just decorate the labels with ACES
        # family + kind for readability.
        self._builtin_uris: list[str] = []
        from img_player.color.ocio_manager import OCIOManager
        try:
            entries = OCIOManager.list_builtin_configs()
        except Exception:  # pragma: no cover — defensive
            log.exception("Failed to enumerate OCIO builtin configs")
            entries = []
        for entry in entries:
            badge = []
            if entry.aces_family != "unknown":
                badge.append(f"ACES {entry.aces_family}")
            badge.append("Studio" if entry.kind == "studio" else "CG")
            if entry.recommended:
                badge.append("recommended")
            label = f"{' · '.join(badge)}  —  {entry.name}"
            self._builtin_combo.addItem(label, entry.uri)
            self._builtin_uris.append(entry.uri)
        # Fallback row when OCIO can't enumerate — keep something
        # selectable so the dropdown isn't visibly empty.
        if not self._builtin_uris:
            self._builtin_combo.addItem(
                "Library default  —  ocio://default",
                "ocio://default",
            )
            self._builtin_uris.append("ocio://default")

        # Env radio (no nested control).
        layout.addWidget(self._radio_env)

        # Custom radio + its path picker.
        layout.addWidget(self._radio_custom)

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
        # NB: the user prefs "Open folder" button used to live here but
        # flick.toml stores cross-section settings (color + disk cache,
        # and any future user-facing pref), so it now lives in the
        # General tab — a single canonical spot for "app data" actions.

        # ---- Pending-changes banner ---------------------------------
        # Two flavours: amber "you have pending changes" before Apply,
        # green "applied" after a successful hot-reload, red on
        # failure. Stays hidden until the user actually edits the
        # mode or path.
        self._restart_banner = QLabel()
        self._restart_banner.setWordWrap(True)
        self._restart_banner.setVisible(False)
        layout.addWidget(self._restart_banner)
        self._set_banner_pending()

        layout.addStretch(1)

        # ---- Initial state + signals --------------------------------
        self._radio_default.setChecked(self._initial_mode == "default")
        self._radio_env.setChecked(self._initial_mode == "env")
        self._radio_custom.setChecked(self._initial_mode == "custom")
        self._path_edit.setText(self._initial_path)
        # Pre-select the builtin matching the saved pref; fall back to
        # the first entry if the saved URI is no longer available (e.g.
        # OCIO downgrade).
        try:
            initial_idx = self._builtin_uris.index(self._initial_builtin)
        except ValueError:
            initial_idx = 0
        self._builtin_combo.setCurrentIndex(initial_idx)
        self._sync_path_enabled()

        self._mode_group.idToggled.connect(self._on_mode_changed)
        self._path_edit.textChanged.connect(self._update_dirty_state)
        self._builtin_combo.currentIndexChanged.connect(self._update_dirty_state)

    # ---------------------------------------------------------------- API

    def apply(self) -> bool:
        """Persist values and trigger hot-reload if anything changed.

        Returns ``True`` if the OCIO settings actually changed
        (regardless of reload outcome) so the parent dialog can
        decide whether to keep the dialog open or close it.

        Each setter is gated by an explicit "did this field change?"
        check. Without that gate, clicking Apply with the dialog open
        but unchanged would call every setter unconditionally —
        materialising a user TOML file with the resolved (= site /
        hardcoded) values even though the user touched nothing. The
        site-config default would then become a user-level override
        on disk, breaking the "user TOML only contains explicit
        choices" invariant.
        """
        new_mode = self._current_mode()
        new_path = self._path_edit.text().strip() or None
        new_builtin = self._current_builtin_uri()

        mode_changed = new_mode != self._initial_mode
        path_changed = (new_path or "") != (self._initial_path or "")
        builtin_changed = new_builtin != self._initial_builtin

        if mode_changed:
            self._prefs.ocio_config_mode = new_mode
            self._initial_mode = new_mode
        if path_changed:
            self._prefs.ocio_config_path = new_path
            self._initial_path = new_path or ""
        if builtin_changed:
            self._prefs.ocio_builtin_uri = new_builtin
            self._initial_builtin = new_builtin

        dirty = mode_changed or path_changed or builtin_changed

        if not dirty:
            self._restart_banner.setVisible(False)
            return False

        if self._on_reload is None:
            # No hot-reload wired — keep the legacy "Restart required"
            # message so the user knows their pref was saved but won't
            # take effect until next launch.
            self._set_banner_legacy_restart()
            return True

        try:
            status = self._on_reload()
        except Exception as err:  # pragma: no cover — defensive
            log.exception("OCIO hot-reload failed")
            self._set_banner_failure(str(err))
            return True

        self._set_banner_success(status)
        # Refresh the read-only "Active config" line so it reflects
        # the just-loaded config (otherwise it still reads the
        # boot-time one until the dialog is reopened).
        self._status.setText(self._describe_active_config())
        return True

    # ---------------------------------------------------------------- Banner helpers

    def _set_banner_pending(self) -> None:
        self._restart_banner.setText(
            "Pending changes — click Apply or OK to load the new OCIO config."
        )
        self._restart_banner.setStyleSheet(
            "background: #4a3a1f; color: #f5c878; "
            "padding: 8px; border-radius: 4px;"
        )

    def _set_banner_legacy_restart(self) -> None:
        self._restart_banner.setText(
            "Restart Flick to apply changes to the OCIO configuration."
        )
        self._restart_banner.setStyleSheet(
            "background: #4a3a1f; color: #f5c878; "
            "padding: 8px; border-radius: 4px;"
        )
        self._restart_banner.setVisible(True)

    def _set_banner_success(self, status: dict[str, object]) -> None:
        name = status.get("config_name", "?")
        # Tally any picks that didn't survive the swap so we can be
        # honest about what just changed under the user's feet.
        invalidated = [
            label for label, key in (
                ("source", "source_preserved"),
                ("display", "display_preserved"),
                ("view", "view_preserved"),
            )
            if status.get(key) is False
        ]
        msg = f"Loaded: {name}"
        if invalidated:
            msg += f" — reset stale {' / '.join(invalidated)} pick(s)"
        else:
            msg += " — picks preserved"
        self._restart_banner.setText(msg)
        self._restart_banner.setStyleSheet(
            "background: #1f3a26; color: #87d98b; "
            "padding: 8px; border-radius: 4px;"
        )
        self._restart_banner.setVisible(True)

    def _set_banner_failure(self, error: str) -> None:
        self._restart_banner.setText(
            f"Hot-reload failed: {error}. The previous config is still active."
        )
        self._restart_banner.setStyleSheet(
            "background: #4a1f1f; color: #f58c8c; "
            "padding: 8px; border-radius: 4px;"
        )
        self._restart_banner.setVisible(True)

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
        default = self._radio_default.isChecked()
        self._path_edit.setEnabled(custom)
        self._browse_btn.setEnabled(custom)
        # The built-in dropdown only matters when "Default" is the
        # active mode — grey it out otherwise so the UI matches the
        # actual resolution path.
        self._builtin_combo.setEnabled(default)

    def _current_builtin_uri(self) -> str:
        """Return the builtin URI currently selected in the dropdown.
        Falls back to the saved pref if the dropdown is somehow empty
        (defensive)."""
        idx = self._builtin_combo.currentIndex()
        if 0 <= idx < len(self._builtin_uris):
            return self._builtin_uris[idx]
        return self._initial_builtin

    def _update_dirty_state(self) -> None:
        new_mode = self._current_mode()
        new_path = self._path_edit.text().strip()
        new_builtin = self._current_builtin_uri()
        dirty = (
            new_mode != self._initial_mode
            or new_path != self._initial_path
            or new_builtin != self._initial_builtin
        )
        if dirty:
            # Reset to the amber pending message; any prior success /
            # failure banner is now stale because the user is editing
            # again.
            self._set_banner_pending()
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
        builtin = self._initial_builtin or DEFAULT_BUILTIN_URI
        if mode == "custom" and path:
            try:
                return ocio.Config.CreateFromFile(path)
            except ocio.Exception:
                return ocio.Config.CreateFromBuiltinConfig(builtin)
        if mode == "env":
            env = os.environ.get("OCIO")
            if env:
                try:
                    return ocio.Config.CreateFromFile(env)
                except ocio.Exception:
                    pass
        try:
            return ocio.Config.CreateFromBuiltinConfig(builtin)
        except ocio.Exception:
            return ocio.Config.CreateFromBuiltinConfig(DEFAULT_BUILTIN_URI)


class _DiskCachePage(QWidget):
    """Disk-cache settings — enable/disable, location, budget, clear.

    The disk cache is a session-spanning second tier above the live
    ``MasterFrameCache``: evicted RAM frames are persisted as
    lz4-compressed half-float blobs so the next session can re-open
    warm. This page exposes the three knobs the user cares about:

    * **Enable** — global on/off. When off the cache acts RAM-only.
    * **Path** — where the blob tree + SQLite index live. Useful to
      move the cache off the system drive (SSD wear) or onto a
      faster NVMe scratch.
    * **Budget** — soft upper bound in **gigabytes**. ``0`` = no
      limit (cache only grows on explicit clear).
    * **Clear** — wipe everything in one click for "my disk is
      full" situations.

    Settings changes only take effect on next session — the running
    cache instance is captured at app init. We tell the user
    explicitly so they don't expect a hot-reload.
    """

    def __init__(
        self,
        prefs: Preferences,
        disk_cache_handle: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._prefs = prefs
        # ``disk_cache_handle`` is the live :class:`DiskCache` instance
        # (or ``None`` if disk caching is disabled / unavailable).
        # Used only for the live-stats display + Clear-now button —
        # config changes route through ``prefs`` and apply next launch.
        self._disk_cache = disk_cache_handle
        self._initial_enabled = prefs.disk_cache_enabled
        self._initial_path = str(prefs.disk_cache_path or "")
        self._initial_budget_gb = prefs.disk_cache_budget_gb
        self._initial_compression = prefs.disk_cache_compression

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Disk cache")
        title.setStyleSheet("font-size: 14px; font-weight: 600;")
        layout.addWidget(title)

        intro = QLabel(
            "Decoded frames evicted from RAM are saved to disk so the "
            "next session can re-open warm — same shot tomorrow loads in "
            "a tenth of the time. Stored as lz4-compressed half-float "
            "blobs (≈ 25 MB per 4K frame).",
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #aaa;")
        layout.addWidget(intro)

        # ---- Enable -----------------------------------------------------
        self._enable_chk = QCheckBox("Enable disk cache")
        self._enable_chk.setChecked(self._initial_enabled)
        self._enable_chk.toggled.connect(self._on_enable_toggled)
        layout.addWidget(self._enable_chk)

        # ---- Path -------------------------------------------------------
        layout.addWidget(self._make_subtitle("Cache location"))

        path_help = QLabel(
            "Leave empty to use the default OS-specific cache directory. "
            "Point to a faster drive (NVMe scratch) or a partition with "
            "more headroom if needed.",
        )
        path_help.setWordWrap(True)
        path_help.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(path_help)

        path_row = QHBoxLayout()
        self._path_edit = QLineEdit(self._initial_path)
        self._path_edit.setPlaceholderText(
            "(default — %LOCALAPPDATA%\\FlickPlayer\\disk_cache\\)",
        )
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(self._path_edit, 1)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        # ---- Budget -----------------------------------------------------
        layout.addWidget(self._make_subtitle("Disk budget"))

        budget_row = QHBoxLayout()
        self._budget_spin = QSpinBox()
        self._budget_spin.setRange(0, 10000)  # 0 = unlimited; 10 TB ceiling
        self._budget_spin.setSuffix(" GB")
        self._budget_spin.setSpecialValueText("Unlimited")  # shown when value=0
        self._budget_spin.setValue(self._initial_budget_gb)
        budget_help = QLabel(
            "Soft upper bound. When the cache exceeds this, the oldest "
            "frames are evicted until total drops to 85 %. Set 0 for "
            "no limit (use the Clear button below when needed).",
        )
        budget_help.setWordWrap(True)
        budget_help.setStyleSheet("color: #888; font-size: 11px;")
        budget_row.addWidget(self._budget_spin)
        budget_row.addStretch(1)
        layout.addWidget(budget_help)
        layout.addLayout(budget_row)

        # ---- Storage / compression -------------------------------------
        layout.addWidget(self._make_subtitle("Storage"))

        self._compress_chk = QCheckBox("Compress blobs (lz4)")
        self._compress_chk.setChecked(self._initial_compression)
        self._compress_chk.setToolTip(
            "lz4 compression trades ~5 ms of decode time per read for "
            "~50 % smaller files. Disable on fast NVMe drives where I/O "
            "is essentially free and lz4 becomes the bottleneck — costs "
            "about 2× more disk space."
        )
        layout.addWidget(self._compress_chk)
        compress_help = QLabel(
            "Only affects new writes. Existing entries stay readable "
            "after toggling — no need to clear the cache.",
        )
        compress_help.setWordWrap(True)
        compress_help.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(compress_help)

        # ---- Clear ------------------------------------------------------
        layout.addWidget(self._make_subtitle("Maintenance"))

        self._usage_label = QLabel("")
        self._usage_label.setStyleSheet("color: #ccc;")
        layout.addWidget(self._usage_label)

        clear_btn = QPushButton("Clear cache now…")
        clear_btn.setToolTip(
            "Delete every cached frame from disk. Useful when the disk "
            "is running out of space. The cache rebuilds automatically "
            "as you decode frames during the next playback.",
        )
        clear_btn.clicked.connect(self._on_clear)
        clear_btn.setMaximumWidth(220)
        layout.addWidget(clear_btn)

        layout.addStretch(1)

        # Initial sync of the enabled/disabled visual state + usage.
        self._on_enable_toggled(self._initial_enabled)
        self._refresh_usage_label()

        # Live refresh — every 1.5 s while the page is visible, repull
        # ``stats()`` so the hit / write counters and disk usage update
        # without the user having to reopen the dialog. Stops itself
        # when the page is hidden / destroyed.
        self._stats_timer = QTimer(self)
        self._stats_timer.setInterval(1500)
        self._stats_timer.timeout.connect(self._refresh_usage_label)
        self._stats_timer.start()

    # ---- Subtitle helper -----------------------------------------------

    def _make_subtitle(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("font-weight: 600; margin-top: 6px;")
        return lbl

    # ---- Slots ---------------------------------------------------------

    def _on_enable_toggled(self, checked: bool) -> None:
        # Grey out path + budget + compression when the cache is off —
        # they have no effect in that state and showing them as active
        # would mislead.
        self._path_edit.setEnabled(checked)
        self._budget_spin.setEnabled(checked)
        self._compress_chk.setEnabled(checked)

    def _on_browse(self) -> None:
        current = self._path_edit.text().strip() or ""
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Pick a disk cache directory",
            current,
        )
        if chosen:
            self._path_edit.setText(chosen)

    def _on_clear(self) -> None:
        if self._disk_cache is None:
            QMessageBox.information(
                self,
                "Disk cache disabled",
                "The disk cache isn't currently running — nothing to "
                "clear. Enable it and restart to populate the cache.",
            )
            return
        reply = QMessageBox.warning(
            self,
            "Clear disk cache?",
            "This will delete every cached frame on disk. The cache "
            "rebuilds automatically as frames are decoded during the "
            "next playback, but the first scrub through the shot "
            "won't be as fast.\n\nClear now?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            freed = self._disk_cache.clear()
        except Exception:  # pragma: no cover — defensive
            log.exception("DiskCache clear failed")
            QMessageBox.critical(
                self, "Clear failed",
                "An error occurred while clearing the disk cache. "
                "Check the log for details.",
            )
            return
        self._refresh_usage_label()
        QMessageBox.information(
            self, "Disk cache cleared",
            f"Freed {freed / (1024 ** 3):.2f} GB.",
        )

    def _refresh_usage_label(self) -> None:
        if self._disk_cache is None:
            self._usage_label.setText(
                "Disk cache is not currently running.",
            )
            return
        try:
            stats = self._disk_cache.stats()
            path = self._disk_cache.cache_dir()
        except Exception:  # pragma: no cover — defensive
            self._usage_label.setText("(unable to read cache stats)")
            return
        used_gb = stats.size_bytes / (1024 ** 3)
        budget_gb = self._initial_budget_gb
        budget_str = "unlimited" if budget_gb == 0 else f"{budget_gb} GB"
        # Two-line readout: disk-usage / budget / path on line 1,
        # session-runtime counters on line 2. Plain text — keeps the
        # status colour the QSS already sets for the label.
        read_mb = stats.bytes_read / (1024 ** 2)
        written_mb = stats.bytes_written / (1024 ** 2)
        total_reads = stats.hits + stats.misses
        hit_pct = 100.0 * stats.hit_rate if total_reads > 0 else 0.0
        # Read-only banner appears as a prefix when another Flick
        # instance owns the cache directory — explains why ``Writes``
        # stays at 0 even though the user is scrubbing fresh frames.
        readonly_prefix = ""
        if getattr(stats, "read_only", False):
            readonly_prefix = (
                "⚠ Read-only — another Flick instance owns this cache.\n"
            )
        self._usage_label.setText(
            f"{readonly_prefix}"
            f"Used: {used_gb:.2f} GB  ·  {stats.entries} entries  ·  "
            f"Budget: {budget_str}\n"
            f"Hits: {stats.hits} / {total_reads}  "
            f"({hit_pct:.1f}%)  ·  "
            f"Writes: {stats.writes}  ·  "
            f"Read: {read_mb:.1f} MB  ·  Written: {written_mb:.1f} MB\n"
            f"Path: {path}",
        )

    # ---- Apply ---------------------------------------------------------

    def apply(self) -> bool:
        """Persist new values to preferences. Returns ``True`` when
        something changed (= a restart hint is worth showing)."""
        new_enabled = self._enable_chk.isChecked()
        new_path = self._path_edit.text().strip()
        new_budget = int(self._budget_spin.value())
        new_compression = self._compress_chk.isChecked()

        changed = False
        if new_enabled != self._initial_enabled:
            self._prefs.disk_cache_enabled = new_enabled
            self._initial_enabled = new_enabled
            changed = True
        if new_path != self._initial_path:
            self._prefs.disk_cache_path = Path(new_path) if new_path else None
            self._initial_path = new_path
            changed = True
        if new_budget != self._initial_budget_gb:
            self._prefs.disk_cache_budget_gb = new_budget
            self._initial_budget_gb = new_budget
            # Live cache picks up the new budget immediately — eviction
            # fires straight away if shrinking past current usage.
            if self._disk_cache is not None:
                try:
                    self._disk_cache.set_budget(new_budget * (1024 ** 3))
                except Exception:  # pragma: no cover — defensive
                    log.exception("DiskCache set_budget failed")
            changed = True
        if new_compression != self._initial_compression:
            self._prefs.disk_cache_compression = new_compression
            self._initial_compression = new_compression
            # Toggle the live cache too — affects new writes immediately.
            # Existing entries remain readable thanks to the multi-magic
            # auto-detection on read.
            if self._disk_cache is not None:
                try:
                    self._disk_cache.set_compress(new_compression)
                except Exception:  # pragma: no cover — defensive
                    log.exception("DiskCache set_compress failed")
            changed = True
        return changed


class PreferencesDialog(QDialog):
    """Modal preferences dialog with a category sidebar.

    Add a new section by:
      1. Creating a ``QWidget`` subclass with an ``apply()`` method
      2. Calling ``self._add_section("Title", widget)`` in ``__init__``
    """

    def __init__(
        self,
        prefs: Preferences,
        on_reload: Callable[[], dict[str, object]] | None = None,
        disk_cache: object | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.resize(720, 520)

        self._prefs = prefs
        self._on_reload = on_reload
        self._disk_cache = disk_cache
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

        # Sections — order: General (paths + diagnostics) first because
        # it's the section users most often want to find when something
        # is wrong (where's my log? where's my prefs file?), then the
        # feature-specific tabs.
        self._add_section(
            "General",
            _GeneralPage(prefs, parent=self),
        )
        self._add_section(
            "Color Management",
            _ColorManagementPage(prefs, on_reload=on_reload, parent=self),
        )
        self._add_section(
            "Disk cache",
            _DiskCachePage(prefs, disk_cache_handle=disk_cache, parent=self),
        )

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
