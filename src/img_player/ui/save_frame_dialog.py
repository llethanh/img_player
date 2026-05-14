"""Save Frame As… dialog — single-frame snapshot of the *full image*.

A lightweight alternative to the full Export pipeline: reads the
active source frame at full resolution (NOT a viewer screenshot), so
zoom / pan don't crop the saved pixels. The user can pick an output
resolution the same way the Export dialog does (Source / preset /
Custom W×H + Lock aspect).

The dialog itself is pure UI — it gathers user intent and returns a
:class:`SaveFrameSettings`. The full-image render + write happens in
:mod:`img_player.save_frame_handler`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from img_player.export.settings import RESOLUTION_PRESETS

# Format catalogue: extension → human label. Order matters for the
# dropdown; the first entry is the fallback default. PNG is first
# because it's the lossless, alpha-aware default for VFX review
# screenshots; JPG second for "share quickly". TIFF + BMP + WebP
# round out the common cases. EXR is intentionally excluded — the
# capture path goes through Qt's image writer (uint8 only); EXR's
# float pipeline belongs to the Export dialog.
FORMATS: tuple[tuple[str, str], ...] = (
    ("png", "PNG (lossless)"),
    ("jpg", "JPEG"),
    ("tif", "TIFF"),
    ("bmp", "BMP"),
    ("webp", "WebP"),
)
_DEFAULT_FORMAT = "png"


@dataclass(frozen=True)
class SaveFrameSettings:
    """User-confirmed Save Frame As… choices.

    ``path`` is the absolute target path (directory + filename +
    extension). The handler writes there directly; if the file
    exists the QFileDialog already prompted the user for confirmation
    on the chooser side, so the handler can overwrite without asking
    again.

    ``width`` / ``height`` are either both ``None`` (= keep source
    resolution) or both positive ints (= resize to that exact size
    via OIIO Lanczos). Same contract as the Export dialog so the
    user's mental model carries over.

    The HUD / brackets / decorative overlays are always excluded
    from the capture — they're UI chrome, not content. Only
    annotations and the A/B compare overlay have user-visible
    toggles (reviewer-authored / live-blend content the user may or
    may not want baked in).
    """

    path: Path
    fmt: str  # extension without dot, e.g. "png"
    with_annotations: bool
    bake_compare: bool
    width: int | None = None
    height: int | None = None


class SaveFrameDialog(QDialog):
    """Compact dialog: filename + format + resolution + toggles."""

    def __init__(
        self,
        *,
        suggested_filename: str,
        suggested_dir: Path,
        source_width: int,
        source_height: int,
        last_format: str = _DEFAULT_FORMAT,
        last_with_annotations: bool = True,
        last_bake_compare: bool = True,
        last_width: int | None = None,
        last_height: int | None = None,
        compare_active: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save Frame As…")
        self.setModal(True)
        # Compact dialog: same vertical rhythm as the rest of the app's
        # modals. We don't enforce a fixed width — the QLineEdit grows
        # with the file path, which is what the user usually wants.
        self.setMinimumWidth(460)

        self._suggested_filename = suggested_filename
        self._dir = Path(suggested_dir)
        if not self._dir.exists() or not self._dir.is_dir():
            # Fallback to the user's home directory if the suggested
            # directory has vanished (sequence on a now-unmounted
            # network share, etc.).
            self._dir = Path.home()

        self._source_w = max(1, int(source_width))
        self._source_h = max(1, int(source_height))

        # ---- Filename + dir ----
        self._filename_edit = QLineEdit(suggested_filename)
        # Browse button picks a different directory; the filename
        # field stays editable on its own so the user can rename
        # without clicking Browse.
        self._browse_btn = QPushButton("Browse…")
        self._browse_btn.clicked.connect(self._on_browse)
        # Compact dir display — just the trailing path component so
        # long absolute paths don't blow out the dialog width. Hover
        # tooltip shows the full path.
        self._dir_label = QLineEdit(str(self._dir))
        self._dir_label.setReadOnly(True)
        self._dir_label.setToolTip(str(self._dir))
        # Visual cue: read-only field looks slightly recessed against
        # the editable filename field above.
        self._dir_label.setStyleSheet("background: #15171B; color: #9098A4;")

        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.addWidget(self._dir_label, 1)
        dir_row.addWidget(self._browse_btn)
        dir_row_widget = QWidget()
        dir_row_widget.setLayout(dir_row)

        # ---- Format dropdown ----
        self._format_combo = QComboBox()
        for ext, label in FORMATS:
            self._format_combo.addItem(label, ext)
        # Restore last-used format if still in the list, else default.
        idx = self._format_combo.findData(last_format)
        if idx < 0:
            idx = self._format_combo.findData(_DEFAULT_FORMAT)
        self._format_combo.setCurrentIndex(max(0, idx))

        # ---- Resolution (mirror Export dialog UX) -------------------
        # Same widget layout + same RESOLUTION_PRESETS tuple as the
        # Export dialog so users transitioning between the two don't
        # have to relearn the controls. "Source" keeps the source's
        # native dimensions; named presets snap to common review
        # sizes; "Custom…" enables the W/H spinboxes for a manual
        # pick. "Lock aspect" mirrors the source W:H ratio while the
        # user types Custom values.
        res_box = QGroupBox("Resolution")
        res_form = QFormLayout(res_box)
        self._res_combo = QComboBox()
        for label, _w, _h in RESOLUTION_PRESETS:
            self._res_combo.addItem(label)
        res_form.addRow("Preset:", self._res_combo)
        wh_row = QHBoxLayout()
        self._width_spin = QSpinBox()
        self._width_spin.setRange(2, 16_384)
        self._width_spin.setSingleStep(2)
        self._height_spin = QSpinBox()
        self._height_spin.setRange(2, 16_384)
        self._height_spin.setSingleStep(2)
        wh_row.addWidget(QLabel("W:"))
        wh_row.addWidget(self._width_spin)
        wh_row.addSpacing(8)
        wh_row.addWidget(QLabel("H:"))
        wh_row.addWidget(self._height_spin)
        wh_row.addSpacing(8)
        self._lock_aspect_chk = QCheckBox("Lock aspect")
        self._lock_aspect_chk.setToolTip(
            "Keep the current W:H ratio while editing one of the "
            "Custom dimensions. The ratio is captured at the moment "
            "the box is checked.",
        )
        wh_row.addWidget(self._lock_aspect_chk)
        wh_row.addStretch(1)
        res_form.addRow(wh_row)

        # Reference aspect — locked at construction; tracked separately
        # so the lock-aspect feedback loop doesn't ping-pong both
        # spinboxes (the inner ``setValue`` would re-emit
        # ``valueChanged`` — the ``_aspect_lock_busy`` guard breaks
        # that). Recomputed only when the user toggles Lock aspect on.
        self._aspect_ratio: float = (
            (self._source_w / self._source_h)
            if self._source_h > 0 else 1.0
        )
        self._aspect_lock_busy: bool = False

        # ---- Toggles ----
        # Annotations are content the reviewer may want to bake in;
        # the bottom HUD / brackets / channel labels are UI chrome
        # and always excluded from the capture (= never useful in a
        # saved snapshot — that's what the screenshot tool is for).
        self._annotations_check = QCheckBox("Include annotations")
        self._annotations_check.setChecked(bool(last_with_annotations))
        self._annotations_check.setToolTip(
            "Bake the on-screen freehand strokes into the saved image.",
        )
        # A/B compare overlay — only meaningful (and only surfaced)
        # when the live wipe is active. With the box checked, the
        # capture keeps whatever blend / wipe / opacity is on screen.
        # Unchecked, the handler renders the underlying composite —
        # useful for delivering a clean plate even though the user is
        # mid-review.
        self._bake_compare_check = QCheckBox("Bake compare overlay")
        self._bake_compare_check.setChecked(bool(last_bake_compare))
        self._bake_compare_check.setToolTip(
            "Save the active A/B wipe / blend. Untick to capture the "
            "active sequence's composite without the overlay.",
        )
        self._bake_compare_check.setVisible(bool(compare_active))
        self._compare_active = bool(compare_active)

        # ---- Buttons ----
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel,
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        # ---- Layout ----
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        form.addRow("Filename:", self._filename_edit)
        form.addRow("Folder:", dir_row_widget)
        form.addRow("Format:", self._format_combo)
        form.addRow("", self._annotations_check)
        if self._compare_active:
            form.addRow("", self._bake_compare_check)

        root = QVBoxLayout(self)
        root.addLayout(form)
        root.addWidget(res_box)
        root.addSpacing(8)
        root.addWidget(buttons)

        # ---- Wire resolution interactions (after layout so the
        # ``_on_res_preset`` call below sees the spinboxes mounted) --
        self._res_combo.currentIndexChanged.connect(self._on_res_preset)
        self._width_spin.valueChanged.connect(self._on_width_changed)
        self._height_spin.valueChanged.connect(self._on_height_changed)
        self._lock_aspect_chk.toggled.connect(self._on_lock_aspect_toggled)

        # Restore last-used resolution choice. None/None → Source.
        # Otherwise match a known preset; if no preset matches the
        # exact W×H, fall back to Custom with those dims.
        self._init_resolution(last_width, last_height)

    # ------------------------------------------------------------------ Init helpers

    def _init_resolution(
        self, last_width: int | None, last_height: int | None,
    ) -> None:
        """Pre-fill the resolution combo + spinboxes from prefs.

        Treats both ``None`` as "Source"; otherwise tries to match a
        named preset on the stored dims and falls back to Custom.
        """
        if last_width is None or last_height is None:
            self._res_combo.setCurrentIndex(0)  # Source
            # Force the preset's side-effects (spin values + enabled)
            # by re-firing the slot — setCurrentIndex on an already-
            # selected index is a no-op so we call directly.
            self._on_res_preset(0)
            return
        for i, (_label, w, h) in enumerate(RESOLUTION_PRESETS):
            if w == last_width and h == last_height:
                self._res_combo.setCurrentIndex(i)
                return
        # Custom path — write the stored dims into the spinboxes and
        # select the Custom preset so they're editable.
        custom_idx = len(RESOLUTION_PRESETS) - 1
        self._res_combo.setCurrentIndex(custom_idx)
        self._width_spin.setValue(int(last_width))
        self._height_spin.setValue(int(last_height))

    # ------------------------------------------------------------------ Slots

    def _on_browse(self) -> None:
        """Pick a destination directory.

        Filename is kept independently editable in the line above so
        the user can hit Browse to change folder without losing the
        suggested filename.
        """
        picked = QFileDialog.getExistingDirectory(
            self,
            "Save frame to folder",
            str(self._dir),
        )
        if not picked:
            return
        self._dir = Path(picked)
        self._dir_label.setText(str(self._dir))
        self._dir_label.setToolTip(str(self._dir))

    def _on_res_preset(self, idx: int) -> None:
        """Drive the W/H spins from the preset combo.

        * **Source** → fill source dims, disable spins.
        * **Named preset** → fill preset dims, disable spins.
        * **Custom…** → leave spins editable (no auto-fill).
        """
        custom_idx = len(RESOLUTION_PRESETS) - 1
        if idx == 0:  # Source
            self._width_spin.setValue(self._source_w)
            self._height_spin.setValue(self._source_h)
            self._width_spin.setEnabled(False)
            self._height_spin.setEnabled(False)
        elif idx == custom_idx:
            self._width_spin.setEnabled(True)
            self._height_spin.setEnabled(True)
        else:
            _label, w, h = RESOLUTION_PRESETS[idx]
            self._width_spin.setValue(int(w) if w else self._source_w)
            self._height_spin.setValue(int(h) if h else self._source_h)
            self._width_spin.setEnabled(False)
            self._height_spin.setEnabled(False)
        # Lock-aspect only matters in Custom mode — disable the box
        # elsewhere so its state can't influence presets.
        self._lock_aspect_chk.setEnabled(idx == custom_idx)

    def _on_lock_aspect_toggled(self, on: bool) -> None:
        """Capture the current W:H as the reference ratio when the
        user checks the box. The box is unchecked at construction so
        this only fires on user interaction."""
        if not on:
            return
        w = self._width_spin.value()
        h = self._height_spin.value()
        if w > 0 and h > 0:
            self._aspect_ratio = w / h

    def _on_width_changed(self, value: int) -> None:
        """Mirror W → H using the locked aspect ratio.

        Skipped when (a) the box isn't checked, (b) we're not in
        Custom mode (other presets disable the spins anyway, but a
        programmatic ``setValue`` from preset selection would also
        fire valueChanged), or (c) we're already inside an aspect
        update — the inner ``setValue`` would re-emit and ping-pong.
        """
        if self._aspect_lock_busy:
            return
        if not self._lock_aspect_chk.isChecked():
            return
        if self._res_combo.currentIndex() != len(RESOLUTION_PRESETS) - 1:
            return
        if self._aspect_ratio <= 0:
            return
        new_h = max(2, round(value / self._aspect_ratio))
        self._aspect_lock_busy = True
        try:
            self._height_spin.setValue(new_h)
        finally:
            self._aspect_lock_busy = False

    def _on_height_changed(self, value: int) -> None:
        """Mirror H → W. Symmetric to :meth:`_on_width_changed`."""
        if self._aspect_lock_busy:
            return
        if not self._lock_aspect_chk.isChecked():
            return
        if self._res_combo.currentIndex() != len(RESOLUTION_PRESETS) - 1:
            return
        if self._aspect_ratio <= 0:
            return
        new_w = max(2, round(value * self._aspect_ratio))
        self._aspect_lock_busy = True
        try:
            self._width_spin.setValue(new_w)
        finally:
            self._aspect_lock_busy = False

    # ------------------------------------------------------------------ Public API

    def settings(self) -> SaveFrameSettings:
        """Compose the user's choices into a :class:`SaveFrameSettings`.

        Strips any extension from the filename field (the format
        combo is the source of truth) and re-appends the chosen one,
        so a user typing ``slate.jpg`` then picking PNG still gets
        ``slate.png``. Empty filenames fall back to the suggested
        default to avoid producing files named just ``.png``.

        Resolution is normalised so the handler can branch on a
        single ``None`` check: Source preset → ``(None, None)``,
        every other preset / Custom → positive integers.
        """
        ext = self._format_combo.currentData() or _DEFAULT_FORMAT
        stem = self._filename_edit.text().strip()
        if not stem:
            stem = self._suggested_filename
        # Drop any extension the user typed manually — the combo wins.
        stem = Path(stem).stem
        path = self._dir / f"{stem}.{ext}"

        res_idx = self._res_combo.currentIndex()
        if res_idx == 0:  # Source
            width: int | None = None
            height: int | None = None
        elif res_idx == len(RESOLUTION_PRESETS) - 1:  # Custom
            width = int(self._width_spin.value())
            height = int(self._height_spin.value())
        else:
            _label, w, h = RESOLUTION_PRESETS[res_idx]
            width = int(w) if w else None
            height = int(h) if h else None

        return SaveFrameSettings(
            path=path,
            fmt=ext,
            with_annotations=self._annotations_check.isChecked(),
            # Always read the checkbox's actual state, even when the
            # row is hidden (= compare wasn't active when the dialog
            # opened). The hidden-but-initialised checkbox carries
            # the user's last-saved choice through __init__'s
            # ``setChecked(last_bake_compare)`` call, so reading it
            # here keeps the persisted pref stable across "open Save
            # Frame while compare is off" sessions. Forcing True
            # here would silently overwrite the user's preference
            # every time the dialog ran without the row visible.
            bake_compare=self._bake_compare_check.isChecked(),
            width=width,
            height=height,
        )
