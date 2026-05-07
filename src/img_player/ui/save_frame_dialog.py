"""Save Frame As… dialog — quick single-frame snapshot of the viewer.

A lightweight alternative to the full Export pipeline: WYSIWYG
capture of whatever's on the GL viewport, with two toggles
(annotations on/off, overlay on/off) and a format dropdown.

The dialog itself is pure UI — it gathers user intent and returns a
:class:`SaveFrameSettings`. The capture + write is done by
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
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

# Format catalogue: extension → human label. Order matters for the
# dropdown; the first entry is the fallback default. PNG is first
# because it's the lossless, alpha-aware default for VFX review
# screenshots; JPG second for "share quickly". TIFF + BMP + WebP
# round out the common cases. EXR is intentionally excluded — the
# capture is screen-pixel uint8, EXR's float pipeline belongs to
# the Export dialog.
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


class SaveFrameDialog(QDialog):
    """Compact dialog: filename + format + annotations/overlay toggles."""

    def __init__(
        self,
        *,
        suggested_filename: str,
        suggested_dir: Path,
        last_format: str = _DEFAULT_FORMAT,
        last_with_annotations: bool = True,
        last_bake_compare: bool = True,
        compare_active: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save Frame As…")
        self.setModal(True)
        # Compact dialog: same vertical rhythm as the rest of the app's
        # modals. We don't enforce a fixed width — the QLineEdit grows
        # with the file path, which is what the user usually wants.
        self.setMinimumWidth(420)

        self._suggested_filename = suggested_filename
        self._dir = Path(suggested_dir)
        if not self._dir.exists() or not self._dir.is_dir():
            # Fallback to the user's home directory if the suggested
            # directory has vanished (sequence on a now-unmounted
            # network share, etc.).
            self._dir = Path.home()

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
        # Unchecked, the handler temporarily disables compare and
        # snapshots the underlying composite — useful for delivering
        # a clean plate even though the user is mid-review.
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
        root.addSpacing(8)
        root.addWidget(buttons)

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

    # ------------------------------------------------------------------ Public API

    def settings(self) -> SaveFrameSettings:
        """Compose the user's choices into a :class:`SaveFrameSettings`.

        Strips any extension from the filename field (the format
        combo is the source of truth) and re-appends the chosen one,
        so a user typing ``slate.jpg`` then picking PNG still gets
        ``slate.png``. Empty filenames fall back to the suggested
        default to avoid producing files named just ``.png``.
        """
        ext = self._format_combo.currentData() or _DEFAULT_FORMAT
        stem = self._filename_edit.text().strip()
        if not stem:
            stem = self._suggested_filename
        # Drop any extension the user typed manually — the combo wins.
        stem = Path(stem).stem
        path = self._dir / f"{stem}.{ext}"
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
        )
