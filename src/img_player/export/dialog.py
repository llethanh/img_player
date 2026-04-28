"""The :class:`ExportDialog` — user-facing :class:`QDialog`.

Layout (top → bottom):

* Output folder picker.
* Format radio (Image sequence / Video) + format dropdown.
* Range fields (in / out) + "Use full range" button.
* Resolution preset + width/height fields.
* Frame rate preset + custom field (video only).
* Color: "Apply display transform" checkbox.
* Annotations: "Bake" + "Copy sidecar" checkboxes.
* Advanced section (collapsible): JPG quality, EXR compression,
  Video CRF, ProRes profile, H.26x preset.
* "Estimated size" label.
* Buttons: Cancel / Export.

Accept produces an :class:`ExportSettings`.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from img_player.export.settings import (
    AVAILABLE_IMAGE_FORMATS,
    AVAILABLE_VIDEO_FORMATS,
    EXR_COMPRESSIONS,
    FPS_PRESETS,
    PRORES_PROFILES,
    RESOLUTION_PRESETS,
    ExportFormat,
    ExportFormatKind,
    ExportSettings,
    estimate_size_bytes,
    format_bytes,
    format_by_key,
)

log = logging.getLogger(__name__)


class ExportDialog(QDialog):  # type: ignore[misc]
    """Modal dialog. Use :meth:`get_settings` after :meth:`exec`."""

    def __init__(
        self,
        *,
        in_frame: int,
        out_frame: int,
        source_in_frame: int,
        source_out_frame: int,
        source_width: int,
        source_height: int,
        source_fps: float,
        initial_settings: ExportSettings | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export sequence")
        self.setModal(True)
        self.setMinimumWidth(520)
        # The global QSS doesn't style QRadioButton — Qt's default
        # indicator is a faint hollow circle that disappears against
        # the dark dialog background. Inject visible radio styling
        # scoped to this dialog so the user can tell which option
        # is selected. Cyan (#4A8DE8) matches the accent used by the
        # ephemeral mode toolbar — cohérent across the app.
        self.setStyleSheet(
            "QRadioButton { spacing: 8px; padding: 2px 0; }"
            "QRadioButton::indicator {"
            "  width: 14px; height: 14px;"
            "  border-radius: 8px;"
            "  border: 2px solid #5A5A5E;"
            "  background: #1B1B1F;"
            "}"
            "QRadioButton::indicator:hover {"
            "  border-color: #8A8A8E;"
            "}"
            "QRadioButton::indicator:checked {"
            "  background: qradialgradient("
            "    cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,"
            "    stop:0 #4A8DE8, stop:0.55 #4A8DE8,"
            "    stop:0.6 #1B1B1F, stop:1 #1B1B1F);"
            "  border-color: #4A8DE8;"
            "}"
        )

        self._source_w = max(1, int(source_width))
        self._source_h = max(1, int(source_height))
        self._source_fps = max(1.0, float(source_fps))
        self._source_in = int(source_in_frame)
        self._source_out = int(source_out_frame)

        # Build the working ``ExportSettings`` we mutate as the user
        # tweaks fields. Seed from ``initial_settings`` if any.
        if initial_settings is not None:
            self._settings = initial_settings.with_changes(
                in_frame=int(in_frame),
                out_frame=int(out_frame),
            )
        else:
            self._settings = ExportSettings(
                output_dir=Path.home(),
                in_frame=int(in_frame),
                out_frame=int(out_frame),
            )

        self._build_ui()
        self._load_state_into_widgets()
        self._refresh_estimate()

    # ------------------------------------------------------------------ Public

    def get_settings(self) -> ExportSettings:
        """Snapshot the widgets into a fresh :class:`ExportSettings`."""
        return self._collect_settings()

    # ------------------------------------------------------------------ UI build

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # --- Output --------------------------------------------------
        out_box = QGroupBox("Output")
        out_form = QFormLayout(out_box)
        self._out_dir_edit = QLineEdit()
        self._out_dir_edit.setPlaceholderText("/path/to/export/folder")
        # Editable on purpose — typing a path is faster than the
        # browse dialog when the user already knows where they want
        # the export to land. The Export button validates that the
        # path resolves to a writable directory before launching.
        self._out_dir_edit.textChanged.connect(self._refresh_estimate)
        out_btn = QPushButton("Browse…")
        out_btn.clicked.connect(self._browse_output_dir)
        out_row = QHBoxLayout()
        out_row.addWidget(self._out_dir_edit, 1)
        out_row.addWidget(out_btn)
        out_form.addRow("Folder:", out_row)
        self._start_frame_spin = QSpinBox()
        self._start_frame_spin.setRange(0, 9_999_999)
        out_form.addRow("Start frame:", self._start_frame_spin)
        root.addWidget(out_box)

        # --- Format --------------------------------------------------
        fmt_box = QGroupBox("Format")
        fmt_layout = QVBoxLayout(fmt_box)
        radio_row = QHBoxLayout()
        self._radio_imgseq = QRadioButton("Image sequence")
        self._radio_video = QRadioButton("Video")
        self._radio_group = QButtonGroup(self)
        self._radio_group.addButton(self._radio_imgseq, 0)
        self._radio_group.addButton(self._radio_video, 1)
        radio_row.addWidget(self._radio_imgseq)
        radio_row.addWidget(self._radio_video)
        radio_row.addStretch(1)
        fmt_layout.addLayout(radio_row)

        self._format_combo = QComboBox()
        fmt_layout.addWidget(self._format_combo)
        self._format_desc = QLabel("")
        self._format_desc.setWordWrap(True)
        self._format_desc.setStyleSheet("color: #8A8A8E; font-size: 11px;")
        fmt_layout.addWidget(self._format_desc)
        root.addWidget(fmt_box)

        # --- Range ---------------------------------------------------
        range_box = QGroupBox("Range")
        range_form = QFormLayout(range_box)
        range_row = QHBoxLayout()
        self._in_spin = QSpinBox()
        self._in_spin.setRange(-9_999_999, 9_999_999)
        self._out_spin = QSpinBox()
        self._out_spin.setRange(-9_999_999, 9_999_999)
        full_btn = QPushButton("Use full range")
        full_btn.clicked.connect(self._use_full_range)
        range_row.addWidget(QLabel("From:"))
        range_row.addWidget(self._in_spin)
        range_row.addSpacing(8)
        range_row.addWidget(QLabel("To:"))
        range_row.addWidget(self._out_spin)
        range_row.addStretch(1)
        range_row.addWidget(full_btn)
        range_form.addRow(range_row)
        root.addWidget(range_box)

        # --- Resolution ----------------------------------------------
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
        wh_row.addStretch(1)
        res_form.addRow(wh_row)
        root.addWidget(res_box)

        # --- FPS (video only) ----------------------------------------
        fps_box = QGroupBox("Frame rate (video)")
        fps_form = QFormLayout(fps_box)
        self._fps_combo = QComboBox()
        for label, _v in FPS_PRESETS:
            self._fps_combo.addItem(label)
        fps_form.addRow("Preset:", self._fps_combo)
        self._fps_spin = QLineEdit()
        self._fps_spin.setPlaceholderText("e.g. 24")
        fps_form.addRow("Custom:", self._fps_spin)
        self._fps_box = fps_box
        root.addWidget(fps_box)

        # --- Color ---------------------------------------------------
        col_box = QGroupBox("Color")
        col_layout = QVBoxLayout(col_box)
        self._display_xform_chk = QCheckBox(
            "Apply display transform (bake what you see in the viewer)"
        )
        col_layout.addWidget(self._display_xform_chk)
        col_hint = QLabel(
            "Off → write linear / passthrough buffer. Recommended for EXR + TIFF."
        )
        col_hint.setStyleSheet("color: #8A8A8E; font-size: 11px;")
        col_hint.setWordWrap(True)
        col_layout.addWidget(col_hint)
        root.addWidget(col_box)

        # --- Annotations ---------------------------------------------
        ann_box = QGroupBox("Annotations")
        ann_layout = QVBoxLayout(ann_box)
        self._bake_chk = QCheckBox("Bake annotations into output")
        self._bake_chk.toggled.connect(self._on_bake_toggled)
        ann_layout.addWidget(self._bake_chk)
        self._copy_sidecar_chk = QCheckBox(
            "Also copy annotations sidecar (.json) next to the export"
        )
        ann_layout.addWidget(self._copy_sidecar_chk)
        root.addWidget(ann_box)

        # --- Advanced (collapsible) ---------------------------------
        self._advanced_btn = QToolButton()
        self._advanced_btn.setText("▶  Advanced")
        self._advanced_btn.setCheckable(True)
        self._advanced_btn.setStyleSheet("QToolButton { border: none; }")
        self._advanced_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._advanced_btn.toggled.connect(self._toggle_advanced)
        root.addWidget(self._advanced_btn)

        self._advanced_panel = QFrame()
        adv_form = QFormLayout(self._advanced_panel)
        self._jpg_quality_spin = QSpinBox()
        self._jpg_quality_spin.setRange(1, 100)
        adv_form.addRow("JPG quality:", self._jpg_quality_spin)
        self._exr_compression_combo = QComboBox()
        for c in EXR_COMPRESSIONS:
            self._exr_compression_combo.addItem(c)
        adv_form.addRow("EXR compression:", self._exr_compression_combo)
        self._video_crf_spin = QSpinBox()
        self._video_crf_spin.setRange(0, 51)
        adv_form.addRow("Video CRF (lower=better):", self._video_crf_spin)
        self._prores_profile_combo = QComboBox()
        for label, value in PRORES_PROFILES:
            self._prores_profile_combo.addItem(label, value)
        adv_form.addRow("ProRes profile:", self._prores_profile_combo)
        self._h26x_preset_combo = QComboBox()
        for p in (
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        ):
            self._h26x_preset_combo.addItem(p)
        adv_form.addRow("H.264/H.265 preset:", self._h26x_preset_combo)
        self._advanced_panel.setVisible(False)
        root.addWidget(self._advanced_panel)

        # --- Estimate label ------------------------------------------
        self._estimate_label = QLabel("Estimated size: —")
        self._estimate_label.setStyleSheet(
            "color: #B5B5B8; font-size: 11px; padding-top: 6px;"
        )
        root.addWidget(self._estimate_label)

        # --- Buttons -------------------------------------------------
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        btns.accepted.connect(self._on_accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

        # --- Wire change-tracking signals ----------------------------
        self._radio_group.idClicked.connect(lambda _i: self._on_kind_changed())
        self._format_combo.currentIndexChanged.connect(self._on_format_changed)
        self._res_combo.currentIndexChanged.connect(self._on_res_preset)
        self._fps_combo.currentIndexChanged.connect(self._on_fps_preset)
        # Re-estimate on every numeric tweak.
        for w in (
            self._in_spin, self._out_spin, self._width_spin, self._height_spin,
            self._start_frame_spin, self._jpg_quality_spin, self._video_crf_spin,
        ):
            w.valueChanged.connect(self._refresh_estimate)
        self._fps_spin.textChanged.connect(self._refresh_estimate)

    # ------------------------------------------------------------------ State sync

    def _load_state_into_widgets(self) -> None:
        s = self._settings
        # Output
        self._out_dir_edit.setText(str(s.output_dir))
        self._start_frame_spin.setValue(int(s.start_frame))
        # Format radio + dropdown
        is_video = s.is_video
        self._radio_video.setChecked(is_video)
        self._radio_imgseq.setChecked(not is_video)
        self._populate_format_combo(is_video)
        # Try to select s.format_key in the combo.
        for i in range(self._format_combo.count()):
            if self._format_combo.itemData(i) == s.format_key:
                self._format_combo.setCurrentIndex(i)
                break
        # Range
        self._in_spin.setValue(int(s.in_frame))
        self._out_spin.setValue(int(s.out_frame))
        # Resolution
        if s.width is None or s.height is None:
            self._res_combo.setCurrentIndex(0)  # "Source"
            self._width_spin.setValue(self._source_w)
            self._height_spin.setValue(self._source_h)
            self._width_spin.setEnabled(False)
            self._height_spin.setEnabled(False)
        else:
            # Find a matching preset, else "Custom".
            matched = False
            for i, (_label, w, h) in enumerate(RESOLUTION_PRESETS):
                if w == s.width and h == s.height:
                    self._res_combo.setCurrentIndex(i)
                    matched = True
                    break
            if not matched:
                # "Custom…" is the last preset
                self._res_combo.setCurrentIndex(len(RESOLUTION_PRESETS) - 1)
            self._width_spin.setValue(int(s.width))
            self._height_spin.setValue(int(s.height))
            self._width_spin.setEnabled(self._res_combo.currentIndex() == len(RESOLUTION_PRESETS) - 1)
            self._height_spin.setEnabled(self._width_spin.isEnabled())
        # FPS
        if s.fps is None:
            self._fps_combo.setCurrentIndex(0)  # "Source"
            self._fps_spin.setText("")
            self._fps_spin.setEnabled(False)
        else:
            matched = False
            for i, (_label, v) in enumerate(FPS_PRESETS):
                if v is not None and abs(v - s.fps) < 1e-3:
                    self._fps_combo.setCurrentIndex(i)
                    matched = True
                    break
            if not matched:
                self._fps_combo.setCurrentIndex(len(FPS_PRESETS) - 1)
            self._fps_spin.setText(str(s.fps))
            self._fps_spin.setEnabled(self._fps_combo.currentIndex() == len(FPS_PRESETS) - 1)
        # Color
        self._display_xform_chk.setChecked(s.apply_display_transform)
        # Annotations
        self._bake_chk.setChecked(s.bake_annotations)
        self._copy_sidecar_chk.setChecked(s.copy_sidecar)
        self._copy_sidecar_chk.setEnabled(s.bake_annotations)
        # Advanced
        self._jpg_quality_spin.setValue(int(s.jpg_quality))
        idx = self._exr_compression_combo.findText(s.exr_compression)
        if idx >= 0:
            self._exr_compression_combo.setCurrentIndex(idx)
        self._video_crf_spin.setValue(int(s.video_crf))
        for i in range(self._prores_profile_combo.count()):
            if self._prores_profile_combo.itemData(i) == s.prores_profile:
                self._prores_profile_combo.setCurrentIndex(i)
                break
        idx = self._h26x_preset_combo.findText(s.h26x_preset)
        if idx >= 0:
            self._h26x_preset_combo.setCurrentIndex(idx)

        # Show/hide FPS box per kind.
        self._fps_box.setVisible(self._radio_video.isChecked())

    def _collect_settings(self) -> ExportSettings:
        # Build a fresh ExportSettings from the widget state.
        is_video = self._radio_video.isChecked()
        format_key = self._format_combo.currentData() or (
            AVAILABLE_VIDEO_FORMATS[0].key if is_video else AVAILABLE_IMAGE_FORMATS[0].key
        )
        # Resolution
        res_idx = self._res_combo.currentIndex()
        if res_idx == 0:  # Source
            width = None
            height = None
        elif res_idx == len(RESOLUTION_PRESETS) - 1:  # Custom
            width = int(self._width_spin.value())
            height = int(self._height_spin.value())
        else:
            _label, w, h = RESOLUTION_PRESETS[res_idx]
            width = int(w) if w else None
            height = int(h) if h else None
        # FPS
        fps_idx = self._fps_combo.currentIndex()
        if fps_idx == 0:
            fps = None
        elif fps_idx == len(FPS_PRESETS) - 1:
            try:
                fps = float(self._fps_spin.text().strip().replace(",", "."))
            except ValueError:
                fps = None
        else:
            fps = FPS_PRESETS[fps_idx][1]

        return ExportSettings(
            output_dir=Path(self._out_dir_edit.text().strip() or "."),
            start_frame=int(self._start_frame_spin.value()),
            format_key=str(format_key),
            in_frame=int(self._in_spin.value()),
            out_frame=int(self._out_spin.value()),
            width=width,
            height=height,
            fps=fps,
            apply_display_transform=self._display_xform_chk.isChecked(),
            bake_annotations=self._bake_chk.isChecked(),
            copy_sidecar=self._copy_sidecar_chk.isChecked(),
            jpg_quality=int(self._jpg_quality_spin.value()),
            exr_compression=str(self._exr_compression_combo.currentText()),
            video_crf=int(self._video_crf_spin.value()),
            prores_profile=int(self._prores_profile_combo.currentData() or 3),
            h26x_preset=str(self._h26x_preset_combo.currentText()),
        )

    # ------------------------------------------------------------------ Slots

    def _populate_format_combo(self, is_video: bool) -> None:
        self._format_combo.blockSignals(True)
        self._format_combo.clear()
        formats = AVAILABLE_VIDEO_FORMATS if is_video else AVAILABLE_IMAGE_FORMATS
        for fmt in formats:
            self._format_combo.addItem(fmt.label, fmt.key)
        self._format_combo.blockSignals(False)
        self._on_format_changed(0)

    def _on_kind_changed(self) -> None:
        is_video = self._radio_video.isChecked()
        self._populate_format_combo(is_video)
        self._fps_box.setVisible(is_video)
        self._refresh_estimate()

    def _on_format_changed(self, _idx: int) -> None:
        key = self._format_combo.currentData()
        if not key:
            return
        try:
            fmt: ExportFormat = format_by_key(key)
        except KeyError:
            return
        self._format_desc.setText(fmt.description)
        # Reset display-transform default per format unless the user
        # already deviates — simpler: always sync to the format
        # default. Power user can override after selection.
        self._display_xform_chk.setChecked(fmt.display_bake_default)
        self._refresh_estimate()

    def _on_res_preset(self, idx: int) -> None:
        if idx == 0:  # Source
            self._width_spin.setValue(self._source_w)
            self._height_spin.setValue(self._source_h)
            self._width_spin.setEnabled(False)
            self._height_spin.setEnabled(False)
        elif idx == len(RESOLUTION_PRESETS) - 1:  # Custom
            self._width_spin.setEnabled(True)
            self._height_spin.setEnabled(True)
        else:
            _label, w, h = RESOLUTION_PRESETS[idx]
            self._width_spin.setValue(int(w) if w else self._source_w)
            self._height_spin.setValue(int(h) if h else self._source_h)
            self._width_spin.setEnabled(False)
            self._height_spin.setEnabled(False)
        self._refresh_estimate()

    def _on_fps_preset(self, idx: int) -> None:
        if idx == len(FPS_PRESETS) - 1:
            self._fps_spin.setEnabled(True)
        else:
            self._fps_spin.setEnabled(False)
            if idx > 0:
                _label, v = FPS_PRESETS[idx]
                self._fps_spin.setText(str(v))
        self._refresh_estimate()

    def _on_bake_toggled(self, checked: bool) -> None:
        self._copy_sidecar_chk.setEnabled(checked)
        if not checked:
            # Sidecar copy makes no sense without bake (the export IS
            # the sidecar use case in that mode).
            self._copy_sidecar_chk.setChecked(False)

    def _toggle_advanced(self, checked: bool) -> None:
        self._advanced_panel.setVisible(checked)
        self._advanced_btn.setText("▼  Advanced" if checked else "▶  Advanced")

    def _use_full_range(self) -> None:
        self._in_spin.setValue(int(self._source_in))
        self._out_spin.setValue(int(self._source_out))

    def _browse_output_dir(self) -> None:
        start = self._out_dir_edit.text().strip() or str(Path.home())
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose export output folder", start,
        )
        if chosen:
            self._out_dir_edit.setText(chosen)
            self._refresh_estimate()

    def _refresh_estimate(self) -> None:
        try:
            settings = self._collect_settings()
        except Exception:
            self._estimate_label.setText("Estimated size: —")
            return
        size = estimate_size_bytes(
            settings,
            source_w=self._source_w,
            source_h=self._source_h,
            source_fps=self._source_fps,
        )
        self._estimate_label.setText(
            f"Estimated size: {format_bytes(size)} · {settings.total_frames} frames"
        )

    # ------------------------------------------------------------------ Accept

    def _on_accept(self) -> None:
        try:
            settings = self._collect_settings()
            settings.validate()
            # Output dir must exist or be creatable.
            settings.output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as err:  # noqa: BLE001 — surface every error to the user
            from PySide6.QtWidgets import QMessageBox  # local import for test envs
            QMessageBox.warning(self, "Invalid export settings", str(err))
            return
        self._settings = settings
        self.accept()
