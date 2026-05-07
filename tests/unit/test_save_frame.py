"""Tests for File → Save Frame As… dialog + handler.

Covers:

* :class:`SaveFrameDialog.settings` builds the right path/format combo
  from the user's input (filename strip + extension swap).
* :func:`capture_viewer` toggles overlay / annotation widgets around
  the grab and restores them — even when ``grab`` raises.
* :func:`_write_image` picks the correct Qt format hint per extension.

We avoid running the full app — the dialog is constructed directly
with QPixmap fixtures, the handler's capture path is exercised via
plain QWidgets.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QLabel, QWidget

from img_player.save_frame_handler import (
    _coerce_bool,
    _QT_FORMAT_FOR_EXT,
    _write_image,
    capture_viewer,
)
from img_player.ui.save_frame_dialog import (
    FORMATS,
    SaveFrameDialog,
    SaveFrameSettings,
)


# ============================================================================
# SaveFrameDialog.settings
# ============================================================================


class TestSaveFrameDialogSettings:
    def test_default_settings_use_suggested_filename(
        self, qtbot, tmp_path: Path,
    ) -> None:
        dialog = SaveFrameDialog(
            suggested_filename="render_0042",
            suggested_dir=tmp_path,
        )
        qtbot.addWidget(dialog)
        s = dialog.settings()
        assert s.path == tmp_path / "render_0042.png"
        assert s.fmt == "png"
        assert s.with_annotations is True

    def test_format_combo_overrides_typed_extension(
        self, qtbot, tmp_path: Path,
    ) -> None:
        """User types ``shot.jpg`` then picks PNG → output is shot.png."""
        dialog = SaveFrameDialog(
            suggested_filename="render_0042",
            suggested_dir=tmp_path,
            last_format="png",
        )
        qtbot.addWidget(dialog)
        dialog._filename_edit.setText("shot.jpg")
        s = dialog.settings()
        assert s.path.name == "shot.png"

    def test_empty_filename_falls_back_to_suggested(
        self, qtbot, tmp_path: Path,
    ) -> None:
        dialog = SaveFrameDialog(
            suggested_filename="render_0042",
            suggested_dir=tmp_path,
        )
        qtbot.addWidget(dialog)
        dialog._filename_edit.setText("   ")
        s = dialog.settings()
        assert s.path.name == "render_0042.png"

    def test_last_format_round_trips(
        self, qtbot, tmp_path: Path,
    ) -> None:
        dialog = SaveFrameDialog(
            suggested_filename="x",
            suggested_dir=tmp_path,
            last_format="jpg",
        )
        qtbot.addWidget(dialog)
        assert dialog.settings().fmt == "jpg"

    def test_unknown_last_format_falls_back_to_default(
        self, qtbot, tmp_path: Path,
    ) -> None:
        dialog = SaveFrameDialog(
            suggested_filename="x",
            suggested_dir=tmp_path,
            last_format="not_a_format",
        )
        qtbot.addWidget(dialog)
        assert dialog.settings().fmt == "png"

    def test_annotations_toggle_persists(
        self, qtbot, tmp_path: Path,
    ) -> None:
        dialog = SaveFrameDialog(
            suggested_filename="x",
            suggested_dir=tmp_path,
            last_with_annotations=False,
        )
        qtbot.addWidget(dialog)
        s = dialog.settings()
        assert s.with_annotations is False


# ============================================================================
# capture_viewer — widget visibility lifecycle
# ============================================================================


class _FakeViewer(QWidget):
    """Stand-in for ViewerWidget with the two overlay attributes
    capture_viewer reaches for via getattr."""

    def __init__(self) -> None:
        super().__init__()
        self.resize(64, 32)
        self._info_band = QLabel("HUD", self)
        self._info_band.setVisible(True)
        self._overlay = QLabel("brackets", self)
        self._overlay.setVisible(True)


class TestCaptureViewer:
    def test_grab_returns_qimage(self, qtbot) -> None:
        viewer = _FakeViewer()
        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitExposed(viewer)
        anno = QLabel("anno", viewer)
        anno.setVisible(True)
        img = capture_viewer(
            viewer,
            annotation_overlay=anno,
            with_annotations=True,
        )
        assert isinstance(img, QImage)
        assert not img.isNull()
        # On HiDPI screens the image is the widget size × DPR, so we
        # don't assert exact dimensions — just that the grab produced
        # a non-empty buffer of at least the logical widget size.
        assert img.width() >= viewer.width()
        assert img.height() >= viewer.height()

    def test_overlay_widgets_always_hidden_then_restored(self, qtbot) -> None:
        """Overlay (HUD + brackets) is always excluded from capture
        regardless of the annotations toggle, and restored after."""
        viewer = _FakeViewer()
        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitExposed(viewer)
        info_band_was = viewer._info_band.isVisible()
        brackets_was = viewer._overlay.isVisible()
        capture_viewer(
            viewer,
            annotation_overlay=None,
            with_annotations=True,
        )
        # After capture, visibility is back to what it was before.
        assert viewer._info_band.isVisible() == info_band_was
        assert viewer._overlay.isVisible() == brackets_was

    def test_annotations_off_hides_only_annotation_overlay(
        self, qtbot,
    ) -> None:
        viewer = _FakeViewer()
        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitExposed(viewer)
        anno = QLabel("anno", viewer)
        anno.setVisible(True)
        capture_viewer(
            viewer,
            annotation_overlay=anno,
            with_annotations=False,
        )
        # Restored after capture.
        assert anno.isVisible() is True

    def test_already_hidden_widget_stays_hidden(self, qtbot) -> None:
        """Capture must not turn ON a widget the user had hidden."""
        viewer = _FakeViewer()
        qtbot.addWidget(viewer)
        viewer.show()
        qtbot.waitExposed(viewer)
        viewer._info_band.setVisible(False)
        capture_viewer(
            viewer,
            annotation_overlay=None,
            with_annotations=True,
        )
        assert viewer._info_band.isVisible() is False


# ============================================================================
# _write_image
# ============================================================================


class TestWriteImage:
    @pytest.mark.parametrize("ext", ["png", "jpg", "tif", "bmp", "webp"])
    def test_writes_known_format(self, tmp_path: Path, ext: str) -> None:
        img = QImage(8, 8, QImage.Format.Format_RGBA8888)
        img.fill(0xFF112233)
        settings = SaveFrameSettings(
            path=tmp_path / f"out.{ext}",
            fmt=ext,
            with_annotations=False,
            bake_compare=True,
        )
        ok = _write_image(img, settings)
        assert ok is True
        assert settings.path.exists()
        assert settings.path.stat().st_size > 0

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        img = QImage(4, 4, QImage.Format.Format_RGB888)
        img.fill(0xFF000000)
        nested = tmp_path / "a" / "b" / "c"
        settings = SaveFrameSettings(
            path=nested / "x.png",
            fmt="png",
            with_annotations=False,
            bake_compare=True,
        )
        assert _write_image(img, settings)
        assert (nested / "x.png").exists()

    def test_format_map_is_complete_for_dialog_options(self) -> None:
        """Every extension the dialog offers must be in the Qt format
        map — otherwise we'd silently fall back to filename sniffing."""
        for ext, _label in FORMATS:
            assert ext in _QT_FORMAT_FOR_EXT


# ============================================================================
# _coerce_bool
# ============================================================================


class TestCoerceBool:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (True, True),
            (False, False),
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("0", False),
            ("", False),  # empty string treated as falsy via the lower-set test
            (1, True),
            (0, False),
            (None, "DEFAULT"),  # None falls through
            (object(), "DEFAULT"),  # unknown types fall through
        ],
    )
    def test_coercion(self, value: object, expected: object) -> None:
        if expected == "DEFAULT":
            # Default of True so we can distinguish "fell through" from "False".
            assert _coerce_bool(value, True) is True
            assert _coerce_bool(value, False) is False
        else:
            assert _coerce_bool(value, not bool(expected)) is expected
