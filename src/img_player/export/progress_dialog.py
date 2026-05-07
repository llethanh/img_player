"""The :class:`ExportProgressDialog` — non-modal progress UI."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class ExportProgressDialog(QDialog):  # type: ignore[misc]
    """Shows a progress bar + ETA + Cancel button.

    The hosting code wires the export worker's signals to the
    :meth:`update_progress`, :meth:`on_finished`, :meth:`on_failed`,
    :meth:`on_canceled` slots."""

    def __init__(
        self,
        *,
        total_frames: int,
        output_path: Path,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Exporting…")
        self.setModal(True)
        self.setMinimumWidth(420)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self._total = max(1, int(total_frames))
        self._output_path = output_path
        self._cancel_requested = False
        self._fps_history: deque[float] = deque(maxlen=20)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        self._main_label = QLabel(f"Rendering frame 0 / {self._total}")
        self._main_label.setStyleSheet("font-weight: 600;")
        layout.addWidget(self._main_label)

        self._progress = QProgressBar()
        self._progress.setRange(0, self._total)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._stats_label = QLabel("Speed: — · ETA: —")
        self._stats_label.setStyleSheet("color: #B5B5B8; font-size: 11px;")
        layout.addWidget(self._stats_label)

        self._path_label = QLabel(f"Output: {output_path}")
        self._path_label.setStyleSheet("color: #8A8A8E; font-size: 11px;")
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._request_cancel)
        btn_row.addWidget(self._cancel_btn)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------ Slots

    def update_progress(self, current: int, total: int, fps_running: float) -> None:
        self._total = max(1, int(total))
        self._progress.setMaximum(self._total)
        self._progress.setValue(int(current))
        self._main_label.setText(f"Rendering frame {current} / {total}")
        self._fps_history.append(float(fps_running))
        avg_fps = sum(self._fps_history) / max(1, len(self._fps_history))
        remaining = max(0, total - current)
        eta_s = remaining / avg_fps if avg_fps > 1e-3 else 0.0
        self._stats_label.setText(
            f"Speed: {avg_fps:.1f} fps · ETA: {self._format_eta(eta_s)}"
        )

    # Shared styling for the three terminal states. Same big bold
    # font on the main label, only the colour + glyph changes:
    # green for success, orange for cancel, red for failure. The
    # consistency makes the panel's outcome obvious at a glance.
    _STYLE_DONE = "color: #5DC46C; font-weight: 700; font-size: 14px;"
    _STYLE_CANCELED = "color: #E8901C; font-weight: 700; font-size: 14px;"
    _STYLE_FAILED = "color: #E84A4A; font-weight: 700; font-size: 14px;"

    def on_finished(self, output_path: str, frames: int, duration_s: float) -> None:
        # Force the bar to 100 % in case the last update_progress
        # call hadn't reached ``self._total`` yet — otherwise the
        # bar reads as "still crunching" alongside the "done" label.
        self._progress.setValue(self._progress.maximum())
        self._main_label.setText(
            f"✅  Done — {frames} frames in {duration_s:.1f} s"
        )
        self._main_label.setStyleSheet(self._STYLE_DONE)
        self._stats_label.setText("")
        self._path_label.setText(f"Saved to: {output_path}")
        self._reset_close_button(connect_to=self.accept)

    def on_failed(self, message: str) -> None:
        self._main_label.setText(f"❌  Export failed: {message}")
        self._main_label.setStyleSheet(self._STYLE_FAILED)
        self._stats_label.setText("")
        self._reset_close_button(connect_to=self.reject)

    def on_canceled(self, output_path: str, frames: int) -> None:
        del output_path
        self._main_label.setText(f"⚠  Canceled after {frames} frames")
        self._main_label.setStyleSheet(self._STYLE_CANCELED)
        self._stats_label.setText("")
        self._reset_close_button(connect_to=self.reject)

    def _reset_close_button(self, *, connect_to) -> None:  # type: ignore[no-untyped-def]
        """Flip the cancel button into its terminal "Close" state.

        Re-enables the button (``_request_cancel`` may have disabled
        it while the worker was winding down) and rewires its
        ``clicked`` signal to ``connect_to`` (= the dialog's
        ``accept`` / ``reject`` slot, depending on outcome). Ensures
        the user can always close the dialog after a cancel /
        finish / fail rather than being stuck with a greyed
        button.
        """
        self._cancel_btn.setEnabled(True)
        self._cancel_btn.setText("Close")
        try:
            self._cancel_btn.clicked.disconnect()
        except (TypeError, RuntimeError):
            pass
        self._cancel_btn.clicked.connect(connect_to)

    # ------------------------------------------------------------------ Cancel

    def cancel_requested(self) -> bool:
        return self._cancel_requested

    def _request_cancel(self) -> None:
        if self._cancel_requested:
            return
        self._cancel_requested = True
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.setText("Canceling…")
        # The hosting code also connects to ``cancel_btn.clicked`` to
        # forward the cancel to the worker — that connection is set
        # up on construction by the orchestrator.

    @property
    def cancel_button(self) -> QPushButton:
        """Public accessor — the orchestrator wires its
        ``worker.cancel`` to this button's ``clicked`` signal."""
        return self._cancel_btn

    # ------------------------------------------------------------------ Helpers

    @staticmethod
    def _format_eta(seconds: float) -> str:
        if seconds <= 0:
            return "—"
        if seconds < 60:
            return f"{seconds:.0f} s"
        m, s = divmod(int(seconds), 60)
        if m < 60:
            return f"{m} m {s:02d} s"
        h, m = divmod(m, 60)
        return f"{h} h {m:02d} m"
