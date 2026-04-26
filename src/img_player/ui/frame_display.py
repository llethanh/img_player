"""Editable frame / timecode field that lives between the navigation
and playback halves of the transport bar.

Behaviour:
* Reads the current frame in the playback state (set via :meth:`set_frame`)
  and renders it as either a frame number (``0042``) or a non-drop-frame
  timecode (``00:00:01:18``) depending on the global display mode set by
  the View menu.
* Click into the field, type a new value, press Enter — emits
  ``frame_seek_requested(int)``. Empty input or unparseable text is
  silently ignored.
* The user can scrub the timeline in the meantime; that updates the
  field via :meth:`set_frame`. We avoid clobbering text the user is
  currently editing by checking ``hasFocus()`` first.
"""

from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QLineEdit, QWidget

from img_player.ui.theme import F, G

DisplayMode = Literal["frames", "tc"]


def _frame_to_timecode(frame: int, fps: float) -> str:
    """Non-drop-frame timecode (HH:MM:SS:FF) for an absolute frame index.

    Mirrors :func:`img_player.ui.timeline.frame_to_timecode` deliberately
    — we duplicate it here rather than import it cross-module so the
    field stays self-contained and the timeline doesn't have to expose a
    helper to the transport.
    """
    fps_int = max(1, round(fps))
    frame = max(0, frame)
    total_seconds = frame // fps_int
    ff = frame % fps_int
    hours = total_seconds // 3600
    minutes = (total_seconds // 60) % 60
    seconds = total_seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{ff:02d}"


def _parse_user_text(text: str, mode: DisplayMode, fps: float) -> int | None:
    """Convert what the user typed into an absolute frame index.

    In ``"frames"`` mode the input is just an integer; in ``"tc"`` mode
    it can be either a full ``HH:MM:SS:FF`` or an integer (handy if the
    user is in TC mode but knows the frame number they want).
    """
    text = text.strip()
    if not text:
        return None
    # Always allow a bare integer regardless of mode.
    try:
        return int(text)
    except ValueError:
        pass
    if mode == "tc":
        parts = text.split(":")
        if len(parts) == 4:
            try:
                hh, mm, ss, ff = (int(p) for p in parts)
            except ValueError:
                return None
            fps_int = max(1, round(fps))
            return ((hh * 3600 + mm * 60 + ss) * fps_int) + ff
    return None


class FrameDisplay(QLineEdit):  # type: ignore[misc]
    """Editable frame indicator for the transport bar."""

    frame_seek_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._frame = 0
        self._fps = 24.0
        self._mode: DisplayMode = "frames"

        self.setFont(F.mono(F.SIZE_SM))
        # Width tailored to fit the longest possible TC string with a
        # bit of breathing room. We measure it via the actual font so
        # the field doesn't reflow when the mode toggles.
        metrics = QFontMetrics(self.font())
        tc_width = metrics.horizontalAdvance("88:88:88:88") + 16
        self.setFixedWidth(tc_width)
        self.setFixedHeight(G.INPUT_H)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # No focus-on-tab — keeps the field out of the keyboard
        # focus ring used by the global shortcuts (Space, J/K/L, …).
        # The user can still click in to type.
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setToolTip("Current frame — click to type a frame number or timecode")

        self.editingFinished.connect(self._on_committed)
        self._refresh_text()

    # ------------------------------------------------------------------ API

    def set_frame(self, frame: int) -> None:
        """Update the displayed frame (called by the controller wiring).

        We *always* refresh — even if the user has focus and was
        typing. Rationale: if a different source moved the playhead
        (timeline scrub, J/L shortcut, prev/next button, the
        playback loop itself), the user's mid-typed value is now a
        stale intent and should not block the new ground truth from
        appearing on screen. Effectively: the user has 1 keystroke +
        Enter to commit a frame seek; any external change cancels
        their edit.
        """
        if frame == self._frame:
            return
        self._frame = frame
        self._refresh_text()

    def set_fps(self, fps: float) -> None:
        if abs(fps - self._fps) < 1e-6:
            return
        self._fps = max(0.1, fps)
        if self._mode == "tc":
            self._refresh_text()

    def set_display_mode(self, mode: DisplayMode) -> None:
        if mode == self._mode:
            return
        self._mode = mode
        self._refresh_text()

    # ------------------------------------------------------------------ Internals

    def _refresh_text(self) -> None:
        if self._mode == "tc":
            self.setText(_frame_to_timecode(self._frame, self._fps))
        else:
            self.setText(f"{self._frame:04d}")

    def _on_committed(self) -> None:
        target = _parse_user_text(self.text(), self._mode, self._fps)
        if target is None:
            # Restore the displayed value so the field doesn't show
            # garbage after a typo.
            self._refresh_text()
            return
        # Re-render too — the controller will eventually push the
        # corrected frame back via set_frame, but this keeps the field
        # in sync immediately for the user's eye.
        self._frame = target
        self._refresh_text()
        self.frame_seek_requested.emit(target)
