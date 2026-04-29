"""Drag-and-drop zones with visual overlays.

The app has two distinct drop semantics:

* **Replace** — drop on the viewer area, the whole sequence is
  replaced. Mirrors the File → Open menu.
* **Add layer** — drop on the layer panel area, the dropped folder
  becomes a new top layer in the stack. Mirrors File → Add layer…

Pre-v1.0, every drop went through a modal "Add / Replace / Cancel"
dialog. That worked but was friction every time. The new model uses
spatial disambiguation à la OpenRV / DaVinci: each zone shows an
overlay during drag-over, the user reads it and lets go in the right
place. No dialog needed.

Both zones share :class:`DropOverlay` for the visual ("Replace" /
"Add to layers" centered on a translucent dark fill with a dashed
border), and a small mixin in :class:`DropZoneMixin` for the QWidget
event plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class DropOverlay(QFrame):  # type: ignore[misc]
    """Semi-transparent overlay shown only during drag-over.

    Lives as a child of the drop zone, sized to fill its parent at
    every paint. Transparent to mouse events so the underlying
    widget keeps receiving the drop position.
    """

    def __init__(
        self,
        label: str,
        accent: str,
        parent: QWidget,
    ) -> None:
        super().__init__(parent)
        # Block mouse / keyboard interaction with the overlay itself —
        # the drag events have to keep flowing to the parent zone.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            "QFrame {"
            f"  background: rgba(0, 0, 0, 140);"
            f"  border: 2px dashed {accent};"
            f"  border-radius: 6px;"
            "}"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(label)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setStyleSheet(
            "QLabel {"
            f"  background: transparent;"
            f"  border: none;"
            f"  color: {accent};"
            f"  font-size: 28px;"
            f"  font-weight: 700;"
            f"  letter-spacing: 1px;"
            "}"
        )
        layout.addWidget(self._label)

        self.hide()

    def show_overlay(self) -> None:
        """Reposition over the parent + bring to top."""
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())
        self.raise_()
        self.show()

    def hide_overlay(self) -> None:
        self.hide()


def install_file_drop_zone(
    widget: QWidget,
    overlay: DropOverlay,
    on_drop: Callable[[Path], None],
) -> None:
    """Wire a QWidget to accept folder/file drops with the given overlay.

    ``on_drop`` receives the resolved local path. The wiring uses
    method-level monkey-patching rather than subclassing so it stays
    reusable on widgets we don't own (``ViewerWidget``,
    ``MasterTimelinePanel``).

    Drag-over shows the overlay; drop or leave hides it. The widget's
    pre-existing drag/drop handlers (e.g. ``_RowsHost`` accepting the
    layer-id mime for intra-panel reorder) keep working — we only
    accept ``hasUrls()`` mimes here, so foreign drag types fall
    through to the original handlers.
    """
    widget.setAcceptDrops(True)

    prev_enter = widget.dragEnterEvent
    prev_move = widget.dragMoveEvent
    prev_leave = widget.dragLeaveEvent
    prev_drop = widget.dropEvent

    def drag_enter(event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            overlay.show_overlay()
            event.acceptProposedAction()
            return
        prev_enter(event)

    def drag_move(event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            return
        prev_move(event)

    def drag_leave(event: QDragLeaveEvent) -> None:
        overlay.hide_overlay()
        prev_leave(event)

    def drop(event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            overlay.hide_overlay()
            urls = event.mimeData().urls()
            if not urls:
                event.ignore()
                return
            local = urls[0].toLocalFile()
            if not local:
                event.ignore()
                return
            event.acceptProposedAction()
            on_drop(Path(local))
            return
        prev_drop(event)

    widget.dragEnterEvent = drag_enter      # type: ignore[method-assign]
    widget.dragMoveEvent = drag_move        # type: ignore[method-assign]
    widget.dragLeaveEvent = drag_leave      # type: ignore[method-assign]
    widget.dropEvent = drop                 # type: ignore[method-assign]


# Visual tokens — same hue family as the rest of the UI accents so
# the overlays read as "img_player UI" rather than generic dark-blue
# Qt boilerplate.
REPLACE_ACCENT = "#F2A23B"      # warm orange — destructive-ish
ADD_LAYER_ACCENT = "#5DC9D2"    # teal — additive cue
