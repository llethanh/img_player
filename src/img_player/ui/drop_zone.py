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

**Both overlays are shown simultaneously** as soon as a drag-with-urls
enters EITHER zone — that way the user always sees both possible
destinations and can pick by moving the cursor. The zone the cursor
is currently over goes "active" (bright accent fill + dashed border);
its peer stays "dim" (low-alpha fill + thin solid border) so it reads
as a secondary option rather than a competing target.

Coordination between the two zones lives in :class:`DropZoneCoordinator`.
A single module-level instance is shared by every call to
:func:`install_file_drop_zone`, so callers don't have to thread an
extra object through their constructors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class DropOverlay(QFrame):  # type: ignore[misc]
    """Semi-transparent overlay shown only during drag-over.

    Lives as a child of the drop zone, sized to fill its parent at
    every paint. Transparent to mouse events so the underlying
    widget keeps receiving the drop position.

    Has two visual states (driven by :meth:`set_active`):

    * **Active** — the cursor is over this zone. Strong fill + 2 px
      dashed accent border + bold accent label. Reads as "drop here
      and this is what will happen".
    * **Dim** — the cursor is over a sibling zone. Low-alpha fill +
      1 px solid border + de-saturated label. Reads as "this is
      another option, slide the cursor here to switch target".
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

        self._accent = accent
        self._label_text = label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(label)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)

        # Default to dim — the coordinator flips us to active the
        # instant the parent zone gets a dragEnter.
        self._is_active = False
        self._apply_style()
        self.hide()

    # -- state ------------------------------------------------------

    def set_active(self, active: bool) -> None:
        """Toggle between bright (cursor here) and dim (cursor on
        peer zone) presentation. Cheap — re-applies the stylesheet
        for ``self`` + the child label, no relayout."""
        if active == self._is_active:
            return
        self._is_active = active
        self._apply_style()

    def _apply_style(self) -> None:
        if self._is_active:
            frame_bg = "rgba(0, 0, 0, 140)"
            border = f"2px dashed {self._accent}"
            label_color = self._accent
            label_alpha = ""  # full opacity
        else:
            frame_bg = "rgba(0, 0, 0, 80)"
            border = f"1px solid {self._accent}88"  # ~53% alpha hex
            label_color = f"{self._accent}A0"       # ~63% alpha hex
            label_alpha = ""
        self.setStyleSheet(
            "QFrame {"
            f"  background: {frame_bg};"
            f"  border: {border};"
            f"  border-radius: 6px;"
            "}"
        )
        self._label.setStyleSheet(
            "QLabel {"
            f"  background: transparent;"
            f"  border: none;"
            f"  color: {label_color};"
            f"  font-size: 28px;"
            f"  font-weight: 700;"
            f"  letter-spacing: 1px;"
            f"  {label_alpha}"
            "}"
        )

    # -- show / hide -----------------------------------------------

    def show_overlay(self) -> None:
        """Reposition over the parent + bring to top."""
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())
        self.raise_()
        self.show()

    def hide_overlay(self) -> None:
        self.hide()


class DropZoneCoordinator:
    """Cross-zone state machine for drag-over.

    One instance is shared across every drop zone in the app (see
    :func:`get_default_coordinator`). It tracks which overlay is
    currently active and ensures a drag-enter on ANY zone shows
    EVERY registered overlay simultaneously — the active one bright,
    the rest dim.

    Why not just rely on per-zone show / hide? Without a shared
    coordinator each zone only knows about its own overlay, so the
    user would see one option at a time and miss the alternative.
    The dim/active split makes the choice legible.

    Drag-leave is deferred by one event-loop tick: when the cursor
    crosses from zone A into zone B, Qt fires A's dragLeave first
    and B's dragEnter immediately after. The 0 ms timer lets B's
    activate() preempt A's pending hide so the overlays don't
    flicker off and back on at the boundary.
    """

    def __init__(self) -> None:
        self._overlays: list[DropOverlay] = []
        # Session-mode overlay(s) — shown alone (REPLACE/ADD hidden)
        # whenever the drag payload contains a ``.session`` file.
        # Registered separately so the URL-sniffing logic in
        # :meth:`activate` can pick which mode to render.
        self._session_overlays: list[DropOverlay] = []
        self._active: DropOverlay | None = None
        self._pending_clear: bool = False
        self._clear_timer = QTimer()
        self._clear_timer.setSingleShot(True)
        self._clear_timer.setInterval(0)
        self._clear_timer.timeout.connect(self._maybe_clear)

    def register(self, overlay: DropOverlay) -> None:
        """Add ``overlay`` to the set the coordinator drives. Idempotent."""
        if overlay not in self._overlays:
            self._overlays.append(overlay)

    def register_session_overlay(self, overlay: DropOverlay) -> None:
        """Register a "drag carries a .session" overlay.

        These overlays replace the regular REPLACE/ADD pair when the
        drop payload is a project file — sequence zones don't apply
        and showing them would be confusing. Idempotent.
        """
        if overlay not in self._session_overlays:
            self._session_overlays.append(overlay)

    @staticmethod
    def _payload_is_session(urls: list[str] | None) -> bool:
        """Return True iff the drag carries at least one ``.session`` URL.

        Sniffs the local-file extension on every URL — a single
        session in a multi-URL drop is enough to flip the mode,
        since the session loader takes the whole stack and the
        other items would be ignored anyway.
        """
        if not urls:
            return False
        for u in urls:
            if u.lower().endswith(".session"):
                return True
        return False

    def activate(
        self,
        overlay: DropOverlay,
        urls: list[str] | None = None,
    ) -> None:
        """The cursor entered ``overlay``'s zone with a urls payload.

        Inspects ``urls`` to pick the rendering mode:

        * **session mode** — payload contains a ``.session``: every
          sequence overlay hides, every registered session overlay
          shows full-window. There's no "active vs dim" because the
          drop target is the whole window — anywhere works.
        * **sequence mode** — the regular REPLACE/ADD spatial
          disambiguation: every registered sequence overlay shows,
          ``overlay`` is the bright one.
        """
        self._pending_clear = False
        self._clear_timer.stop()
        self._active = overlay
        if self._payload_is_session(urls):
            for o in self._overlays:
                o.hide_overlay()
            for o in self._session_overlays:
                o.set_active(True)
                o.show_overlay()
            return
        for o in self._session_overlays:
            o.hide_overlay()
        for o in self._overlays:
            o.set_active(o is overlay)
            o.show_overlay()

    def deactivate(self, overlay: DropOverlay) -> None:
        """The cursor left ``overlay``'s zone.

        Defer the hide so a sibling :meth:`activate` (= cursor
        sliding into the other zone) can take over without a flicker.
        If no activate arrives within the timer tick, every overlay
        hides via :meth:`_maybe_clear`.
        """
        # Only schedule a clear if this overlay was the active one —
        # spurious dragLeave on a non-active overlay (e.g. parent
        # widget churn) shouldn't trigger a hide.
        if self._active is overlay:
            self._pending_clear = True
            self._clear_timer.start()

    def force_clear(self) -> None:
        """Hide every overlay immediately — used on drop so the
        post-drop state is clean without waiting for the timer."""
        self._clear_timer.stop()
        self._pending_clear = False
        self._active = None
        for o in self._overlays:
            o.hide_overlay()
        for o in self._session_overlays:
            o.hide_overlay()

    def _maybe_clear(self) -> None:
        if not self._pending_clear:
            return
        self._pending_clear = False
        self._active = None
        for o in self._overlays:
            o.hide_overlay()
        for o in self._session_overlays:
            o.hide_overlay()


# Module-level shared coordinator. Lazy so it's only created once a
# drop zone is actually installed (= GUI mode), not at import time.
_default_coordinator: DropZoneCoordinator | None = None


def get_default_coordinator() -> DropZoneCoordinator:
    """Return the shared coordinator, creating it on first access."""
    global _default_coordinator
    if _default_coordinator is None:
        _default_coordinator = DropZoneCoordinator()
    return _default_coordinator


def install_file_drop_zone(
    widget: QWidget,
    overlay: DropOverlay,
    on_drop: Callable[[list[Path]], None],
    coordinator: DropZoneCoordinator | None = None,
) -> None:
    """Wire a QWidget to accept folder/file drops with the given overlay.

    ``on_drop`` receives the resolved local path. The wiring uses
    method-level monkey-patching rather than subclassing so it stays
    reusable on widgets we don't own (``ViewerWidget``,
    ``MasterTimelinePanel``).

    Drag-over activates this zone in the coordinator (= our overlay
    goes bright, peers go dim and also show); drag-leave defers a
    hide; drop clears everything. The widget's pre-existing drag/drop
    handlers (e.g. ``_RowsHost`` accepting the layer-id mime for
    intra-panel reorder) keep working — we only accept ``hasUrls()``
    mimes here, so foreign drag types fall through to the original
    handlers.

    ``coordinator`` defaults to the shared module-level instance, so
    every drop zone in a normal app run lives in one shared state
    machine. Pass an explicit coordinator only for tests that need
    isolation between zones.
    """
    if coordinator is None:
        coordinator = get_default_coordinator()
    coordinator.register(overlay)

    widget.setAcceptDrops(True)

    prev_enter = widget.dragEnterEvent
    prev_move = widget.dragMoveEvent
    prev_leave = widget.dragLeaveEvent
    prev_drop = widget.dropEvent

    def _payload_urls(event) -> list[str]:  # type: ignore[no-untyped-def]
        """Extract the local-file string from every URL in the payload.

        We pass these into the coordinator so it can sniff for a
        ``.session`` extension and switch the overlay rendering mode
        accordingly. Lower-cased on the way out so the suffix check
        on the other side stays case-insensitive.
        """
        return [u.toLocalFile() for u in event.mimeData().urls()]

    def drag_enter(event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            coordinator.activate(overlay, _payload_urls(event))
            event.acceptProposedAction()
            return
        prev_enter(event)

    def drag_move(event: QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            # The cursor stayed inside our zone — keep us active even
            # if a stale ``deactivate`` from an earlier boundary
            # crossing is still pending in the timer.
            coordinator.activate(overlay, _payload_urls(event))
            event.acceptProposedAction()
            return
        prev_move(event)

    def drag_leave(event: QDragLeaveEvent) -> None:
        coordinator.deactivate(overlay)
        prev_leave(event)

    def drop(event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            coordinator.force_clear()
            urls = event.mimeData().urls()
            paths: list[Path] = []
            for u in urls:
                local = u.toLocalFile()
                if local:
                    paths.append(Path(local))
            if not paths:
                event.ignore()
                return
            event.acceptProposedAction()
            on_drop(paths)
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
SESSION_ACCENT = "#9F7BFF"      # purple — distinct "project" cue
