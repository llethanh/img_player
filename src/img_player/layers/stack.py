"""The :class:`LayerStack` — ordered Layers + master-timeline resolution.

The stack is the single source of truth for the multi-layer state.
Every UI surface (layer panel, viewport, channel menu, color panel)
reads from it; every mutation goes through one of its public
methods which then emits a fine-grained signal so listeners only
react to what they care about:

* ``layers_changed`` — composition (add / remove / reorder) — anyone
  drawing the stack visually needs to redraw.
* ``visibility_changed`` — œil toggle — the cache invalidates the
  affected master-frame region; the viewport re-displays.
* ``layer_modified(id)`` — per-layer state mutation (channel,
  colorspace, exposure, trim, offset, name). Listeners that care
  about specific layers (e.g. the layer panel row, the cache) read
  the layer back from the stack on receipt.
* ``focus_changed(id)`` — which layer the user is currently editing.
  The channel menu / color panel / annotation overlay rebind to it.

Order convention: index 0 = top of stack = highest priority. The
class is iterable in that order.
"""

from __future__ import annotations

import copy
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from PySide6.QtCore import QObject, Signal

from img_player.layers.models import Layer

log = logging.getLogger(__name__)


# History is bounded so a long session of nudges doesn't grow the
# undo list unboundedly. 100 steps comfortably covers a typical edit
# session — comparable to most NLEs' default. Each snapshot is a
# small dict of layer fields, so memory cost is negligible.
_MAX_HISTORY = 100


@dataclass(frozen=True)
class _StackSnapshot:
    """Immutable point-in-time of the LayerStack.

    Kept tiny on purpose: a tuple of shallow-copied Layer dataclasses
    + the focused id. The underlying ``SequenceInfo`` is frozen so
    sharing the reference across snapshots is safe; only the Layer
    itself (whose mutable scalars carry edits) is duplicated. Equality
    is structural — ``__eq__`` walks every layer's fields, used to
    suppress duplicate "did nothing" pushes from idempotent calls.
    """

    layers: tuple[Layer, ...]
    focused_id: str


class LayerStack(QObject):  # type: ignore[misc]
    """Ordered list of :class:`Layer` + signals on every mutation."""

    # Composition changed (add / remove / reorder). Carries no payload —
    # listeners re-read the full stack via :meth:`layers`.
    layers_changed = Signal()
    # Visibility (œil) toggled on one layer. Carries the layer id so
    # the cache can decide whether to invalidate.
    visibility_changed = Signal(str)
    # Per-layer state mutated (trim, offset, channel, colorspace, …).
    # Carries the layer id; specific signal granularity per field
    # would multiply API surface for little gain.
    layer_modified = Signal(str)
    # The "focused" layer (= what the user is currently editing in
    # the side panels) changed. Empty string when no layer is focused
    # (e.g. all layers were removed).
    focus_changed = Signal(str)
    # Undo / redo availability flipped. Listeners (the Edit menu,
    # toolbar buttons) read :attr:`can_undo` / :attr:`can_redo` on
    # receipt to refresh their enabled state.
    history_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._layers: list[Layer] = []
        self._focused_id: str = ""
        # --- History ---------------------------------------------------
        # ``_history_enabled`` is flipped off during ``undo`` / ``redo``
        # restoration so the mutations they perform don't push more
        # entries onto the undo stack (otherwise undo would be
        # impossible to escape from).
        # ``_batch_depth`` lets callers group several mutations into a
        # single undo step (e.g. "replace sequence" = remove-all +
        # add-one). The first push inside a batch records the state
        # BEFORE the batch; further pushes inside the same batch are
        # suppressed. Re-entrant via :meth:`batch`'s context manager.
        self._undo_stack: list[_StackSnapshot] = []
        self._redo_stack: list[_StackSnapshot] = []
        self._history_enabled: bool = True
        self._batch_depth: int = 0
        self._batch_pending_snap: _StackSnapshot | None = None

    # ------------------------------------------------------------------ Mutation

    # ------------------------------------------------------------------ History API

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def undo(self) -> None:
        """Restore the previous snapshot. No-op if the undo stack is
        empty. The current state is pushed onto the redo stack so
        :meth:`redo` can replay it."""
        if not self._undo_stack:
            return
        current = self._snapshot()
        snap = self._undo_stack.pop()
        self._redo_stack.append(current)
        self._restore(snap)
        self.history_changed.emit()

    def redo(self) -> None:
        """Inverse of :meth:`undo`."""
        if not self._redo_stack:
            return
        current = self._snapshot()
        snap = self._redo_stack.pop()
        self._undo_stack.append(current)
        self._restore(snap)
        self.history_changed.emit()

    def clear_history(self) -> None:
        """Drop both undo + redo stacks. Call when starting a fresh
        session (loaded a new file from a clean app state) so the
        user can't undo across unrelated session boundaries."""
        had_history = bool(self._undo_stack or self._redo_stack)
        self._undo_stack.clear()
        self._redo_stack.clear()
        if had_history:
            self.history_changed.emit()

    @contextmanager
    def batch(self):
        """Group several mutations into a single undo step.

        Use for high-level actions that decompose into many low-level
        mutations under the hood (e.g. "replace sequence" =
        remove-every-existing + add-new; "load session" = clear +
        add-N). The first ``_push_undo`` inside the batch records the
        state *before* any change; subsequent pushes are suppressed.
        Re-entrant — nested batches collapse into the outermost one.
        """
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._batch_pending_snap is not None:
                snap = self._batch_pending_snap
                self._batch_pending_snap = None
                if not self._undo_stack or self._undo_stack[-1] != snap:
                    self._undo_stack.append(snap)
                    if len(self._undo_stack) > _MAX_HISTORY:
                        self._undo_stack.pop(0)
                    self._redo_stack.clear()
                    self.history_changed.emit()

    # ------------------------------------------------------------------ History internals

    def _snapshot(self) -> _StackSnapshot:
        """Capture the current stack as a ``_StackSnapshot``.

        Layers are shallow-copied (``copy.copy``) — their mutable
        scalars (offset, layer_in/out, alpha flags, exposure, gamma)
        are duplicated; the frozen :class:`SequenceInfo` /
        :class:`ChannelSelection` are shared by reference. Cheap and
        correct: undo only needs to roll back layer-level state, not
        the underlying scanned files."""
        return _StackSnapshot(
            layers=tuple(copy.copy(layer) for layer in self._layers),
            focused_id=self._focused_id,
        )

    def _push_undo(self) -> None:
        """Record the current state as the next undo target.

        Called by every mutating method *before* it changes anything.
        Suppressed during ``undo`` / ``redo`` restoration and when a
        :meth:`batch` is open (the batch records one snapshot total
        for all the mutations it covers).
        """
        if not self._history_enabled:
            return
        if self._batch_depth > 0:
            if self._batch_pending_snap is None:
                self._batch_pending_snap = self._snapshot()
            return
        snap = self._snapshot()
        if self._undo_stack and self._undo_stack[-1] == snap:
            return  # idempotent: nothing actually changed
        self._undo_stack.append(snap)
        if len(self._undo_stack) > _MAX_HISTORY:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self.history_changed.emit()

    def _restore(self, snap: _StackSnapshot) -> None:
        """Apply a snapshot to the live stack and re-emit signals so
        every listener (panel, cache, viewport) refreshes."""
        old_focus = self._focused_id
        self._history_enabled = False
        try:
            # Materialise the snapshot's layers as fresh mutable copies
            # so future edits don't accidentally write back through to
            # other snapshots that share the reference.
            self._layers = [copy.copy(layer) for layer in snap.layers]
            self._focused_id = snap.focused_id
        finally:
            self._history_enabled = True
        # Composition + per-layer state both potentially changed —
        # ``layers_changed`` triggers a full panel rebuild + cache
        # nuke-and-rebuild, which is the right hammer here.
        self.layers_changed.emit()
        if self._focused_id != old_focus:
            self.focus_changed.emit(self._focused_id)

    # ------------------------------------------------------------------ Mutation

    def add(self, layer: Layer, position: int = 0) -> None:
        """Insert ``layer`` at ``position`` (default = top of stack).

        Out-of-range positions clamp to the nearest valid index
        rather than raising — drag-drop UX is forgiving and we
        prefer "almost where the user dropped" over exceptions.
        New layer auto-focuses unless one is already focused; that
        way the first layer of a fresh session always becomes
        focused without an extra click, while subsequent adds don't
        steal focus mid-edit.
        """
        position = max(0, min(position, len(self._layers)))
        self._push_undo()
        self._layers.insert(position, layer)
        self.layers_changed.emit()
        if not self._focused_id:
            self.set_focus(layer.id)

    def remove(self, layer_id: str) -> None:
        """Remove the layer with this id. No-op if not found.

        If the focused layer is removed, focus shifts to the next
        layer in stack order (top first), or clears when the stack
        becomes empty.
        """
        for i, layer in enumerate(self._layers):
            if layer.id == layer_id:
                self._push_undo()
                del self._layers[i]
                if self._focused_id == layer_id:
                    new_focus = self._layers[0].id if self._layers else ""
                    self._focused_id = new_focus
                    self.focus_changed.emit(new_focus)
                self.layers_changed.emit()
                return

    def reorder(self, layer_id: str, new_position: int) -> None:
        """Move ``layer_id`` to ``new_position`` in stack order.

        ``new_position`` is the destination index *after* removal —
        so passing ``0`` always lands the layer at the very top
        regardless of its previous position. Out-of-range clamps.
        """
        for i, layer in enumerate(self._layers):
            if layer.id == layer_id:
                if i == new_position:
                    return  # idempotent — no signal
                self._push_undo()
                self._layers.pop(i)
                clamped = max(0, min(new_position, len(self._layers)))
                self._layers.insert(clamped, layer)
                self.layers_changed.emit()
                return

    def toggle_visible(self, layer_id: str) -> None:
        """Flip the œil. Emits :attr:`visibility_changed`."""
        layer = self._find(layer_id)
        if layer is None:
            return
        self._push_undo()
        layer.visible = not layer.visible
        self.visibility_changed.emit(layer_id)

    def set_visible(self, layer_id: str, visible: bool) -> None:
        """Set the œil to a specific value. Idempotent."""
        layer = self._find(layer_id)
        if layer is None or layer.visible == bool(visible):
            return
        self._push_undo()
        layer.visible = bool(visible)
        self.visibility_changed.emit(layer_id)

    def set_focus(self, layer_id: str) -> None:
        """Mark ``layer_id`` as the focused layer. Idempotent.

        Empty string clears focus (no per-layer panel context).
        """
        if layer_id and self._find(layer_id) is None:
            return  # silently ignore unknown id — defensive
        if layer_id == self._focused_id:
            return
        self._focused_id = layer_id
        self.focus_changed.emit(layer_id)

    def update(self, layer_id: str, **fields: object) -> None:
        """Mutate a layer's per-layer state in bulk.

        Each ``fields`` entry must match a Layer attribute name; we
        ``setattr`` each one and emit a single
        :attr:`layer_modified` so multi-field updates (e.g. exposure
        + gamma at once from the color panel) don't fire N signals.
        Unknown attributes are logged and ignored.

        Suppresses the signal emit when **none** of the fields
        actually change value. Without this, ``restore_channel_state``
        on a fresh layer fires ``layer_modified`` even when the
        new selection happens to match the layer's defaults — which
        invalidates the cache mid-prefetch and forces a costly
        re-decode of every frame the controller had just queued.
        """
        layer = self._find(layer_id)
        if layer is None or not fields:
            return
        # Pre-scan to determine whether any field actually changes —
        # we want one undo entry per *real* edit, not one per call.
        # This also matches the existing "suppress no-op signals"
        # contract documented above.
        will_change = False
        for name, value in fields.items():
            if not hasattr(layer, name):
                log.warning("LayerStack.update: unknown field %r", name)
                continue
            if getattr(layer, name) != value:
                will_change = True
                break
        if not will_change:
            return
        self._push_undo()
        changed = False
        for name, value in fields.items():
            if not hasattr(layer, name):
                continue  # already warned in the pre-scan
            if getattr(layer, name) == value:
                continue
            setattr(layer, name, value)
            changed = True
        if changed:
            self.layer_modified.emit(layer_id)

    # ------------------------------------------------------------------ Queries

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self) -> Iterator[Layer]:
        return iter(self._layers)

    def __bool__(self) -> bool:
        return bool(self._layers)

    def layers(self) -> tuple[Layer, ...]:
        """Snapshot tuple in stack order (top → bottom)."""
        return tuple(self._layers)

    def find(self, layer_id: str) -> Layer | None:
        """Public lookup by id. ``None`` when absent."""
        return self._find(layer_id)

    def focused(self) -> Layer | None:
        """The layer the user is currently editing, or ``None``."""
        if not self._focused_id:
            return None
        return self._find(self._focused_id)

    @property
    def focused_id(self) -> str:
        return self._focused_id

    def topmost_visible_at(self, master_frame: int) -> Layer | None:
        """Return the highest-priority visible layer covering
        ``master_frame``, or ``None`` if every covering layer is
        hidden (or none cover the frame at all → black screen).
        """
        for layer in self._layers:
            if layer.visible and layer.covers(master_frame):
                return layer
        return None

    def covers(self, master_frame: int) -> tuple[Layer, ...]:
        """Every layer that has a frame at ``master_frame`` (any
        visibility). Useful for the cache's pre-fetch policy and for
        the panel's "click visibility" UI."""
        return tuple(layer for layer in self._layers if layer.covers(master_frame))

    def master_range(self) -> tuple[int, int]:
        """Inclusive ``(first, last)`` master frames covered by the
        union of every layer.

        Returns ``(0, 0)`` when the stack is empty — callers that
        care about emptiness should check :meth:`__bool__` first.
        """
        if not self._layers:
            return (0, 0)
        first = min(layer.master_start for layer in self._layers)
        last = max(layer.master_end for layer in self._layers)
        return (first, last)

    def gap_frames(
        self, bounds: tuple[int, int] | None = None,
    ) -> frozenset[int]:
        """Master frames in ``bounds`` that no *visible* layer covers.

        Used by the timeline to paint gaps distinctly from cached /
        missing frames so the user can tell at a glance why playback
        would stall there (= nothing to decode rather than slow
        decode). Hidden layers count as not-covering, so toggling
        visibility live updates the gap visualisation.

        ``bounds`` defaults to :meth:`master_range` (= the trim-bounded
        range), but callers should usually pass the panel's
        ``broad_master_range`` — that's the range the timeline draws,
        and it includes the post-OUT-trim void where no layer reaches.
        Without that override, frames past the last layer's
        ``master_end`` aren't flagged as gaps and end up painted as
        empty cache slots instead of the distinct grey.
        """
        if not self._layers:
            return frozenset()
        if bounds is None:
            first, last = self.master_range()
        else:
            first, last = bounds
        gaps: list[int] = []
        for f in range(first, last + 1):
            if self.topmost_visible_at(f) is None:
                gaps.append(f)
        return frozenset(gaps)

    def master_length(self) -> int:
        """Total span of the master timeline in frames."""
        if not self._layers:
            return 0
        first, last = self.master_range()
        return last - first + 1

    # ------------------------------------------------------------------ Internals

    def _find(self, layer_id: str) -> Layer | None:
        for layer in self._layers:
            if layer.id == layer_id:
                return layer
        return None
