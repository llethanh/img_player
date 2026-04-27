"""The :class:`AnnotationStore` — per-frame strokes + per-frame undo.

This is the model layer. It knows nothing about Qt widgets or
rendering: it's a plain data store that emits ``QObject`` signals when
its state changes. The overlay reads it, the toolbar mutates it, the
timeline observes the signals to repaint markers.

Design points (see spec ``2026-04-27-annotations-design.md`` §6):

* **Per-frame undo stacks.** Each frame has its own undo and redo
  stacks. Hitting ``Ctrl+Z`` while on frame 42 only undoes actions
  performed on frame 42 — no surprise frame jumps.
* **Stacks are NOT persisted.** Fresh on every session. Persisting
  them would complicate the data model (interleaved with strokes)
  and open the door to "I undo a stroke from a previous session,
  but the stroke I'm undoing was already saved" inconsistencies.
* **Actions are reversible.** :class:`Action` carries everything
  needed to invert itself: an ``ADD`` becomes a ``REMOVE`` (same
  frame, same idx, same stroke), and vice versa. Redo is just the
  forward Action.
* **Signals.** ``annotated_frames_changed`` fires when the set of
  frames-with-at-least-one-stroke changes. ``frame_annotated(N)``
  fires whenever frame ``N``'s strokes list changes (new stroke,
  removed stroke, undo, redo) — used by the overlay to know when
  to repaint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from PySide6.QtCore import QObject, Signal

from img_player.annotate.stroke import Stroke


class ActionKind(Enum):
    """Direction of a reversible :class:`Action`."""

    ADD = "add"
    REMOVE = "remove"


@dataclass(frozen=True)
class Action:
    """A reversible state mutation on a single frame's stroke list.

    The inverse is constructed by flipping ``kind``: if you applied
    ``Action(ADD, 42, 3, stroke)``, the inverse is
    ``Action(REMOVE, 42, 3, stroke)``. :meth:`AnnotationStore.undo`
    pops the most recent Action, builds its inverse, applies it, and
    pushes the original onto the redo stack — so a subsequent
    :meth:`AnnotationStore.redo` re-applies it.
    """

    kind: ActionKind
    frame: int
    idx: int
    stroke: Stroke

    def inverse(self) -> Action:
        opposite = ActionKind.REMOVE if self.kind == ActionKind.ADD else ActionKind.ADD
        return Action(kind=opposite, frame=self.frame, idx=self.idx, stroke=self.stroke)


@dataclass
class _FrameState:
    """Per-frame container. Internal — do not import."""

    strokes: list[Stroke] = field(default_factory=list)
    undo_stack: list[Action] = field(default_factory=list)
    redo_stack: list[Action] = field(default_factory=list)


class AnnotationStore(QObject):
    """Per-frame strokes + per-frame undo. Qt signals for the UI."""

    annotated_frames_changed = Signal()
    """Emitted when the set ``{f : len(strokes_at(f)) > 0}`` mutates.

    Consumers: the timeline (markers) and the transport bar
    (prev/next buttons enabled state).
    """

    frame_annotated = Signal(int)
    """Emitted when a specific frame's stroke list mutates.

    Consumers: the overlay (repaints if the frame is currently shown).
    Fires for ADD, REMOVE, undo, and redo on that frame.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._frames: dict[int, _FrameState] = {}
        # Slice 4 hooks this — for now just the field, no toggle wiring.
        self._show_during_playback: bool = False

    # ------------------------------------------------------------------ Read

    def strokes_at(self, frame: int) -> tuple[Stroke, ...]:
        """Strokes at ``frame`` in draw order. Empty if no annotations."""
        state = self._frames.get(frame)
        return tuple(state.strokes) if state is not None else ()

    def annotated_frames(self) -> frozenset[int]:
        """Set of frame indices that carry at least one stroke."""
        return frozenset(f for f, s in self._frames.items() if s.strokes)

    @property
    def show_during_playback(self) -> bool:
        return self._show_during_playback

    @show_during_playback.setter
    def show_during_playback(self, value: bool) -> None:
        self._show_during_playback = bool(value)

    # ------------------------------------------------------------------ Mutate

    def add_stroke(self, frame: int, stroke: Stroke) -> None:
        """Append a stroke to ``frame``. Pushes an undo entry, clears redo."""
        state = self._frames.setdefault(frame, _FrameState())
        was_empty = not state.strokes
        idx = len(state.strokes)
        state.strokes.append(stroke)
        state.undo_stack.append(Action(ActionKind.ADD, frame, idx, stroke))
        # User action — the redo stack is no longer reachable.
        state.redo_stack.clear()
        self.frame_annotated.emit(frame)
        if was_empty:
            self.annotated_frames_changed.emit()

    def remove_stroke(self, frame: int, idx: int) -> None:
        """Remove the stroke at index ``idx`` of ``frame``. Symmetric undo entry."""
        state = self._frames.get(frame)
        if state is None or not (0 <= idx < len(state.strokes)):
            raise IndexError(
                f"No stroke at frame={frame}, idx={idx} (have "
                f"{0 if state is None else len(state.strokes)} strokes)"
            )
        stroke = state.strokes.pop(idx)
        state.undo_stack.append(Action(ActionKind.REMOVE, frame, idx, stroke))
        state.redo_stack.clear()
        self.frame_annotated.emit(frame)
        if not state.strokes:
            self.annotated_frames_changed.emit()

    # ------------------------------------------------------------------ Undo/redo

    def undo(self, frame: int) -> bool:
        """Undo the most recent action on ``frame``. Returns whether anything changed."""
        state = self._frames.get(frame)
        if state is None or not state.undo_stack:
            return False
        action = state.undo_stack.pop()
        self._apply(action.inverse(), record_to=None)
        state.redo_stack.append(action)
        return True

    def redo(self, frame: int) -> bool:
        """Redo the most recently undone action on ``frame``. Returns whether anything changed."""
        state = self._frames.get(frame)
        if state is None or not state.redo_stack:
            return False
        action = state.redo_stack.pop()
        self._apply(action, record_to=None)
        state.undo_stack.append(action)
        return True

    def _apply(self, action: Action, *, record_to: str | None) -> None:
        """Internal: apply ``action`` to its frame's strokes list.

        ``record_to`` is unused at the moment (we always pass None)
        but the parameter is here so the method signature is honest
        about the fact that callers control whether to log the
        action onto a stack — :meth:`add_stroke` / :meth:`remove_stroke`
        record onto undo themselves; :meth:`undo` / :meth:`redo` move
        actions between the two stacks manually. Keeps :meth:`_apply`
        side-effect-free w.r.t. the stacks.
        """
        state = self._frames.setdefault(action.frame, _FrameState())
        was_empty = not state.strokes
        if action.kind == ActionKind.ADD:
            state.strokes.insert(action.idx, action.stroke)
        else:  # REMOVE
            del state.strokes[action.idx]
        is_empty = not state.strokes
        self.frame_annotated.emit(action.frame)
        if was_empty != is_empty:
            self.annotated_frames_changed.emit()

    # ------------------------------------------------------------------ Persistence

    def to_dict(self) -> dict[str, dict[str, list[dict[str, object]]]]:
        """Serialise the current strokes (without undo stacks).

        Shape: ``{<basename>: {"frames": {<frame_str>: [<stroke>, ...]}}}``.
        The basename layer is filled in by the persistence module
        (this method returns a single-sequence shape and the caller
        wraps it under a basename key).
        """
        # Slice 1 stores everything under a single virtual basename.
        # Slice 2+ will pass the real basename when wiring to app.py.
        # Either way the structure is the same — only the key changes.
        out: dict[str, list[dict[str, object]]] = {}
        for frame, state in self._frames.items():
            if state.strokes:
                out[str(frame)] = [s.to_dict() for s in state.strokes]
        return {"frames": out}  # type: ignore[return-value]

    def load_from_dict(self, data: dict[str, list[dict[str, object]]]) -> None:
        """Replace state from a JSON dict. Clears existing strokes and stacks.

        Expects the ``"frames"`` shape returned by :meth:`to_dict`.
        Skips strokes that fail :meth:`Stroke.from_dict` validation
        (the persistence layer logs a warning; here we just no-op).
        """
        self._frames.clear()
        for frame_str, stroke_dicts in data.items():
            try:
                frame = int(frame_str)
            except (TypeError, ValueError):
                continue
            kept: list[Stroke] = []
            for sd in stroke_dicts:
                try:
                    kept.append(Stroke.from_dict(sd))
                except (KeyError, TypeError, ValueError):
                    # Skip malformed strokes silently — load is
                    # best-effort and one bad stroke shouldn't
                    # invalidate the rest of the file.
                    continue
            if kept:
                self._frames[frame] = _FrameState(strokes=kept)
        # Loading state is not undoable.
        self.annotated_frames_changed.emit()

    # ------------------------------------------------------------------ Test helpers

    def _undo_stack_size(self, frame: int) -> int:
        """Test-only: peek at the undo stack depth without exposing it."""
        state = self._frames.get(frame)
        return 0 if state is None else len(state.undo_stack)

    def _redo_stack_size(self, frame: int) -> int:
        """Test-only: peek at the redo stack depth without exposing it."""
        state = self._frames.get(frame)
        return 0 if state is None else len(state.redo_stack)
