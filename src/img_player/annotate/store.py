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
    """Per-layer, per-frame strokes + per-frame undo. Qt signals for the UI.

    **Layer-scoping (v1.5.15+).** The store partitions strokes by
    ``layer_id`` internally so an annotation written while reviewing
    layer A doesn't leak onto layer B's frames. The public API stays
    "frame-keyed" — every read / mutate routes to the
    :attr:`current_layer_id`, which the app updates from the layer
    panel's focus changes. This keeps existing call sites unchanged
    (no ``layer_id`` parameter pollution) while enabling the
    "annotations survive source swaps" workflow: replacing the
    underlying sequence on a layer keeps ``layer_id`` (and therefore
    the layer's strokes) — only the pixels behind the strokes change.

    Empty string is the default ``current_layer_id`` to keep tests +
    bare-app behaviour identical to the pre-refactor state when no
    layer focus has been pushed yet.
    """

    annotated_frames_changed = Signal()
    """Emitted when the set ``{f : len(strokes_at(f)) > 0}`` mutates
    for the **current** layer.

    Also fires when :meth:`set_current_layer_id` switches the active
    layer — consumers (timeline markers, transport prev/next
    enabled-state) repaint as if the entire set was recomputed.
    """

    frame_annotated = Signal(int)
    """Emitted when a specific frame's stroke list mutates on the
    **current** layer.

    Consumers: the overlay (repaints if the frame is currently shown).
    Fires for ADD, REMOVE, undo, and redo on that frame. Mutations
    on a non-current layer (= e.g. an export bake reaches across
    layers) don't fire this signal; consumers tracking specific
    cross-layer events should listen to :attr:`layer_frame_annotated`.
    """

    layer_frame_annotated = Signal(str, int)
    """Cross-layer variant of :attr:`frame_annotated` — carries the
    ``layer_id`` of the affected layer. Fires for every mutation
    regardless of which layer is currently focused.

    Consumers: export / save-frame bakery that needs to read strokes
    of a specific layer regardless of focus. The overlay + timeline
    stick to the focus-filtered :attr:`frame_annotated` signal.
    """

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # Two-level dict: layer_id → frame → _FrameState. The
        # current layer's nested dict is what the public API exposes
        # — calling code reads / writes through the same frame-keyed
        # methods it always did.
        self._layers: dict[str, dict[int, _FrameState]] = {}
        # Empty string = "no layer focused yet" (e.g. app just
        # started, no sequence opened). Strokes still flow in / out
        # under this key so a stand-alone test can use the store
        # without setting up a layer.
        self._current_layer_id: str = ""
        # Slice 4 hooks this — for now just the field, no toggle wiring.
        self._show_during_playback: bool = False
        # Tracks whether the in-memory state has changed since the
        # last load_from_dict / mark_clean. The app reads this at
        # close-time to decide whether to prompt the user about
        # saving. Note: undo/redo also flip this, even if the net
        # effect brings the state back to "as loaded" — we err on
        # the side of asking. The user can pick "Don't save" if
        # they really meant to discard.
        self._dirty: bool = False

    # ------------------------------------------------------------------ Layer scope

    @property
    def current_layer_id(self) -> str:
        """The layer id all frame-keyed methods read / write against.
        Empty string when no layer has focus yet."""
        return self._current_layer_id

    def set_current_layer_id(self, layer_id: str) -> None:
        """Re-target every frame-keyed read / write to ``layer_id``.

        Idempotent. On change, emits :attr:`annotated_frames_changed`
        so consumers (timeline markers, transport buttons) refresh as
        if the whole set was recomputed — which is true: the set of
        annotated frames for the new layer is a different one.

        The empty string is reserved for "no layer focus yet".
        """
        lid = str(layer_id)
        if lid == self._current_layer_id:
            return
        self._current_layer_id = lid
        self.annotated_frames_changed.emit()

    def _frames(self) -> dict[int, _FrameState]:
        """The frame dict for the current layer, created on first
        access. Private — public reads / writes go through the
        frame-keyed methods below."""
        return self._layers.setdefault(self._current_layer_id, {})

    def _frames_for(self, layer_id: str) -> dict[int, _FrameState]:
        """Read-only-ish view on another layer's frames. Used by
        cross-layer consumers (export bake, persistence dump)
        without mutating the ``_layers`` dict if the layer has no
        data."""
        return self._layers.get(layer_id, {})

    def layers_with_strokes(self) -> frozenset[str]:
        """Layer ids that have at least one stroke on at least one
        frame. Used by the persistence layer to know which slots to
        write into the v2 sidecar shape."""
        return frozenset(
            lid for lid, frames in self._layers.items()
            if any(s.strokes for s in frames.values())
        )

    # ------------------------------------------------------------------ Read

    def strokes_at(self, frame: int) -> tuple[Stroke, ...]:
        """Strokes at ``frame`` (on the current layer) in draw order.
        Empty if no annotations."""
        state = self._frames().get(frame)
        return tuple(state.strokes) if state is not None else ()

    def strokes_at_for(
        self, layer_id: str, frame: int,
    ) -> tuple[Stroke, ...]:
        """Cross-layer variant of :meth:`strokes_at` — used by the
        export / save-frame bakery that needs to read strokes for a
        specific layer regardless of focus."""
        state = self._frames_for(layer_id).get(frame)
        return tuple(state.strokes) if state is not None else ()

    def annotated_frames(self) -> frozenset[int]:
        """Set of frame indices on the current layer that carry at
        least one stroke."""
        return frozenset(f for f, s in self._frames().items() if s.strokes)

    @property
    def show_during_playback(self) -> bool:
        return self._show_during_playback

    @show_during_playback.setter
    def show_during_playback(self, value: bool) -> None:
        self._show_during_playback = bool(value)

    def is_dirty(self) -> bool:
        """``True`` if any mutation happened since the last
        :meth:`load_from_dict` or :meth:`mark_clean` — used by the
        app's close-time prompt to skip the dialog when nothing has
        changed."""
        return self._dirty

    def mark_clean(self) -> None:
        """Reset the dirty flag — call after successfully writing
        the in-memory state to disk."""
        self._dirty = False

    # ------------------------------------------------------------------ Mutate

    def add_stroke(self, frame: int, stroke: Stroke) -> None:
        """Append a stroke to ``frame`` (on the current layer). Pushes
        an undo entry, clears redo."""
        frames = self._frames()
        state = frames.setdefault(frame, _FrameState())
        was_empty = not state.strokes
        idx = len(state.strokes)
        state.strokes.append(stroke)
        state.undo_stack.append(Action(ActionKind.ADD, frame, idx, stroke))
        # User action — the redo stack is no longer reachable.
        state.redo_stack.clear()
        self._dirty = True
        self.frame_annotated.emit(frame)
        self.layer_frame_annotated.emit(self._current_layer_id, frame)
        if was_empty:
            self.annotated_frames_changed.emit()

    def remove_stroke(self, frame: int, idx: int) -> None:
        """Remove the stroke at index ``idx`` of ``frame`` (on the
        current layer). Symmetric undo entry."""
        frames = self._frames()
        state = frames.get(frame)
        if state is None or not (0 <= idx < len(state.strokes)):
            raise IndexError(
                f"No stroke at frame={frame}, idx={idx} (have "
                f"{0 if state is None else len(state.strokes)} strokes)"
            )
        stroke = state.strokes.pop(idx)
        state.undo_stack.append(Action(ActionKind.REMOVE, frame, idx, stroke))
        state.redo_stack.clear()
        self._dirty = True
        self.frame_annotated.emit(frame)
        self.layer_frame_annotated.emit(self._current_layer_id, frame)
        if not state.strokes:
            self.annotated_frames_changed.emit()

    def clear_frame(self, frame: int) -> int:
        """Remove every stroke on ``frame`` (on the current layer).
        Returns the count removed.

        Each removal is recorded as its own undo entry, so the user
        can walk back stroke-by-stroke with Ctrl+Z. We remove from
        the back so each Action's ``idx`` matches the position the
        stroke occupied at removal time — that's what :meth:`undo`
        re-inserts into.
        """
        state = self._frames().get(frame)
        if state is None or not state.strokes:
            return 0
        count = len(state.strokes)
        while state.strokes:
            self.remove_stroke(frame, len(state.strokes) - 1)
        return count

    # ------------------------------------------------------------------ Undo/redo

    def undo(self, frame: int) -> bool:
        """Undo the most recent action on ``frame`` (on the current
        layer). Returns whether anything changed."""
        state = self._frames().get(frame)
        if state is None or not state.undo_stack:
            return False
        action = state.undo_stack.pop()
        self._apply(action.inverse(), record_to=None)
        state.redo_stack.append(action)
        return True

    def redo(self, frame: int) -> bool:
        """Redo the most recently undone action on ``frame`` (on the
        current layer). Returns whether anything changed."""
        state = self._frames().get(frame)
        if state is None or not state.redo_stack:
            return False
        action = state.redo_stack.pop()
        self._apply(action, record_to=None)
        state.undo_stack.append(action)
        return True

    def _apply(self, action: Action, *, record_to: str | None) -> None:
        """Internal: apply ``action`` to its frame's strokes list (on
        the current layer).

        ``record_to`` is unused at the moment (we always pass None)
        but the parameter is here so the method signature is honest
        about the fact that callers control whether to log the
        action onto a stack — :meth:`add_stroke` / :meth:`remove_stroke`
        record onto undo themselves; :meth:`undo` / :meth:`redo` move
        actions between the two stacks manually. Keeps :meth:`_apply`
        side-effect-free w.r.t. the stacks.
        """
        frames = self._frames()
        state = frames.setdefault(action.frame, _FrameState())
        was_empty = not state.strokes
        if action.kind == ActionKind.ADD:
            state.strokes.insert(action.idx, action.stroke)
        else:  # REMOVE
            del state.strokes[action.idx]
        is_empty = not state.strokes
        # Undo / redo are state-changing too — flag the store dirty.
        self._dirty = True
        self.frame_annotated.emit(action.frame)
        self.layer_frame_annotated.emit(self._current_layer_id, action.frame)
        if was_empty != is_empty:
            self.annotated_frames_changed.emit()

    # ------------------------------------------------------------------ Persistence

    def to_dict(self) -> dict[str, dict[str, list[dict[str, object]]]]:
        """Serialise the **current layer's** strokes (without undo stacks).

        Shape: ``{"frames": {<frame_str>: [<stroke>, ...]}}``.

        This intentionally only dumps the current layer's frames so
        the existing v1 sidecar writer can be reused unchanged. The
        multi-layer dump used by the v2 sidecar lives in
        :meth:`to_dict_multi`.
        """
        out: dict[str, list[dict[str, object]]] = {}
        for frame, state in self._frames().items():
            if state.strokes:
                out[str(frame)] = [s.to_dict() for s in state.strokes]
        return {"frames": out}  # type: ignore[return-value]

    def to_dict_multi(self) -> dict[str, dict[str, list[dict[str, object]]]]:
        """Serialise **all** layers' strokes for the v2 sidecar.

        Shape: ``{<layer_id>: {<frame_str>: [<stroke>, ...]}}``.

        Skips layers with no strokes so the file stays small. The
        persistence layer wraps this under a top-level ``"layers"``
        key with a ``"schema_version"`` sibling — see
        :mod:`img_player.annotate.persistence`.
        """
        out: dict[str, dict[str, list[dict[str, object]]]] = {}
        for layer_id, frames in self._layers.items():
            layer_frames: dict[str, list[dict[str, object]]] = {}
            for frame, state in frames.items():
                if state.strokes:
                    layer_frames[str(frame)] = [s.to_dict() for s in state.strokes]
            if layer_frames:
                out[layer_id] = layer_frames
        return out

    def load_from_dict(self, data: dict[str, list[dict[str, object]]]) -> None:
        """Replace **the current layer's** state from a v1-shaped dict.

        Expects the ``"frames"``-payload shape (i.e. the value of
        ``to_dict()["frames"]``). Skips strokes that fail
        :meth:`Stroke.from_dict` validation (the persistence layer
        logs a warning; here we just no-op).

        Other layers' state is left untouched — so a v2 reader that
        wants to load each layer separately can call this once per
        layer after setting :attr:`current_layer_id` to that layer.
        """
        # Reset only the current layer's frames + stacks. Other
        # layers' state stays intact.
        self._layers[self._current_layer_id] = {}
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
                self._layers[self._current_layer_id][frame] = _FrameState(strokes=kept)
        # Loading state is not undoable. We just synced with disk →
        # by definition the in-memory state matches the file we
        # loaded from, so the dirty flag resets.
        self._dirty = False
        self.annotated_frames_changed.emit()

    def load_from_dict_multi(
        self, data: dict[str, dict[str, list[dict[str, object]]]],
    ) -> None:
        """Replace **all** layers' state from a v2-shaped dict.

        Expects the ``{<layer_id>: {<frame_str>: [<stroke>, ...]}}``
        shape returned by :meth:`to_dict_multi`. Mirrors
        :meth:`load_from_dict`'s best-effort tolerance for malformed
        strokes / non-integer frame keys.
        """
        self._layers.clear()
        for layer_id, frames_dict in data.items():
            layer_frames: dict[int, _FrameState] = {}
            for frame_str, stroke_dicts in frames_dict.items():
                try:
                    frame = int(frame_str)
                except (TypeError, ValueError):
                    continue
                kept: list[Stroke] = []
                for sd in stroke_dicts:
                    try:
                        kept.append(Stroke.from_dict(sd))
                    except (KeyError, TypeError, ValueError):
                        continue
                if kept:
                    layer_frames[frame] = _FrameState(strokes=kept)
            if layer_frames:
                self._layers[str(layer_id)] = layer_frames
        # Make sure the current layer's slot exists so subsequent
        # writes don't trip on a KeyError. setdefault is idempotent.
        self._layers.setdefault(self._current_layer_id, {})
        self._dirty = False
        self.annotated_frames_changed.emit()

    # ------------------------------------------------------------------ Test helpers

    def _undo_stack_size(self, frame: int) -> int:
        """Test-only: peek at the undo stack depth (on the current
        layer) without exposing it."""
        state = self._frames().get(frame)
        return 0 if state is None else len(state.undo_stack)

    def _redo_stack_size(self, frame: int) -> int:
        """Test-only: peek at the redo stack depth (on the current
        layer) without exposing it."""
        state = self._frames().get(frame)
        return 0 if state is None else len(state.redo_stack)
