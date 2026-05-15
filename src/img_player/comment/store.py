"""The :class:`CommentStore` — per-layer, per-frame ordered list of comments.

Mirrors :class:`~img_player.annotate.store.AnnotationStore` in
shape: a `QObject` keyed by frame index, with Qt signals for the
UI to react. Adds the CRUD operations for individual comments
(add / edit / delete) since unlike strokes, comments stay user-
addressable after creation — the panel needs a stable handle to
edit or remove a specific row.

Design notes:

* No undo stack. Comments are textual, edits are deliberate, and
  Slack / Kitsu / etc. don't ship undo for comment edits either.
  Worst case the user re-types — much simpler model.
* Dirty tracking, same contract as ``AnnotationStore``: any
  mutation flips ``_dirty``, ``load_from_dict`` clears it.
* The shared sidecar (with annotations) is loaded / saved by the
  app at sequence open / close — see
  :mod:`img_player.comment.persistence`.

**Layer-scoping (v1.5.15+).** Same model as ``AnnotationStore``:
comments are partitioned by ``layer_id`` internally so swapping
the source on a layer preserves its comments. The public API
stays frame-keyed — every read / mutate routes through the
:attr:`current_layer_id`, pushed by the app on layer-focus
changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from PySide6.QtCore import QObject, Signal

from img_player.comment.comment import Comment

log = logging.getLogger(__name__)


@dataclass
class _FrameState:
    """Per-frame container. Internal — do not import."""

    comments: list[Comment] = field(default_factory=list)


class CommentStore(QObject):
    """Per-layer, per-frame textual comments. Qt signals for the UI."""

    commented_frames_changed = Signal()
    """Emitted when the set ``{f : len(comments_at(f)) > 0}`` mutates
    for the **current** layer. Also fires on
    :meth:`set_current_layer_id` so consumers refresh when focus
    swaps.
    """

    frame_comments_changed = Signal(int)
    """Emitted when a specific frame's comment list mutates on the
    **current** layer."""

    layer_frame_comments_changed = Signal(str, int)
    """Cross-layer variant: ``(layer_id, frame)``. Fires for every
    mutation regardless of focus. Used by consumers that need
    per-layer events (e.g. a future "all layers' notes" panel)."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # Two-level dict: layer_id → frame → _FrameState. Same shape
        # as AnnotationStore so the symmetry is obvious at a glance.
        self._layers: dict[str, dict[int, _FrameState]] = {}
        self._current_layer_id: str = ""
        self._dirty: bool = False

    # ------------------------------------------------------------------ Layer scope

    @property
    def current_layer_id(self) -> str:
        """The layer id every frame-keyed method routes through."""
        return self._current_layer_id

    def set_current_layer_id(self, layer_id: str) -> None:
        """Re-target every frame-keyed read / write to ``layer_id``.

        Idempotent. On change, emits
        :attr:`commented_frames_changed` so the timeline + comment
        panel refresh as if the whole set was recomputed (it is — a
        different layer has a different set of commented frames)."""
        lid = str(layer_id)
        if lid == self._current_layer_id:
            return
        self._current_layer_id = lid
        self.commented_frames_changed.emit()

    def _frames(self) -> dict[int, _FrameState]:
        """The frame dict for the current layer, created on first access."""
        return self._layers.setdefault(self._current_layer_id, {})

    def _frames_for(self, layer_id: str) -> dict[int, _FrameState]:
        """Read-only-ish access to another layer's frames."""
        return self._layers.get(layer_id, {})

    def layers_with_comments(self) -> frozenset[str]:
        """Layer ids that have at least one comment on at least one
        frame. Used by the persistence layer."""
        return frozenset(
            lid for lid, frames in self._layers.items()
            if any(s.comments for s in frames.values())
        )

    # ------------------------------------------------------------------ Read

    def comments_at(self, frame: int) -> tuple[Comment, ...]:
        """All comments on ``frame`` (on the current layer) in
        chronological (insertion) order."""
        state = self._frames().get(frame)
        return tuple(state.comments) if state is not None else ()

    def comments_at_for(
        self, layer_id: str, frame: int,
    ) -> tuple[Comment, ...]:
        """Cross-layer read — bypasses ``current_layer_id``. Used by
        the persistence dump + any future cross-layer panel."""
        state = self._frames_for(layer_id).get(frame)
        return tuple(state.comments) if state is not None else ()

    def commented_frames(self) -> frozenset[int]:
        """Set of frame indices on the current layer that carry at
        least one comment."""
        return frozenset(f for f, s in self._frames().items() if s.comments)

    def is_dirty(self) -> bool:
        """``True`` if any mutation happened since the last
        :meth:`load_from_dict` or :meth:`mark_clean`."""
        return self._dirty

    def mark_clean(self) -> None:
        """Reset the dirty flag — call after successful save."""
        self._dirty = False

    # ------------------------------------------------------------------ Mutate

    def add_comment(self, frame: int, text: str) -> Comment:
        """Append a fresh comment to ``frame`` (on the current layer).
        Returns the new :class:`Comment` so the caller can dismiss
        the edit field knowing the id."""
        text = text.strip()
        if not text:
            raise ValueError("Comment text cannot be empty")
        frames = self._frames()
        state = frames.setdefault(frame, _FrameState())
        was_empty = not state.comments
        comment = Comment.new(text)
        state.comments.append(comment)
        self._dirty = True
        self.frame_comments_changed.emit(frame)
        self.layer_frame_comments_changed.emit(self._current_layer_id, frame)
        if was_empty:
            self.commented_frames_changed.emit()
        return comment

    def edit_comment(self, frame: int, comment_id: str, new_text: str) -> bool:
        """Replace the text of the comment with the given id (on the
        current layer). Returns ``True`` on success, ``False`` if
        the id wasn't found.

        Raises ``ValueError`` for an empty new text — same contract
        as :meth:`add_comment`. The store does not silently delete
        a comment via an empty edit; the user must use
        :meth:`delete_comment` explicitly.
        """
        new_text = new_text.strip()
        if not new_text:
            raise ValueError("Comment text cannot be empty")
        state = self._frames().get(frame)
        if state is None:
            return False
        for i, existing in enumerate(state.comments):
            if existing.id == comment_id:
                state.comments[i] = existing.edited(new_text)
                self._dirty = True
                self.frame_comments_changed.emit(frame)
                self.layer_frame_comments_changed.emit(
                    self._current_layer_id, frame,
                )
                return True
        return False

    def delete_comment(self, frame: int, comment_id: str) -> bool:
        """Remove the comment with the given id (on the current
        layer). Returns ``True`` on success, ``False`` if the id
        wasn't found.

        Unlike :meth:`AnnotationStore.remove_stroke`, deletes are
        **not undoable** — comment text is small and re-typing is
        cheap; carrying an undo stack just for comments would
        complicate the model with little user benefit.
        """
        state = self._frames().get(frame)
        if state is None:
            return False
        for i, existing in enumerate(state.comments):
            if existing.id == comment_id:
                del state.comments[i]
                self._dirty = True
                self.frame_comments_changed.emit(frame)
                self.layer_frame_comments_changed.emit(
                    self._current_layer_id, frame,
                )
                if not state.comments:
                    self.commented_frames_changed.emit()
                return True
        return False

    # ------------------------------------------------------------------ Persistence

    def to_dict(self) -> dict[str, list[dict[str, str]]]:
        """Serialise the **current layer's** comments.

        Shape: ``{<frame_str>: [<comment dict>, ...]}``. The
        persistence module wraps this under
        ``sequences[<basename>]["comments"]`` in the on-disk JSON.

        Multi-layer dump for the upcoming v2 sidecar lives in
        :meth:`to_dict_multi`.
        """
        out: dict[str, list[dict[str, str]]] = {}
        for frame, state in self._frames().items():
            if state.comments:
                out[str(frame)] = [c.to_dict() for c in state.comments]
        return out

    def to_dict_multi(self) -> dict[str, dict[str, list[dict[str, str]]]]:
        """Serialise **all** layers' comments for the v2 sidecar.

        Shape: ``{<layer_id>: {<frame_str>: [<comment dict>, ...]}}``.
        """
        out: dict[str, dict[str, list[dict[str, str]]]] = {}
        for layer_id, frames in self._layers.items():
            layer_frames: dict[str, list[dict[str, str]]] = {}
            for frame, state in frames.items():
                if state.comments:
                    layer_frames[str(frame)] = [c.to_dict() for c in state.comments]
            if layer_frames:
                out[layer_id] = layer_frames
        return out

    def load_from_dict(self, data: dict[str, list[dict[str, object]]]) -> None:
        """Replace **the current layer's** state from a v1-shaped dict.
        Skips malformed rows."""
        self._layers[self._current_layer_id] = {}
        for frame_str, comment_dicts in data.items():
            try:
                frame = int(frame_str)
            except (TypeError, ValueError):
                continue
            kept: list[Comment] = []
            for cd in comment_dicts:
                try:
                    kept.append(Comment.from_dict(cd))
                except (KeyError, TypeError, ValueError):
                    log.warning(
                        "[comment] dropping malformed comment on frame %d",
                        frame,
                    )
                    continue
            if kept:
                self._layers[self._current_layer_id][frame] = _FrameState(comments=kept)
        # Loading is the "now matches disk" moment; reset dirty.
        self._dirty = False
        self.commented_frames_changed.emit()

    def load_from_dict_multi(
        self, data: dict[str, dict[str, list[dict[str, object]]]],
    ) -> None:
        """Replace **all** layers' state from a v2-shaped dict.

        Mirrors :meth:`AnnotationStore.load_from_dict_multi`.
        """
        self._layers.clear()
        for layer_id, frames_dict in data.items():
            layer_frames: dict[int, _FrameState] = {}
            for frame_str, comment_dicts in frames_dict.items():
                try:
                    frame = int(frame_str)
                except (TypeError, ValueError):
                    continue
                kept: list[Comment] = []
                for cd in comment_dicts:
                    try:
                        kept.append(Comment.from_dict(cd))
                    except (KeyError, TypeError, ValueError):
                        log.warning(
                            "[comment] dropping malformed comment on frame %d",
                            frame,
                        )
                        continue
                if kept:
                    layer_frames[frame] = _FrameState(comments=kept)
            if layer_frames:
                self._layers[str(layer_id)] = layer_frames
        self._layers.setdefault(self._current_layer_id, {})
        self._dirty = False
        self.commented_frames_changed.emit()
