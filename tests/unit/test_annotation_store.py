"""Tests for :class:`img_player.annotate.store.AnnotationStore`.

Covers per-frame strokes, per-frame undo/redo isolation, signal
emission counts, and the JSON round-trip via to_dict / load_from_dict.
"""

from __future__ import annotations

import pytest

from img_player.annotate.store import Action, ActionKind, AnnotationStore
from img_player.annotate.stroke import Stroke


def _stroke(color: str = "#FF0000", size: float = 5.0) -> Stroke:
    """Quick stroke factory for tests that don't care about the geometry."""
    return Stroke(points=((0.0, 0.0), (10.0, 10.0)), color=color, size=size)


# ============================================================================
# Read API
# ============================================================================


class TestReadAPI:
    def test_empty_store_has_no_annotated_frames(self) -> None:
        store = AnnotationStore()
        assert store.annotated_frames() == frozenset()
        assert store.strokes_at(0) == ()

    def test_strokes_at_returns_tuple_not_list(self) -> None:
        """The read API returns an immutable view so callers can't
        accidentally mutate the store from outside."""
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        result = store.strokes_at(42)
        assert isinstance(result, tuple)

    def test_annotated_frames_only_includes_non_empty(self) -> None:
        """A frame that had a stroke and then was cleared via
        remove_stroke must NOT appear in annotated_frames()."""
        store = AnnotationStore()
        store.add_stroke(10, _stroke())
        store.add_stroke(20, _stroke())
        store.remove_stroke(10, 0)
        assert store.annotated_frames() == frozenset({20})


# ============================================================================
# Mutation
# ============================================================================


class TestAddStroke:
    def test_add_appends_in_order(self) -> None:
        store = AnnotationStore()
        s1 = _stroke(color="#FF0000")
        s2 = _stroke(color="#00FF00")
        store.add_stroke(42, s1)
        store.add_stroke(42, s2)
        assert store.strokes_at(42) == (s1, s2)

    def test_add_to_distinct_frames_does_not_cross_contaminate(self) -> None:
        store = AnnotationStore()
        s1 = _stroke(color="#FF0000")
        s2 = _stroke(color="#00FF00")
        store.add_stroke(10, s1)
        store.add_stroke(20, s2)
        assert store.strokes_at(10) == (s1,)
        assert store.strokes_at(20) == (s2,)

    def test_add_pushes_undo_clears_redo(self) -> None:
        """User actions invalidate the redo stack — pressing Ctrl+Z,
        drawing again, then Ctrl+Y must NOT bring back the undone
        stroke (it would be confusing)."""
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        store.undo(42)
        assert store._redo_stack_size(42) == 1
        store.add_stroke(42, _stroke(color="#00FF00"))
        # The new add wiped the redo stack.
        assert store._redo_stack_size(42) == 0
        assert store._undo_stack_size(42) == 1


class TestRemoveStroke:
    def test_remove_by_index(self) -> None:
        store = AnnotationStore()
        s1 = _stroke(color="#FF0000")
        s2 = _stroke(color="#00FF00")
        s3 = _stroke(color="#0000FF")
        store.add_stroke(42, s1)
        store.add_stroke(42, s2)
        store.add_stroke(42, s3)
        store.remove_stroke(42, 1)  # remove the green one
        assert store.strokes_at(42) == (s1, s3)

    def test_remove_invalid_index_raises(self) -> None:
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        with pytest.raises(IndexError):
            store.remove_stroke(42, 5)

    def test_remove_from_unknown_frame_raises(self) -> None:
        store = AnnotationStore()
        with pytest.raises(IndexError):
            store.remove_stroke(99, 0)


# ============================================================================
# Undo/redo (per-frame isolation is the headline)
# ============================================================================


class TestUndoRedo:
    def test_undo_reverts_add(self) -> None:
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        assert store.undo(42) is True
        assert store.strokes_at(42) == ()

    def test_undo_reverts_remove(self) -> None:
        store = AnnotationStore()
        s = _stroke()
        store.add_stroke(42, s)
        store.remove_stroke(42, 0)
        store.undo(42)
        assert store.strokes_at(42) == (s,)

    def test_redo_replays_undone_action(self) -> None:
        store = AnnotationStore()
        s = _stroke()
        store.add_stroke(42, s)
        store.undo(42)
        assert store.redo(42) is True
        assert store.strokes_at(42) == (s,)

    def test_undo_on_empty_stack_returns_false(self) -> None:
        store = AnnotationStore()
        assert store.undo(42) is False

    def test_redo_on_empty_stack_returns_false(self) -> None:
        store = AnnotationStore()
        assert store.redo(42) is False

    def test_per_frame_stacks_are_isolated(self) -> None:
        """The headline contract: Ctrl+Z while on frame 42 only
        affects frame 42, no surprise frame-jumps."""
        store = AnnotationStore()
        s10 = _stroke(color="#FF0000")
        s20 = _stroke(color="#00FF00")
        store.add_stroke(10, s10)
        store.add_stroke(20, s20)
        # Undo on 20 leaves 10 alone.
        store.undo(20)
        assert store.strokes_at(10) == (s10,)
        assert store.strokes_at(20) == ()
        # Undo on 20 again is a no-op (its stack is empty).
        assert store.undo(20) is False
        # Frame 10's stack is still intact.
        store.undo(10)
        assert store.strokes_at(10) == ()

    def test_redo_after_partial_redo_chain(self) -> None:
        """Undo three, redo one — the next redo brings back the next
        action in chronological order, not the most recent."""
        store = AnnotationStore()
        sa = _stroke(color="#A00000")
        sb = _stroke(color="#00A000")
        sc = _stroke(color="#0000A0")
        store.add_stroke(42, sa)
        store.add_stroke(42, sb)
        store.add_stroke(42, sc)
        store.undo(42)
        store.undo(42)
        store.undo(42)
        assert store.strokes_at(42) == ()
        store.redo(42)
        assert store.strokes_at(42) == (sa,)
        store.redo(42)
        assert store.strokes_at(42) == (sa, sb)


# ============================================================================
# Signals
# ============================================================================


class TestSignals:
    def test_add_emits_frame_annotated(self) -> None:
        store = AnnotationStore()
        events: list[int] = []
        store.frame_annotated.connect(events.append)
        store.add_stroke(42, _stroke())
        assert events == [42]

    def test_add_to_empty_frame_emits_annotated_frames_changed(self) -> None:
        """Going from "no annotation on this frame" to "one
        annotation" changes the timeline-marker set, so listeners
        should be notified."""
        store = AnnotationStore()
        ev: list[None] = []
        store.annotated_frames_changed.connect(lambda: ev.append(None))
        store.add_stroke(42, _stroke())
        assert len(ev) == 1

    def test_add_to_already_annotated_frame_does_not_re_emit_set(self) -> None:
        """Adding a SECOND stroke to frame 42 doesn't change the SET
        of annotated frames — the marker is already there. Only
        frame_annotated fires."""
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        ev: list[None] = []
        store.annotated_frames_changed.connect(lambda: ev.append(None))
        store.add_stroke(42, _stroke(color="#00FF00"))
        assert ev == []

    def test_remove_last_stroke_emits_annotated_frames_changed(self) -> None:
        """Going from "annotated" to "not annotated" — the marker
        must disappear from the timeline."""
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        ev: list[None] = []
        store.annotated_frames_changed.connect(lambda: ev.append(None))
        store.remove_stroke(42, 0)
        assert len(ev) == 1

    def test_undo_emits_correct_signals(self) -> None:
        """Undoing the only stroke on a frame removes it from the
        annotated set — both signals fire."""
        store = AnnotationStore()
        store.add_stroke(42, _stroke())
        frame_events: list[int] = []
        set_events: list[None] = []
        store.frame_annotated.connect(frame_events.append)
        store.annotated_frames_changed.connect(lambda: set_events.append(None))
        store.undo(42)
        assert frame_events == [42]
        assert len(set_events) == 1


# ============================================================================
# Action helpers
# ============================================================================


class TestAction:
    def test_inverse_flips_kind(self) -> None:
        s = _stroke()
        add = Action(ActionKind.ADD, frame=42, idx=0, stroke=s)
        rm = Action(ActionKind.REMOVE, frame=42, idx=0, stroke=s)
        assert add.inverse() == rm
        assert rm.inverse() == add

    def test_inverse_preserves_other_fields(self) -> None:
        s = _stroke()
        add = Action(ActionKind.ADD, frame=42, idx=3, stroke=s)
        inv = add.inverse()
        assert inv.frame == 42
        assert inv.idx == 3
        assert inv.stroke is s


# ============================================================================
# Persistence helpers (to_dict / load_from_dict)
# ============================================================================


class TestStoreSerialization:
    def test_to_dict_only_includes_non_empty_frames(self) -> None:
        """A frame whose strokes were all removed must not appear in
        the serialised payload — it would resurrect as an empty list
        on load and give the wrong annotated_frames() set."""
        store = AnnotationStore()
        store.add_stroke(10, _stroke())
        store.add_stroke(20, _stroke())
        store.remove_stroke(10, 0)
        out = store.to_dict()
        assert "10" not in out["frames"]
        assert "20" in out["frames"]

    def test_to_dict_load_from_dict_round_trip(self) -> None:
        store = AnnotationStore()
        s1 = _stroke(color="#FF0000")
        s2 = _stroke(color="#00FF00")
        store.add_stroke(42, s1)
        store.add_stroke(42, s2)
        store.add_stroke(87, _stroke(color="#0000FF"))

        payload = store.to_dict()

        restored = AnnotationStore()
        restored.load_from_dict(payload["frames"])
        assert restored.annotated_frames() == frozenset({42, 87})
        assert restored.strokes_at(42) == (s1, s2)

    def test_load_clears_existing_state(self) -> None:
        """A fresh load replaces the in-memory state — opening a new
        sequence shouldn't see the previous sequence's strokes."""
        store = AnnotationStore()
        store.add_stroke(42, _stroke(color="#FF0000"))
        store.load_from_dict({})  # empty payload
        assert store.annotated_frames() == frozenset()

    def test_load_skips_malformed_strokes_silently(self) -> None:
        """One bad stroke in a sidecar shouldn't invalidate the rest
        of the file — the broken stroke is dropped, the rest loads."""
        store = AnnotationStore()
        payload = {
            "42": [
                {"color": "#FF0000", "size": 5.0, "points": [[0, 0], [1, 1]]},
                {"color": "not-a-color", "size": 5.0, "points": [[0, 0]]},
            ],
        }
        store.load_from_dict(payload)
        # The valid stroke survived; the bad one was dropped.
        assert len(store.strokes_at(42)) == 1

    def test_load_from_dict_emits_annotated_frames_changed(self) -> None:
        """The timeline must repaint markers after a load — the
        signal fires once at the end (regardless of how many frames
        were loaded)."""
        store = AnnotationStore()
        ev: list[None] = []
        store.annotated_frames_changed.connect(lambda: ev.append(None))
        store.load_from_dict(
            {"42": [{"color": "#FF0000", "size": 5.0, "points": [[0, 0]]}]}
        )
        assert len(ev) == 1
