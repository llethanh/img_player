"""Tests for :class:`img_player.comment.store.CommentStore`.

Covers per-frame add / edit / delete, signal emission counts,
dirty tracking, and the JSON round-trip via to_dict / load_from_dict.
"""

from __future__ import annotations

import time

import pytest

from img_player.comment.comment import Comment
from img_player.comment.store import CommentStore


def _tick() -> None:
    """See ``test_comment.py::_tick`` — guarantees the ``_now()``
    call inside :meth:`Comment.edited` returns a strictly later
    timestamp than the comment's ``created_at`` on fast hardware."""
    time.sleep(0.002)


# ============================================================================
# Read API
# ============================================================================


class TestReadAPI:
    def test_empty_store_has_no_commented_frames(self) -> None:
        store = CommentStore()
        assert store.commented_frames() == frozenset()
        assert store.comments_at(0) == ()

    def test_comments_at_returns_tuple(self) -> None:
        """Read API returns an immutable view so callers can't
        accidentally mutate the store from outside."""
        store = CommentStore()
        store.add_comment(42, "hello")
        assert isinstance(store.comments_at(42), tuple)


# ============================================================================
# add / edit / delete
# ============================================================================


class TestAddComment:
    def test_add_appends_in_order(self) -> None:
        store = CommentStore()
        c1 = store.add_comment(42, "first")
        c2 = store.add_comment(42, "second")
        comments = store.comments_at(42)
        assert (comments[0], comments[1]) == (c1, c2)

    def test_add_returns_the_new_comment(self) -> None:
        """The caller (the panel) needs the new comment back to
        clear its input field, etc."""
        store = CommentStore()
        c = store.add_comment(42, "hello")
        assert isinstance(c, Comment)
        assert c.text == "hello"

    def test_add_strips_whitespace(self) -> None:
        store = CommentStore()
        c = store.add_comment(42, "   spaced   ")
        assert c.text == "spaced"

    def test_add_empty_text_raises(self) -> None:
        store = CommentStore()
        with pytest.raises(ValueError):
            store.add_comment(42, "")
        with pytest.raises(ValueError):
            store.add_comment(42, "   ")

    def test_add_distinct_frames_isolated(self) -> None:
        store = CommentStore()
        store.add_comment(10, "ten")
        store.add_comment(20, "twenty")
        assert store.commented_frames() == frozenset({10, 20})


class TestEditComment:
    def test_edit_replaces_text_and_advances_updated_at(self) -> None:
        store = CommentStore()
        c = store.add_comment(42, "first")
        _tick()  # ensure the updated_at of the edited copy differs
        ok = store.edit_comment(42, c.id, "second")
        assert ok is True
        revised = store.comments_at(42)[0]
        assert revised.text == "second"
        assert revised.id == c.id
        assert revised.is_edited is True

    def test_edit_unknown_id_returns_false(self) -> None:
        store = CommentStore()
        store.add_comment(42, "hello")
        assert store.edit_comment(42, "non-existent-id", "new") is False

    def test_edit_unknown_frame_returns_false(self) -> None:
        store = CommentStore()
        assert store.edit_comment(99, "anything", "new") is False

    def test_edit_empty_text_raises(self) -> None:
        store = CommentStore()
        c = store.add_comment(42, "hello")
        with pytest.raises(ValueError):
            store.edit_comment(42, c.id, "")


class TestDeleteComment:
    def test_delete_by_id(self) -> None:
        store = CommentStore()
        c1 = store.add_comment(42, "first")
        c2 = store.add_comment(42, "second")
        ok = store.delete_comment(42, c1.id)
        assert ok is True
        assert store.comments_at(42) == (c2,)

    def test_delete_unknown_id_returns_false(self) -> None:
        store = CommentStore()
        store.add_comment(42, "hello")
        assert store.delete_comment(42, "nope") is False

    def test_delete_last_removes_frame_from_set(self) -> None:
        store = CommentStore()
        c = store.add_comment(42, "hello")
        store.delete_comment(42, c.id)
        assert store.commented_frames() == frozenset()


# ============================================================================
# Signals
# ============================================================================


class TestSignals:
    def test_add_emits_frame_comments_changed(self) -> None:
        store = CommentStore()
        events: list[int] = []
        store.frame_comments_changed.connect(events.append)
        store.add_comment(42, "hello")
        assert events == [42]

    def test_add_to_empty_frame_emits_commented_frames_changed(self) -> None:
        store = CommentStore()
        ev: list[None] = []
        store.commented_frames_changed.connect(lambda: ev.append(None))
        store.add_comment(42, "hello")
        assert len(ev) == 1

    def test_add_to_non_empty_does_not_re_emit_set(self) -> None:
        store = CommentStore()
        store.add_comment(42, "first")
        ev: list[None] = []
        store.commented_frames_changed.connect(lambda: ev.append(None))
        store.add_comment(42, "second")
        assert ev == []

    def test_delete_last_emits_commented_frames_changed(self) -> None:
        store = CommentStore()
        c = store.add_comment(42, "hello")
        ev: list[None] = []
        store.commented_frames_changed.connect(lambda: ev.append(None))
        store.delete_comment(42, c.id)
        assert len(ev) == 1


# ============================================================================
# Dirty tracking
# ============================================================================


class TestDirtyTracking:
    def test_initially_clean(self) -> None:
        assert CommentStore().is_dirty() is False

    def test_add_marks_dirty(self) -> None:
        store = CommentStore()
        store.add_comment(42, "hello")
        assert store.is_dirty() is True

    def test_edit_marks_dirty(self) -> None:
        store = CommentStore()
        c = store.add_comment(42, "first")
        store.mark_clean()
        store.edit_comment(42, c.id, "second")
        assert store.is_dirty() is True

    def test_delete_marks_dirty(self) -> None:
        store = CommentStore()
        c = store.add_comment(42, "hello")
        store.mark_clean()
        store.delete_comment(42, c.id)
        assert store.is_dirty() is True

    def test_load_from_dict_resets_clean(self) -> None:
        store = CommentStore()
        store.add_comment(42, "hello")
        assert store.is_dirty()
        store.load_from_dict({})
        assert store.is_dirty() is False


# ============================================================================
# Persistence helpers (to_dict / load_from_dict)
# ============================================================================


class TestStoreSerialization:
    def test_to_dict_only_includes_non_empty_frames(self) -> None:
        store = CommentStore()
        c = store.add_comment(10, "hello")
        store.add_comment(20, "world")
        store.delete_comment(10, c.id)
        out = store.to_dict()
        assert "10" not in out
        assert "20" in out

    def test_round_trip(self) -> None:
        store = CommentStore()
        c1 = store.add_comment(42, "first")
        c2 = store.add_comment(42, "second")

        payload = store.to_dict()

        restored = CommentStore()
        restored.load_from_dict(payload)
        assert restored.commented_frames() == frozenset({42})
        assert restored.comments_at(42) == (c1, c2)

    def test_load_skips_malformed_rows(self) -> None:
        store = CommentStore()
        payload = {
            "42": [
                {
                    "id": "a",
                    "text": "good",
                    "author": "alice",
                    "created_at": "2026-04-27T18:00:00+00:00",
                    "updated_at": "2026-04-27T18:00:00+00:00",
                },
                {  # malformed: missing 'text'
                    "id": "b",
                    "author": "alice",
                    "created_at": "2026-04-27T18:00:00+00:00",
                    "updated_at": "2026-04-27T18:00:00+00:00",
                },
            ],
        }
        store.load_from_dict(payload)
        assert len(store.comments_at(42)) == 1

    def test_load_emits_commented_frames_changed(self) -> None:
        store = CommentStore()
        ev: list[None] = []
        store.commented_frames_changed.connect(lambda: ev.append(None))
        store.load_from_dict({
            "42": [
                {
                    "id": "a",
                    "text": "x",
                    "author": "alice",
                    "created_at": "2026-04-27T18:00:00+00:00",
                    "updated_at": "2026-04-27T18:00:00+00:00",
                },
            ],
        })
        assert len(ev) == 1


# ============================================================================
# Layer scoping (v1.5.15+) — current_layer_id partitions comments
# ============================================================================


class TestLayerScoping:
    def test_default_layer_id_is_empty_string(self) -> None:
        """No layer focused yet → reads / writes route through the
        ``""`` key. Same convention as ``AnnotationStore``."""
        store = CommentStore()
        assert store.current_layer_id == ""

    def test_comments_partition_by_layer(self) -> None:
        """Adding a comment under layer A doesn't leak into layer B's
        view; switching ``current_layer_id`` swaps the visible set."""
        store = CommentStore()
        store.set_current_layer_id("layer-A")
        store.add_comment(42, "hero take")
        assert len(store.comments_at(42)) == 1

        store.set_current_layer_id("layer-B")
        assert store.comments_at(42) == ()

        store.add_comment(42, "alt take")
        assert store.comments_at(42)[0].text == "alt take"

        store.set_current_layer_id("layer-A")
        assert store.comments_at(42)[0].text == "hero take"

    def test_set_current_layer_id_emits_changed_signal(self, qtbot) -> None:
        """Focus switch fires ``commented_frames_changed`` so the
        timeline + comment panel refresh as if the set was
        recomputed."""
        store = CommentStore()
        store.set_current_layer_id("layer-A")
        store.add_comment(42, "x")

        with qtbot.waitSignal(store.commented_frames_changed, timeout=500):
            store.set_current_layer_id("layer-B")

    def test_layer_frame_comments_changed_carries_layer_id(
        self, qtbot,
    ) -> None:
        store = CommentStore()
        store.set_current_layer_id("layer-A")
        with qtbot.waitSignal(
            store.layer_frame_comments_changed, timeout=500,
        ) as blocker:
            store.add_comment(42, "hero")
        assert blocker.args == ["layer-A", 42]

    def test_comments_at_for_reads_any_layer(self) -> None:
        store = CommentStore()
        store.set_current_layer_id("layer-A")
        store.add_comment(42, "hero")
        store.set_current_layer_id("layer-B")
        assert store.comments_at_for("layer-A", 42)[0].text == "hero"
        assert store.comments_at_for("layer-B", 42) == ()
        # Unknown layer → empty, no KeyError.
        assert store.comments_at_for("layer-nonexistent", 42) == ()

    def test_layers_with_comments_enumerates_non_empty_keys(self) -> None:
        store = CommentStore()
        store.set_current_layer_id("layer-A")
        store.add_comment(1, "x")
        store.set_current_layer_id("layer-B")
        store.add_comment(2, "y")
        store.set_current_layer_id("layer-C")  # no comments
        assert store.layers_with_comments() == frozenset({"layer-A", "layer-B"})

    def test_to_dict_multi_round_trip_preserves_layers(self) -> None:
        store = CommentStore()
        store.set_current_layer_id("layer-A")
        store.add_comment(1, "hero")
        store.set_current_layer_id("layer-B")
        store.add_comment(2, "alt")
        dump = store.to_dict_multi()
        assert set(dump.keys()) == {"layer-A", "layer-B"}

        # Reload into a fresh store; both layers' comments come back.
        fresh = CommentStore()
        fresh.load_from_dict_multi(dump)
        fresh.set_current_layer_id("layer-A")
        assert fresh.comments_at(1)[0].text == "hero"
        fresh.set_current_layer_id("layer-B")
        assert fresh.comments_at(2)[0].text == "alt"
