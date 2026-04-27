"""Tests for slice 4 — timeline markers + transport buttons.

The visual paint of the markers can't be unit-tested without a Qt
event loop AND a real paint surface — verified manually in the PR
test plan. What IS unit-testable:

* :meth:`Timeline.set_annotated_frames` — idempotent guard, state
  storage.
* :meth:`TransportBar.set_annotation_nav_enabled` — button enabled
  states reflect the input flags.
* :meth:`TransportBar.set_annotation_toggle_active` — checked state
  reflects the input flag, signal-blocked while we mutate.
* The 3 transport-bar signals (``annotation_*_clicked``) fire when
  the user clicks the buttons.
"""

from __future__ import annotations

import pytest

from img_player.ui.timeline import Timeline
from img_player.ui.transport import TransportBar


# ============================================================================
# Timeline annotation markers
# ============================================================================


class TestTimelineAnnotatedFrames:
    def test_initial_set_is_empty(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        timeline = Timeline()
        qtbot.addWidget(timeline)
        assert timeline._annotated_frames == frozenset()

    def test_set_stores_state(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        timeline = Timeline()
        qtbot.addWidget(timeline)
        frames = frozenset({1, 5, 10})
        timeline.set_annotated_frames(frames)
        assert timeline._annotated_frames == frames

    def test_same_set_is_noop(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """Idempotency — a no-op set call must not trigger an update.
        ``QWidget.update()`` is hard to mock, so we just verify the
        guard short-circuits before mutating state."""
        timeline = Timeline()
        qtbot.addWidget(timeline)
        frames = frozenset({42})
        timeline.set_annotated_frames(frames)
        # Re-passing the same set returns without changing state — we
        # confirm the same instance is held (frozenset is hashable).
        timeline.set_annotated_frames(frames)
        assert timeline._annotated_frames is frames


# ============================================================================
# Transport annotation buttons — enabled state
# ============================================================================


class TestTransportNavEnabled:
    def test_both_disabled_initially(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        transport = TransportBar()
        qtbot.addWidget(transport)
        assert transport._annotation_prev_btn.isEnabled() is False
        assert transport._annotation_next_btn.isEnabled() is False

    def test_set_enables_each_independently(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        transport = TransportBar()
        qtbot.addWidget(transport)
        transport.set_annotation_nav_enabled(prev_avail=True, next_avail=False)
        assert transport._annotation_prev_btn.isEnabled() is True
        assert transport._annotation_next_btn.isEnabled() is False

        transport.set_annotation_nav_enabled(prev_avail=False, next_avail=True)
        assert transport._annotation_prev_btn.isEnabled() is False
        assert transport._annotation_next_btn.isEnabled() is True

    def test_set_both_true(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        transport = TransportBar()
        qtbot.addWidget(transport)
        transport.set_annotation_nav_enabled(prev_avail=True, next_avail=True)
        assert transport._annotation_prev_btn.isEnabled() is True
        assert transport._annotation_next_btn.isEnabled() is True


# ============================================================================
# Transport annotation toggle button — checked state
# ============================================================================


class TestTransportToggleActive:
    def test_initially_unchecked(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        transport = TransportBar()
        qtbot.addWidget(transport)
        assert transport._annotation_toggle_btn.isChecked() is False

    def test_set_active_true(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        transport = TransportBar()
        qtbot.addWidget(transport)
        transport.set_annotation_toggle_active(True)
        assert transport._annotation_toggle_btn.isChecked() is True

    def test_set_active_does_not_emit_clicked(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """Programmatic activation (via ``set_annotation_toggle_active``)
        must not feedback-loop into ``annotation_toggle_clicked`` — it
        would otherwise re-trigger toolbar visibility flipping in
        the app."""
        transport = TransportBar()
        qtbot.addWidget(transport)
        events: list[None] = []
        transport.annotation_toggle_clicked.connect(lambda: events.append(None))
        transport.set_annotation_toggle_active(True)
        transport.set_annotation_toggle_active(False)
        assert events == []

    def test_idempotent_set(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """Setting to the same state does nothing harmful."""
        transport = TransportBar()
        qtbot.addWidget(transport)
        transport.set_annotation_toggle_active(True)
        transport.set_annotation_toggle_active(True)
        assert transport._annotation_toggle_btn.isChecked() is True


# ============================================================================
# Click signals
# ============================================================================


class TestTransportClickSignals:
    def test_prev_button_emits(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        transport = TransportBar()
        qtbot.addWidget(transport)
        transport._annotation_prev_btn.setEnabled(True)
        with qtbot.waitSignal(transport.annotation_prev_clicked, timeout=500):
            transport._annotation_prev_btn.click()

    def test_next_button_emits(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        transport = TransportBar()
        qtbot.addWidget(transport)
        transport._annotation_next_btn.setEnabled(True)
        with qtbot.waitSignal(transport.annotation_next_clicked, timeout=500):
            transport._annotation_next_btn.click()

    def test_toggle_button_emits_on_user_click(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """The user clicking the ✏ button DOES emit annotation_toggle_clicked
        — only the programmatic ``set_annotation_toggle_active`` path
        is signal-blocked."""
        transport = TransportBar()
        qtbot.addWidget(transport)
        with qtbot.waitSignal(transport.annotation_toggle_clicked, timeout=500):
            transport._annotation_toggle_btn.click()

    def test_disabled_prev_button_does_not_emit(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """Sanity: a disabled button shouldn't reach its slot — Qt
        normally enforces this, but it's worth pinning so a future
        styling change can't accidentally re-enable click delivery."""
        transport = TransportBar()
        qtbot.addWidget(transport)
        transport._annotation_prev_btn.setEnabled(False)
        events: list[None] = []
        transport.annotation_prev_clicked.connect(lambda: events.append(None))
        transport._annotation_prev_btn.click()
        assert events == []
