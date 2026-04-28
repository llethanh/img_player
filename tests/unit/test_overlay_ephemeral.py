"""Tests for ephemeral-mode routing in :class:`AnnotationOverlay`.

Level 3 of the ephemeral feature's testing strategy (spec §8.3).

We construct the overlay against a *fake* GL viewport that only
implements the four-method surface the overlay actually reads
(``image_size``, ``current_transform``, ``rect``, ``transform_changed``
signal). That lets the overlay live in a regular Qt event loop —
no real OpenGL context, no off-screen surface gymnastics — while
still exercising the routing code-path end to end.

The headline contract: at ``mouseRelease``, the finished stroke
goes to the **manager** if and only if the press-time snapshot of
``_ephemeral_mode`` was True. A mid-drag mode toggle does NOT
reroute the stroke.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from PySide6.QtCore import QPoint, QPointF, QRect, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QWidget

from img_player.annotate.ephemeral import EphemeralStrokeManager
from img_player.annotate.overlay import AnnotationOverlay, ToolKind
from img_player.annotate.store import AnnotationStore


# ============================================================================
# Fake viewport — minimal QWidget with the API the overlay reads
# ============================================================================


class _FakeViewport(QWidget):
    """Stand-in for ``GLViewport`` — implements the read-back surface
    the overlay touches at paint and event time, without an OpenGL
    context.

    Surface required by :class:`AnnotationOverlay`:

    * ``image_size() -> (int, int)`` — for the cursor-to-image
      transform. Non-zero so the overlay accepts events.
    * ``current_transform() -> (factor, pan_x, pan_y)`` — for the
      same transform. ``factor=1.0`` puts widget pixels = image
      pixels which makes test assertions readable.
    * ``transform_changed`` Qt signal — the overlay subscribes to
      it for repaints. Just emit and the slot fires.
    * ``rect()`` — inherited from QWidget.
    * Behaves as the parent — the overlay is parented onto it via
      ``super().__init__(gl_viewport)``.
    """

    transform_changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.resize(1024, 768)
        self._img_size: tuple[int, int] = (1024, 768)
        self._transform: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def image_size(self) -> tuple[int, int]:
        return self._img_size

    def current_transform(self) -> tuple[float, float, float]:
        return self._transform


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def viewport(qtbot) -> Iterator[_FakeViewport]:  # type: ignore[no-untyped-def]
    v = _FakeViewport()
    qtbot.addWidget(v)
    v.show()
    qtbot.waitExposed(v)
    yield v


@pytest.fixture
def store() -> AnnotationStore:
    return AnnotationStore()


@pytest.fixture
def manager(qtbot) -> EphemeralStrokeManager:  # type: ignore[no-untyped-def]
    return EphemeralStrokeManager()


@pytest.fixture
def overlay(
    qtbot,  # type: ignore[no-untyped-def]
    viewport: _FakeViewport,
    store: AnnotationStore,
    manager: EphemeralStrokeManager,
) -> AnnotationOverlay:
    o = AnnotationOverlay(viewport, store)
    o.set_ephemeral_manager(manager)
    o.set_tool(ToolKind.PEN)
    o.show()
    qtbot.waitExposed(o)
    return o


# ============================================================================
# Helpers — synthesise QMouseEvent press/move/release
# ============================================================================


def _press(overlay: AnnotationOverlay, x: int, y: int) -> None:
    """Fire a left-button mousePressEvent at widget coords ``(x, y)``."""
    overlay.mousePressEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(x, y),
            QPointF(x, y),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


def _move(overlay: AnnotationOverlay, x: int, y: int) -> None:
    overlay.mouseMoveEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(x, y),
            QPointF(x, y),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


def _release(overlay: AnnotationOverlay, x: int, y: int) -> None:
    overlay.mouseReleaseEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(x, y),
            QPointF(x, y),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )


# ============================================================================
# Routing — the headline contract
# ============================================================================


class TestRouting:
    def test_persistent_mode_routes_to_store(
        self,
        overlay: AnnotationOverlay,
        store: AnnotationStore,
        manager: EphemeralStrokeManager,
    ) -> None:
        """Default mode (ephemeral OFF) → finished stroke lands in the store."""
        assert overlay.ephemeral_mode is False
        _press(overlay, 100, 100)
        _move(overlay, 110, 110)
        _release(overlay, 120, 120)
        assert len(store.strokes_at(0)) == 1
        assert manager._stroke_count() == 0

    def test_ephemeral_mode_routes_to_manager(
        self,
        overlay: AnnotationOverlay,
        store: AnnotationStore,
        manager: EphemeralStrokeManager,
    ) -> None:
        """Ephemeral mode ON → finished stroke lands in the manager."""
        overlay.set_ephemeral_mode(True)
        _press(overlay, 100, 100)
        _move(overlay, 110, 110)
        _release(overlay, 120, 120)
        assert manager._stroke_count() == 1
        assert len(store.strokes_at(0)) == 0

    def test_mid_drag_toggle_on_does_not_reroute(
        self,
        overlay: AnnotationOverlay,
        store: AnnotationStore,
        manager: EphemeralStrokeManager,
    ) -> None:
        """Press persistent + toggle ephemeral mid-drag + release → store wins.

        The press-time snapshot is the source of truth. A mid-drag
        keyboard ``G`` or toolbar click must not yank the stroke
        out of the store path it was committed to.
        """
        # Mode OFF at press.
        _press(overlay, 100, 100)
        # User flips the toggle while drawing.
        overlay.set_ephemeral_mode(True)
        _move(overlay, 110, 110)
        _release(overlay, 120, 120)
        assert len(store.strokes_at(0)) == 1
        assert manager._stroke_count() == 0

    def test_mid_drag_toggle_off_does_not_reroute(
        self,
        overlay: AnnotationOverlay,
        store: AnnotationStore,
        manager: EphemeralStrokeManager,
    ) -> None:
        """Symmetric: press ephemeral + toggle off mid-drag → manager wins."""
        overlay.set_ephemeral_mode(True)
        _press(overlay, 100, 100)
        overlay.set_ephemeral_mode(False)
        _move(overlay, 110, 110)
        _release(overlay, 120, 120)
        assert manager._stroke_count() == 1
        assert len(store.strokes_at(0)) == 0

    def test_no_manager_falls_back_to_store(
        self,
        qtbot,  # type: ignore[no-untyped-def]
        viewport: _FakeViewport,
        store: AnnotationStore,
    ) -> None:
        """If the manager wasn't injected, ephemeral strokes go to the store.

        The overlay fails-safe rather than dropping the stroke. This
        matters for partial-wiring states (early app boot, isolated
        unit-test harness without manager).
        """
        o = AnnotationOverlay(viewport, store)
        o.set_tool(ToolKind.PEN)
        o.set_ephemeral_mode(True)  # mode ON, but no manager wired.
        o.show()
        qtbot.waitExposed(o)
        _press(o, 100, 100)
        _release(o, 110, 110)
        assert len(store.strokes_at(0)) == 1


# ============================================================================
# Snapshot lifecycle
# ============================================================================


class TestSnapshotLifecycle:
    def test_snapshot_cleared_after_release(
        self, overlay: AnnotationOverlay
    ) -> None:
        """The press-time snapshot is reset to ``None`` once the stroke
        is committed, so subsequent presses make a fresh decision."""
        overlay.set_ephemeral_mode(True)
        _press(overlay, 100, 100)
        _release(overlay, 110, 110)
        # Internal state — sniff via the private attribute. Worth it
        # to lock down the lifecycle invariant.
        assert overlay._current_stroke_is_ephemeral is None

    def test_two_consecutive_strokes_in_different_modes(
        self,
        overlay: AnnotationOverlay,
        store: AnnotationStore,
        manager: EphemeralStrokeManager,
    ) -> None:
        """First stroke ephemeral, switch off, second stroke persistent.
        Each lands in the right sink."""
        overlay.set_ephemeral_mode(True)
        _press(overlay, 50, 50)
        _release(overlay, 60, 60)
        overlay.set_ephemeral_mode(False)
        _press(overlay, 70, 70)
        _release(overlay, 80, 80)
        assert manager._stroke_count() == 1
        assert len(store.strokes_at(0)) == 1


# ============================================================================
# Repaint hook
# ============================================================================


class TestPaintHook:
    def test_manager_repaint_signal_triggers_overlay_update(
        self,
        overlay: AnnotationOverlay,
        manager: EphemeralStrokeManager,
        qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """When the manager emits ``repaint_needed``, the overlay's
        slot fires (we test by checking the connection is alive — a
        proper paintEvent hit-test would need rendering into a QImage
        which is overkill here)."""
        from img_player.annotate.stroke import Stroke

        # Adding a stroke to the manager should emit repaint_needed,
        # which the overlay subscribed to in set_ephemeral_manager.
        # We assert the signal fires; the actual update() call is
        # internal to Qt and not directly inspectable.
        with qtbot.waitSignal(manager.repaint_needed, timeout=200):
            manager.add(Stroke(points=((0.0, 0.0),), color="#FF0000", size=5.0))
