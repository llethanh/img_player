"""Tests for :class:`img_player.annotate.toolbar.AnnotationToolbar`.

The toolbar is mostly UI, but several behaviours are public-API
contracts worth pinning so a future refactor of the inner widgets
(when we replace text labels with proper SVG icons) can't silently
change them. We use pytest-qt's ``qtbot`` fixture to drive the widget
without bringing up the whole app.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

from img_player.annotate import (
    DEFAULT_COLOR,
    DEFAULT_SIZE,
    MAX_SIZE,
    PALETTE,
    AnnotationToolbar,
    ToolbarMode,
    ToolKind,
)


# ============================================================================
# Fixture: build a toolbar with stand-in viewport + dock so we don't
# have to spin up a full GLViewport (which would need a GL context).
# ============================================================================


@pytest.fixture
def toolbar_setup(qtbot):  # type: ignore[no-untyped-def]
    """Return a tuple of ``(toolbar, fake_viewport, fake_dock, main_window)``.

    The ``fake_viewport`` is a plain ``QWidget`` that the toolbar
    re-parents into when in float mode. It needs to be a real widget
    with non-zero size so the drag-clamp math has bounds to work
    with — we don't actually paint into it.
    """
    main_window = QMainWindow()
    main_window.resize(1024, 768)
    qtbot.addWidget(main_window)

    fake_viewport = QWidget(main_window)
    fake_viewport.resize(800, 600)

    fake_dock = QDockWidget("Annotations", main_window)

    toolbar = AnnotationToolbar(
        fake_viewport,  # stands in for GLViewport — only needs .width/.height/.mapFromGlobal
        fake_dock,
        initial_mode=ToolbarMode.FLOAT,
        initial_floating_pos=(12, 12),
        parent=main_window,
    )
    return toolbar, fake_viewport, fake_dock, main_window


# ============================================================================
# Construction
# ============================================================================


class TestConstruction:
    def test_initial_defaults(self, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        assert toolbar.mode() == ToolbarMode.FLOAT
        assert toolbar.current_tool() == ToolKind.NONE
        assert toolbar.current_color() == DEFAULT_COLOR
        assert toolbar.current_size() == DEFAULT_SIZE
        assert toolbar.floating_pos() == (12, 12)

    def test_initial_dock_mode_parents_into_dock(self, qtbot) -> None:  # type: ignore[no-untyped-def]
        """When constructed with DOCK mode, the toolbar should already
        be inside the QDockWidget — not free-floating."""
        main_window = QMainWindow()
        main_window.resize(1024, 768)
        qtbot.addWidget(main_window)
        fake_viewport = QWidget(main_window)
        fake_viewport.resize(800, 600)
        fake_dock = QDockWidget("A", main_window)

        toolbar = AnnotationToolbar(
            fake_viewport,
            fake_dock,
            initial_mode=ToolbarMode.DOCK,
            parent=main_window,
        )
        assert toolbar.mode() == ToolbarMode.DOCK
        assert fake_dock.widget() is toolbar


# ============================================================================
# Tool selection (pen / eraser, with mutual exclusion)
# ============================================================================


class TestToolSelection:
    def test_set_current_tool_emits_signal(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        with qtbot.waitSignal(toolbar.tool_changed, timeout=500) as blocker:
            toolbar.set_current_tool(ToolKind.PEN)
        assert blocker.args == [ToolKind.PEN]
        assert toolbar.current_tool() == ToolKind.PEN

    def test_set_same_tool_does_not_re_emit(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        """No-op idempotency: setting the active tool twice should not
        spam connected slots."""
        toolbar, *_ = toolbar_setup
        toolbar.set_current_tool(ToolKind.PEN)
        events: list[ToolKind] = []
        toolbar.tool_changed.connect(events.append)
        toolbar.set_current_tool(ToolKind.PEN)
        assert events == []

    def test_switching_pen_to_eraser_keeps_one_active(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        """Pen and eraser are mutually exclusive — activating one
        deactivates the other (the pen button must un-check)."""
        toolbar, *_ = toolbar_setup
        toolbar.set_current_tool(ToolKind.PEN)
        assert toolbar._pen_btn.isChecked()
        assert not toolbar._eraser_btn.isChecked()

        toolbar.set_current_tool(ToolKind.ERASER)
        assert not toolbar._pen_btn.isChecked()
        assert toolbar._eraser_btn.isChecked()

    def test_set_current_tool_to_none_unchecks_all(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        toolbar.set_current_tool(ToolKind.PEN)
        toolbar.set_current_tool(ToolKind.NONE)
        assert not toolbar._pen_btn.isChecked()
        assert not toolbar._eraser_btn.isChecked()


# ============================================================================
# Color palette
# ============================================================================


class TestColorPalette:
    def test_palette_has_seven_colors(self, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        assert len(PALETTE) == 7
        assert len(toolbar._swatches) == 7

    def test_set_current_color_emits_signal(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        new_color = PALETTE[2]  # green
        with qtbot.waitSignal(toolbar.color_changed, timeout=500) as blocker:
            toolbar.set_current_color(new_color)
        assert blocker.args == [new_color]
        assert toolbar.current_color() == new_color

    def test_set_non_palette_color_is_ignored(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        """The toolbar advertises a fixed palette — silently rejecting
        off-palette colors prevents drift if a future caller passes
        something unexpected."""
        toolbar, *_ = toolbar_setup
        events: list[str] = []
        toolbar.color_changed.connect(events.append)
        toolbar.set_current_color("#123456")  # not in PALETTE
        assert events == []
        assert toolbar.current_color() == DEFAULT_COLOR

    def test_swatch_check_state_reflects_active(self, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        toolbar.set_current_color(PALETTE[3])  # blue
        for sw in toolbar._swatches:
            assert sw.isChecked() == (sw.color_hex == PALETTE[3])


# ============================================================================
# Size slider
# ============================================================================


class TestSize:
    def test_set_size_clamps_within_bounds(self, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        toolbar.set_current_size(0.5)  # below MIN_SIZE
        assert toolbar.current_size() == 1.0
        toolbar.set_current_size(99.0)  # above MAX_SIZE
        assert toolbar.current_size() == MAX_SIZE

    def test_set_size_emits_signal(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        with qtbot.waitSignal(toolbar.size_changed, timeout=500) as blocker:
            toolbar.set_current_size(20.0)
        assert blocker.args == [20.0]


# ============================================================================
# Mode switching (float ⇄ dock)
# ============================================================================


class TestModeSwitching:
    def test_set_mode_float_to_dock_reparents(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, _viewport, dock, _ = toolbar_setup
        with qtbot.waitSignal(toolbar.mode_changed, timeout=500) as blocker:
            toolbar.set_mode(ToolbarMode.DOCK)
        assert blocker.args == [ToolbarMode.DOCK]
        assert toolbar.mode() == ToolbarMode.DOCK
        assert dock.widget() is toolbar

    def test_set_mode_dock_to_float_reparents_to_viewport(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, viewport, dock, _ = toolbar_setup
        toolbar.set_mode(ToolbarMode.DOCK)
        assert dock.widget() is toolbar
        toolbar.set_mode(ToolbarMode.FLOAT)
        assert toolbar.mode() == ToolbarMode.FLOAT
        assert toolbar.parent() is viewport
        assert dock.widget() is None

    def test_set_same_mode_is_noop(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        events: list[ToolbarMode] = []
        toolbar.mode_changed.connect(events.append)
        toolbar.set_mode(ToolbarMode.FLOAT)  # already FLOAT
        assert events == []


# ============================================================================
# Undo/redo signals
# ============================================================================


class TestUndoRedoSignals:
    def test_undo_button_emits_request(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        with qtbot.waitSignal(toolbar.undo_requested, timeout=500):
            toolbar._undo_btn.click()

    def test_redo_button_emits_request(self, qtbot, toolbar_setup) -> None:  # type: ignore[no-untyped-def]
        toolbar, *_ = toolbar_setup
        with qtbot.waitSignal(toolbar.redo_requested, timeout=500):
            toolbar._redo_btn.click()
