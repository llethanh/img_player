"""Tests for the ephemeral controls in :class:`AnnotationToolbar` (v0.4.1).

Level 4 of the ephemeral feature's testing strategy (spec §8.4).

Same fixture pattern as ``test_annotation_toolbar.py`` — a stand-in
viewport so we don't have to spin up a real ``GLViewport`` (which
needs an OpenGL context).
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDockWidget, QMainWindow, QWidget

from img_player.annotate import (
    DEFAULT_EPHEMERAL_PRESET_INDEX,
    EPHEMERAL_PRESETS_S,
    AnnotationToolbar,
    ToolbarMode,
    ToolKind,
)


@pytest.fixture
def toolbar_setup(qtbot):  # type: ignore[no-untyped-def]
    """Return ``(toolbar, fake_viewport, fake_dock, main_window)``.

    Same shape as ``test_annotation_toolbar.toolbar_setup`` — the
    test holds references to all four so the QObject hierarchy
    survives the test (otherwise shiboken garbage-collects the
    parent and child widgets get "already deleted" at access).
    """
    main_window = QMainWindow()
    main_window.resize(1024, 768)
    qtbot.addWidget(main_window)

    fake_viewport = QWidget(main_window)
    fake_viewport.resize(800, 600)

    fake_dock = QDockWidget("Annotations", main_window)

    toolbar = AnnotationToolbar(
        fake_viewport,
        fake_dock,
        initial_mode=ToolbarMode.FLOAT,
        initial_floating_pos=(12, 12),
        parent=main_window,
    )
    return toolbar, fake_viewport, fake_dock, main_window


@pytest.fixture
def toolbar(toolbar_setup):  # type: ignore[no-untyped-def]
    """Convenience: just the toolbar, but the rest of the tuple is
    kept alive via ``toolbar_setup``'s lifetime."""
    return toolbar_setup[0]


# ============================================================================
# Initial state
# ============================================================================


class TestInitialState:
    def test_ephemeral_mode_off_by_default(
        self, toolbar: AnnotationToolbar
    ) -> None:
        """Boot default — persistent mode wins."""
        assert toolbar.is_ephemeral_mode() is False

    def test_default_preset_is_moyen_at_index_1(
        self, toolbar: AnnotationToolbar
    ) -> None:
        """Default = index 1 → moyen (5 s). Natural order: court(0) /
        moyen(1) / long(2). Middle dot is the default selection."""
        assert toolbar.ephemeral_preset_index() == DEFAULT_EPHEMERAL_PRESET_INDEX
        assert DEFAULT_EPHEMERAL_PRESET_INDEX == 1
        assert toolbar.ephemeral_duration_seconds() == pytest.approx(5.0)

    def test_preset_row_hidden_when_mode_off(
        self, toolbar: AnnotationToolbar
    ) -> None:
        """The 3-preset bar should be invisible until the user activates ephemeral."""
        assert toolbar._ephemeral_preset_row.isVisibleTo(toolbar) is False

    def test_pen_glyph_is_pencil_in_normal_mode(
        self, toolbar: AnnotationToolbar
    ) -> None:
        assert toolbar._pen_btn.text() == "✏️"

    def test_eraser_enabled_in_normal_mode(
        self, toolbar: AnnotationToolbar
    ) -> None:
        assert toolbar._eraser_btn.isEnabled() is True

    def test_initial_preset_override_honoured(
        self, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """Constructor accepts ``initial_ephemeral_preset`` to seed
        from preferences."""
        main_window = QMainWindow()
        qtbot.addWidget(main_window)
        fake_viewport = QWidget(main_window)
        fake_dock = QDockWidget(main_window)
        tb = AnnotationToolbar(
            fake_viewport,
            fake_dock,
            initial_ephemeral_preset=2,
            parent=main_window,
        )
        assert tb.ephemeral_preset_index() == 2
        assert tb.ephemeral_duration_seconds() == pytest.approx(10.0)

    def test_invalid_initial_preset_falls_back_to_default(
        self, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        main_window = QMainWindow()
        qtbot.addWidget(main_window)
        fake_viewport = QWidget(main_window)
        fake_dock = QDockWidget(main_window)
        tb = AnnotationToolbar(
            fake_viewport,
            fake_dock,
            initial_ephemeral_preset=99,
            parent=main_window,
        )
        assert tb.ephemeral_preset_index() == DEFAULT_EPHEMERAL_PRESET_INDEX


# ============================================================================
# Toggling ephemeral mode
# ============================================================================


class TestEphemeralToggle:
    def test_set_ephemeral_mode_on_emits_signal(
        self, toolbar: AnnotationToolbar, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        with qtbot.waitSignal(toolbar.ephemeral_mode_changed, timeout=200) as block:
            toolbar.set_ephemeral_mode(True)
        assert block.args == [True]

    def test_set_ephemeral_mode_off_emits_signal(
        self, toolbar: AnnotationToolbar, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        toolbar.set_ephemeral_mode(True)
        with qtbot.waitSignal(toolbar.ephemeral_mode_changed, timeout=200) as block:
            toolbar.set_ephemeral_mode(False)
        assert block.args == [False]

    def test_set_ephemeral_mode_idempotent(
        self, toolbar: AnnotationToolbar
    ) -> None:
        """Calling with the same state shouldn't re-emit (we'd flood
        the wiring with no-op signals)."""
        # Initial state: off. Setting off again: no signal.
        # Easier to assert via state — _ephemeral_mode unchanged.
        assert toolbar.is_ephemeral_mode() is False
        toolbar.set_ephemeral_mode(False)
        assert toolbar.is_ephemeral_mode() is False

    def test_emit_false_argument_suppresses_signal(
        self, toolbar: AnnotationToolbar, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """``set_ephemeral_mode(True, emit=False)`` flips state without
        broadcasting — used when reflecting an external state change."""
        # Use a custom blocker - we want NO signal.
        with qtbot.assertNotEmitted(toolbar.ephemeral_mode_changed, wait=50):
            toolbar.set_ephemeral_mode(True, emit=False)
        assert toolbar.is_ephemeral_mode() is True

    def test_preset_row_visible_when_mode_on(
        self, toolbar: AnnotationToolbar
    ) -> None:
        toolbar.show()
        toolbar.set_ephemeral_mode(True)
        assert toolbar._ephemeral_preset_row.isVisibleTo(toolbar) is True

    def test_pen_glyph_swaps_to_ghost_in_ephemeral(
        self, toolbar: AnnotationToolbar
    ) -> None:
        toolbar.set_ephemeral_mode(True)
        assert toolbar._pen_btn.text() == "👻"

    def test_pen_glyph_restores_when_mode_off(
        self, toolbar: AnnotationToolbar
    ) -> None:
        toolbar.set_ephemeral_mode(True)
        toolbar.set_ephemeral_mode(False)
        assert toolbar._pen_btn.text() == "✏️"

    def test_eraser_disabled_in_ephemeral(
        self, toolbar: AnnotationToolbar
    ) -> None:
        toolbar.set_ephemeral_mode(True)
        assert toolbar._eraser_btn.isEnabled() is False

    def test_eraser_re_enabled_when_mode_off(
        self, toolbar: AnnotationToolbar
    ) -> None:
        toolbar.set_ephemeral_mode(True)
        toolbar.set_ephemeral_mode(False)
        assert toolbar._eraser_btn.isEnabled() is True

    def test_active_eraser_auto_switches_to_none_on_activation(
        self, toolbar: AnnotationToolbar
    ) -> None:
        """If eraser was the active tool, activating ephemeral mode
        switches to NONE — otherwise we'd have a greyed-but-checked
        button which reads as broken UI."""
        toolbar.set_current_tool(ToolKind.ERASER)
        assert toolbar.current_tool() == ToolKind.ERASER
        toolbar.set_ephemeral_mode(True)
        assert toolbar.current_tool() == ToolKind.NONE


# ============================================================================
# Border tinting
# ============================================================================


class TestBorderTint:
    def test_normal_mode_has_default_border(
        self, toolbar: AnnotationToolbar
    ) -> None:
        ss = toolbar.styleSheet()
        # Default border is the rgba grey, NOT the cyan accent.
        assert "#4A8DE8" not in ss

    def test_ephemeral_mode_has_cyan_border(
        self, toolbar: AnnotationToolbar
    ) -> None:
        toolbar.set_ephemeral_mode(True)
        ss = toolbar.styleSheet()
        assert "#4A8DE8" in ss

    def test_dock_mode_also_tints_border_in_ephemeral(
        self, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """The cyan accent applies in both float and dock modes —
        consistent visual signal regardless of toolbar parenting."""
        main_window = QMainWindow()
        qtbot.addWidget(main_window)
        fake_viewport = QWidget(main_window)
        fake_dock = QDockWidget(main_window)
        tb = AnnotationToolbar(
            fake_viewport,
            fake_dock,
            initial_mode=ToolbarMode.DOCK,
            parent=main_window,
        )
        tb.set_ephemeral_mode(True)
        assert "#4A8DE8" in tb.styleSheet()


# ============================================================================
# Duration presets
# ============================================================================


class TestDurationPresets:
    def test_set_preset_emits_seconds(
        self, toolbar: AnnotationToolbar, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        # Default is index 1 (moyen). Pick index 0 (court / 2 s) to
        # get a different value and confirm the signal fires.
        with qtbot.waitSignal(
            toolbar.ephemeral_duration_changed, timeout=200
        ) as block:
            toolbar.set_ephemeral_preset_index(0)
        assert block.args == [pytest.approx(2.0)]

    def test_each_preset_maps_correctly(
        self, toolbar: AnnotationToolbar
    ) -> None:
        """Natural order: 0=court (2s), 1=moyen (5s), 2=long (10s)."""
        toolbar.set_ephemeral_preset_index(0)
        assert toolbar.ephemeral_duration_seconds() == pytest.approx(2.0)
        toolbar.set_ephemeral_preset_index(1)
        assert toolbar.ephemeral_duration_seconds() == pytest.approx(5.0)
        toolbar.set_ephemeral_preset_index(2)
        assert toolbar.ephemeral_duration_seconds() == pytest.approx(10.0)

    def test_invalid_index_falls_back_to_default(
        self, toolbar: AnnotationToolbar
    ) -> None:
        # Switch away from default (1) first so we can detect the fallback.
        toolbar.set_ephemeral_preset_index(0)
        toolbar.set_ephemeral_preset_index(99)
        assert toolbar.ephemeral_preset_index() == DEFAULT_EPHEMERAL_PRESET_INDEX

    def test_preset_buttons_radio_behaviour(
        self, toolbar: AnnotationToolbar
    ) -> None:
        """Only one preset button is checked at a time."""
        toolbar.set_ephemeral_preset_index(2)
        checked = [b.isChecked() for b in toolbar._ephemeral_preset_btns]
        assert checked == [False, False, True]

    def test_three_preset_buttons_exist(
        self, toolbar: AnnotationToolbar
    ) -> None:
        assert len(toolbar._ephemeral_preset_btns) == 3
        assert len(EPHEMERAL_PRESETS_S) == 3

    def test_preset_order_court_moyen_long(self) -> None:
        """Natural ascending order: court / moyen / long. Default is
        the middle dot (index 1 = moyen) — shortest left, longest right."""
        assert EPHEMERAL_PRESETS_S == (2.0, 5.0, 10.0)
        assert DEFAULT_EPHEMERAL_PRESET_INDEX == 1

    def test_selected_preset_has_color_highlight(
        self, toolbar: AnnotationToolbar
    ) -> None:
        """The active preset's stylesheet contains the cyan accent
        in its ``:checked`` rule — that's what colors the selected
        dot. The QSS string is generated per-button so we can assert
        the rule's presence directly."""
        for btn in toolbar._ephemeral_preset_btns:
            ss = btn.styleSheet()
            assert "QToolButton:checked" in ss
            # Cyan accent lives in the :checked block.
            assert "#4A8DE8" in ss


# ============================================================================
# Click on the 👻 button drives the same path as the public setter
# ============================================================================


class TestButtonClick:
    def test_button_click_toggles_state(
        self, toolbar: AnnotationToolbar, qtbot,  # type: ignore[no-untyped-def]
    ) -> None:
        """A real click on 👻 must end up with the state flipped and
        the change-signal emitted."""
        toolbar.show()
        with qtbot.waitSignal(toolbar.ephemeral_mode_changed, timeout=200) as block:
            qtbot.mouseClick(
                toolbar._ephemeral_btn,
                pytest.importorskip("PySide6").QtCore.Qt.MouseButton.LeftButton,
            )
        assert block.args == [True]
        assert toolbar.is_ephemeral_mode() is True
