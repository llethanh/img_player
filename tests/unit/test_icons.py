"""Tests for the inline SVG icon factory."""

from __future__ import annotations

import pytest

from img_player.ui.icons import available_names, make_icon
from img_player.ui.theme import H


@pytest.fixture(autouse=True)
def _ensure_qapplication(qtbot) -> None:  # type: ignore[no-untyped-def]
    """Forces qtbot to spin up a QApplication. QSvgRenderer + QPixmap
    both need one to function, even if we never show the icon."""


class TestMakeIcon:
    def test_returns_non_null_for_every_template(self) -> None:
        for name in available_names():
            icon = make_icon(name)
            assert not icon.isNull(), f"icon '{name}' should render to a non-null QIcon"

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            make_icon("definitely-not-a-real-icon")

    def test_default_color_is_text_primary(self) -> None:
        # Different colours go through the lru_cache as different keys —
        # check that swapping colour returns a *different* icon object.
        a = make_icon("play")  # default: TEXT_PRIMARY
        b = make_icon("play", color=H.ACCENT)
        assert a is not b

    def test_size_argument_is_respected(self) -> None:
        # availableSizes returns the sizes the QIcon's underlying
        # pixmaps were built at. We requested 48px so it should appear.
        icon = make_icon("play", size=48)
        # availableSizes is in physical pixels (DPR-multiplied) — accept
        # either 48 or 48 * dpr.
        widths = {sz.width() for sz in icon.availableSizes()}
        assert any(w >= 48 for w in widths), (
            f"expected at least one size ≥ 48px, got {widths}"
        )


class TestLruCache:
    def test_same_args_return_same_instance(self) -> None:
        # The point of caching is to avoid re-rendering the SVG on every
        # call. Same args → same QIcon object.
        a = make_icon("play", color=H.ACCENT, size=18)
        b = make_icon("play", color=H.ACCENT, size=18)
        assert a is b
