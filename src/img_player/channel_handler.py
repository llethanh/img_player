"""Channel-selection handlers extracted from app.py.

Two free functions taking the :class:`ImgPlayerApp` as first arg —
they read / write the focused layer's channel selection and the
app-level fallback used until the first layer focuses.

Kept as free functions rather than methods on a mixin to avoid an
extra class hierarchy for what is, at heart, a thin imperative
binding between the transport menu and the layer stack.

Historical note (v1.2): an earlier version of this module also held
contact-sheet handlers (``on_tile_isolate_requested``,
``toggle_contact_sheet``, ``on_channel_layout_mode_changed``,
``on_channel_labels_visible_changed``). They were retired with the
contact-sheet feature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from img_player.sequence.channels import ChannelSelection

if TYPE_CHECKING:
    from img_player.app import ImgPlayerApp


def set_channel_selection(app: ImgPlayerApp, selection: ChannelSelection) -> None:
    """Switch the FOCUSED layer's channel selection.

    The selection lives on the layer itself so that adding multiple
    sequences each preserves its own choice. The legacy
    ``app._channel_selection`` attribute is kept in sync as a
    fallback for code paths that haven't been migrated yet (e.g.
    the export-time snapshot).

    Mutating the focused layer fires ``layer_modified`` → cache
    invalidates that layer's master range → the wired
    ``_refresh_after_stack_change`` re-issues prefetch + display.
    """
    focused = app._layer_stack.focused()
    app._channel_selection = selection  # legacy fallback
    if focused is None:
        # No sequence loaded yet — keep the selection on app state
        # so it's there for the first layer when it lands.
        return
    app._layer_stack.update(focused.id, channel_selection=selection)


def on_channel_selection_changed(app: ImgPlayerApp, selection: object) -> None:
    """Apply a fresh :class:`ChannelSelection` from the transport menu."""
    if not isinstance(selection, ChannelSelection):
        return  # signal carrier mismatch — defensive guard
    set_channel_selection(app, selection)
