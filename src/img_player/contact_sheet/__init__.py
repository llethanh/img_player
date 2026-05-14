"""Contact sheet — multi-layer grid view.

Displays every loaded layer as a tile in a grid, with each layer
re-aligned to the same "frame 0" regardless of its timeline offset.
The user can pick the column / row count (or let it auto-fit to
preserve the source image aspect ratio) and toggle a per-tile name
overlay.

The feature plugs into the existing ``_on_frame_changed`` dispatch
between compare-mode and the regular cache lookup: when enabled,
the GL upload is hijacked with a numpy-composed grid frame, the
master cache + composite pipeline are bypassed (each tile is
decoded independently via :class:`ContactSheetDecoder`).
"""

from img_player.contact_sheet.compose import (
    auto_grid_dimensions,
    render_contact_sheet,
)
from img_player.contact_sheet.decoder import ContactSheetDecoder
from img_player.contact_sheet.state import ContactSheetState

__all__ = [
    "ContactSheetDecoder",
    "ContactSheetState",
    "auto_grid_dimensions",
    "render_contact_sheet",
]
