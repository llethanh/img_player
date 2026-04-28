"""Per-frame annotations — drawing tool for VFX review.

See ``docs/specs/2026-04-27-annotations-design.md`` for the full design,
and ``docs/specs/2026-04-28-ephemeral-annotations-design.md`` for the
ephemeral-mode addition shipped in v0.4.1.

* :class:`Stroke` — frozen dataclass: one pen-down/drag/pen-up gesture
  as a polyline in image-space.
* :class:`AnnotationStore` — per-frame strokes + per-frame undo/redo,
  with Qt signals to wire to the UI.
* :func:`save_annotations` / :func:`load_annotations` — atomic sidecar
  JSON next to the sequence (``.img_player_annotations.json``).
* :class:`AnnotationOverlay` — transparent ``QWidget`` above the GL
  viewport; captures pen strokes and paints them via ``QPainter``.
* :class:`ToolKind` — tool selector enum (NONE / PEN / ERASER).
* :class:`EphemeralStrokeManager` — Google-Meet-style fading strokes,
  not persisted, for live presentations during a video call.
"""

from img_player.annotate.ephemeral import (
    EphemeralStrokeManager,
    alpha_at,
)
from img_player.annotate.overlay import (
    AnnotationOverlay,
    ToolKind,
    image_to_widget,
    nearest_stroke_index,
    widget_to_image,
)
from img_player.annotate.persistence import (
    SCHEMA_VERSION,
    load_annotations,
    save_annotations,
)
from img_player.annotate.store import Action, ActionKind, AnnotationStore
from img_player.annotate.stroke import Stroke
from img_player.annotate.toolbar import (
    DEFAULT_COLOR,
    DEFAULT_EPHEMERAL_PRESET_INDEX,
    DEFAULT_SIZE,
    EPHEMERAL_PRESETS_S,
    MAX_SIZE,
    MIN_SIZE,
    PALETTE,
    AnnotationToolbar,
    ToolbarMode,
)

__all__ = [
    "Action",
    "ActionKind",
    "AnnotationOverlay",
    "AnnotationStore",
    "AnnotationToolbar",
    "DEFAULT_COLOR",
    "DEFAULT_EPHEMERAL_PRESET_INDEX",
    "DEFAULT_SIZE",
    "EPHEMERAL_PRESETS_S",
    "EphemeralStrokeManager",
    "MAX_SIZE",
    "MIN_SIZE",
    "PALETTE",
    "SCHEMA_VERSION",
    "Stroke",
    "ToolKind",
    "ToolbarMode",
    "alpha_at",
    "image_to_widget",
    "load_annotations",
    "nearest_stroke_index",
    "save_annotations",
    "widget_to_image",
]
