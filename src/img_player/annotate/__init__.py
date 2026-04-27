"""Per-frame annotations — drawing tool for VFX review.

See ``docs/specs/2026-04-27-annotations-design.md`` for the full design.

Slice 1 ships the pure-Python core (no Qt widgets):

* :class:`Stroke` — frozen dataclass: one pen-down/drag/pen-up gesture
  as a polyline in image-space.
* :class:`AnnotationStore` — per-frame strokes + per-frame undo/redo,
  with Qt signals to wire to the UI in later slices.
* :func:`save_annotations` / :func:`load_annotations` — atomic sidecar
  JSON next to the sequence (``.img_player_annotations.json``).
"""

from img_player.annotate.persistence import (
    SCHEMA_VERSION,
    load_annotations,
    save_annotations,
)
from img_player.annotate.store import Action, ActionKind, AnnotationStore
from img_player.annotate.stroke import Stroke

__all__ = [
    "Action",
    "ActionKind",
    "AnnotationStore",
    "SCHEMA_VERSION",
    "Stroke",
    "load_annotations",
    "save_annotations",
]
