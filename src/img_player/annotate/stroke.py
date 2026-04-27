"""The :class:`Stroke` — one pen gesture as a polyline in image-space.

A stroke is the atomic unit of annotation. Created by the overlay when
the user releases the mouse button after a drag. Stored in the
:class:`~img_player.annotate.store.AnnotationStore` keyed by frame
index. Persisted to JSON at shutdown via
:mod:`img_player.annotate.persistence`.

Design notes (see spec):

* **Frozen dataclass.** The undo stack holds references to strokes —
  immutability guarantees the stack can't be silently corrupted by
  mutating a stroke after the fact.
* **Tuples, not lists.** Reinforces the immutability contract and
  makes ``Stroke`` hashable (useful for future deduplication / set
  operations).
* **No ID.** A stroke is identified by its index in the per-frame
  list. The eraser deletes by index. Identity is positional, not
  intrinsic.
* **Coords in image-pixels.** A stroke at (1024.5, 532.0) means
  "image pixel (1024.5, 532.0)" — independent of widget size, zoom
  factor, pan offset. The viewport's transform translates image
  coords to widget coords at paint time.
* **Size in image-pixels.** A 5 px-image stroke stays visually
  proportional to the image at any zoom: the ``QPen`` width is
  scaled by the current zoom factor at paint time.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Stroke:
    """One pen-down, drag, pen-up gesture.

    Attributes
    ----------
    points
        Polyline in image-space pixels. Tuple of ``(x, y)`` 2-tuples.
        Always at least one point — a click without movement still
        produces a one-point stroke (rendered as a dot).
    color
        Hex color string from the toolbar's fixed palette
        (e.g. ``"#E84A4A"``). Stored as a string rather than a
        ``QColor`` so the dataclass stays Qt-free and JSON-trivial.
    size
        Brush diameter in image-pixels. Range 1.0–50.0 in v1
        (enforced by the toolbar's slider, not by this class).
    """

    points: tuple[tuple[float, float], ...]
    color: str
    size: float

    def __post_init__(self) -> None:
        if len(self.points) == 0:
            raise ValueError("Stroke must have at least one point")
        if self.size <= 0:
            raise ValueError(f"Stroke size must be positive, got {self.size}")
        if not (self.color.startswith("#") and len(self.color) in (4, 7, 9)):
            # #RGB, #RRGGBB, #RRGGBBAA. We don't validate the hex
            # digits themselves — Qt's QColor constructor is the
            # ultimate arbiter at render time.
            raise ValueError(
                f"Stroke color must be a hex string '#RGB' / '#RRGGBB' / "
                f"'#RRGGBBAA', got {self.color!r}"
            )

    def to_dict(self) -> dict[str, object]:
        """Serialise for JSON. Inverse of :meth:`from_dict`."""
        return {
            "color": self.color,
            "size": self.size,
            "points": [list(p) for p in self.points],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Stroke:
        """Build from a JSON dict. Raises ``KeyError`` / ``TypeError``
        / ``ValueError`` on a malformed payload — callers are expected
        to wrap in a try/except (the persistence layer does this and
        treats failures as "skip this stroke")."""
        return cls(
            points=tuple((float(p[0]), float(p[1])) for p in data["points"]),  # type: ignore[index]
            color=str(data["color"]),
            size=float(data["size"]),  # type: ignore[arg-type]
        )
