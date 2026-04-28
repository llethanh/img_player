"""Immutable data models describing image sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FrameInfo:
    path: Path
    frame_number: int
    # File mtime in seconds-since-epoch (Path.stat().st_mtime). Used by
    # the FrameCache to detect when a file on disk has changed since the
    # frame was cached, so a "Reload" only re-decodes what's actually
    # different. ``0.0`` means "unknown" — callers that don't care
    # (legacy code, tests) leave the default; the scanner populates it
    # at sequence-detection time.
    mtime: float = 0.0


@dataclass(frozen=True)
class SequenceInfo:
    """A contiguous-or-sparse run of image files sharing a numeric index.

    Frames are always stored sorted by `frame_number`. Padding is the
    zero-pad width observed in the filename (0 means no padding).
    """

    base_name: str
    extension: str
    directory: Path
    padding: int
    frames: tuple[FrameInfo, ...]
    fps_default: float = 24.0
    width: int | None = None
    height: int | None = None
    channel_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("SequenceInfo requires at least one frame")

    @property
    def first_frame(self) -> int:
        return self.frames[0].frame_number

    @property
    def last_frame(self) -> int:
        return self.frames[-1].frame_number

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def missing_frames(self) -> tuple[int, ...]:
        present = {f.frame_number for f in self.frames}
        expected = range(self.first_frame, self.last_frame + 1)
        return tuple(n for n in expected if n not in present)

    @property
    def is_contiguous(self) -> bool:
        return not self.missing_frames

    def display_pattern(self) -> str:
        """Human-readable pattern e.g. 'render.####.exr'.

        `base_name` already captures whatever separator preceded the digits
        (dot, underscore, nothing), so we don't add one here.
        """
        hashes = "#" * self.padding if self.padding > 0 else "#"
        return f"{self.base_name}{hashes}{self.extension}"
