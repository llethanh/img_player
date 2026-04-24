"""Detect image sequences from a file or directory path."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from img_player.io.formats import is_supported
from img_player.io.reader import FrameReadError, read_header
from img_player.sequence.models import FrameInfo, SequenceInfo

_FRAME_PATTERN = re.compile(r"^(.*?)(\d+)\.([^.]+)$")


class SequenceNotFoundError(FileNotFoundError):
    """Raised when no sequence can be derived from the given path."""


@dataclass(frozen=True)
class _ParsedName:
    base: str
    frame: int
    padding: int
    extension: str  # lowercase, no leading dot


def _parse(filename: str) -> _ParsedName | None:
    """Parse a filename into its sequence parts, or None if it isn't a frame."""
    match = _FRAME_PATTERN.match(filename)
    if not match:
        return None
    base, digits, ext = match.groups()
    return _ParsedName(base=base, frame=int(digits), padding=len(digits), extension=ext.lower())


def _probe_first_frame(
    frames: tuple[FrameInfo, ...],
) -> tuple[int | None, int | None, tuple[str, ...]]:
    """Read the first frame's header to fill width/height/channels. Non-fatal."""
    try:
        spec = read_header(frames[0].path)
    except FrameReadError:
        return (None, None, ())
    return (spec.width, spec.height, tuple(spec.channelnames))


def scan(path: Path | str, *, probe: bool = True) -> SequenceInfo:
    """Detect a sequence at `path`.

    - If `path` is a single file, returns the sequence it belongs to
      (same base + padding + extension) in its parent directory.
    - If `path` is a directory, returns the largest sequence in it.

    Parameters
    ----------
    probe:
        When True (default) opens the first frame's header to fill
        ``width``, ``height`` and ``channel_names``. Set to False to skip
        this: useful on slow / lazy filesystems (Google Drive Stream,
        NAS) where the first read triggers a full file download. The UI
        can populate those fields later from the first decoded frame.

    Raises
    ------
    SequenceNotFoundError
        If the path doesn't exist or no sequence can be derived.
    """
    path = Path(path)
    if path.is_file():
        return _scan_from_file(path, probe=probe)
    if path.is_dir():
        return _scan_from_dir(path, probe=probe)
    raise SequenceNotFoundError(f"Path does not exist: {path}")


def scan_all(directory: Path | str, *, probe: bool = True) -> list[SequenceInfo]:
    """Return every distinct sequence found in `directory`, largest first.

    See :func:`scan` for the ``probe`` flag.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise SequenceNotFoundError(f"Not a directory: {directory}")

    groups: dict[tuple[str, int, str], list[FrameInfo]] = defaultdict(list)
    for entry in directory.iterdir():
        if not entry.is_file() or entry.name.startswith("."):
            continue
        if not is_supported(entry):
            continue
        parsed = _parse(entry.name)
        if parsed is None:
            continue
        key = (parsed.base, parsed.padding, parsed.extension)
        groups[key].append(FrameInfo(path=entry, frame_number=parsed.frame))

    sequences: list[SequenceInfo] = []
    for (base, padding, ext), frames in groups.items():
        frames.sort(key=lambda f: f.frame_number)
        width, height, channels = _probe_first_frame(tuple(frames)) if probe else (None, None, ())
        sequences.append(
            SequenceInfo(
                base_name=base,
                extension=f".{ext}",
                directory=directory,
                padding=padding,
                frames=tuple(frames),
                width=width,
                height=height,
                channel_names=channels,
            )
        )

    sequences.sort(key=lambda s: s.frame_count, reverse=True)
    return sequences


def _scan_from_file(file: Path, *, probe: bool = True) -> SequenceInfo:
    parsed = _parse(file.name)
    if parsed is None:
        raise SequenceNotFoundError(f"Filename is not a sequence frame: {file.name}")

    directory = file.parent
    matching_frames: list[FrameInfo] = []
    for entry in directory.iterdir():
        if not entry.is_file() or entry.name.startswith("."):
            continue
        candidate = _parse(entry.name)
        if candidate is None:
            continue
        if (
            candidate.base == parsed.base
            and candidate.padding == parsed.padding
            and candidate.extension == parsed.extension
        ):
            matching_frames.append(FrameInfo(path=entry, frame_number=candidate.frame))

    if not matching_frames:
        raise SequenceNotFoundError(f"No frames matching {file.name} in {directory}")

    matching_frames.sort(key=lambda f: f.frame_number)
    width, height, channels = (
        _probe_first_frame(tuple(matching_frames)) if probe else (None, None, ())
    )
    return SequenceInfo(
        base_name=parsed.base,
        extension=f".{parsed.extension}",
        directory=directory,
        padding=parsed.padding,
        frames=tuple(matching_frames),
        width=width,
        height=height,
        channel_names=channels,
    )


def _scan_from_dir(directory: Path, *, probe: bool = True) -> SequenceInfo:
    sequences = scan_all(directory, probe=probe)
    if not sequences:
        raise SequenceNotFoundError(f"No sequence found in {directory}")
    return sequences[0]
