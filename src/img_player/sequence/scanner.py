"""Detect image sequences from a file or directory path."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from img_player.io.formats import is_supported
from img_player.io.reader import FrameReadError, read_header
from img_player.sequence.models import FrameInfo, SequenceInfo


@dataclass(frozen=True)
class FolderGroup:
    """A bucket of sequences for the multi-source picker.

    ``folder`` is ``None`` when the group is filled from raw files
    dropped on the player (loose files appear at the root of the
    picker, with no folder header). ``empty`` flags a folder the user
    dropped that contained zero detectable sequences — kept in the
    result so the picker can render a greyed entry rather than
    silently swallowing the drop.
    """

    folder: Path | None
    sequences: tuple[SequenceInfo, ...] = field(default_factory=tuple)
    empty: bool = False

_FRAME_PATTERN = re.compile(r"^(.*?)(\d+)\.([^.]+)$")


class SequenceNotFoundError(FileNotFoundError):
    """Raised when no sequence can be derived from the given path."""


def _safe_mtime(path: Path) -> float:
    """Return ``path.stat().st_mtime``, or ``0.0`` if the file is gone.

    The cache uses mtime to detect "this file changed since I cached
    it" — a missing file is just a special case of "changed", so a
    fallback of 0.0 is fine: the next stat() will return a non-zero
    value and the cache will see the change.
    """
    try:
        return path.stat().st_mtime
    except (OSError, FileNotFoundError):
        return 0.0


def _iter_image_entries(directory: Path) -> Iterator[tuple[Path, float]]:
    """Yield ``(path, mtime)`` for every regular file in ``directory``
    whose name doesn't start with a dot.

    Uses :func:`os.scandir` so each entry's :meth:`stat` is **cached**
    by the OS — on Windows this is ~3× faster than the
    ``Path.iterdir() + Path.stat()`` pair we used to do per file. The
    cached stat is what powers cheap ``mtime`` reads without a second
    syscall per file. Matters most on Drive-synced / network folders
    where every stat round-trip is a noticeable.

    Errors during scandir (deleted directory, perm error, …) raise
    OSError up to the caller — matches the old behaviour of
    ``Path.iterdir()``.
    """
    with os.scandir(directory) as it:
        for entry in it:
            name = entry.name
            if name.startswith("."):
                continue
            try:
                if not entry.is_file():
                    continue
                mtime = entry.stat().st_mtime
            except OSError:
                # Entry vanished between scandir and stat — skip.
                continue
            yield Path(entry.path), mtime


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
    for entry_path, mtime in _iter_image_entries(directory):
        if not is_supported(entry_path):
            continue
        parsed = _parse(entry_path.name)
        if parsed is None:
            continue
        key = (parsed.base, parsed.padding, parsed.extension)
        groups[key].append(
            FrameInfo(path=entry_path, frame_number=parsed.frame, mtime=mtime),
        )

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
        # No numeric frame pattern in the name — treat as a still
        # image (slate, lookdev ref, single matte). We still
        # validate the extension is one OIIO can read; otherwise
        # the user gets a clear "unsupported format" rather than a
        # confusing "not a sequence frame" message.
        if not is_supported(file):
            raise SequenceNotFoundError(
                f"Unsupported image format: {file.name}",
            )
        frame = FrameInfo(path=file, frame_number=0, mtime=_safe_mtime(file))
        width, height, channels = (
            _probe_first_frame((frame,)) if probe else (None, None, ())
        )
        # base_name gets the filename stem; padding 0 indicates "no
        # numeric pattern" so ``display_pattern`` falls back to a
        # plain filename rendering.
        return SequenceInfo(
            base_name=file.stem,
            extension=file.suffix,
            directory=file.parent,
            padding=0,
            frames=(frame,),
            width=width,
            height=height,
            channel_names=channels,
        )

    directory = file.parent
    matching_frames: list[FrameInfo] = []
    for entry_path, mtime in _iter_image_entries(directory):
        candidate = _parse(entry_path.name)
        if candidate is None:
            continue
        if (
            candidate.base == parsed.base
            and candidate.padding == parsed.padding
            and candidate.extension == parsed.extension
        ):
            matching_frames.append(
                FrameInfo(path=entry_path, frame_number=candidate.frame, mtime=mtime),
            )

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


def scan_paths(
    paths: list[Path] | tuple[Path, ...], *, probe: bool = False,
) -> list[FolderGroup]:
    """Scan a heterogeneous drop (folders + loose files) into groups.

    Behaviour
    ---------
    * For every directory in ``paths``: enumerate sequences at level 1
      only (no recursion into sub-folders) — produces one
      :class:`FolderGroup` per directory, possibly with ``empty=True``
      when nothing was detected.
    * For every file in ``paths``: resolve its sequence via
      :func:`scan` and add it to the special "loose files" group whose
      ``folder`` is ``None``. Multiple loose files that resolve to the
      same sequence are de-duplicated.

    Sort order: the loose group comes first (None header), then the
    folder groups sorted alphabetically by folder name. Within each
    group, sequences are sorted alphabetically by display pattern.
    Missing paths are silently skipped — the picker can't show a
    folder that doesn't exist.
    """
    folder_to_seqs: dict[Path, list[SequenceInfo]] = {}
    folder_seen_keys: dict[Path, set[tuple[str, int, str]]] = {}
    loose: list[SequenceInfo] = []
    loose_seen_keys: set[tuple[str, int, str, Path]] = set()
    empty_folders: set[Path] = set()

    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            try:
                seqs = scan_all(p, probe=probe)
            except SequenceNotFoundError:
                seqs = []
            if not seqs:
                empty_folders.add(p)
                folder_to_seqs.setdefault(p, [])
                folder_seen_keys.setdefault(p, set())
                continue
            keys = folder_seen_keys.setdefault(p, set())
            bucket = folder_to_seqs.setdefault(p, [])
            for s in seqs:
                k = (s.base_name, s.padding, s.extension.lstrip(".").lower())
                if k in keys:
                    continue
                keys.add(k)
                bucket.append(s)
        elif p.is_file():
            try:
                seq = scan(p, probe=probe)
            except SequenceNotFoundError:
                continue
            k = (
                seq.base_name,
                seq.padding,
                seq.extension.lstrip(".").lower(),
                seq.directory,
            )
            if k in loose_seen_keys:
                continue
            loose_seen_keys.add(k)
            loose.append(seq)
        # else: vanished path — skip.

    groups: list[FolderGroup] = []
    if loose:
        loose_sorted = tuple(
            sorted(loose, key=lambda s: s.display_pattern().lower())
        )
        groups.append(FolderGroup(folder=None, sequences=loose_sorted))

    for folder in sorted(folder_to_seqs.keys(), key=lambda p: p.name.lower()):
        seqs = folder_to_seqs[folder]
        seqs_sorted = tuple(
            sorted(seqs, key=lambda s: s.display_pattern().lower())
        )
        groups.append(
            FolderGroup(
                folder=folder,
                sequences=seqs_sorted,
                empty=(folder in empty_folders and not seqs_sorted),
            )
        )
    return groups


def _scan_from_dir(directory: Path, *, probe: bool = True) -> SequenceInfo:
    sequences = scan_all(directory, probe=probe)
    if not sequences:
        raise SequenceNotFoundError(f"No sequence found in {directory}")
    return sequences[0]


def rescan(sequence: SequenceInfo) -> SequenceInfo:
    """Cheap re-detection of an already-loaded sequence.

    Re-globs the source directory for the same ``base_name`` /
    ``padding`` / ``extension`` triplet, refreshes ``mtime`` on every
    surviving frame, picks up any new frames added to disk, and drops
    any frames that vanished. Width / height / channels are preserved
    (no header probe — would cost an extra OIIO open per frame).

    Used by the "Reload cache" action: the caller diffs the new
    SequenceInfo against the old one (per-frame mtime) and tells the
    cache which frames to drop / re-decode. Cheap enough to call
    interactively (one ``iterdir`` + one ``stat`` per file).
    """
    directory = sequence.directory
    matching: list[FrameInfo] = []
    if not directory.is_dir():
        # Directory itself disappeared — return an empty rescan;
        # the caller marks every frame as missing.
        return SequenceInfo(
            base_name=sequence.base_name,
            extension=sequence.extension,
            directory=directory,
            padding=sequence.padding,
            frames=sequence.frames,  # keep the old frame list (all missing)
            fps_default=sequence.fps_default,
            width=sequence.width,
            height=sequence.height,
            channel_names=sequence.channel_names,
        )
    target_ext = sequence.extension.lstrip(".").lower()
    for entry_path, mtime in _iter_image_entries(directory):
        parsed = _parse(entry_path.name)
        if parsed is None:
            continue
        if (
            parsed.base != sequence.base_name
            or parsed.padding != sequence.padding
            or parsed.extension != target_ext
        ):
            continue
        matching.append(
            FrameInfo(path=entry_path, frame_number=parsed.frame, mtime=mtime),
        )
    matching.sort(key=lambda f: f.frame_number)
    if not matching:
        # Files all gone — keep the old frame list so the caller
        # can paint every slot as missing.
        return SequenceInfo(
            base_name=sequence.base_name,
            extension=sequence.extension,
            directory=directory,
            padding=sequence.padding,
            frames=sequence.frames,
            fps_default=sequence.fps_default,
            width=sequence.width,
            height=sequence.height,
            channel_names=sequence.channel_names,
        )
    return SequenceInfo(
        base_name=sequence.base_name,
        extension=sequence.extension,
        directory=directory,
        padding=sequence.padding,
        frames=tuple(matching),
        fps_default=sequence.fps_default,
        width=sequence.width,
        height=sequence.height,
        channel_names=sequence.channel_names,
    )
