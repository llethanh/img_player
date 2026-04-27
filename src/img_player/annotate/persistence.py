"""Sidecar JSON persistence for annotations.

A sequence at ``<dir>/<basename>.<frame>.<ext>`` gets a sidecar at
``<dir>/.img_player_annotations.json``. Multiple sequences sharing the
same dir cohabit under different ``basename`` keys.

Atomic save (``.tmp`` + rename), schema-versioned, best-effort load
(any failure mode returns ``None`` rather than raising). Mirrors the
patterns established in :mod:`img_player.perf.calibration` for
``profile.json``.

This module is the boundary between the in-memory store (Qt-aware,
mutable) and the on-disk JSON (Qt-free, declarative). Neither layer
knows about the other directly.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from img_player import __version__ as IMG_PLAYER_VERSION
from img_player.annotate.store import AnnotationStore

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
"""Bump when the on-disk shape changes. Loader rejects unknown versions
gracefully (returns an empty store rather than raising or guessing)."""

SIDECAR_FILENAME = ".img_player_annotations.json"
"""Filename of the sidecar inside the sequence's directory.

Dot-prefixed so it's hidden on Linux/macOS and less visible on Windows.
Discoverable in any explorer that shows hidden files — but doesn't
clutter visually.
"""


def sidecar_path(sequence_dir: Path) -> Path:
    """Path to the sidecar JSON for the sequence in ``sequence_dir``."""
    return sequence_dir / SIDECAR_FILENAME


def save_annotations(
    path: Path,
    store: AnnotationStore,
    *,
    basename: str,
) -> bool:
    """Atomically write the store to ``path``.

    The store is wrapped under ``sequences[<basename>]`` in the
    on-disk format so multiple sequences sharing one dir can cohabit.
    Existing sidecars at ``path`` are merged: other basenames'
    annotations are preserved, only the matching basename is updated.

    Returns ``True`` on success, ``False`` on any I/O failure (the
    error is logged at WARNING level — never raised user-facing,
    because a read-only Drive Stream session at shutdown shouldn't
    crash the app).

    Implementation: write the full payload to ``path.tmp``, then
    ``Path.replace`` it onto ``path`` (atomic on POSIX, near-atomic
    on Windows — a torn write would leave the previous good file
    intact, which is what we want).
    """
    try:
        # Merge: read existing payload (if any) so we don't clobber
        # other basenames.
        existing_sequences: dict[str, dict[str, object]] = {}
        if path.exists():
            try:
                prev = json.loads(path.read_text(encoding="utf-8"))
                if prev.get("schema_version") == SCHEMA_VERSION:
                    existing_sequences = prev.get("sequences", {}) or {}
            except (json.JSONDecodeError, OSError):
                # Treat a corrupt or unreadable existing file as if it
                # didn't exist — we're about to overwrite it anyway.
                log.warning(
                    "[annotations] existing sidecar at %s is unreadable; "
                    "overwriting",
                    path,
                )

        existing_sequences[basename] = store.to_dict()

        payload = {
            "schema_version": SCHEMA_VERSION,
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "img_player_version": IMG_PLAYER_VERSION,
            "sequences": existing_sequences,
        }

        tmp = path.with_suffix(path.suffix + ".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)
        return True
    except OSError as err:  # pragma: no cover — best-effort I/O
        log.warning(
            "[annotations] save failed at %s: %s. Annotations will be "
            "lost on close — likely a read-only directory (Drive Stream "
            "offline, USB write-protect, NAS).",
            path,
            err,
        )
        return False


def load_annotations(
    path: Path,
    *,
    basename: str,
) -> AnnotationStore | None:
    """Return a freshly populated :class:`AnnotationStore` from ``path``.

    Returns ``None`` when:

    * the file is missing,
    * the JSON is malformed,
    * the schema version is unknown,
    * the requested ``basename`` is not present,
    * any I/O error.

    Never raises. Callers (typically :class:`~img_player.app.App`) treat
    ``None`` as "no annotations for this sequence" and start with an
    empty store.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as err:
        log.warning(
            "[annotations] %s is not valid JSON (%s). Starting with "
            "empty annotations; the file is left untouched for you to "
            "investigate.",
            path,
            err,
        )
        return None

    if data.get("schema_version") != SCHEMA_VERSION:
        log.warning(
            "[annotations] %s has schema_version=%r, this build expects "
            "%d. Starting with empty annotations.",
            path,
            data.get("schema_version"),
            SCHEMA_VERSION,
        )
        return None

    sequences = data.get("sequences", {})
    payload = sequences.get(basename)
    if not isinstance(payload, dict):
        return None

    frames = payload.get("frames", {})
    if not isinstance(frames, dict):
        return None

    store = AnnotationStore()
    store.load_from_dict(frames)
    return store
