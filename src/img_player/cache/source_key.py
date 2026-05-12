"""Compute session-independent cache keys for the on-disk tier.

The in-memory cache in :class:`MasterFrameCache` keys frames on
session-local ids (``layer.id`` is a fresh UUID per session). That's
fine for the RAM tier — entries don't outlive the process. But the
**disk tier** must survive close + reopen, ideally even across
different sessions referencing the same underlying media. So its key
needs to be **source-canonical**: derived purely from "what is this
frame on disk and how is it interpreted".

Components that go into the key
-------------------------------

* **Canonical absolute path** — resolves symlinks, normalises slashes,
  case-fold on case-insensitive filesystems. Two sessions pointing at
  the same file get the same key.
* **File mtime** — invalidates the cache when an artist overwrites
  the frame on disk. Without this we'd serve stale pixels after a
  re-render.
* **File size** — extra safety net against mtime-collision attacks
  (e.g. ``touch -m`` on a different content). Cheap to read, costs
  nothing on the typical path.
* **Active channels** — RGB vs RGBA vs ``albedo`` vs ``Z`` change
  what ``read_frame`` returns. Different keys for different channel
  selections so the cache stays coherent when the user toggles AOVs.
* **Alpha flags** — ``alpha_composite`` + ``alpha_is_straight`` change
  the post-read interpretation (premultiply flag, stripping). Folded
  into the key so a layer with αS toggled gets its own cache slot.

What is **NOT** in the key
--------------------------

* **OCIO transform** — applied in the GPU shader at display time, not
  during cache fill. Changing the OCIO config doesn't invalidate the
  disk cache (intentional: it's free to re-display under a different
  config).
* **Layer offset / trim** — these are session-level remappings; the
  cached pixels are the raw frame buffer at the source frame number,
  not the post-remap composite.
* **Stack order / visibility** — composite-level state, not per-layer.
  The disk cache stores per-layer reads; composition is redone on the
  fly when the RAM tier reassembles.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def source_key_for_layer_frame(
    *,
    source_path: Path | str,
    mtime: float,
    size: int,
    channels: Iterable[str] | None,
    alpha_composite: bool,
    alpha_is_straight: bool,
) -> str:
    """Build the SHA-1 hex key for one layer's contribution at one frame.

    Returns a 40-character lowercase hex string suitable as the key
    parameter to :meth:`img_player.cache.disk_cache.DiskCache.get` /
    :meth:`put`.

    Parameters
    ----------
    source_path
        Path to the source file. The caller should pass the **already-
        resolved** absolute path so symlinks / relative bits collapse
        before hashing — otherwise two sessions navigating the same
        file via different paths get different keys.
    mtime
        File modification time, typically ``path.stat().st_mtime``.
        Float seconds since epoch. Folded into the key so an artist
        re-rendering the same filename invalidates the cache slot.
    size
        File size in bytes (``path.stat().st_size``). Defensive — two
        different files with the same mtime are extremely rare but
        not impossible; size guards against it.
    channels
        Iterable of channel names that were passed to ``read_frame``.
        Sorted before hashing so ``["R", "G", "B"]`` and ``["G", "R",
        "B"]`` produce the same key. ``None`` is treated as "default"
        (= whatever the loader picks for this format).
    alpha_composite
        The layer's ``alpha_composite`` flag — controls whether the
        decoder produces a premultiplied RGBA or an opaque RGB.
    alpha_is_straight
        The layer's ``alpha_is_straight`` flag — controls premultiply
        conversion at read time.
    """
    if channels is None:
        channel_str = "DEFAULT"
    else:
        channel_str = ",".join(sorted(channels))
    # Path normalisation: ``str(Path(...).resolve())`` should already
    # be canonical on POSIX. On Windows we additionally lower-case it
    # since NTFS is case-insensitive — two paths that differ only in
    # case must hash the same.
    canonical = str(source_path).replace("\\", "/")
    if canonical and canonical[1:2] == ":":  # "C:/Users/..." → windows-ish
        canonical = canonical.lower()
    # Version prefix bumped to ``v2`` when we fixed the channel-key
    # mismatch in 1.5.1 (the live-state read in ``_source_key_at``
    # caused alt-channel decodes to write blobs under the wrong key,
    # mixing channels in the disk cache). Old v1 blobs stay on disk
    # but are effectively orphaned — LRU eviction sweeps them on the
    # next budget overrun, or the user clears manually via
    # Preferences > Disk cache > Clear cache now.
    payload = (
        f"v2|{canonical}|{int(mtime * 1000)}|{int(size)}|"
        f"{channel_str}|{int(alpha_composite)}{int(alpha_is_straight)}"
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def composite_source_key(per_layer_keys: Iterable[str]) -> str:
    """Build a composite key for a multi-layer over-composite frame.

    A multi-layer composite is uniquely identified by the **ordered**
    list of its contributing layers' source keys (order matters — A
    over B differs from B over A). This wraps the per-layer keys
    into a single hash so a composite can be cached / retrieved as
    one blob.

    Use when the caller has already computed each layer's key via
    :func:`source_key_for_layer_frame`.
    """
    h = hashlib.sha1()
    h.update(b"composite-v2|")  # bumped alongside the per-layer v2 above
    for k in per_layer_keys:
        h.update(k.encode("ascii"))
        h.update(b"|")
    return h.hexdigest()
