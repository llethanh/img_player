"""Internal shared helpers + constants used by both :mod:`frame_cache`
and :mod:`master_frame_cache`.

These were originally duplicated between the two cache modules.
Centralising them here keeps both implementations in sync when the
filename convention, budget defaults, or eviction-scoring constants
need to evolve.

This module is intentionally tiny and dependency-light — only
:class:`SequenceInfo` so :func:`expected_filename` works. The rest is
plain constants. Anything richer (e.g. ``CacheStats``) lives next to
its primary consumer; cache flavours that need a different stats
shape don't pay for fields they don't use.
"""

from __future__ import annotations

from dataclasses import dataclass

from img_player.sequence.models import SequenceInfo


# --- Tuning defaults --------------------------------------------------------

# RAM budget the cache will not exceed (worker pool's prefetch is
# throttled by ``_evict_if_over_budget`` once this is hit). 8 GiB is
# the conservative floor; the auto-tune in ``perf`` widens it on
# workstations with more memory.
DEFAULT_BUDGET_BYTES: int = 8 * 1024**3

# Default number of decode worker threads. The autotune overrides this
# based on CPU thread count, but the bare ``FrameCache()`` constructor
# without args (tests, ad-hoc tools) gets this floor.
DEFAULT_NUM_WORKERS: int = 4

# Eviction multiplier for frames that lie *behind* the playhead in the
# current playback direction. They cost more because we'll only revisit
# them after a full loop wrap — so we throw them out first to free space
# for what's coming up next.
BEHIND_PLAYHEAD_PENALTY: float = 3.0


# --- Shared helpers ---------------------------------------------------------


def expected_filename(seq: SequenceInfo, frame_number: int) -> str:
    """Reconstruct the filename a missing slot *would* have on disk.

    Uses the sequence's ``base_name`` + observed zero-pad width +
    ``extension``. Mirrors the convention the scanner uses to detect
    holes, so any "missing frame" overlay the user sees in the cache
    bar or timeline corresponds 1-to-1 with a file path they can
    look for on disk.
    """
    pad = max(0, int(seq.padding))
    digits = f"{frame_number:0{pad}d}" if pad > 0 else str(frame_number)
    return f"{seq.base_name}{digits}{seq.extension}"


# --- Stats ------------------------------------------------------------------


@dataclass(frozen=True)
class CacheStats:
    """Snapshot of a frame cache's hit/miss counters + memory state.

    Used by both :class:`FrameCache` and :class:`MasterFrameCache` —
    same shape, same semantics. UI code (status bar, Preferences >
    Disk cache page) duck-types on this dataclass.
    """

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    decode_errors: int = 0
    bytes_used: int = 0
    bytes_budget: int = 0
    frames_cached: int = 0
