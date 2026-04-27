"""Runtime memory pressure snapshot — pure logic + a tiny psutil call.

Section 6 of the spec: ``compute_tune`` looks at *total* RAM, but
that is the wrong number when the user has Nuke / DaVinci / Blender
already eating 60 GB. This module snapshots the *available* RAM at
boot and clamps the cache budget so img_player doesn't reserve
memory it doesn't have — preventing the swap-induced freeze the
user would otherwise blame on the app.

Two halves:

* :class:`RuntimeState` — frozen dataclass, plus a single side-
  effecting factory :meth:`snapshot` that calls ``psutil``. Falls
  back to optimistic numbers if ``psutil`` is broken (better to
  risk a swap than to needlessly cripple a healthy machine).
* :func:`apply_runtime_constraints` — pure function ``(tune, state)
  → tune``. Easy to unit-test without mocking ``psutil``.

The pipeline in ``__main__._resolve_tune`` calls these in this order::

    auto      = compute_tune(hw)                   # heuristics
    overrides = apply_cli_overrides(auto, ...)     # CLI wins
    final     = apply_runtime_constraints(         # runtime safety
                    overrides, RuntimeState.snapshot()
                )

Runtime constraints come **after** CLI overrides on purpose: even
if the user explicitly asks for a 16 GB cache, we still refuse to
swap. The log line will tell them their override was clamped — they
can close other apps and retry, but the app stays responsive in the
meantime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

from img_player.perf.hardware import PerformanceTune

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Constants — sourced from spec §6.
# ----------------------------------------------------------------------------

# Of the *currently available* RAM, the cache claims this fraction.
# 0.6 leaves 40 % headroom for the OS and any new app the user might
# launch during a session. Hardcoded here (not in `hardware.py`)
# because it's a runtime-pressure number, not a static heuristic.
_AVAILABLE_RAM_FRACTION = 0.6

# Never shrink the cache below this — it stops being a cache.
_CACHE_MIN_GB = 2.0

# Above this swap usage at boot we log a warning. The number is
# intentionally low: even 1 GB of swap means the OS is already
# trading memory for disk somewhere, and the user should know.
_SWAP_WARN_THRESHOLD_GB = 1.0


# ----------------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RuntimeState:
    """Snapshot of live memory pressure at a single point in time.

    Used to clamp the cache budget *for this session*. The live
    monitoring (cache shrink during playback) belongs to slice 5's
    ``RuntimeMonitor``; this dataclass is just a one-shot picture.
    """

    available_ram_gb: float
    swap_used_gb: float

    @classmethod
    def snapshot(cls) -> RuntimeState:
        """Build a snapshot from the current ``psutil`` reading.

        The only side-effecting function in this module — kept as a
        classmethod so callers can substitute it in tests via
        ``monkeypatch.setattr(RuntimeState, "snapshot", ...)``.

        Falls back to *optimistic* numbers if ``psutil`` raises:
        ``available_ram_gb`` set to a very large value (effectively
        disabling the cache shrink), ``swap_used_gb`` set to 0.
        Rationale: a broken ``psutil`` shouldn't punish a healthy
        machine. The auto-tune ceiling on ``cache_gb`` (64 GB) still
        applies, so we won't actually allocate something insane.
        """
        try:
            import psutil

            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            return cls(
                available_ram_gb=vm.available / (1024**3),
                swap_used_gb=sw.used / (1024**3),
            )
        except Exception:
            logger.warning(
                "psutil unavailable for runtime snapshot — assuming ample RAM",
            )
            return cls(available_ram_gb=1024.0, swap_used_gb=0.0)


# ----------------------------------------------------------------------------
# The pure constraint function
# ----------------------------------------------------------------------------


def apply_runtime_constraints(
    tune: PerformanceTune,
    state: RuntimeState,
) -> PerformanceTune:
    """Clamp ``tune.cache_gb`` against currently-available RAM.

    Pure function: deterministic on inputs, no I/O, no side effects.
    Returns a new ``PerformanceTune`` (frozen dataclass + ``replace``);
    the input is never mutated.

    The rule: at most 60 % of currently-available RAM. If the
    pre-existing tune already fits, this is a no-op. The 2 GB floor
    is preserved — never shrink below it, the cache stops being
    useful at that point.

    Other tune fields (``num_workers``, ``oiio_threads``, ``use_pbo``)
    are returned untouched — runtime memory pressure has no effect
    on them at boot. The runtime monitor in slice 5 will reduce the
    prefetch window dynamically if it observes degraded playback.
    """
    safe_cache = state.available_ram_gb * _AVAILABLE_RAM_FRACTION
    if safe_cache >= tune.cache_gb:
        # Plenty of room — no-op.
        return tune
    return replace(tune, cache_gb=max(_CACHE_MIN_GB, safe_cache))


# ----------------------------------------------------------------------------
# Logging helper
# ----------------------------------------------------------------------------


def log_runtime_state(
    state: RuntimeState,
    before: PerformanceTune,
    after: PerformanceTune,
    log: logging.Logger | None = None,
) -> None:
    """Emit the ``[hw-tune]`` runtime lines documented in spec §6.

    Always logs the snapshot. Logs the cache-shrink line only when
    the tune was actually reduced. Logs the swap warning only when
    swap is non-trivially in use already.

    The message strings match the spec verbatim so a tail-the-log
    consumer can pattern-match them.
    """
    log = log or logger
    log.info(
        "[hw-tune] runtime check: available_ram=%.1f GB, swap_used=%.1f GB",
        state.available_ram_gb,
        state.swap_used_gb,
    )
    if after.cache_gb < before.cache_gb:
        log.info(
            "[hw-tune] reduced cache from %.1f→%.1f GB "
            "(only %.1f GB available, leaving headroom for other apps)",
            before.cache_gb,
            after.cache_gb,
            state.available_ram_gb,
        )
    if state.swap_used_gb > _SWAP_WARN_THRESHOLD_GB:
        log.warning(
            "[hw-tune] swap is already in use (%.1f GB) — "
            "system is under memory pressure before img_player even started",
            state.swap_used_gb,
        )
