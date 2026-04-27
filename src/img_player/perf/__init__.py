"""Hardware detection and performance tuning.

This package contains the logic that decides how to dimension the
runtime (worker pool, frame cache, OIIO threads, PBO usage) on a
given machine.

The submodules form three layers:

* `hardware` — *pure logic*: detect the running machine and apply
  heuristics. No Qt, no OIIO, no GL imports. Trivially unit-testable.
* `runtime_state` (slice 3) — *pure logic*: snapshot live memory
  pressure and clamp the static tune accordingly.
* `runtime_monitor` (slice 5) — Qt-aware: 1 Hz watchdog that emits
  warnings and shrinks the cache under load.
* `calibration` (slice 6) — Qt + GL aware: first-launch self-bench
  that persists a per-machine profile.

See `docs/specs/2026-04-26-hw-adaptive-perf-design.md` for the full
design.
"""

from img_player.perf.calibration import (
    CalibrationProfile,
    apply_profile_to_tune,
    build_profile,
    hw_signature,
    load_profile,
    profile_path,
    save_profile,
)
from img_player.perf.hardware import (
    GpuKind,
    HardwareProfile,
    PerformanceTune,
    apply_cli_overrides,
    classify_gpu,
    compute_tune,
    detect_hardware,
    log_applied_tune,
    log_tune_resolution,
)
from img_player.perf.runtime_monitor import RuntimeMonitor
from img_player.perf.runtime_state import (
    RuntimeState,
    apply_runtime_constraints,
    log_runtime_state,
)

__all__ = [
    "CalibrationProfile",
    "GpuKind",
    "HardwareProfile",
    "PerformanceTune",
    "RuntimeMonitor",
    "RuntimeState",
    "apply_cli_overrides",
    "apply_profile_to_tune",
    "apply_runtime_constraints",
    "build_profile",
    "classify_gpu",
    "compute_tune",
    "detect_hardware",
    "hw_signature",
    "load_profile",
    "log_applied_tune",
    "log_runtime_state",
    "log_tune_resolution",
    "profile_path",
    "save_profile",
]
