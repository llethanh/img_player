"""Per-machine ``PerformanceTune`` persistence — spec section 9.

Slice 6 of the hw-adaptive perf work, implemented as a *minimal*
foundation:

* **Persists the resolved tune** at shutdown into a per-user JSON
  file (``profile.json`` under the platform's cache dir).
* **Reuses it at boot** if the machine signature matches: skipping
  ``compute_tune`` and going straight to the previously-applied
  values. This gives the user (a) stability across versions of the
  heuristics, and (b) a hand-editable file to pin values they
  prefer over the auto-tune.

What this slice deliberately does **not** do (yet):

* No real self-bench (no synthetic 4K frames pushed through the GL
  upload path to *measure* the machine). The dataclass leaves room
  for it (``calibration`` field) so a future slice can fill in
  empirically-measured corrections without breaking the schema.
* No splash screen during the first launch. That would require a
  separate GL context and a careful bundle of QSplashScreen +
  asset loading; out of scope for the bare-minimum delivery here.

The two CLI flags from spec §9 (``--skip-calibration`` and
``--recalibrate``) are wired all the way through, so once the
self-bench step lands they snap into place without further changes
to the public surface.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from img_player.perf.hardware import HardwareProfile, PerformanceTune

logger = logging.getLogger(__name__)


# Schema version. Bump every time we add or remove a top-level field
# in CalibrationProfile. Old files become unreadable (treated as
# missing) until the user re-calibrates — that's safer than silently
# loading a partial profile.
_SCHEMA_VERSION = 1


# ----------------------------------------------------------------------------
# CalibrationProfile dataclass + (de)serialization
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationProfile:
    """Persisted per-machine performance state.

    Mirrors the JSON layout exactly (modulo the schema_version key,
    which is added during serialization). The two "logical" parts
    are:

    * **hw_signature** — what machine this profile is for. Both
      raw fields (for human inspection) and a SHA-1 digest (for
      cheap equality at load time).
    * **applied_tune** — the four ``PerformanceTune`` fields that
      this profile resolves to. The runtime uses these directly
      instead of re-running ``compute_tune`` when the profile is
      found valid for the current machine.
    * **ran_at** — ISO-8601 UTC timestamp of when the profile was
      last written. Diagnostic only; the runtime doesn't read it.
    """

    cpu_threads: int
    ram_total_gb: float
    gpu_renderer: str
    digest: str  # SHA-1 hex of the three above
    num_workers: int
    cache_gb: float
    oiio_threads: int
    use_pbo: bool
    ran_at: str  # ISO-8601 UTC

    def matches(self, hw: HardwareProfile) -> bool:
        """True if this profile was computed for the given machine."""
        return self.digest == hw_signature(hw)

    def to_tune(self) -> PerformanceTune:
        """Extract the ``PerformanceTune`` from a stored profile."""
        return PerformanceTune(
            num_workers=self.num_workers,
            cache_gb=self.cache_gb,
            oiio_threads=self.oiio_threads,
            use_pbo=self.use_pbo,
        )


# ----------------------------------------------------------------------------
# Hardware signature
# ----------------------------------------------------------------------------


def hw_signature(hw: HardwareProfile) -> str:
    """Return a stable SHA-1 hex digest identifying the machine.

    Includes ``cpu_threads`` and ``total_ram_gb`` (rounded to the
    nearest GB so noise from ``psutil`` doesn't invalidate the
    profile). **Does not include** ``gpu_renderer``.

    Why exclude the GPU: ``hw_signature()`` is computed both at
    boot (when the GL context doesn't exist — ``gpu_renderer`` is
    ``None``) and after late-bind (when we know the real renderer).
    If the GPU were part of the signature the boot-time lookup
    would never match a profile saved at shutdown, and the boot is
    the only moment where ``cache_gb`` and ``num_workers`` can
    be applied (the cache and worker pool become live right after).

    Trade-off: a user who toggles their laptop's iGPU↔dGPU routing
    between sessions will see the same profile reused —
    ``use_pbo=True`` saved on the dGPU is wrong on the iGPU.
    Mitigation: pass ``--recalibrate`` after the switch.

    The signature has nothing to do with security; it's just a
    cheap way to detect "machine changed".
    """
    raw = "|".join([
        str(hw.cpu_threads),
        str(round(hw.total_ram_gb)),
    ])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------
# Cache location
# ----------------------------------------------------------------------------


def profile_path() -> Path:
    """Return where ``profile.json`` lives.

    Native per-OS conventions, **without** going through
    ``QStandardPaths``. We deliberately don't use Qt because the
    boot-time call site is BEFORE ``QApplication`` exists — at that
    point ``QStandardPaths`` returns a path that lacks the
    organisation/application name (the Qt fallback path) and
    diverges from the path resolved later in the same session
    (after ``QApplication.setApplicationName`` ran). That mismatch
    silently makes the profile invisible at boot — which is the
    one moment where the persisted ``cache_gb`` matters most.

    Resolution rules:

    * **Windows** — ``%LOCALAPPDATA%\\img_player\\profile.json``,
      falling back to ``~/AppData/Local/img_player/profile.json``.
    * **macOS** — ``~/Library/Caches/img_player/profile.json``.
    * **Linux + others** — ``$XDG_CACHE_HOME/img_player/profile.json``,
      falling back to ``~/.cache/img_player/profile.json``.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "img_player" / "profile.json"
        return Path.home() / "AppData" / "Local" / "img_player" / "profile.json"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "img_player" / "profile.json"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache) / "img_player" / "profile.json"
    return Path.home() / ".cache" / "img_player" / "profile.json"


# ----------------------------------------------------------------------------
# Load / save
# ----------------------------------------------------------------------------


def load_profile(path: Path | None = None) -> CalibrationProfile | None:
    """Read ``profile.json`` and return the profile, or ``None`` if
    missing / malformed / wrong schema version.

    Failures are logged at INFO level (not WARNING) because a missing
    profile is the *expected* state at first launch — not a problem.
    Malformed JSON does log a WARNING because that is anomalous.
    """
    p = path or profile_path()
    if not p.exists():
        logger.info("[calibration] no profile found at %s (first launch?)", p)
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as err:
        logger.warning("[calibration] could not read profile: %s", err)
        return None

    if raw.get("schema_version") != _SCHEMA_VERSION:
        logger.warning(
            "[calibration] profile schema version %s does not match expected %s — ignoring",
            raw.get("schema_version"),
            _SCHEMA_VERSION,
        )
        return None
    try:
        return CalibrationProfile(
            cpu_threads=int(raw["cpu_threads"]),
            ram_total_gb=float(raw["ram_total_gb"]),
            gpu_renderer=str(raw["gpu_renderer"]),
            digest=str(raw["digest"]),
            num_workers=int(raw["num_workers"]),
            cache_gb=float(raw["cache_gb"]),
            oiio_threads=int(raw["oiio_threads"]),
            use_pbo=bool(raw["use_pbo"]),
            ran_at=str(raw["ran_at"]),
        )
    except (KeyError, ValueError, TypeError) as err:
        logger.warning("[calibration] profile field error (%s) — ignoring", err)
        return None


def save_profile(
    profile: CalibrationProfile, path: Path | None = None,
) -> None:
    """Write ``profile.json`` atomically.

    Creates parent dirs if needed. Uses a temp file + rename so a
    partial write doesn't leave an unreadable file behind. Errors
    are caught and logged — this is best-effort, the app must boot
    even if disk is read-only.
    """
    p = path or profile_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {"schema_version": _SCHEMA_VERSION}
        payload.update(asdict(profile))
        # Atomic write: dump to a sibling .tmp, then rename. Avoids
        # the "PC crashed mid-save" failure mode that would leave
        # a half-written profile that load_profile would treat as
        # malformed.
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(p)
        logger.info("[calibration] wrote profile to %s", p)
    except OSError as err:
        logger.warning("[calibration] could not save profile: %s", err)


# ----------------------------------------------------------------------------
# Profile factory
# ----------------------------------------------------------------------------


def build_profile(hw: HardwareProfile, tune: PerformanceTune) -> CalibrationProfile:
    """Construct a ``CalibrationProfile`` from a HW snapshot + a tune.

    Called at shutdown to persist what the session actually used,
    so the next boot can skip ``compute_tune`` and reuse the same
    values verbatim.
    """
    return CalibrationProfile(
        cpu_threads=hw.cpu_threads,
        ram_total_gb=hw.total_ram_gb,
        gpu_renderer=hw.gpu_renderer if hw.gpu_renderer is not None else "None",
        digest=hw_signature(hw),
        num_workers=tune.num_workers,
        cache_gb=tune.cache_gb,
        oiio_threads=tune.oiio_threads,
        use_pbo=tune.use_pbo,
        ran_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )


# ----------------------------------------------------------------------------
# Boot integration
# ----------------------------------------------------------------------------


def apply_profile_to_tune(
    tune: PerformanceTune,
    profile: CalibrationProfile | None,
    hw: HardwareProfile,
) -> PerformanceTune:
    """Replace ``tune`` with the persisted one if the profile matches.

    The contract: at most one of two things happens —

    * The profile is None or doesn't match the current machine →
      ``tune`` is returned unchanged. This is the first-launch path
      and the "machine changed" path.
    * The profile matches → its ``applied_tune`` becomes the new
      tune. This intentionally overrides the freshly-computed
      heuristics, because a previous session on this exact machine
      already settled on values the user hasn't complained about.

    The resulting tune still goes through ``apply_cli_overrides``
    and ``apply_runtime_constraints`` afterwards in the boot
    pipeline (see ``__main__._resolve_tune``), so explicit CLI
    flags and live RAM pressure still win.
    """
    if profile is None or not profile.matches(hw):
        return tune
    logger.info(
        "[calibration] reusing profile from %s (matches signature %s…)",
        profile.ran_at,
        profile.digest[:8],
    )
    return replace(
        tune,
        num_workers=profile.num_workers,
        cache_gb=profile.cache_gb,
        oiio_threads=profile.oiio_threads,
        use_pbo=profile.use_pbo,
    )
