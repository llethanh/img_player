"""1 Hz watchdog that auto-corrects mid-playback under memory pressure.

Section 7 of the spec. The boot-time health check (slice 3) deals
with conditions known *before* playback starts; this module deals
with conditions that develop *during* playback — a long shot opens
in Nuke, a Chrome tab takes 4 GB, the OS starts trading RAM for
disk. The runtime monitor detects each of these and reacts:

* **Cache hit rate falls below 80 % for 5 s** → shrink the prefetch
  window so workers focus on what we'll actually display next, not
  on far-ahead frames the cache can't hold anyway.
* **Cache hit rate stays low for 10 s** → emit ``playback_struggle``,
  the UI surfaces a status-bar message in plain French ("la machine
  ne suit pas le rythme").
* **Swap usage grows by > 500 MB during playback** → shrink the
  cache budget by 25 % and force an eviction. Emit ``memory_pressure``
  with a recoverable suggestion ("fermez d'autres applications").

What we do NOT do:

* **Auto-grow** — once shrunk, values stay shrunk for the session.
  This is a deliberate choice (spec §7) to avoid oscillation under
  bursty pressure. The user gets a chance to close apps and restart
  for a roomier setup.
* **Frame-pacing detection** (``paint_p99``) — the spec lists a
  ``frame_pacing_drop`` signal, but its source (``bench/recorder.py``
  rolling 100-paint window) requires the recorder to be on
  permanently. We define the signal for forward-compatibility but
  the implementation is deferred to a follow-up; see the docstring
  on the signal.

Lifecycle: hooked to ``controller.state_changed``. The 1 Hz timer
runs only while ``state.is_playing`` is True — there's nothing to
monitor when playback is paused, and a stopped timer doesn't fire.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QTimer, Signal

if TYPE_CHECKING:
    from img_player.cache.frame_cache import FrameCache
    from img_player.player.controller import PlayerController
    from img_player.player.state import PlaybackState


log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Tunables — exposed as class constants so a test can override them by
# subclass / monkeypatch without touching the runtime instance.
# ----------------------------------------------------------------------------


_HIT_RATE_THRESHOLD = 0.80
_HIT_RATE_LOW_BEFORE_SHRINK_S = 5.0
_HIT_RATE_LOW_BEFORE_WARN_S = 10.0
_PREFETCH_SHRINK_FACTOR = 0.75
_PREFETCH_FLOOR = 8
_SWAP_DELTA_THRESHOLD_GB = 0.5  # 500 MB
_CACHE_SHRINK_FACTOR = 0.75


# Plain-French user-facing messages that the monitor emits as the
# payload of its three Qt signals. They are kept inline (not in a
# resources file) because they're a fixed contract and easy to grep
# from a bug report.
_MSG_STRUGGLE = (
    "Lecture irrégulière sur cette séquence — la machine ne suit pas le rythme."
)
_MSG_MEMORY_PRESSURE = (
    "Mémoire insuffisante — cache réduit automatiquement. "
    "Fermez d'autres applications pour de meilleures performances."
)
_MSG_FRAME_PACING = (
    "Décrochement de lecture détecté — paint p99 au-delà du budget."
)


def _read_swap_used_gb() -> float | None:
    """Return current swap usage in GB, or ``None`` if psutil isn't usable.

    Wrapped here so the rest of the class doesn't import psutil
    directly — easier to mock in tests.
    """
    try:
        import psutil

        return psutil.swap_memory().used / (1024**3)
    except Exception:  # pragma: no cover — fallback path
        return None


# ============================================================================
# RuntimeMonitor
# ============================================================================


class RuntimeMonitor(QObject):  # type: ignore[misc]
    """1 Hz watchdog hooked to a :class:`PlayerController` + :class:`FrameCache`.

    Construct once at app boot, parented to the main window so it
    shares its lifetime. Hook the three signals to whatever UI layer
    surfaces user-facing warnings (status bar set_status, toast, etc).
    """

    # Emitted when sustained low cache hit rate suggests the machine
    # can't decode fast enough to keep the prefetch window full. The
    # payload is a French message ready to display verbatim.
    playback_struggle = Signal(str)
    # Emitted when swap_used grew by more than the threshold during
    # playback (= the OS started paging). The cache budget has been
    # auto-reduced before this signal fires.
    memory_pressure = Signal(str)
    # Reserved for future paint p99 integration. Defined now so the
    # status-bar layer can wire it permanently; not yet emitted.
    frame_pacing_drop = Signal(str)

    def __init__(
        self,
        controller: PlayerController,
        cache: FrameCache,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._cache = cache

        self._timer = QTimer(self)
        self._timer.setInterval(1000)  # 1 Hz
        self._timer.timeout.connect(self._tick)

        # Subscribe to play / pause transitions. The monitor only does
        # work while playing — paused playback has nothing to monitor
        # and a free-running timer would just waste cycles.
        self._controller.state_changed.connect(self._on_state_changed)

        # State that resets at every play_started. The "_emitted"
        # latches stay True for the whole session: once we've warned
        # the user about a problem, we don't spam them every second.
        self._timer_running = False
        self._swap_at_play_start: float | None = None
        self._hit_rate_low_since: float | None = None
        self._struggle_emitted: bool = False
        self._memory_pressure_emitted: bool = False
        # Track whether we've already shrunk the prefetch on this
        # session — if so, don't keep shrinking on every threshold
        # crossing (would zero out the prefetch).
        self._prefetch_shrunk: bool = False

    # -- Public hook for tests / debug --------------------------------------

    def is_running(self) -> bool:
        """Whether the 1 Hz tick is currently armed."""
        return self._timer_running

    # -- State transitions --------------------------------------------------

    def _on_state_changed(self, state: PlaybackState) -> None:
        if state.is_playing and not self._timer_running:
            self._on_play_started()
        elif not state.is_playing and self._timer_running:
            self._on_play_stopped()

    def _on_play_started(self) -> None:
        # Capture swap baseline so deltas are play-relative, not
        # boot-relative — this distinguishes "machine is gradually
        # paging because of img_player itself" from "user opened
        # Nuke between two playbacks".
        self._swap_at_play_start = _read_swap_used_gb()
        self._hit_rate_low_since = None
        # The "_emitted" latches reset per playback session: a single
        # issue gets one toast, but a fresh play after a pause gets a
        # new chance to warn (in case the user fixed the problem).
        self._struggle_emitted = False
        self._memory_pressure_emitted = False
        self._prefetch_shrunk = False
        self._timer.start()
        self._timer_running = True

    def _on_play_stopped(self) -> None:
        self._timer.stop()
        self._timer_running = False

    # -- Per-tick monitoring ------------------------------------------------

    def _tick(self) -> None:
        """Run the three checks. Called at 1 Hz while playing."""
        self._check_cache_hit_rate()
        self._check_swap_pressure()
        # frame_pacing_drop deferred — see module docstring.

    def _check_cache_hit_rate(self) -> None:
        rate = self._controller.cache_hit_rate()
        if rate is None:
            # Not enough samples yet — the controller needs at least
            # 4 ticks before reporting. Reset the "low since" timer
            # so a transient None doesn't lengthen the count.
            return

        now = time.monotonic()

        if rate >= _HIT_RATE_THRESHOLD:
            # Recovered. Clear the timer but DON'T un-emit warnings —
            # the user already saw them, and silently going green
            # would suggest the issue self-resolved (which is true,
            # but we leave the messaging to whoever consumes the
            # signals; we just don't fire new ones).
            self._hit_rate_low_since = None
            return

        if self._hit_rate_low_since is None:
            self._hit_rate_low_since = now
            return

        elapsed = now - self._hit_rate_low_since
        if elapsed >= _HIT_RATE_LOW_BEFORE_WARN_S and not self._struggle_emitted:
            # Sustained low hit rate — escalate to the user.
            self.playback_struggle.emit(_MSG_STRUGGLE)
            self._struggle_emitted = True
        elif elapsed >= _HIT_RATE_LOW_BEFORE_SHRINK_S and not self._prefetch_shrunk:
            # First-tier intervention: shrink the prefetch window.
            # The workers will then focus on imminent frames instead
            # of pre-decoding far ones the cache can't hold.
            old = self._controller.get_prefetch_ahead()
            new = max(_PREFETCH_FLOOR, int(old * _PREFETCH_SHRINK_FACTOR))
            if new < old:
                log.info(
                    "[runtime] prefetch window shrink: %d → %d "
                    "(low cache hit rate %.0f%% sustained for %.0fs)",
                    old, new, rate * 100, elapsed,
                )
                self._controller.set_prefetch_ahead(new)
            self._prefetch_shrunk = True

    def _check_swap_pressure(self) -> None:
        if self._swap_at_play_start is None or self._memory_pressure_emitted:
            return
        current = _read_swap_used_gb()
        if current is None:
            return
        delta = current - self._swap_at_play_start
        if delta < _SWAP_DELTA_THRESHOLD_GB:
            return

        # Swap grew during playback — shrink the cache to give memory
        # back to the OS, then warn the user. We do the shrink BEFORE
        # the emit so the user-facing message is honest ("cache
        # réduit") rather than a future tense.
        old_budget = self._cache._budget
        new_budget = int(old_budget * _CACHE_SHRINK_FACTOR)
        log.info(
            "[runtime] cache shrink: %.1f → %.1f GB "
            "(swap pressure: +%.1f GB during playback)",
            old_budget / 1024**3, new_budget / 1024**3, delta,
        )
        self._cache.shrink_budget(new_budget)
        self.memory_pressure.emit(_MSG_MEMORY_PRESSURE)
        self._memory_pressure_emitted = True
