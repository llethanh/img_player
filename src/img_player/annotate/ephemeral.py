"""The :class:`EphemeralStrokeManager` — live, fading, never-saved strokes.

Companion to :class:`~img_player.annotate.store.AnnotationStore`. Where
``AnnotationStore`` is the persistent half (per-frame strokes saved to
the sidecar JSON), this module owns the **ephemeral** half: strokes
drawn during a live presentation (think Google Meet's screen-share
annotations) that fade linearly over a few seconds and never touch
the disk.

Two artifacts live here:

* :func:`alpha_at` — pure module-level helper computing the opacity
  of a stroke given its birth timestamp, the current time, and the
  fade duration. Unit-testable without spinning up Qt — same pattern
  as :func:`~img_player.annotate.overlay.widget_to_image`.
* :class:`EphemeralStrokeManager` — ``QObject`` owning the live list,
  driving a ``QTimer`` for repaints, and emitting ``repaint_needed``.

Design points (see spec ``2026-04-28-ephemeral-annotations-design.md``):

* **Image-space, frame-agnostic.** Strokes are positioned in image
  pixels (same as persistent strokes) but visible regardless of the
  current frame. Presenter model — gestures comment whatever's on
  screen, not a specific frame.
* **Strict zero persistence.** No sidecar write, no dirty flag, no
  close prompt. Strokes die with the process.
* **Mode-agnostic manager.** The manager doesn't know whether
  "ephemeral mode" is active in the toolbar. It simply collects
  whatever ``add()`` is called with. The overlay decides where to
  route a finished stroke based on a press-time snapshot.
* **Auto start/stop timer.** ``add()`` starts the timer when the
  list was empty; the per-tick sweep stops it when the list becomes
  empty. No idle polling.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QTimer, Signal

from img_player.annotate.stroke import Stroke


# ============================================================================
# Pure helper (module-level for testability)
# ============================================================================


def alpha_at(birth_ts: float, now_ts: float, duration_s: float) -> float:
    """Linear opacity for a stroke born at ``birth_ts``, at time ``now_ts``.

    Goes from ``1.0`` at birth to ``0.0`` at ``birth + duration_s``,
    linearly. Clamped to ``[0.0, 1.0]`` outside that window.

    Two edge cases worth pinning down:

    * ``now_ts < birth_ts`` (clock jitter — NTP correction during a
      video call, or suspend/resume with a slightly off
      ``time.monotonic()``): treated as age = 0, returns ``1.0``. We
      never want a stroke to render as a faint ghost just because
      the clock blipped backwards.
    * ``duration_s <= 0`` (slider somehow at zero): instant death,
      returns ``0.0``. The stroke never had a chance.
    """
    age = max(0.0, now_ts - birth_ts)
    if duration_s <= 0.0:
        return 0.0
    return max(0.0, 1.0 - age / duration_s)


# ============================================================================
# The manager
# ============================================================================


# 33 ms ≈ 30 FPS. Fast enough that the linear fade looks smooth, slow
# enough not to burn CPU. The manager only ticks while strokes are
# alive, so this is zero-cost when nothing is being drawn.
_TICK_INTERVAL_MS = 33


class EphemeralStrokeManager(QObject):
    """Owns live ephemeral strokes + drives their fade lifecycle.

    Public surface (7 methods + 1 signal) is the entire integration
    contract — the overlay queries it, the toolbar/app push duration
    changes and admin commands. Nothing else should touch the
    internal list.
    """

    repaint_needed = Signal()
    """Emitted when the overlay should repaint.

    Fires on every state change: ``add()``, ``kill_last()``,
    ``clear_all()``, and on every timer tick (so the alpha animates
    smoothly). The overlay connects this to its ``update()`` slot.
    """

    def __init__(self, *, parent: QObject | None = None) -> None:
        super().__init__(parent)
        # Birth timestamp captured via time.monotonic() — robust to
        # system clock adjustments during a live call.
        self._strokes: list[tuple[Stroke, float]] = []
        # Default 5s — the spec's "moyen" preset. Overwritten by
        # set_duration() at toolbar boot once preferences are read.
        self._duration_s: float = 5.0

        # Single shared timer. Start/stop in lock-step with
        # ``has_live_strokes()`` so we never poll while idle.
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_INTERVAL_MS)
        self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------ Public API

    def set_duration(self, seconds: float) -> None:
        """Set the fade duration in seconds. Affects all live strokes.

        Already-born strokes keep their ``birth_ts`` but their alpha
        is recomputed against the new duration on the next read. So
        shortening from 10 s → 2 s mid-presentation will visibly
        accelerate the fade of any in-flight stroke. Useful when the
        screen gets cluttered.
        """
        # Defensive: a non-positive duration would mean instant death
        # for every stroke. We allow it (matches alpha_at's contract)
        # but the toolbar never sends one.
        try:
            self._duration_s = float(seconds)
        except (TypeError, ValueError):
            return

    def duration(self) -> float:
        """Current fade duration in seconds — for tests and diagnostics."""
        return self._duration_s

    def add(self, stroke: Stroke) -> None:
        """Insert a fresh stroke. Auto-starts the timer if it was idle.

        Called by the overlay's ``mouseReleaseEvent`` when the user
        finished drawing in ephemeral mode. ``time.monotonic()`` is
        captured *here* (release time) — not at press, not at the
        first move sample. The user's intent is "the stroke exists
        at the moment I lifted the pen".
        """
        was_empty = not self._strokes
        self._strokes.append((stroke, time.monotonic()))
        if was_empty:
            # First stroke after an idle period — wake the timer.
            self._timer.start()
        self.repaint_needed.emit()

    def kill_last(self) -> bool:
        """Remove the most recently born stroke. Returns whether anything happened.

        Wired to the ``Ctrl+Z`` shortcut while ephemeral mode is
        active. We chose "kill last" rather than "undo to previous
        state" because ephemeral strokes have no notion of history —
        once you start a fade, the stroke is on a one-way trip.
        Killing the youngest is the closest analog to undo.
        """
        if not self._strokes:
            return False
        self._strokes.pop()
        if not self._strokes:
            self._timer.stop()
        self.repaint_needed.emit()
        return True

    def clear_all(self) -> int:
        """Wipe every live stroke. Returns the number removed.

        Wired to the ``Clear`` button while ephemeral mode is active,
        and to the sequence-change callback in ``app.py`` (so ghosts
        from a previous sequence don't bleed over the new one).
        """
        if not self._strokes:
            return 0
        count = len(self._strokes)
        self._strokes.clear()
        self._timer.stop()
        self.repaint_needed.emit()
        return count

    def live_strokes_with_alpha(
        self, *, now_ts: float | None = None
    ) -> tuple[tuple[Stroke, float], ...]:
        """Snapshot of every live stroke + its current alpha.

        Insertion order is preserved so the overlay paints earlier
        strokes underneath later ones — same convention as
        :class:`AnnotationStore`. Strokes whose alpha would compute
        to ``0.0`` are filtered out (they're effectively dead but
        the timer hasn't swept them yet — paying for the GC immediately
        avoids a one-frame ghost).

        ``now_ts`` is exposed for tests; production callers leave it
        ``None`` and we read ``time.monotonic()`` ourselves.
        """
        if now_ts is None:
            now_ts = time.monotonic()
        out: list[tuple[Stroke, float]] = []
        for stroke, birth in self._strokes:
            a = alpha_at(birth, now_ts, self._duration_s)
            if a > 0.0:
                out.append((stroke, a))
        return tuple(out)

    def has_live_strokes(self) -> bool:
        """``True`` if at least one stroke is in the list (regardless of alpha).

        Note: a stroke that's alpha=0 but not yet swept by the timer
        is still "live" by this definition. The timer will catch it
        on the next tick. This method is mainly used by tests; the
        overlay reads :meth:`live_strokes_with_alpha` directly.
        """
        return bool(self._strokes)

    # ------------------------------------------------------------------ Timer

    def _on_tick(self) -> None:
        """Periodic sweep: drop expired strokes, request a repaint, idle if empty.

        Runs every ``_TICK_INTERVAL_MS`` while the timer is active.
        Three things happen, in order:

        1. Compute ``now`` once (we want all alpha-checks in this tick
           to share the same instant).
        2. Filter out expired strokes (alpha == 0).
        3. Emit ``repaint_needed`` so the overlay re-renders the
           still-fading strokes at their new alpha.
        4. If the list is empty after the sweep, stop the timer so
           we don't poll while idle.
        """
        now = time.monotonic()
        # Filter in place — keep only the still-alive strokes.
        kept = [
            (stroke, birth)
            for stroke, birth in self._strokes
            if alpha_at(birth, now, self._duration_s) > 0.0
        ]
        self._strokes = kept
        self.repaint_needed.emit()
        if not self._strokes:
            self._timer.stop()

    # ------------------------------------------------------------------ Test helpers

    def _is_timer_active(self) -> bool:
        """Test-only: peek at the timer state without exposing the QTimer."""
        return self._timer.isActive()

    def _stroke_count(self) -> int:
        """Test-only: peek at the live list size without exposing it."""
        return len(self._strokes)
