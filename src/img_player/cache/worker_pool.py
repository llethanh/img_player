"""Priority-based thread pool with task deduplication by key.

Used by the frame cache to run image decoding off the Qt thread.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections.abc import Callable
from itertools import count
from typing import Any

log = logging.getLogger(__name__)

# Sentinel used to unblock workers on shutdown.
_SHUTDOWN_PRIORITY = -(10**9)


class WorkerPool:
    """Thread pool that consumes a priority queue.

    Submitters provide (priority, key, fn). Lower `priority` runs sooner.
    If a task with the same `key` is already pending, a new submit is dropped
    (dedup). This matches the decode-a-frame use case: no point queuing the
    same frame twice.
    """

    def __init__(self, num_workers: int = 4, name: str = "worker") -> None:
        self._num_workers = max(1, num_workers)
        self._queue: queue.PriorityQueue[tuple[int, int, Any, Callable[[], None] | None]] = (
            queue.PriorityQueue()
        )
        self._counter = count()
        self._lock = threading.Lock()
        self._pending: set[Any] = set()
        self._shutdown_flag = False
        self._threads = [
            threading.Thread(target=self._run, name=f"{name}-{i}", daemon=True)
            for i in range(self._num_workers)
        ]
        for t in self._threads:
            t.start()

    def submit(self, priority: int, key: Any, fn: Callable[[], None]) -> bool:
        """Enqueue `fn`. Returns False if `key` is already pending."""
        with self._lock:
            if self._shutdown_flag or key in self._pending:
                return False
            self._pending.add(key)
        self._queue.put((priority, next(self._counter), key, fn))
        return True

    def clear(self) -> int:
        """Drop all queued tasks that haven't started yet. Returns the count dropped.

        Tasks currently being executed by workers continue to
        completion (we can't safely interrupt arbitrary Python /
        OIIO calls). Their result is expected to be discarded by the
        caller via an epoch check at store time.

        The dedup ``_pending`` set is wiped entirely — including the
        keys of in-flight tasks — so new submissions for the same
        keys aren't silently dropped while the old (about-to-be-
        ghosted) decode finishes. Without this, the very frames the
        caller most wants to re-decode (i.e. the ones close to the
        playhead, which were busy when the user e.g. swapped layers)
        end up un-cached and the viewer freezes black at startup.
        In-flight workers harmlessly call ``_pending.discard(key)``
        at the end of their task — a no-op once the key is gone.
        """
        # Keep only shutdown sentinels; drop everything else.
        return self._drain_filter(
            keep=lambda prio, _key: prio == _SHUTDOWN_PRIORITY,
            wipe_all_pending=True,
        )

    def drop_above_priority(self, threshold: int) -> int:
        """Drop every queued task whose priority is **>= threshold**.

        Used by the cache to discard pending alt-channel decodes in
        bulk once the RAM budget is reached: those tasks have very
        high priority numbers (= low priority in the queue), and
        without dropping them the worker pool keeps churning through
        them one at a time even though their decoded results would
        be evicted on store, fragmenting the Python heap.

        Returns the number of tasks dropped. In-flight tasks are
        unaffected — they finish to completion. Shutdown sentinels
        (priority = ``_SHUTDOWN_PRIORITY`` = highly negative) are
        always preserved.
        """
        return self._drain_filter(
            keep=lambda prio, _key: prio == _SHUTDOWN_PRIORITY or prio < threshold,
        )

    def _drain_filter(
        self,
        keep: Callable[[int, Any], bool],
        *,
        wipe_all_pending: bool = False,
    ) -> int:
        """Common drain-then-rebuild loop used by :meth:`clear` and
        :meth:`drop_above_priority`. ``keep(priority, key)`` returns
        ``True`` to re-queue the item, ``False`` to drop it.

        When ``wipe_all_pending`` is True the entire ``_pending`` set
        is cleared regardless of which items were dropped — used by
        :meth:`clear` so in-flight tasks' keys are freed too. When
        False (the default), only the dropped items' keys are
        released, so in-flight + still-queued keys keep their dedup
        slot.
        """
        dropped = 0
        dropped_keys: list[Any] = []
        drained: list[tuple[int, int, Any, Callable[[], None] | None]] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            prio, _, key, _fn = item
            if keep(prio, key):
                drained.append(item)
                continue
            if key is not None:
                dropped += 1
                dropped_keys.append(key)
        # Wipe ``_pending`` after the queue drain so we don't reopen
        # dedup slots while we're still re-queuing surviving items.
        if wipe_all_pending:
            with self._lock:
                self._pending.clear()
        elif dropped_keys:
            with self._lock:
                for k in dropped_keys:
                    self._pending.discard(k)
        for item in drained:
            self._queue.put(item)
        return dropped

    def pending(self) -> int:
        """Number of submitted-but-not-done tasks (queued + running)."""
        with self._lock:
            return len(self._pending)

    def shutdown(self, wait: bool = True, timeout: float = 2.0) -> None:
        """Signal workers to exit. If `wait`, joins the threads."""
        with self._lock:
            self._shutdown_flag = True
        for _ in self._threads:
            self._queue.put((_SHUTDOWN_PRIORITY, -1, None, None))
        if wait:
            for t in self._threads:
                t.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            _prio, _counter, key, fn = self._queue.get()
            if fn is None:  # shutdown sentinel
                return
            try:
                fn()
            except Exception:
                log.exception("worker task raised (key=%r)", key)
            finally:
                with self._lock:
                    self._pending.discard(key)
