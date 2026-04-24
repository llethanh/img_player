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

        Tasks currently being executed by workers continue to completion.
        """
        dropped = 0
        drained: list[tuple[int, int, Any, Callable[[], None] | None]] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            prio, _, key, _fn = item
            if prio == _SHUTDOWN_PRIORITY:
                # Preserve shutdown sentinels — do not eat them.
                drained.append(item)
                continue
            if key is not None:
                with self._lock:
                    self._pending.discard(key)
                dropped += 1
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
