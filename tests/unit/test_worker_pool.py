"""Tests for cache/worker_pool.py — priority queue + dedup + shutdown."""

from __future__ import annotations

import threading
import time

from img_player.cache.worker_pool import WorkerPool


def test_submitted_tasks_run() -> None:
    pool = WorkerPool(num_workers=2, name="test")
    try:
        hits: list[int] = []
        lock = threading.Lock()

        def make_task(i: int):
            def inner() -> None:
                with lock:
                    hits.append(i)

            return inner

        for i in range(10):
            pool.submit(priority=0, key=i, fn=make_task(i))

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and pool.pending() > 0:
            time.sleep(0.01)

        assert sorted(hits) == list(range(10))
    finally:
        pool.shutdown()


def test_submit_with_same_key_is_deduped() -> None:
    pool = WorkerPool(num_workers=1, name="test")
    try:
        calls = 0

        def slow_task() -> None:
            nonlocal calls
            calls += 1
            time.sleep(0.05)

        # first submit enqueues; second should be rejected (same key)
        assert pool.submit(0, "same_key", slow_task) is True
        assert pool.submit(0, "same_key", slow_task) is False

        while pool.pending() > 0:
            time.sleep(0.01)

        assert calls == 1
    finally:
        pool.shutdown()


def test_priority_order_is_respected() -> None:
    # Single worker so we can observe ordering deterministically.
    pool = WorkerPool(num_workers=1, name="test")
    try:
        # Blocking task to make sure priorities queue up before execution starts.
        gate = threading.Event()
        pool.submit(100, "blocker", lambda: gate.wait(timeout=1.0))

        order: list[int] = []

        def make(i: int):
            def inner() -> None:
                order.append(i)

            return inner

        pool.submit(5, "low", make(5))
        pool.submit(1, "high", make(1))
        pool.submit(3, "mid", make(3))

        gate.set()

        while pool.pending() > 0:
            time.sleep(0.01)

        assert order == [1, 3, 5]
    finally:
        pool.shutdown()


def test_clear_drops_pending_tasks() -> None:
    pool = WorkerPool(num_workers=1, name="test")
    try:
        started = threading.Event()
        gate = threading.Event()

        def blocker() -> None:
            started.set()
            gate.wait(timeout=1.0)

        # blocker keeps the worker busy so queued tasks stay queued
        pool.submit(0, "blocker", blocker)
        # Wait until the single worker has actually picked up the blocker,
        # otherwise clear() races and may drop it too.
        assert started.wait(timeout=1.0)

        for i in range(5):
            pool.submit(1, f"task_{i}", lambda: None)

        dropped = pool.clear()
        assert dropped == 5
        assert pool.pending() == 1  # only the blocker is still "in flight"

        gate.set()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and pool.pending() > 0:
            time.sleep(0.01)
        assert pool.pending() == 0
    finally:
        pool.shutdown()


def test_shutdown_unblocks_workers() -> None:
    pool = WorkerPool(num_workers=3, name="test")
    pool.submit(0, "task", lambda: None)
    time.sleep(0.05)
    pool.shutdown(wait=True, timeout=2.0)
    # All threads should have exited. If they're still alive after shutdown,
    # something is stuck.
    for t in pool._threads:
        assert not t.is_alive()


def test_submit_after_shutdown_returns_false() -> None:
    pool = WorkerPool(num_workers=1, name="test")
    pool.shutdown()
    assert pool.submit(0, "x", lambda: None) is False
