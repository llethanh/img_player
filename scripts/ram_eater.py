"""Simulate memory pressure for the bench D scenario.

Allocates ``--gb`` gigabytes of RAM (default 6) into a single
NumPy float32 array and sleeps until killed (Ctrl-C or process
termination). Used to reproduce the "user has Nuke open" condition
when validating the boot-time health check (spec §6 / bench D).

Intended use::

    # Terminal A — eat 6 GB of RAM:
    python scripts/ram_eater.py --gb 6

    # Terminal B — bench img_player with the eater running:
    python -m img_player --benchmark \\
        --bench-output perf/postopti_memory_pressure.json \\
        <SEQUENCE_PATH>

The bench should show the [hw-tune] reduced cache log line and no
swap delta during playback. If swap grows, the constraint isn't
firing and slice 3 has a bug.

Note: ``np.zeros`` is used over ``np.empty`` to ensure the OS
actually commits the pages — ``np.empty`` would only reserve
virtual address space without touching physical RAM.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gb",
        type=float,
        default=6.0,
        help="GB of RAM to allocate (default: 6).",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=600.0,
        help="How long to hold the allocation (default: 600 = 10 min).",
    )
    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=256,
        help=(
            "Allocate in chunks of this many MB (default: 256). Smaller "
            "chunks tolerate fragmented memory; large monolithic allocs "
            "can fail even when total free RAM is enough."
        ),
    )
    args = parser.parse_args()
    print(f"[ram_eater] target = {args.gb} GB, chunk = {args.chunk_mb} MB", flush=True)

    chunk_bytes = args.chunk_mb * 1024 * 1024
    n_per_chunk = chunk_bytes // 4  # float32 = 4 bytes
    total_bytes = int(args.gb * 1024**3)
    n_chunks = max(1, total_bytes // chunk_bytes)

    bufs: list[np.ndarray] = []
    try:
        for i in range(n_chunks):
            bufs.append(np.zeros(n_per_chunk, dtype=np.float32))
            allocated_gb = (i + 1) * chunk_bytes / 1024**3
            print(
                f"[ram_eater] chunk {i+1}/{n_chunks} done, total {allocated_gb:.2f} GB",
                flush=True,
            )
    except MemoryError as err:
        print(
            f"[ram_eater] hit MemoryError at chunk {len(bufs)+1}: {err}. "
            f"Holding {len(bufs) * args.chunk_mb / 1024:.2f} GB anyway.",
            flush=True,
        )

    print(
        f"[ram_eater] holding {len(bufs)} chunks for {args.seconds}s — Ctrl-C to release",
        flush=True,
    )
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        print("[ram_eater] interrupted, releasing.", flush=True)
    finally:
        del bufs
        print("[ram_eater] done.", flush=True)


if __name__ == "__main__":
    main()
