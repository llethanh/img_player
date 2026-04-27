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
    args = parser.parse_args()

    bytes_per_float32 = 4
    n_elements = int(args.gb * 1024**3 // bytes_per_float32)

    print(f"[ram_eater] allocating {args.gb} GB ({n_elements:,} float32 elements)...")
    buf = np.zeros(n_elements, dtype=np.float32)
    print(f"[ram_eater] allocated. id={id(buf)}. holding for {args.seconds}s...")
    print("[ram_eater] (Ctrl-C to release early)")

    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        print("[ram_eater] interrupted, releasing.")
    finally:
        del buf
        print("[ram_eater] done.")


if __name__ == "__main__":
    main()
