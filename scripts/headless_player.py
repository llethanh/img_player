"""Headless smoke test for cache + controller (no UI).

Loads a sequence, plays it, and prints frame numbers + cache stats to stdout.

Usage:
    python scripts/headless_player.py <path-to-sequence-dir-or-file> [num-frames]
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QTimer

from img_player.cache.frame_cache import FrameCache
from img_player.player.controller import PlayerController
from img_player.sequence.scanner import scan


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if len(sys.argv) < 2:
        print("usage: python scripts/headless_player.py <path> [num-frames]", file=sys.stderr)
        return 2

    target = Path(sys.argv[1])
    num_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    seq = scan(target)
    print(f"Loaded {seq.display_pattern()} ({seq.frame_count} frames)")

    app = QCoreApplication(sys.argv)
    cache = FrameCache(budget_bytes=512 * 1024 * 1024, num_workers=4)
    controller = PlayerController(cache)

    frames_shown: list[int] = []

    def on_frame(frame: int) -> None:
        arr = cache.get(frame)
        status = "HIT " if arr is not None else "MISS"
        frames_shown.append(frame)
        print(f"[{status}] frame {frame:4d}  ({len(frames_shown)}/{num_frames})")

    controller.frame_changed.connect(on_frame)
    controller.load_sequence(seq)

    # wait a bit for initial prefetch so we don't drop every frame
    time.sleep(0.5)
    controller.play()

    def stop_after() -> None:
        controller.pause()
        app.quit()

    # stop after num_frames ticks
    def maybe_stop() -> None:
        if len(frames_shown) >= num_frames:
            stop_after()

    # check periodically
    check_timer = QTimer()
    check_timer.setInterval(50)
    check_timer.timeout.connect(maybe_stop)
    check_timer.start()

    try:
        app.exec()
    finally:
        controller.shutdown()
        stats = cache.stats()
        cache.shutdown()

        print("\n=== cache stats ===")
        print(f"  hits          : {stats.hits}")
        print(f"  misses        : {stats.misses}")
        print(f"  evictions     : {stats.evictions}")
        print(f"  decode errors : {stats.decode_errors}")
        print(f"  frames cached : {stats.frames_cached}")
        print(
            f"  bytes used    : {stats.bytes_used / 1024**2:.1f} MB of {stats.bytes_budget / 1024**2:.1f} MB"
        )
        print(f"  dropped       : {controller.state.dropped_frames}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
