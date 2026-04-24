"""Entry point for `python -m img_player` and the `img_player` console script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from img_player import __version__

if TYPE_CHECKING:
    from img_player.sequence.models import SequenceInfo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="img_player",
        description="VFX-grade image sequence player.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"img_player {__version__}",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Print a CLI summary of the sequence at PATH instead of launching the GUI.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="With --scan, list every sequence in the directory, not just the largest.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force the Qt GUI (implied when a PATH is given and --scan is not).",
    )
    parser.add_argument(
        "--cache-gb",
        type=float,
        default=None,
        help="RAM cache budget in GiB (default: 8). Bigger = more frames kept.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of decode workers (default: 6). Bump for heavy-disk loads.",
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="File or directory. With no flag, launches the GUI on that sequence.",
    )
    args = parser.parse_args(argv)

    if args.scan:
        if args.path is None:
            parser.error("--scan requires a PATH.")
        return _cmd_scan(args.path, list_all=args.all)

    # Default: launch the GUI (empty if no path, opening the given
    # sequence otherwise). Users can still drag & drop once the window
    # is open.
    from img_player.app import DEFAULT_CACHE_BUDGET_BYTES, DEFAULT_NUM_WORKERS, run_gui

    budget = (
        int(args.cache_gb * 1024**3) if args.cache_gb is not None else DEFAULT_CACHE_BUDGET_BYTES
    )
    workers = args.workers if args.workers is not None else DEFAULT_NUM_WORKERS
    return run_gui(initial_path=args.path, cache_budget_bytes=budget, num_workers=workers)


def _cmd_scan(path: Path, *, list_all: bool) -> int:
    from img_player.sequence.scanner import SequenceNotFoundError, scan, scan_all

    try:
        if list_all:
            sequences = scan_all(path)
            if not sequences:
                print(f"No sequences found in {path}.")
                return 1
            print(f"Found {len(sequences)} sequence(s) in {path}:")
            for seq in sequences:
                _print_sequence(seq, indent="  ")
        else:
            seq = scan(path)
            _print_sequence(seq)
    except SequenceNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1
    return 0


def _print_sequence(seq: SequenceInfo, indent: str = "") -> None:
    print(f"{indent}{seq.display_pattern()}")
    print(f"{indent}  directory  : {seq.directory}")
    print(f"{indent}  frames     : {seq.frame_count} ({seq.first_frame}..{seq.last_frame})")
    if not seq.is_contiguous:
        print(f"{indent}  missing    : {list(seq.missing_frames)}")
    if seq.width and seq.height:
        print(f"{indent}  resolution : {seq.width}x{seq.height}")
    if seq.channel_names:
        print(f"{indent}  channels   : {', '.join(seq.channel_names)}")


if __name__ == "__main__":
    sys.exit(main())
