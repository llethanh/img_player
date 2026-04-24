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
        "--gui",
        action="store_true",
        help="Launch the Qt GUI.",
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="Optional file or directory to open (launches the GUI automatically).",
    )

    subparsers = parser.add_subparsers(dest="command")
    scan_parser = subparsers.add_parser(
        "scan", help="Detect sequences at PATH and print a summary."
    )
    scan_parser.add_argument("path", type=Path, help="File or directory to scan.")
    scan_parser.add_argument(
        "--all",
        action="store_true",
        help="List every sequence in the directory, not just the largest.",
    )

    args = parser.parse_args(argv)

    if args.command == "scan":
        return _cmd_scan(args.path, list_all=args.all)

    if args.gui or args.path is not None:
        from img_player.app import run_gui

        return run_gui(initial_path=args.path)

    print(
        f"img_player {__version__} — pass --gui to launch the window, "
        "a PATH to open one, or `scan PATH` for a CLI summary."
    )
    return 0


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
