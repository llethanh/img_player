"""Entry point for `python -m img_player` and the `img_player` console script."""

from __future__ import annotations

import argparse
import sys

from img_player import __version__


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
        help="Launch the Qt GUI (smoke test at this stage).",
    )
    args = parser.parse_args(argv)

    if args.gui:
        from img_player.app import run_gui

        return run_gui()

    print(f"img_player {__version__} — CLI placeholder. Use --gui to launch the window.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
