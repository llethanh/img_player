"""Entry point for `python -m img_player` and the `img_player` console script."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from img_player import __version__
from img_player.perf import (
    PerformanceTune,
    RuntimeState,
    apply_cli_overrides,
    apply_profile_to_tune,
    apply_runtime_constraints,
    compute_tune,
    detect_hardware,
    load_profile,
    log_applied_tune,
    log_runtime_state,
    log_tune_resolution,
)

if TYPE_CHECKING:
    from img_player.sequence.models import SequenceInfo


# ----------------------------------------------------------------------------
# Argument parser — extracted so tests can exercise it without invoking main().
# ----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser used by ``main``.

    Extracted from ``main`` so tests (notably ``test_cli_overrides.py``)
    can drive it directly — particularly to verify the
    ``--no-pbo`` / ``--force-pbo`` mutual exclusion enforced by
    ``add_mutually_exclusive_group``.
    """
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
    # All performance flags default to None so the auto-tune layer can
    # tell "user passed nothing" from "user explicitly asked for X".
    parser.add_argument(
        "--cache-gb",
        type=float,
        default=None,
        help="RAM cache budget in GiB. Default: auto-tuned from total RAM (40 %%, clamped 2-64 GB).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of decode workers. Default: auto-tuned from CPU thread count (capped at 12).",
    )
    parser.add_argument(
        "--oiio-threads",
        type=int,
        default=None,
        help="Threads for OIIO's internal decode pool. Default: auto-tuned (1 on iGPU, 2-6 on dGPU).",
    )

    # PBO flags are mutually exclusive — argparse enforces this.
    pbo_group = parser.add_mutually_exclusive_group()
    pbo_group.add_argument(
        "--no-pbo",
        action="store_true",
        help="Force the synchronous upload path (overrides the dGPU auto-detect that would enable PBO).",
    )
    pbo_group.add_argument(
        "--force-pbo",
        action="store_true",
        help="Force the PBO async upload path even on integrated GPU. Mostly for debugging.",
    )

    # Calibration flags (slice 6). The profile.json under the user's
    # cache dir reuses the previously-applied tune across boots.
    # --skip-calibration ignores it entirely (debugging / CI).
    # --recalibrate deletes/ignores the existing one and forces a
    # fresh compute_tune for this session, which then becomes the
    # new persisted profile at shutdown.
    cal_group = parser.add_mutually_exclusive_group()
    cal_group.add_argument(
        "--skip-calibration",
        action="store_true",
        help="Don't read or write the per-machine profile.json. Useful for CI and debugging.",
    )
    cal_group.add_argument(
        "--recalibrate",
        action="store_true",
        help="Ignore the existing profile.json and force a fresh tune computation this session.",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run a timed playback of PATH and write a JSON report. Quits when done.",
    )
    parser.add_argument(
        "--passes",
        type=int,
        default=3,
        help="Number of full sequence loops to time during --benchmark (default: 3).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to prefetch before the bench timer starts (default: 30).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=24.0,
        help="Playback rate during --benchmark (default: 24.0).",
    )
    parser.add_argument(
        "--bench-output",
        type=Path,
        default=None,
        help="Path to write the JSON report (default: perf/bench_<timestamp>.json).",
    )
    parser.add_argument(
        "path",
        type=Path,
        nargs="?",
        default=None,
        help="File or directory. With no flag, launches the GUI on that sequence.",
    )
    return parser


def _resolve_tune(args: argparse.Namespace) -> PerformanceTune:
    """Run the auto-tune pipeline for these CLI args.

    At boot time the GL context isn't alive yet, so we pass
    ``gpu_renderer=None`` — that yields ``gpu_kind="unknown"`` and
    the conservative fallback heuristics (``oiio_threads=1``,
    ``use_pbo=False``). Slice 4 will re-run this same pipeline once
    the renderer is known, giving us the dGPU-tuned values.

    Precedence (highest wins last):

    1. ``compute_tune`` — heuristics from the static profile.
    2. ``apply_cli_overrides`` — user-passed flags (``--workers`` etc.).
    3. ``apply_runtime_constraints`` — refuses to swap, even if the
       user asked for a cache that doesn't fit. The user can close
       Nuke / DaVinci and retry; the app stays responsive in the
       meantime.

    Side effects: emits the ``[hw-tune]`` log lines that surface to
    the user (and to bug reports) what the resolver actually decided.
    """
    hw = detect_hardware(gpu_renderer=None)
    auto = compute_tune(hw)
    # Calibration profile (slice 6): if a previous session on this
    # exact hardware persisted its tune, replace the heuristic-
    # computed one with it BEFORE we apply CLI overrides. CLI flags
    # still win after this. With --skip-calibration or
    # --recalibrate, the profile is intentionally bypassed for this
    # boot; --recalibrate also means the freshly-computed tune
    # becomes the new persisted profile at shutdown.
    if not args.skip_calibration and not args.recalibrate:
        profile = load_profile()
        post_profile = apply_profile_to_tune(auto, profile, hw)
    else:
        post_profile = auto
    after_cli = apply_cli_overrides(
        post_profile,
        cache_gb=args.cache_gb,
        num_workers=args.workers,
        oiio_threads=args.oiio_threads,
        no_pbo=args.no_pbo,
        force_pbo=args.force_pbo,
    )
    log_tune_resolution(hw, auto, after_cli)

    state = RuntimeState.snapshot()
    final = apply_runtime_constraints(after_cli, state)
    log_runtime_state(state, after_cli, final)

    log_applied_tune(final)
    return final


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # `--scan` is a CLI-only path — no GUI, no cache, no need to auto-tune.
    if args.scan:
        if args.path is None:
            parser.error("--scan requires a PATH.")
        return _cmd_scan(args.path, list_all=args.all)

    # Configure root logger so our [hw-tune] lines actually appear.
    # Other modules call `logging.getLogger(__name__).info(...)` and
    # rely on a handler being installed. We use the same single-line
    # format the bench runner uses for consistency.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Auto-tune layer: the hard-coded defaults from earlier versions
    # (8 GB cache, 6 workers, 1 OIIO thread) are now produced as a
    # special case of compute_tune() when gpu_kind is "unknown" —
    # which it is at this stage, before the GL context is alive.
    tune = _resolve_tune(args)

    budget = int(tune.cache_gb * 1024**3)
    workers = tune.num_workers
    oiio_threads = tune.oiio_threads
    # tune.use_pbo is computed but not yet consumed — slice 4 wires
    # it into the GL viewport. We log it for the user's reference
    # via log_tune_resolution() above.

    if args.benchmark:
        if args.path is None:
            parser.error("--benchmark requires a PATH.")
        from img_player.bench.runner import run_benchmark

        return run_benchmark(
            args.path,
            passes=args.passes,
            warmup_frames=args.warmup_frames,
            target_fps=args.target_fps,
            output=args.bench_output,
            cache_budget_bytes=budget,
            num_workers=workers,
            oiio_threads=oiio_threads,
            cli_args=args,
        )

    # Default: launch the GUI (empty if no path, opening the given
    # sequence otherwise). Users can still drag & drop once the window
    # is open.
    from img_player.app import run_gui

    return run_gui(
        initial_path=args.path,
        cache_budget_bytes=budget,
        num_workers=workers,
        oiio_threads=oiio_threads,
        cli_args=args,
    )


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
