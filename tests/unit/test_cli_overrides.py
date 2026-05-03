"""Tests for CLI override precedence — `apply_cli_overrides` and the
argparse mutual exclusion of `--no-pbo` / `--force-pbo`.

The precedence rule from spec §3 is:

    explicit CLI flag  >  auto-tune  >  hardcoded fallback

Each test pins one branch of that rule. The mutex test exercises
the argparse parser directly (built via the helper extracted in
slice 2) to make sure a future re-organisation of the parser can't
silently lose the mutex.
"""

from __future__ import annotations

import pytest

from img_player.__main__ import _build_parser, _resolve_tune
from img_player.perf.hardware import (
    PerformanceTune,
    apply_cli_overrides,
)

# A representative auto-tuned starting point: laptop with discrete GPU.
# Most tests build on this so the deltas read clearly.
_BASE_DGPU = PerformanceTune(
    num_workers=8,
    cache_gb=6.1,
    oiio_threads=4,
    use_pbo=True,
)
_BASE_IGPU = PerformanceTune(
    num_workers=8,
    cache_gb=6.1,
    oiio_threads=1,
    use_pbo=False,
)


# ============================================================================
# apply_cli_overrides — the field-level precedence
# ============================================================================


def test_no_flags_returns_tune_unchanged() -> None:
    """If the user passed nothing, the auto-tune is returned as-is."""
    out = apply_cli_overrides(_BASE_DGPU)
    assert out == _BASE_DGPU


def test_explicit_workers_overrides_autotune() -> None:
    out = apply_cli_overrides(_BASE_DGPU, num_workers=12)
    assert out.num_workers == 12
    # Other fields untouched.
    assert out.cache_gb == _BASE_DGPU.cache_gb
    assert out.oiio_threads == _BASE_DGPU.oiio_threads
    assert out.use_pbo == _BASE_DGPU.use_pbo


def test_explicit_cache_gb_overrides_autotune() -> None:
    out = apply_cli_overrides(_BASE_DGPU, cache_gb=4.0)
    assert out.cache_gb == 4.0
    assert out.num_workers == _BASE_DGPU.num_workers


def test_explicit_oiio_threads_overrides_autotune() -> None:
    out = apply_cli_overrides(_BASE_DGPU, oiio_threads=2)
    assert out.oiio_threads == 2


def test_zero_oiio_threads_is_a_valid_override() -> None:
    """`0` is a meaningful value (= disable OIIO threading) — must not
    be confused with the "user passed nothing" None sentinel."""
    out = apply_cli_overrides(_BASE_DGPU, oiio_threads=0)
    assert out.oiio_threads == 0


def test_no_pbo_disables_pbo_even_on_discrete() -> None:
    """The user asked to force the sync path — auto-tune said dGPU
    means PBO, but the explicit flag wins."""
    out = apply_cli_overrides(_BASE_DGPU, no_pbo=True)
    assert out.use_pbo is False


def test_force_pbo_enables_pbo_on_integrated() -> None:
    """Symmetric: force PBO on an iGPU input. Useful for power-user
    debugging when we want to see how the PBO path behaves on shared
    memory hardware."""
    out = apply_cli_overrides(_BASE_IGPU, force_pbo=True)
    assert out.use_pbo is True


def test_no_pbo_and_force_pbo_raises_value_error() -> None:
    """Defensive guard: argparse normally catches this, but if a
    future programmatic caller bypasses argparse we still refuse."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        apply_cli_overrides(_BASE_DGPU, no_pbo=True, force_pbo=True)


def test_multiple_overrides_combine() -> None:
    out = apply_cli_overrides(
        _BASE_DGPU,
        num_workers=10,
        cache_gb=20.0,
        oiio_threads=2,
        no_pbo=True,
    )
    assert out == PerformanceTune(
        num_workers=10,
        cache_gb=20.0,
        oiio_threads=2,
        use_pbo=False,
    )


# ============================================================================
# argparse mutex on --no-pbo / --force-pbo
# ============================================================================


def test_argparse_no_pbo_and_force_pbo_is_systemexit() -> None:
    """The argparse-level mutual exclusion fires before our function
    is even called. We test it via the parser directly so a future
    re-org of the flags can't drop the mutex without us noticing."""
    parser = _build_parser()
    # argparse exits with code 2 on argument errors.
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--no-pbo", "--force-pbo", "/tmp/path"])
    assert exc_info.value.code == 2


def test_argparse_no_pbo_alone_parses_cleanly() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--no-pbo"])
    assert args.no_pbo is True
    assert args.force_pbo is False


def test_argparse_force_pbo_alone_parses_cleanly() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--force-pbo"])
    assert args.force_pbo is True
    assert args.no_pbo is False


def test_argparse_neither_flag_defaults_false() -> None:
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.no_pbo is False
    assert args.force_pbo is False


# ============================================================================
# Integration: _resolve_tune ties it all together
# ============================================================================


def test_resolve_tune_no_flags_yields_unknown_safe_defaults(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Without any flags, the resolver builds a tune for the current
    machine. Since the GL context isn't alive at this stage, gpu_kind
    is "unknown", which means oiio_threads=1 and use_pbo=False — the
    same conservative defaults today's hardcoded path produces."""
    parser = _build_parser()
    args = parser.parse_args([])
    caplog.set_level("INFO", logger="img_player.perf.hardware")

    _after_cli, tune = _resolve_tune(args)

    assert tune.oiio_threads == 1
    assert tune.use_pbo is False
    # workers and cache_gb are CPU/RAM-dependent — pin the safe minimums
    # so the test runs on any machine.
    assert tune.num_workers >= 2
    assert tune.cache_gb >= 2.0
    # The [hw-tune] log lines were emitted.
    log_text = "\n".join(caplog.messages)
    assert "[hw-tune]" in log_text


def test_resolve_tune_workers_override_propagates() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--workers", "4"])
    _after_cli, tune = _resolve_tune(args)
    assert tune.num_workers == 4


def test_resolve_tune_no_pbo_override_propagates() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--no-pbo"])
    _after_cli, tune = _resolve_tune(args)
    assert tune.use_pbo is False


def test_resolve_tune_force_pbo_override_propagates() -> None:
    """Even though gpu_kind="unknown" gives use_pbo=False from the
    auto-tune, --force-pbo must override that to True."""
    parser = _build_parser()
    args = parser.parse_args(["--force-pbo"])
    _after_cli, tune = _resolve_tune(args)
    assert tune.use_pbo is True
