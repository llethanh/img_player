"""Tests for ``img_player.perf.runtime_state`` — the boot-time
memory-pressure check that prevents img_player from reserving a
cache that would force the OS to swap.

Two groups:

* :func:`apply_runtime_constraints` — pure logic, no psutil mocking.
  Covers the no-op (ample RAM), shrink (tight RAM), floor clamp
  (very tight RAM), and the "other fields stay untouched"
  invariant.
* :class:`RuntimeState.snapshot` — the only side-effecting bit;
  monkey-patches ``psutil.virtual_memory`` and ``psutil.swap_memory``
  to exercise both the happy path and the fallback when psutil
  itself is broken.
"""

from __future__ import annotations

import builtins

import pytest

from img_player.perf.hardware import PerformanceTune
from img_player.perf.runtime_state import (
    RuntimeState,
    apply_runtime_constraints,
)

# A representative starting point that mirrors what `compute_tune`
# would yield on the laptop dGPU (16 threads, 16 GB RAM, RTX 5070).
_BASE_TUNE = PerformanceTune(
    num_workers=8,
    cache_gb=6.1,
    oiio_threads=4,
    use_pbo=True,
)


# ============================================================================
# apply_runtime_constraints — pure logic
# ============================================================================


def test_ample_ram_is_a_no_op() -> None:
    """If 60 % of available RAM still exceeds the requested cache,
    the tune is returned unchanged."""
    state = RuntimeState(available_ram_gb=20.0, swap_used_gb=0.0)
    out = apply_runtime_constraints(_BASE_TUNE, state)
    assert out == _BASE_TUNE


def test_tight_ram_shrinks_cache() -> None:
    """When only 4 GB are free, the cache is clamped to 60 % of that
    (= 2.4 GB), which is below the 6.1 GB the tune asked for."""
    state = RuntimeState(available_ram_gb=4.0, swap_used_gb=0.0)
    out = apply_runtime_constraints(_BASE_TUNE, state)
    # 4.0 * 0.6 = 2.4
    assert out.cache_gb == pytest.approx(2.4, abs=0.01)


def test_very_tight_ram_clamped_to_floor() -> None:
    """If the safe budget falls below 2 GB the function clamps to
    the floor — never below. A 2 GB cache barely holds two 4K UHD
    frames but that's the smallest we'll ever ship."""
    state = RuntimeState(available_ram_gb=1.0, swap_used_gb=0.0)
    out = apply_runtime_constraints(_BASE_TUNE, state)
    assert out.cache_gb == 2.0


def test_other_fields_are_untouched() -> None:
    """Memory pressure only affects the cache budget. Workers,
    OIIO threads and use_pbo are unrelated and must stay as-is."""
    state = RuntimeState(available_ram_gb=2.0, swap_used_gb=0.0)
    out = apply_runtime_constraints(_BASE_TUNE, state)
    assert out.num_workers == _BASE_TUNE.num_workers
    assert out.oiio_threads == _BASE_TUNE.oiio_threads
    assert out.use_pbo == _BASE_TUNE.use_pbo


def test_returns_new_instance_input_not_mutated() -> None:
    """`PerformanceTune` is frozen, but the test pins the contract
    explicitly: input is unchanged, output is a fresh instance."""
    state = RuntimeState(available_ram_gb=4.0, swap_used_gb=0.0)
    out = apply_runtime_constraints(_BASE_TUNE, state)
    assert _BASE_TUNE.cache_gb == 6.1
    assert out is not _BASE_TUNE


def test_exactly_at_threshold_is_no_op() -> None:
    """Edge case: ``available * 0.6 == cache_gb`` — keep, don't shrink.
    Fence-post check on the >= comparison."""
    # 6.1 / 0.6 = 10.166...  → at exactly 10.167 GB available, the
    # safe budget is 6.1, equal to the requested. Should be no-op.
    state = RuntimeState(available_ram_gb=6.1 / 0.6, swap_used_gb=0.0)
    out = apply_runtime_constraints(_BASE_TUNE, state)
    assert out == _BASE_TUNE


# ============================================================================
# RuntimeState.snapshot — the only side-effecting path
# ============================================================================


def test_snapshot_reads_from_psutil(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: snapshot returns the values psutil reports,
    converted to gigabytes."""

    class _FakeVM:
        available = 8 * 1024**3  # 8 GB

    class _FakeSwap:
        used = 256 * 1024**2  # 0.25 GB

    import psutil

    monkeypatch.setattr(psutil, "virtual_memory", lambda: _FakeVM())
    monkeypatch.setattr(psutil, "swap_memory", lambda: _FakeSwap())

    state = RuntimeState.snapshot()
    assert state.available_ram_gb == pytest.approx(8.0, abs=0.01)
    assert state.swap_used_gb == pytest.approx(0.25, abs=0.01)


def test_snapshot_falls_back_optimistically_when_psutil_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``import psutil`` raises, snapshot returns optimistic
    values — never wants to *cause* a cache shrink because the
    detector itself is broken. The auto-tune ceiling on cache_gb
    (64 GB hard cap) still applies, so we won't allocate something
    insane."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "psutil":
            raise ImportError("psutil unavailable in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    state = RuntimeState.snapshot()
    # Optimistic: very large available RAM, no swap.
    assert state.available_ram_gb >= 1024.0
    assert state.swap_used_gb == 0.0


def test_snapshot_falls_back_when_psutil_raises_at_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`psutil.virtual_memory()` itself can raise (rare but possible
    inside containers / sandboxes). The snapshot must still return
    a usable value, not crash the boot."""
    import psutil

    def boom() -> None:
        raise OSError("denied by sandbox")

    monkeypatch.setattr(psutil, "virtual_memory", boom)

    state = RuntimeState.snapshot()
    assert state.available_ram_gb > 0


def test_snapshot_optimistic_fallback_does_not_shrink_cache() -> None:
    """Property test: combining the optimistic fallback with the
    constraint function must yield a no-op. This is the load-bearing
    safety: a broken psutil cannot punish a healthy machine."""
    optimistic = RuntimeState(available_ram_gb=1024.0, swap_used_gb=0.0)
    out = apply_runtime_constraints(_BASE_TUNE, optimistic)
    assert out == _BASE_TUNE


# ============================================================================
# Integration with _resolve_tune (slice 2 + slice 3 wiring)
# ============================================================================


def test_resolve_tune_logs_runtime_check_and_applied(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """End-to-end check that the new pipeline emits both the
    ``runtime check`` line (always) and the final ``applied`` line
    (always last, reflecting the post-runtime tune)."""
    from img_player.__main__ import _build_parser, _resolve_tune

    caplog.set_level("INFO")
    parser = _build_parser()
    args = parser.parse_args([])

    _resolve_tune(args)

    text = "\n".join(caplog.messages)
    assert "[hw-tune] runtime check:" in text
    assert "[hw-tune] applied:" in text


def test_resolve_tune_logs_applied_after_runtime_check(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The order matters: ``applied`` is the LAST tune-related line
    so a user grepping the log finds the truthful final value at
    the bottom, not the pre-runtime intermediate."""
    from img_player.__main__ import _build_parser, _resolve_tune

    caplog.set_level("INFO")
    parser = _build_parser()
    args = parser.parse_args([])

    _resolve_tune(args)

    runtime_idx = next(
        (i for i, m in enumerate(caplog.messages) if "[hw-tune] runtime check:" in m),
        -1,
    )
    applied_idx = next(
        (i for i, m in enumerate(caplog.messages) if "[hw-tune] applied:" in m),
        -1,
    )
    assert runtime_idx >= 0 and applied_idx >= 0
    assert applied_idx > runtime_idx, (
        f"'applied' must come after 'runtime check' (got {applied_idx} vs {runtime_idx})"
    )
