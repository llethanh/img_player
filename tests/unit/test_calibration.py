"""Tests for ``perf.calibration`` — per-machine profile persistence (slice 6).

Three groups:

* :func:`hw_signature` — same machine yields the same digest, even
  on minor RAM jitter from psutil; different CPU / GPU / RAM yields
  different digests.
* JSON round-trip — write a profile, read it back, assert equality;
  malformed and missing files yield ``None`` rather than raising.
* :func:`apply_profile_to_tune` — pure logic, the integration point
  with the boot pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from img_player.perf.calibration import (
    apply_profile_to_tune,
    build_profile,
    hw_signature,
    load_profile,
    save_profile,
)
from img_player.perf.hardware import HardwareProfile, PerformanceTune

# Reference HW + tune used across the suite — mirrors the laptop
# dGPU profile from earlier slices so cases read clearly.
_HW = HardwareProfile(
    cpu_threads=16,
    total_ram_gb=15.3,
    gpu_renderer="NVIDIA GeForce RTX 5070 Laptop GPU",
    gpu_kind="discrete_nvidia",
)
_TUNE = PerformanceTune(num_workers=8, cache_gb=6.1, oiio_threads=1, use_pbo=True)


# ============================================================================
# hw_signature
# ============================================================================


class TestHwSignature:
    def test_same_hw_yields_same_digest(self) -> None:
        assert hw_signature(_HW) == hw_signature(_HW)

    def test_ram_jitter_within_a_gb_is_invisible(self) -> None:
        """psutil can report 15.27 GB on one boot, 15.31 GB on the
        next. The signature rounds to whole GB so this jitter doesn't
        falsely invalidate the profile."""
        hw_a = _HW
        hw_b = HardwareProfile(
            cpu_threads=16,
            total_ram_gb=15.49,  # rounds to 15
            gpu_renderer=_HW.gpu_renderer,
            gpu_kind=_HW.gpu_kind,
        )
        assert hw_signature(hw_a) == hw_signature(hw_b)

    def test_different_cpu_threads_changes_digest(self) -> None:
        hw_b = HardwareProfile(
            cpu_threads=32,
            total_ram_gb=_HW.total_ram_gb,
            gpu_renderer=_HW.gpu_renderer,
            gpu_kind=_HW.gpu_kind,
        )
        assert hw_signature(hw_b) != hw_signature(_HW)

    def test_different_gpu_does_not_change_digest(self) -> None:
        """The GPU is *deliberately* excluded from the signature so the
        same profile is reusable at boot (gpu=None) and post-late-bind
        (real GPU). See ``hw_signature`` docstring for the rationale."""
        hw_b = HardwareProfile(
            cpu_threads=_HW.cpu_threads,
            total_ram_gb=_HW.total_ram_gb,
            gpu_renderer="NVIDIA GeForce RTX 4090",
            gpu_kind="discrete_nvidia",
        )
        assert hw_signature(hw_b) == hw_signature(_HW)

    def test_none_renderer_matches_real_renderer(self) -> None:
        """**The whole point of slice 6.** Boot-time HW (renderer=None)
        must produce the same digest as the same machine post-late-bind
        (with real renderer). Otherwise the profile saved at shutdown
        is silently invisible at boot — making ``cache_gb`` and
        ``num_workers`` from the profile unusable."""
        hw_pre = HardwareProfile(
            cpu_threads=_HW.cpu_threads,
            total_ram_gb=_HW.total_ram_gb,
            gpu_renderer=None,
            gpu_kind="unknown",
        )
        assert hw_signature(hw_pre) == hw_signature(_HW)


# ============================================================================
# JSON round-trip
# ============================================================================


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "profile.json"
        profile = build_profile(_HW, _TUNE)
        save_profile(profile, path)
        out = load_profile(path)
        assert out is not None
        assert out.num_workers == _TUNE.num_workers
        assert out.cache_gb == pytest.approx(_TUNE.cache_gb)
        assert out.oiio_threads == _TUNE.oiio_threads
        assert out.use_pbo is _TUNE.use_pbo
        assert out.digest == hw_signature(_HW)

    def test_load_missing_file_returns_none(self, tmp_path: Path) -> None:
        out = load_profile(tmp_path / "does_not_exist.json")
        assert out is None

    def test_load_malformed_json_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "profile.json"
        path.write_text("{ this is not json", encoding="utf-8")
        assert load_profile(path) is None

    def test_load_wrong_schema_version_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "profile.json"
        path.write_text(
            json.dumps({"schema_version": 999, "cpu_threads": 16}),
            encoding="utf-8",
        )
        assert load_profile(path) is None

    def test_load_missing_field_returns_none(self, tmp_path: Path) -> None:
        """A future schema bump that adds a required field must not
        crash the loader on old files."""
        path = tmp_path / "profile.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "cpu_threads": 16,
                    # missing rest of the fields
                },
            ),
            encoding="utf-8",
        )
        assert load_profile(path) is None

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Profile lives under a deep cache dir — parents must be created."""
        path = tmp_path / "deep" / "nested" / "profile.json"
        save_profile(build_profile(_HW, _TUNE), path)
        assert path.exists()

    def test_save_is_atomic(self, tmp_path: Path) -> None:
        """A successful save leaves no .tmp file behind."""
        path = tmp_path / "profile.json"
        save_profile(build_profile(_HW, _TUNE), path)
        assert path.exists()
        assert not (tmp_path / "profile.json.tmp").exists()


# ============================================================================
# apply_profile_to_tune — boot-pipeline integration
# ============================================================================


class TestApplyProfileToTune:
    def test_none_profile_returns_tune_unchanged(self) -> None:
        out = apply_profile_to_tune(_TUNE, None, _HW)
        assert out == _TUNE

    def test_matching_profile_replaces_tune(self) -> None:
        """The whole point: profile values override the freshly-
        computed ones, because a previous session settled on them
        for this exact machine."""
        custom = PerformanceTune(num_workers=4, cache_gb=2.0, oiio_threads=2, use_pbo=False)
        profile = build_profile(_HW, custom)
        out = apply_profile_to_tune(_TUNE, profile, _HW)
        assert out == custom

    def test_mismatching_profile_is_ignored(self) -> None:
        """If the user moved their config to a new machine, the old
        profile must not corrupt the new auto-tune."""
        other_hw = HardwareProfile(
            cpu_threads=4,
            total_ram_gb=8.0,
            gpu_renderer="Intel UHD",
            gpu_kind="integrated_intel",
        )
        custom = PerformanceTune(num_workers=2, cache_gb=2.0, oiio_threads=1, use_pbo=False)
        profile = build_profile(other_hw, custom)
        out = apply_profile_to_tune(_TUNE, profile, _HW)
        # Tune unchanged because the profile is for a different machine.
        assert out == _TUNE
