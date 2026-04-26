# Hardware-adaptive playback performance — auto-tune + PBO async upload

*Spec — 2026-04-26 · author: img_player team · status: draft awaiting user review*

## Context

img_player v0.1.0 ships a working VFX-grade image-sequence player with
solid playback performance on the target workstation. Phase 1 perf work
(documented in `perf/BASELINE.md`) was conducted **on a laptop iGPU
(AMD Radeon 780M)** and concluded with `DEFAULT_OIIO_THREADS = 1` and
PBO async upload reverted because both *hurt* perf on integrated
graphics with unified memory.

A follow-up benchmark on the very same laptop, after re-routing
`python.exe` to the discrete **NVIDIA GeForce RTX 5070 Laptop GPU** via
Windows Graphics Settings, reveals a profile that is fundamentally
different from the iGPU one and demands a different optimisation
strategy:

| Metric | iGPU 780M | **RTX 5070 Laptop** | Delta |
|---|---|---|---|
| effective fps (tick) | 18.14 | **23.14** / 24 | +27 % |
| effective fps (paint) | (close to tick) | **17.73** | paint can't keep up with tick |
| upload mean (µs) | 31 560 | 24 252 | -23 % |
| upload p99 (µs) | 176 160 | 61 250 | -65 % |
| paint p99 (µs) | 184 220 | 62 950 | -66 % |
| decode mean (ms) | 1 793 | 1 138 | -37 % |
| cache hit rate | 61.5 % | 71.1 % | +9.6 pts |
| warmup 30 frames (s) | 8.1 | 4.3 | -47 % |

Two structural observations follow from this data:

1. **`effective fps tick (23.14) > effective fps paint (17.73)`** — the
   controller fires at 23 Hz but `paintGL()` body runs at 17.7 Hz
   because each paint exceeds the 41 ms budget. Qt coalesces paints.
   The synchronous `glTexSubImage2D` is the cause: 24 ms mean upload
   on a PCIe Gen4 dGPU is *5× more than physically necessary*. PBO
   async upload — wrong on iGPU — is the right answer here.
2. **Decode is still the secondary bottleneck**. At 1.14 s/frame with
   6 workers we sustain ~5 fps decode steady-state — the cache cannot
   keep up under cold-cache play. `oiio_threads = 1` is the wrong
   default on a 16-thread machine with a discrete GPU; on the iGPU it
   was right (memory-bus contention). The choice depends on hardware.

The user also confirmed img_player must run *on two very different
profiles*:

* the laptop **ASUS TUF A16** — 16 CPU threads, 16 GB RAM, RTX 5070
  Laptop dGPU + Radeon 780M iGPU (Optimus-routed);
* a workstation **HP Z2 Tower G9** — 32 CPU threads, **128 GB RAM**,
  NVIDIA workstation dGPU.

A fixed configuration cannot serve both. What's optimal on the
workstation (12 workers, 51 GB cache, 6 OIIO threads) is overkill on
the laptop (12 workers exhaust 16 threads); what's safe on the iGPU
(`use_pbo = False`) leaves the dGPU 5× short of its capacity.

This spec captures a **hardware-adaptive** approach: detect the
runtime profile at startup, dimension worker pool / cache / OIIO
threads / PBO usage accordingly, and let CLI flags override anything
the user wants to pin manually.

## Goals

1. **Auto-tune `num_workers`, `cache_gb`, `oiio_threads`, `use_pbo`**
   from `(cpu_threads, total_ram, gpu_kind)` detected at startup, so
   the same binary delivers near-optimal defaults on the laptop iGPU,
   the laptop dGPU, and the workstation without manual tuning.
2. **Reach 24 fps stable on RTX 5070 Laptop** on the reference
   sequence (`SH0010_Rendered_RGB`, 4K UHD multichannel EXR) — the
   gap to 24 today is 0.86 fps and is dominated by `paintGL` overrun
   on synchronous upload.
3. **No regression on integrated GPU** — running the same code on the
   780M iGPU must stay within ±5 % of the previous Phase 1 numbers
   (`DEFAULT_OIIO_THREADS = 1`, no PBO).
4. **Honest measurement under PBO** — paint / upload timings must
   distinguish "main-thread blocking time" (what costs us fps) from
   "DMA wall time" (out-of-band on dGPU). Without this the bench
   would *look* faster while hiding stalls.
5. **Backwards-compatible CLI** — every `--workers` / `--cache-gb` /
   `--oiio-threads` invocation in existing user scripts keeps the
   exact same meaning.

## Non-goals

Explicitly **out of scope** for this spec — these are larger items
that will get their own design later:

* **Channel preselection at decode** (skip non-RGBA AOVs in
  multichannel EXR). Tracked as a Phase 2 follow-up; the 68-channel
  test EXR makes the case obvious but the implementation touches
  `reader.py` and the channel-switch code path significantly.
* **Memory-pool numpy buffers**. Marginal p99 win, Phase 2.
* **Lazy shader recompile** on display-transform switch. Polish,
  Phase 2.
* **Disk-side proxy transcode** (EXR → DPX/JPEG-XR). Significant
  feature, separate spec entirely.
* **Render thread / GL context separation**. Architecturally
  desirable on dGPU but Qt + OpenGL multithreading is risky; Phase 3.
* **IPGraph (à la OpenRV)**. C++ rewrite territory, not in this spec.

## Architecture

### Component diagram

```
                              ┌─────────────────────────────┐
                              │  HardwareProfile (dataclass) │
                              │  ├── cpu_threads             │
                              │  ├── total_ram_gb            │
                              │  ├── gpu_renderer (raw)      │
                              │  └── gpu_kind                │
                              └──────────────┬──────────────┘
                                             │
                       ┌─────────────────────┼─────────────────────┐
                       │                     │                     │
              ┌────────▼─────────┐  ┌────────▼─────────┐  ┌───────▼────────┐
              │  classify_gpu()  │  │  PerformanceTune │  │  CLI overrides │
              │  GL_RENDERER →   │  │  (computed from  │  │  --workers     │
              │  gpu_kind        │  │  HW + overrides) │  │  --cache-gb    │
              └──────────────────┘  └──────────────────┘  │  --oiio-threads│
                                             │            │  --no-pbo      │
                                             │            │  --force-pbo   │
                                             │            └────────────────┘
                ┌────────────────────────────┼────────────────────────────┐
                │                            │                            │
       ┌────────▼─────────┐         ┌────────▼─────────┐         ┌───────▼─────────┐
       │  FrameCache      │         │  PlayerController │         │  GL Viewport     │
       │  budget = cache  │         │  (no change)      │         │  if use_pbo:     │
       │  workers = N     │         │                   │         │    ring of 3 PBOs│
       │  oiio thr at init│         │                   │         │  else:           │
       └──────────────────┘         └───────────────────┘         │    sync path     │
                                                                  └──────────────────┘
```

### Module boundaries

| Module | Responsibility |
|---|---|
| `img_player/perf/hardware.py` (new) | `HardwareProfile`, `classify_gpu`, `PerformanceTune`, the heuristics |
| `img_player/render/gl_viewport.py` (modified) | Detects `gpu_renderer` at first `initializeGL()`, exposes it via signal/property; PBO ring path gated on `use_pbo` |
| `img_player/app.py` (modified) | At boot: build `HardwareProfile`, compute `PerformanceTune` (with CLI overrides), pass values to `FrameCache` and `gl_viewport` |
| `img_player/__main__.py` (modified) | Add `--no-pbo` / `--force-pbo` flags (mutually exclusive) |

The `perf/` module is pure logic with **zero Qt / OIIO / GL imports** —
it only takes inputs and returns the computed `PerformanceTune`. This
keeps it trivially unit-testable.

## Data flow

### Startup sequence

```
1. argparse runs                          → argv with optional overrides
2. QApplication boot
3. MainWindow / GLViewport.initializeGL   → glGetString(GL_RENDERER) captured
                                            → emit gpu_renderer_detected
4. app.py receives renderer:
   - build HardwareProfile
   - compute PerformanceTune (with overrides applied)
   - log the resolved values
5. FrameCache(budget=cache_gb, workers=num_workers) created
6. oiio.attribute("threads", oiio_threads)
7. gl_viewport switches path:
   - if use_pbo and gpu_kind.startswith("discrete"): allocate PBO ring
   - else: keep synchronous path (today's behaviour)
8. load_sequence(...) proceeds normally
```

This means **the very first paintGL is always on the sync path** — we
can't allocate PBOs before we know the renderer. From the second paint
onwards, the chosen path is stable for the session.

### Per-frame paintGL (PBO path)

```
on paintGL():
    pbo_idx = (last_pbo_idx + 1) % 3

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbos[pbo_idx])
    glBufferData(..., nbytes, NULL, GL_STREAM_DRAW)        # orphan
    ptr = glMapBufferRange(..., MAP_WRITE | INVALIDATE | UNSYNCHRONIZED)
    ctypes.memmove(ptr, frame_pixels.ctypes.data, nbytes)
    glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER)

    glBindTexture(GL_TEXTURE_2D, image_tex)
    t0 = perf_counter()
    glTexSubImage2D(..., None)                              # async DMA dispatched
    upload_cpu_us = (perf_counter() - t0) * 1e6             # ~few hundred µs

    fence = glFenceSync(GPU_COMMANDS_COMPLETE, 0)            # for upload_gpu_us

    # ... bind shader, draw fullscreen quad, etc ...
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    last_pbo_idx = pbo_idx
```

`upload_gpu_us` is read **at the next paint** via
`glClientWaitSync(prev_fence, 0, 0)` non-blocking. If the fence is
already signalled the elapsed time is recorded, otherwise we mark the
sample as "pending" and try again the paint after.

### Per-frame paintGL (sync path — unchanged)

```
on paintGL():
    glBindTexture(GL_TEXTURE_2D, image_tex)
    t0 = perf_counter()
    glTexSubImage2D(..., frame_pixels.ctypes.data)          # blocks
    upload_cpu_us = (perf_counter() - t0) * 1e6
    upload_gpu_us = upload_cpu_us                           # same on sync path
    # ... draw ...
```

## Detailed designs

### 1. `HardwareProfile` and GPU classification

```python
@dataclass(frozen=True)
class HardwareProfile:
    cpu_threads: int
    total_ram_gb: float
    gpu_renderer: str          # raw GL_RENDERER for logs
    gpu_kind: GpuKind          # classified

GpuKind = Literal[
    "discrete_nvidia", "discrete_amd",
    "integrated_amd", "integrated_intel",
    "unknown",
]

def classify_gpu(renderer: str) -> GpuKind:
    r = renderer.lower()
    if any(tok in r for tok in ("geforce", "rtx", "quadro", "tesla")):
        return "discrete_nvidia"
    if any(tok in r for tok in ("radeon pro", "radeon rx", "fire")):
        return "discrete_amd"
    if "radeon" in r and any(tok in r for tok in ("780m", "vega", "graphics")):
        return "integrated_amd"
    if "intel" in r and any(tok in r for tok in ("hd graphics", "iris", "uhd", "arc")):
        return "integrated_intel"
    return "unknown"
```

Fallbacks:
* `glGetString` failure → `gpu_kind = "unknown"`, all heuristics
  treat it as "safe" (= today's hardcoded values, no PBO).
* `psutil` failure → `total_ram_gb = cpu_threads * 4.0` (pessimistic
  but always non-zero).
* `cpu_count()` returning `None` → fallback `8`.

### 2. `PerformanceTune` heuristics

```python
@dataclass(frozen=True)
class PerformanceTune:
    num_workers: int
    cache_gb: float
    oiio_threads: int
    use_pbo: bool

def compute_tune(hw: HardwareProfile) -> PerformanceTune:
    # Workers: half the CPU, capped to avoid lock contention on the cache.
    num_workers = max(2, min(hw.cpu_threads // 2, 12))

    # Cache: 40 % of total RAM, clamped to a sensible range.
    cache_gb = max(2.0, min(hw.total_ram_gb * 0.4, 64.0))

    # OIIO threads: 1 on integrated (memory-bus contention), scale on discrete.
    if hw.gpu_kind.startswith("integrated") or hw.gpu_kind == "unknown":
        oiio_threads = 1
    else:
        oiio_threads = min(max(hw.cpu_threads // 4, 2), 6)

    # PBO async: only on discrete; iGPU was measured slower with PBO.
    use_pbo = hw.gpu_kind.startswith("discrete")

    return PerformanceTune(num_workers, cache_gb, oiio_threads, use_pbo)
```

Concrete output for the three reference setups:

| Setup | num_workers | cache_gb | oiio_threads | use_pbo |
|---|---|---|---|---|
| Laptop ASUS iGPU 780M | 8 | 6.1 | 1 | False |
| Laptop ASUS RTX 5070 | 8 | 6.1 | 4 | True |
| Workstation HP Z2 dGPU | 12 | 51.2 | 6 | True |

Justification of bounds:
* `workers ≤ 12` — beyond that, `_lock` contention in `FrameCache`
  dominates (existing code is RLock-guarded around dict + LRU heap).
* `cache_gb ≤ 64` — LRU eviction latency rises with heap size; over
  64 GB also exceeds practical needs (5+ hours of 4K UHD float16).
* `oiio_threads ≤ 6` — `BASELINE.md` already showed 16 threads
  saturating the bus; 6 is a safe ceiling on current
  generation hardware. Bumpable on proof.
* `cache_gb × 0.4` (vs `× 0.5`) — VFX users routinely have
  Nuke / DaVinci / Blender open simultaneously; 40 % leaves headroom.

### 3. CLI overrides

Order of precedence: `CLI flag (explicit) > auto-tune > hardcoded fallback`.

The existing `argparse` defaults stay at `None` so we can detect
"user passed nothing" vs "user passed a value":

```python
parser.add_argument("--workers",      type=int,   default=None)
parser.add_argument("--cache-gb",     type=float, default=None)
parser.add_argument("--oiio-threads", type=int,   default=None)

pbo = parser.add_mutually_exclusive_group()
pbo.add_argument("--no-pbo",    action="store_true",
                 help="Force synchronous upload path (overrides auto-detect).")
pbo.add_argument("--force-pbo", action="store_true",
                 help="Force PBO async path even on integrated GPU (debug).")
```

After computing the auto-tune values, each one is overridden if
the corresponding flag is non-`None`. The `--no-pbo` / `--force-pbo`
pair is wired into the `use_pbo` field.

Logging at startup:

```
[hw-tune] cpu_threads=16, ram_total=15.3 GB, gpu_renderer="NVIDIA GeForce RTX 5070 Laptop GPU"
[hw-tune] gpu_kind=discrete_nvidia
[hw-tune] auto: num_workers=8, cache_gb=6.1, oiio_threads=4, use_pbo=True
[hw-tune] CLI overrides: cache_gb=4.0    (← only printed if any override applied)
[hw-tune] applied: num_workers=8, cache_gb=4.0, oiio_threads=4, use_pbo=True
```

### 4. PBO ring of 3 buffers

Three PBOs in rotation rather than ping-pong (2):
* if DMA latency > inter-paint interval (worst case spike), 2 PBOs
  stalls at the next remap; 3 buys headroom;
* VRAM cost: `3 × 64 MB = 192 MB` on an 8 GB card — negligible.

Allocation timing:
* On first `attach_to_sequence(...)` after we know the resolution,
  `glBufferData(GL_PIXEL_UNPACK_BUFFER, nbytes, None, GL_STREAM_DRAW)`
  is called on each PBO.
* On a sequence change with different resolution, the PBOs are
  `glBufferData`'d again with the new size. Cost ~1 ms, hidden in
  the `attach()` latency.

The map flags must include `GL_MAP_UNSYNCHRONIZED_BIT` — the
ring guarantees the PBO we're about to map is no longer in flight,
so we can safely tell the driver not to insert a sync point. (The
previous experiment in `PBO_NOTES.md` used only `INVALIDATE_BUFFER_BIT`
without `UNSYNCHRONIZED_BIT`, which keeps the driver guarding against
the previous DMA — re-introducing a stall.)

Fallback: any exception from the PBO path (e.g. `glMapBufferRange`
returning NULL on a surprising driver) logs a warning, sets
`use_pbo = False` for the rest of the session, and the next paint
takes the synchronous path. We don't retry.

### 5. Honest timing under PBO

Two metrics:
* **`upload_cpu_us`** — wall-clock spent on the main thread, measured
  from before `glBindBuffer(GL_PIXEL_UNPACK_BUFFER, ...)` to after
  `glTexSubImage2D(...)` returns. Captures the cost the main thread
  actually pays: orphan + map + memcpy + unmap + dispatch. This is
  what drives effective fps. On PBO with a 63 MB float16 4K UHD
  frame, expected ~3-7 ms (the memcpy at ~15 GB/s dominates). On
  the sync path it's ~24 ms today on RTX 5070 Laptop because the
  whole DMA also happens here.
* **`upload_gpu_us`** — wall-clock of the DMA itself, out-of-band
  on the dGPU. Implemented as: record `t_dispatch = perf_counter()`
  immediately after `glFenceSync(...)`; on the *next* paint, before
  doing anything else, call `glClientWaitSync(prev_fence, 0, 0)`
  with a 0 timeout. If the return value is `GL_ALREADY_SIGNALED` or
  `GL_CONDITION_SATISFIED`, we record `perf_counter() - t_dispatch`
  as the upper-bound estimate (true completion happened somewhere
  between dispatch and now). If `GL_TIMEOUT_EXPIRED`, the sample is
  flagged as "still pending" — this means the GPU genuinely couldn't
  finish the DMA before our next paint, which is a real warning sign
  on a discrete GPU. Diagnostic-only metric, not used for the pass
  criteria.

The existing bench `bench/recorder.py` (which we already trust per
`BASELINE.md`) gets a new `upload_gpu_us` field; the JSON schema is
versioned via a `bench_format_version` key so older reports remain
readable.

## Testing strategy

### Unit tests (no GPU required, run in CI)

| File | Cases |
|---|---|
| `tests/unit/test_hw_profile.py` | `classify_gpu` table (12-15 GL_RENDERER strings → kind), `compute_tune` clamps on extreme profiles, integrated → no PBO + 1 OIIO, discrete → PBO + scaled OIIO |
| `tests/unit/test_cli_overrides.py` | No flags = auto-tune used, explicit flag wins, `--no-pbo` disables PBO, `--force-pbo` enables on integrated, `--no-pbo` + `--force-pbo` is `argparse` error |
| `tests/unit/test_pbo_ring.py` | Index advances modulo 3, resolution change re-allocates, exception on map sets `use_pbo=False` for the session (mocked GL) |

### Integration smoke (skipped in headless CI)

`tests/integration/test_gl_smoke.py` — guarded by
`pytest.mark.skipif(no_display)`:

* `QOpenGLWidget` initialised with `use_pbo=True`, first paint succeeds, PBOs allocated.
* Same with `use_pbo=False`, first paint succeeds, no PBOs allocated.

### Bench validation (manual, before merge)

Three benchmarks, same parameters as `baseline_rtx5070.json`
(3 passes / warmup 30 / target 24 fps / `SH0010_Rendered_RGB`):

| Bench | Output file | Configuration | Pass criteria |
|---|---|---|---|
| **A** | `perf/postopti_rtx5070_autotune.json` | Auto-tune ON, **PBO OFF** | `decode_mean` < 800 ms (vs 1138), `effective fps tick` ≥ 23.5 |
| **B** | `perf/postopti_rtx5070_autotune_pbo.json` | Auto-tune **+ PBO ON** | `upload_cpu_mean` ≤ 8 ms (vs 24), `paint p99` ≤ 20 ms (vs 63), `effective fps paint` ≥ 23.5, no `upload_gpu_us` samples flagged as still-pending |
| **C** | `perf/postopti_igpu_autotune.json` | iGPU 780M (re-routed), `use_pbo` auto-detected as False | No regression vs iGPU baseline measured 2026-04-26 (±5 %) |

Bench C is the **non-regression gate**. Refactoring needed to
introduce the PBO path must not penalise the synchronous path.

### Deferred bench (when user is at the workstation)

`perf/postopti_workstation_autotune_pbo.json` — same protocol on
HP Z2 / dGPU. Validates the high-end profile (12 workers / 51 GB
cache / 6 OIIO threads).

## What this spec does not cover

* The **shape of the workstation auto-tune output** — coefficients
  (`× 0.4`, `// 4`, etc.) may need re-calibration once we have
  real bench data on the workstation. That tuning is part of the
  follow-up bench task, not of the initial implementation.
* **Multi-GPU machines** with two discrete GPUs — `classify_gpu`
  returns one value per GL context; we don't try to switch contexts
  at runtime.
* **Linux / macOS specifics**. `psutil` is cross-platform and
  `GL_RENDERER` works everywhere, but we haven't validated on
  non-Windows platforms — flagged as a follow-up.

## Open questions

* **Does the PBO path need a separate code path for `GL_HALF_FLOAT`
  on certain drivers?** Some older driver revisions reportedly stall
  on half-float upload via PBO. If we hit this on testing we'll add a
  driver-version probe and a per-driver opt-out — not in scope for
  the initial slice.
* **Should `upload_gpu_us` samples that stay "pending" (fence not
  signalled by next paint) bubble up as a warning?** A persistent
  pending state means the GPU genuinely can't keep up. For now we
  just count them and expose the count in the bench output.

---

*Once approved, this spec is handed to the writing-plans skill to
produce a slice-by-slice implementation plan.*
