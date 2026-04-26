# `perf/` — Performance benchmarks

This folder holds reproducible playback benchmarks for img_player. Each
report is a JSON dump (`bench_<timestamp>.json` or named via
`--bench-output`) that captures:

* **Tick stream** — controller QTimer firings: requested frame, cache hit
  flag, pending decodes.
* **Paint stream** — every `paintGL()` execution: upload + paint timings,
  resolution.
* **Decode stream** — every successful frame decode: duration, size.

Reports include hardware/OS metadata (`platform.system`, `release`,
`python`, `machine`) so older numbers stay self-describing.

## Running a benchmark

```bash
# Basic — 3 passes at 24 fps with 30 frames of warmup
python -m img_player --benchmark <PATH_TO_SEQUENCE>

# Tune for a heavier sequence (more warmup, longer measurement)
python -m img_player --benchmark \
  --passes 5 --warmup-frames 60 --target-fps 24 \
  --bench-output perf/optim_pbo.json \
  "C:\Users\lam\PERSO\images\SH0010_Rendered_RGB"
```

Flags:
* `--passes N`        — number of full sequence loops to time (default 3).
* `--warmup-frames N` — wait until N frames are cached before timing starts (default 30).
* `--target-fps F`    — playback rate the controller will request (default 24).
* `--bench-output P`  — JSON output path (default `perf/bench_<timestamp>.json`).

Cache budget and worker count are inherited from the regular flags
(`--cache-gb`, `--workers`).

## Interpreting the numbers

* **Effective FPS (tick)** vs **target FPS** — how close the play loop
  comes to the requested rate. If this is well below target, frames are
  arriving slower than the QTimer wants to display them.
* **Cache hit rate** — fraction of ticks where the requested frame was
  already in RAM. Below ~95% means the prefetcher isn't keeping up.
* **Upload (µs)** — wall-clock spent inside `glTexSubImage2D` /
  `glTexImage2D`. With a synchronous upload path this directly lengthens
  every paint. PBO async upload should reduce this dramatically.
* **Paint total (µs)** — the full body of `paintGL()`. If this is greater
  than 41 ms (24 fps budget), the GL widget itself is the bottleneck.
* **Decode (ms)** — per-frame decode wall-clock from a worker thread.
  Multiply by the number of workers in parallel to estimate sustainable
  throughput (e.g. 1.8 s × 6 workers ≈ 3.3 frames/sec).

## Files

* [`BASELINE.md`](BASELINE.md) — first measured baseline (April 2026)
  with detailed analysis and the optimisation roadmap that follows from
  it.
* `baseline.json` — machine-readable companion to BASELINE.md.
* `bench_<timestamp>.json` — ad-hoc runs.
