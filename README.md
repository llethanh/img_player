# img_player

VFX-grade image sequence player, built in Python with OCIO color management.

- 🎬 Plays image sequences (EXR multichannel, DPX, TIFF, PNG, JPG, TGA, ...)
- 🎨 Proper color management (sRGB / Rec709 / ACES) via OpenColorIO on GPU
- ⚡ Multi-threaded decode + RAM LRU cache for real-time playback
- 🔍 Built for VFX workflows (multichannel, exposure/gamma, channel isolation)

> **Status:** V1 in development. See [docs/specs/](docs/specs/) for the full design and [docs/plans/](docs/plans/) for the implementation plan.

## Install

**Prerequisites:**
- [Miniforge](https://github.com/conda-forge/miniforge) (conda + mamba, conda-forge channel)
- Git

**Create the environment:**

```bash
conda env create -f environment.yml
conda activate img_player
```

This installs OpenImageIO, OpenColorIO, PySide6, and all dev dependencies.

## Run

```bash
conda activate img_player
img_player
```

Or as a module:
```bash
python -m img_player
```

## Development

**Tests:**
```bash
pytest
```

**Lint & type:**
```bash
ruff check .
ruff format .
mypy src/img_player
```

## Features (V1 — planned)

- Play / pause / stop / scrub / loop / ping-pong / in-out
- Auto-detect sequences from drag & drop (`frame.####.exr`)
- OCIO input & display colorspace selection (sRGB, Rec709, ACES, ...)
- Channel / layer selection for EXR multichannel
- Exposure & gamma sliders
- Keyboard shortcuts (space, J/K/L, arrows, ...)
- 4K-capable architecture (HD-optimized cache defaults)

## Roadmap

- **V1** (current): core viewer (Level 2 features)
- **V2**: A/B compare, scopes, pixel inspector, playlists
- **V3**: annotations & review comments (supervisor workflow)
- **V4+**: video playback

See [docs/specs/2026-04-24-img-player-v1-design.md](docs/specs/2026-04-24-img-player-v1-design.md) for details.

## Troubleshooting

**`conda env create` fails to resolve packages:**
Make sure you use conda-forge channel (default with Miniforge). If you use Anaconda, run:
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

**`ImportError: No module named OpenImageIO`:**
OpenImageIO is not available on PyPI for Windows. It MUST be installed via conda from conda-forge.

**`ImportError: No module named PyOpenColorIO`:**
OpenColorIO (on Windows) is not available on conda-forge. We install it via pip (`opencolorio` PyPI package from ASWF). This is wired into `environment.yml` automatically.

**Windows: `conda` not found in PowerShell:**
From the Miniforge Prompt, run `conda init powershell` and restart PowerShell.

## License

[MIT](LICENSE) © 2026 llethanh
