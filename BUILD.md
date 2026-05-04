# Building a standalone Windows bundle

This document explains how to package img_player into a self-contained
Windows folder you can copy to any machine — no Python, no conda, no
admin rights required on the target.

> **TL;DR — just want the binary?** Grab the latest pre-built bundle
> from **[GitHub Releases](https://github.com/llethanh/img_player/releases)**.
> Unzip, double-click `img_player.exe`. The rest of this document is
> only useful if you want to rebuild yourself.

## ⚠️ Don't build inside Google Drive / OneDrive / Dropbox

PyInstaller produces an `img_player.exe` that contains a small
"bootloader" stub. **Windows Defender flags this stub as suspicious**
(false positive — extremely well-documented; see [PyInstaller FAQ][1]).
On its own, that's manageable — you whitelist the folder.

The catastrophic combo is **PyInstaller output sitting in a synced
cloud folder** (Google Drive Stream, OneDrive, Dropbox):

1. Defender deletes `img_player.exe` minutes after the build.
2. The cloud client syncs the deletion across all your machines.
3. The whole bundle becomes useless.

`build_exe.bat` now refuses to run inside such folders. **Clone the
repo to a local SSD path first**:

```
git clone https://github.com/llethanh/img_player.git C:\Users\%USERNAME%\dev\img_player
cd C:\Users\%USERNAME%\dev\img_player
build_exe.bat
```

[1]: https://pyinstaller.org/en/stable/when-things-go-wrong.html#my-anti-virus-quarantines-my-frozen-app

## Why a bundle?

Installing img_player on a locked-down work machine often hits a wall:
no admin rights for Miniforge, antivirus blocks downloaded installers,
no internet access on the production network. The bundle sidesteps all
of that — it ships its own Python interpreter, OpenImageIO, OpenColorIO,
PySide6/Qt6, and every native dependency. The target machine just runs
the `.exe`.

The bundle is **~700 MB**, mostly Qt6 + the native VFX libraries. That's
the cost of making the thing portable. UPX compression is intentionally
disabled because it triggers antivirus heuristics on locked-down
corporate machines.

## Prerequisites (on the build machine, not the target)

* Miniforge / Miniconda installed
* The `img_player` conda env created from `environment.yml` and working
  (i.e. `python -m img_player --version` succeeds)
* The `build` extras installed:

      pip install -e .[build]

  (this pulls in PyInstaller 6.x — kept out of the regular `dev` extras
  since it's only needed for packaging.)

## One-command build

From the repo root, double-click **`build_exe.bat`** or run it from a
developer prompt. It activates the conda env, installs the `build`
extras if missing, cleans `build/` and `dist/`, then invokes
PyInstaller against `img_player.spec`.

Equivalent manual call:

```
conda activate img_player
pyinstaller img_player.spec --noconfirm
```

Build time: 1-3 minutes on an SSD. Output goes to
**`dist/img_player/`** — that's the folder you'll deploy.

### Folder layout produced

```
dist/img_player/
├── img_player.exe          ← the launcher
└── _internal/
    ├── python311.dll       ← embedded interpreter
    ├── PySide6/...         ← Qt6 + plugins
    ├── OpenImageIO.dll     ← native VFX libs
    ├── OpenColorIO.dll
    ├── img_player/...      ← our Python sources (compiled)
    │   └── render/shaders/ ← GLSL templates loaded at runtime
    └── ... (~3000 other DLLs/PYDs)
```

## Deploying to a target machine

1. **Zip** `dist/img_player/` into a single archive (helps with
   transfers and preserves the directory structure):

       cd dist
       7z a img_player-windows-x64.zip img_player\

2. **Transfer** the zip to the target machine (USB, network share, etc).

3. **Unzip** anywhere on the target — the user's Documents folder, a
   USB key, `C:\Tools\`, anything writable. The bundle is fully
   relocatable.

4. **Run** by double-clicking `img_player.exe` or from a terminal:

       img_player.exe                      # GUI on empty window
       img_player.exe path\to\sequence     # GUI on that sequence
       img_player.exe --benchmark path\... # bench mode (writes JSON)

That's it. No admin, no install, no PATH changes. If the user wants a
shortcut on the desktop, right-click → Send to → Desktop (create
shortcut) on `img_player.exe`.

## Optional: wrap into a Windows installer

For wider distribution (Start menu shortcut, `.session` file
association, Add/Remove Programs entry), use the Inno Setup script in
**[`installer/flick.iss`](installer/README.md)**. Two-step build:

```
build_exe.bat                                                  # 1. produce dist\img_player\
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\flick.iss   # 2. wrap into installer
```

Output: `installer\Output\flick-setup-X.Y.Z.exe`. Per-user install
(no admin required). See `installer/README.md` for code-signing,
sharing strategy, and when to switch from raw zip → installer.

## What the spec includes

`img_player.spec` is a Python file PyInstaller reads. It declares:

* **Entrypoint** — `src/img_player/__main__.py`
* **Datas** — GLSL shader templates (`vertex.glsl`,
  `fragment_template.glsl`) read via `importlib.resources.files()`,
  the bundled font under `src/img_player/assets/fonts/`, and the
  pip `sounddevice` wheel's bundled PortAudio DLL
  (`_sounddevice_data/portaudio-binaries/`). Python `resources` /
  `ctypes.CDLL` can't see files inside a frozen .exe unless they're
  explicitly added.
* **Binaries** — every `.dll`/`.pyd` shipped with `OpenImageIO`,
  `PyOpenColorIO`, plus the FFmpeg + codec stack PyAV depends on
  (avcodec/avformat/avutil/sw{resample,scale}/avfilter, x264, x265,
  aom, dav1d, openh264, opus, vorbis/ogg, lame, …) scooped from the
  conda env's `Library/bin/` because conda-forge PyAV doesn't bundle
  FFmpeg next to the `av` package. OIIO + OCIO are discovered
  automatically via `collect_dynamic_libs()`.
* **Hidden imports** — modules imported lazily (inside functions or via
  `importlib`) that PyInstaller's static analyser can miss:
  `OpenGL.platform.win32`, `OpenGL.arrays.*`, all submodules of
  `img_player`, `OpenImageIO`, and `PyOpenColorIO`.
* **Excludes** — Qt6 modules we never use (Qt3D, QtNetwork, QtMultimedia,
  QtWebEngine, etc.). Saves ~80 MB and reduces antivirus surface.

If you ever need to bundle extra files (icons, OCIO config files, sample
LUTs), add them to the `shader_datas` or a new `datas` list at the top
of the spec — keep paths relative to `PROJECT_ROOT`.

## Troubleshooting on the target

### The `.exe` opens then closes immediately

The console window stays open thanks to `console=True` in the spec. If
you want to capture output, run from `cmd.exe` so the window stays:

    cmd /k img_player.exe --version

### Missing DLL error on launch

Usually means the `_internal` folder didn't travel with the `.exe`.
Both must stay together. Re-zip and re-deploy.

### Black viewport / GL errors

The target's GPU driver is too old or doesn't expose OpenGL 4.1 Core.
img_player needs a 2014+ AMD/NVIDIA/Intel driver. On a clean Windows 10
install with no graphics driver update, you'll get the Microsoft Basic
Display generic driver, which is GL 1.1. Update the GPU driver first.

### "Windows protected your PC" SmartScreen warning

Click "More info" → "Run anyway". The `.exe` is unsigned (we don't have
a code-signing certificate yet). For corporate deployment, ask IT to
either whitelist the bundle or sign it with a company certificate.

### Antivirus deletes `img_player.exe`

PyInstaller bootloader is sometimes flagged by heuristic AV. Whitelist
the folder, or sign the binary. If neither is possible, switch to a
one-folder layout (already what we use) since one-file bundles look
even more suspicious — they self-extract to `%TEMP%`.

## Future improvements

* **Code signing** with an EV certificate would eliminate every
  SmartScreen warning. ~$300/year from DigiCert/Sectigo.
* **MSI / NSIS installer** instead of a zip — nicer for non-technical
  users, allows uninstall via Control Panel.
* **CI build** — GitHub Actions on a `windows-latest` runner could
  produce signed bundles on every release tag.
* **Linux bundle** via PyInstaller (same spec works) for studios with
  Linux pipelines.
