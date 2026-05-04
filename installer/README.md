# Flick Player — installer (Inno Setup)

This folder contains the Inno Setup script that wraps the PyInstaller
`--onedir` bundle into a proper Windows installer.

## When to use which

* **`build_exe.bat`** alone (no installer) → produces
  `dist/img_player/`. Zip it, share it, users unzip and double-click
  `img_player.exe`. Perfect for beta testing with 5–10 people.
* **`build_exe.bat` + `flick.iss`** → produces
  `installer/Output/flick-setup-X.Y.Z.exe`. Single-file installer with
  Start menu shortcut, `.session` file association, uninstaller. Right
  choice when you start sharing more widely.

## Building the installer

1. **Build the bundle first** so `dist\img_player\img_player.exe`
   exists:

       build_exe.bat

2. **Install Inno Setup 6+** (one-time): https://jrsoftware.org/isinfo.php

3. **Compile the installer**, either:

   * GUI: open `flick.iss` in Inno Setup Compiler, click "Compile".
   * CLI: `"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer\flick.iss`

4. Output: `installer\Output\flick-setup-1.0.2.exe` (~250 MB).

## What the installer does

* Per-user install (`{autopf}` → `%LOCALAPPDATA%\Programs\Flick Player\`).
  No admin elevation required.
* Start menu shortcut.
* Optional desktop shortcut (default off, user opts in).
* Optional `.session` file association — double-click opens in Flick.
  Registered under `HKCU` so it doesn't collide with system associations.
* Add/Remove Programs entry that uninstalls cleanly + removes the
  registry keys.

## Code signing

Without a code signing certificate, Windows SmartScreen warns users
about "unrecognized publisher" on first run. The installer still
works, but it's a UX speed-bump.

When you have a cert (~$300/year for a standard EV), uncomment the
`SignTool` line in `flick.iss` and supply the path/password.

## Why per-user, not all-users?

Studio review machines often have locked-down admin policies. A
per-user install bypasses that — every artist can install their own
copy under `%LOCALAPPDATA%`. Switching to system-wide is a one-line
change (`PrivilegesRequired=admin`, `DefaultDirName={autopf}` →
`{commonpf}`) when you actually need it.
