@echo off
REM build_exe.bat — produces dist\FlickPlayer\FlickPlayer.exe (and friends).
REM Folder name and exe name are set in img_player.spec.
REM Double-click or run from a developer prompt. Activates the conda env
REM and invokes PyInstaller via the spec file at the repo root.

setlocal enableextensions
set ENV_NAME=img_player

REM ---- Move into the bat's own directory FIRST ------------------------
REM We pushd here (not later) so that the Google-Drive check below is
REM about *the repo's location*, not about the calling shell's cwd.
REM Otherwise calling this bat with its full path from a shell that
REM happens to be inside Google Drive would falsely abort the build.
pushd "%~dp0"

REM ---- Refuse to build inside a synced cloud folder -------------------
REM Google Drive Stream / OneDrive will fight Windows Defender and end
REM up deleting img_player.exe (or worse, the OIIO DLLs). PyInstaller
REM bootloader is a known false-positive AV target. Build on a *local*
REM SSD path instead, e.g. C:\Users\you\dev\img_player\.
echo %CD% | findstr /I /C:"\Mon Drive" /C:"\My Drive" /C:"\OneDrive" /C:"\Dropbox" >nul
if not errorlevel 1 (
    echo.
    echo [build_exe] ERROR: This folder is inside a synced cloud drive
    echo            ^(Google Drive / OneDrive / Dropbox^). PyInstaller
    echo            output gets corrupted by the sync + antivirus
    echo            interaction. Build will fail or produce an
    echo            incomplete bundle.
    echo.
    echo            Move ^(or clone^) the repo to a local SSD path first:
    echo              git clone https://github.com/llethanh/img_player.git C:\Users\%USERNAME%\dev\img_player
    echo            Then re-run build_exe.bat from there.
    echo.
    pause
    exit /b 1
)

REM ---- Locate conda activate script -----------------------------------
set ACTIVATE=
if exist "%USERPROFILE%\miniforge3\Scripts\activate.bat" set ACTIVATE="%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not defined ACTIVATE if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set ACTIVATE="%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not defined ACTIVATE if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set ACTIVATE="%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not defined ACTIVATE if exist "%ProgramData%\miniforge3\Scripts\activate.bat" set ACTIVATE="%ProgramData%\miniforge3\Scripts\activate.bat"
if not defined ACTIVATE (
    echo.
    echo [build_exe] No conda / miniforge install found in the usual places.
    echo Install Miniforge from https://github.com/conda-forge/miniforge
    pause
    exit /b 1
)

REM ---- Activate env ----------------------------------------------------
call %ACTIVATE% %ENV_NAME%
if errorlevel 1 (
    echo.
    echo [build_exe] Failed to activate conda env "%ENV_NAME%".
    echo Create it once with:
    echo   conda env create -f environment.yml
    echo Then install the build extras:
    echo   pip install -e .[build]
    pause
    exit /b 1
)

REM ---- Make sure PyInstaller is installed ------------------------------
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
    echo [build_exe] PyInstaller is not installed. Installing the build extras now...
    pip install -e .[build]
    if errorlevel 1 (
        echo [build_exe] pip install failed. Aborting.
        pause
        exit /b 1
    )
)

REM ---- Clean previous build outputs -----------------------------------
REM (pushd already done at the top of the file.)
if exist build  rmdir /s /q build
if exist dist   rmdir /s /q dist

REM ---- Run PyInstaller -------------------------------------------------
echo.
echo [build_exe] Running PyInstaller (this takes a few minutes)...
echo.
pyinstaller img_player.spec --noconfirm
set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% EQU 0 (
    REM ---- Zip the bundle for portable distribution ---------------
    REM A folder-by-folder copy to a target machine via Drive Stream
    REM / OneDrive / SMB occasionally drops bytes from individual
    REM DLLs, surfacing on the target as cryptic "Failed to load
    REM Python DLL" / "DLL load failed" errors. A single zip transfer
    REM is atomic, sidesteps cloud-sync FUSE quirks, and lets the
    REM end user extract anywhere they have write access — true
    REM portable, zero-install. Uses .NET ZipFile (much faster than
    REM Compress-Archive on ~400 MB folders).
    echo.
    echo [build_exe] Zipping bundle for portable distribution...
    for /d %%d in ("dist\FlickPlayer_v*") do (
        powershell -NoProfile -Command ^
            "Add-Type -AssemblyName System.IO.Compression.FileSystem; if (Test-Path '%%d.zip') { Remove-Item '%%d.zip' -Force }; [System.IO.Compression.ZipFile]::CreateFromDirectory('%%d', '%%d.zip', [System.IO.Compression.CompressionLevel]::Optimal, $true)"
        echo [build_exe] Wrote %%d.zip
    )
    echo.
    echo [build_exe] Done. Outputs:
    echo   %CD%\dist\FlickPlayer_v^<version^>\         ^(unzipped, for local testing^)
    echo   %CD%\dist\FlickPlayer_v^<version^>.zip      ^(portable, for distribution^)
    echo.
    echo Test it with:
    echo   dist\FlickPlayer_v^<version^>\FlickPlayer.exe --version
    echo.
    echo To distribute: send the .zip. The user extracts anywhere and
    echo double-clicks FlickPlayer.exe — no install needed.
) else (
    echo.
    echo [build_exe] PyInstaller FAILED with exit code %EXIT_CODE%.
    echo Common causes:
    echo   - Building inside a synced cloud folder ^(handled above, but
    echo     check for symlinks pointing into one^).
    echo   - Antivirus removing the bootloader. Whitelist the dist folder
    echo     in Windows Defender and re-run.
    echo   - Out of disk space ^(PyInstaller needs ~2 GB free temporarily^).
    echo   - The conda env "img_player" missing PyInstaller. Run:
    echo       pip install -e .[build]
)

REM Always pause so the user can read the message — the cmd window
REM auto-closes on double-click otherwise.
echo.
pause

popd
endlocal & exit /b %EXIT_CODE%
