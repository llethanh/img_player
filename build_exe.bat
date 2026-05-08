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

REM ---- Drop the WPF splash launcher next to the .exe ------------------
REM ``flick.bat`` + ``splash_launcher.ps1`` form the user-facing entry
REM point: clicking ``flick.bat`` brings the splash up in ~200 ms,
REM then spawns FlickPlayer.exe in the background. They have to sit
REM at the bundle root (not under _internal/) so the .ps1's path
REM arithmetic resolves to FlickPlayer.exe and the splash PNG.
if %EXIT_CODE% EQU 0 (
    for /d %%d in ("dist\FlickPlayer_v*") do (
        copy /y "splash_launcher.ps1" "%%d\" >nul
        copy /y "flick.bat" "%%d\" >nul
        echo [build_exe] Copied splash_launcher.ps1 + flick.bat into %%d\
    )
)

if %EXIT_CODE% EQU 0 (
    echo.
    echo [build_exe] Done. Bundle is in:
    echo   %CD%\dist\FlickPlayer\
    echo.
    echo Test it with:
    echo   dist\FlickPlayer\FlickPlayer.exe --version
    echo.
    echo To deploy: copy the entire dist\FlickPlayer\ folder to the target
    echo machine. The .exe finds its DLLs via the _internal subfolder.
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
