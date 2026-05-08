@echo off
REM Flick Player launcher — drives the PowerShell splash + spawns
REM FlickPlayer.exe. Use this instead of double-clicking
REM FlickPlayer.exe directly so the splash appears within ~200 ms
REM of your click. (FlickPlayer.exe still works on its own; the
REM splash code falls back to a Qt-rendered splash when run without
REM the launcher, so the artwork shows up at ~2 s instead.)

setlocal
pushd "%~dp0"

REM ``-WindowStyle Hidden`` keeps the PowerShell console invisible.
REM ``-ExecutionPolicy Bypass`` sidesteps machine-wide PS execution
REM lock-downs without touching the user's policy. ``-NoProfile``
REM skips $PROFILE loading — saves ~100 ms on shells that load
REM oh-my-posh / Starship at startup.
start "" /b powershell.exe ^
    -NoProfile ^
    -ExecutionPolicy Bypass ^
    -WindowStyle Hidden ^
    -File "%~dp0splash_launcher.ps1"

popd
endlocal
