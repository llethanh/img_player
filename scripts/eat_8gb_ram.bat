@echo off
REM Helper to allocate ~4 GB of RAM for testing slice 3's boot-time memory
REM pressure check. Activates the same conda env as img_player.bat first
REM (otherwise numpy crashes on import — DLLs not in PATH).

setlocal
set ENV_NAME=img_player

set ACTIVATE=
if exist "%USERPROFILE%\miniforge3\Scripts\activate.bat" set ACTIVATE="%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not defined ACTIVATE if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set ACTIVATE="%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not defined ACTIVATE if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set ACTIVATE="%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not defined ACTIVATE (
    echo Could not find a conda install.
    pause
    exit /b 1
)

call %ACTIVATE% %ENV_NAME%
if errorlevel 1 (
    echo Failed to activate conda env "%ENV_NAME%".
    pause
    exit /b 1
)

echo Launching ram_eater (4 GB, 10 min)...
echo.
python "%~dp0ram_eater.py" --gb 4 --seconds 600
echo.
echo --- ram_eater exited with code %ERRORLEVEL% ---
pause
