@echo off
REM img_player launcher — double-click or drag a file/folder onto this .bat.
REM Activates the conda env then runs `python -m img_player` with any
REM arguments that were passed in (or dropped on the .bat).

setlocal
set ENV_NAME=img_player

REM Locate a conda install in a few common places.
set ACTIVATE=
if exist "%USERPROFILE%\miniforge3\Scripts\activate.bat" set ACTIVATE="%USERPROFILE%\miniforge3\Scripts\activate.bat"
if not defined ACTIVATE if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" set ACTIVATE="%USERPROFILE%\miniconda3\Scripts\activate.bat"
if not defined ACTIVATE if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" set ACTIVATE="%USERPROFILE%\anaconda3\Scripts\activate.bat"
if not defined ACTIVATE if exist "%ProgramData%\miniforge3\Scripts\activate.bat" set ACTIVATE="%ProgramData%\miniforge3\Scripts\activate.bat"
if not defined ACTIVATE (
    echo.
    echo [img_player] Could not find a conda / miniforge install.
    echo Looked in:
    echo   %%USERPROFILE%%\miniforge3
    echo   %%USERPROFILE%%\miniconda3
    echo   %%USERPROFILE%%\anaconda3
    echo   %%ProgramData%%\miniforge3
    echo.
    echo Install Miniforge from https://github.com/conda-forge/miniforge
    pause
    exit /b 1
)

call %ACTIVATE% %ENV_NAME%
if errorlevel 1 (
    echo.
    echo [img_player] Failed to activate conda env "%ENV_NAME%".
    echo Create it once with:
    echo   conda env create -f environment.yml
    pause
    exit /b 1
)

pushd "%~dp0"
python -m img_player %*
set EXIT_CODE=%ERRORLEVEL%
popd

endlocal & exit /b %EXIT_CODE%
