@echo off
setlocal
cd /d "%~dp0"
title Air Quality Trend Predictor

echo ================================================================
echo  AIR QUALITY TREND PREDICTOR - LINEAR REGRESSION ONLY
echo ================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found.
    echo Install Python 3.10 or newer from https://www.python.org/downloads/
    echo During installation, select "Add Python to PATH".
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating a project-specific Python environment...
    python -m venv .venv
    if errorlevel 1 goto :error
)

call ".venv\Scripts\activate.bat"
echo.
echo Installing or checking required packages...
python -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Running the complete project. This may take a few minutes on the first run...
python main.py
if errorlevel 1 goto :error

echo.
echo Done. Open the outputs folder to view your results.
pause
exit /b 0

:error
echo.
echo The run did not complete. Read the error above and check README.md.
pause
exit /b 1
