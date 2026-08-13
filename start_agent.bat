@echo off
REM ============================================
REM 🤖 OpenMythos Agent - Windows Launcher
REM One-click start for Windows
REM ============================================

cd /d "%~dp0"

echo.
echo.
echo     ============================================
echo     OpenMythos Agent - Windows Launcher
echo     ============================================
echo.

REM Check virtual environment
if exist ".venv\" (
    echo Activating virtual environment...
    call .venv\Scripts\activate
) else (
    echo Virtual environment not found.
    echo Creating one...
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -e . --quiet
)

REM Check config
findstr /c:"api_key:" config.yaml >nul 2>&1
if errorlevel 1 (
    echo WARNING: NIM API key not set in config.yaml
    echo Edit config.yaml and set agent.nim.api_key
    echo.
)

echo Starting OpenMythos Agent...
echo Press Ctrl+C to stop
echo.
echo -------------------------------------------------------------
echo.

REM Forward all arguments
python -m agent_project %*

pause
