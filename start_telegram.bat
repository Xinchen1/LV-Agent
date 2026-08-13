@echo off
REM ============================================
REM 🤖 OpenMythos Agent - Telegram Bot Launcher
REM ============================================

cd /d "%~dp0"

echo.
echo.
echo     ============================================
echo     OpenMythos Agent + Telegram Bot
echo     ============================================
echo.

REM Activate venv
if exist ".venv\" (
    call .venv\Scripts\activate
) else (
    python -m venv .venv
    call .venv\Scripts\activate
    pip install -e . --quiet
)

REM Check token
findstr /c:"bot_token: \"\"" config.yaml >nul 2>&1
if errorlevel 1 (
    echo ✓ Configuration OK
) else (
    echo ERROR: Telegram bot token not set!
    echo.
    echo Please set your bot token:
    echo   1. Edit config.yaml
    echo   2. Under tools.telegram.bot_token, add your token
    echo.
    echo Or set environment variable:
    echo   set TELEGRAM_BOT_TOKEN=your_token
    echo.
    pause
    exit /b 1
)

echo Starting Telegram Bot...
echo Press Ctrl+C to stop
echo.
echo -------------------------------------------------------------
echo.

python start_telegram.py

echo.
echo Telegram bot stopped.
pause
