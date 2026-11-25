@echo off
REM BlockN setup for Windows.

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not on PATH.
    exit /b 1
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if errorlevel 1 (
    echo Python 3.10 or newer is required.
    exit /b 1
)

if not exist ".venv" (
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[gui,dev]"

if not exist ".env" (
    copy .env.example .env
    echo Created .env from .env.example.
)

if not exist "exports" mkdir exports
if not exist "saves" mkdir saves

python -m pytest
if errorlevel 1 (
    echo Setup finished, but pytest reported failures.
    exit /b 1
)

echo Setup complete.
echo   .venv\Scripts\activate.bat
echo   python main.py
echo   blockn generate --classical-order Doric --seed 7 --width 10 --depth 8 --summary
