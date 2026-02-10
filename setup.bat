@echo off
REM LegoGen Setup Script (Windows)
REM This script sets up the LegoGen development environment

echo =========================================
echo LegoGen Setup Script
echo =========================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    exit /b 1
)

python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Skipping creation.
) else (
    python -m venv venv
    echo Virtual environment created.
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated.
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt
echo.

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo .env file created. Please edit it to add your API keys.
    echo.
)

REM Create exports and saves directories
echo Creating export and save directories...
if not exist "exports" mkdir exports
if not exist "saves" mkdir saves
echo Directories created.
echo.

REM Run configuration check
echo Checking configuration...
python config.py
echo.

REM Run tests to verify installation
echo Running tests to verify installation...
python -m unittest discover tests -v
set TEST_EXIT_CODE=%ERRORLEVEL%
echo.

if %TEST_EXIT_CODE% EQU 0 (
    echo =========================================
    echo Setup Complete!
    echo =========================================
    echo.
    echo To get started:
    echo   1. Edit .env file and add your API keys ^(optional^)
    echo   2. Activate virtual environment: venv\Scripts\activate.bat
    echo   3. Run the application: python main.py
    echo.
    echo For more information, see README.md
    echo.
) else (
    echo =========================================
    echo Setup completed with test failures
    echo =========================================
    echo.
    echo Some tests failed, but the environment is set up.
    echo This may be due to missing optional dependencies.
    echo The application should still work for basic functionality.
    echo.
)

pause
