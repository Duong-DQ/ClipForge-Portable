@echo off
title Subtitle Rewriter
cd /d "%~dp0"

echo ============================================
echo   Subtitle Rewriter - Starting...
echo ============================================
echo.

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.10+ from: https://www.python.org/downloads/
    echo IMPORTANT: Check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Python found:
python --version
echo.

REM Check and install dependencies
echo Checking dependencies...
python -c "import flask, flask_sqlalchemy, google.genai, dotenv" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INSTALL] Installing required packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies.
        echo Try running manually: pip install flask flask-sqlalchemy google-genai python-dotenv
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed.
) else (
    echo [OK] All dependencies present.
)
echo.

REM Start server
echo Starting server at http://localhost:5000 ...
echo.
start "" http://localhost:5000
python app.py

echo.
echo Server stopped.
pause
