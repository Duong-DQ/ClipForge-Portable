@echo off
title Subtitle Rewriter
cd /d "%~dp0"

echo Starting Subtitle Rewriter server...
start /B python app.py
timeout /t 2 >nul
start http://localhost:5000
echo.
echo Server running at http://localhost:5000
echo Close this window to stop the server.
pause
