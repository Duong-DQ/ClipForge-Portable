@echo off
cd /d "%~dp0"
where python >nul 2>nul
if %errorlevel%==0 (
    python setup_and_run.py %*
) else (
    py -3 setup_and_run.py %*
)
pause
