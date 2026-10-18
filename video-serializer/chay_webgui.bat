@echo off
setlocal
chcp 65001 >nul
title Video Serializer - Web GUI
set "TOOL_DIR=%~dp0"
cd /d "%TOOL_DIR%"

where python >nul 2>nul
if errorlevel 1 (
  echo [LOI] Khong thay Python trong PATH.
  echo Cai Python 3.10+ roi chay lai.
  pause
  exit /b 1
)

echo ============================================
echo   Video Serializer - Web GUI
echo   Dang khoi dong server...
echo   Trinh duyet se tu dong mo.
echo   De dung: giu cay cua so nay.
echo   Dong cua so hoac nhan Ctrl+C de dung.
echo ============================================
echo.
python webgui.py
echo.
echo Server da dong.
pause
