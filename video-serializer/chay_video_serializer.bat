@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Video Serializer - Tach video dai thanh series clip

REM ============================================================
REM  Video Serializer - phien ban bat file (double-click de chay)
REM  CACH DUNG:
REM    (A) Double-click file nay -> nhap duong dan theo huong dan
REM    (B) Kem (drag & drop) 2 file vao icon: VIDEO truoc, TRANSCRIPT sau
REM  CAU HINH CAN: ffmpeg trong PATH + Python 3.10+
REM  GHI CHU: file .bat phai nam TRONG thu muc video-serializer\
REM ============================================================

set "TOOL_DIR=%~dp0"
if not exist "%TOOL_DIR%video_serializer\__init__.py" (
    echo [LOI] Khong tim thay thu muc video_serializer gan voi file .bat nay.
    echo       File .bat phai nam TRONG thu muc: video-serializer\
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo [LOI] Khong tim thay Python. Hay cai dat Python 3.10+ va boc "Add Python to PATH".
    pause
    exit /b 1
)

REM --- Fast path: 2 file duoc kem vao (video, transcript) ---
if not "%~1"=="" if not "%~2"=="" (
    set "VIDEO=%~1"
    set "TRANSCRIPT=%~2"
    goto RUN
)

REM --- 1 file: nhan dang theo phan moi (.srt/.vtt = transcript, con lai = video) ---
if not "%~1"=="" (
    if /i "%~x1"==".srt" set "TRANSCRIPT=%~1"
    if /i "%~x1"==".vtt" set "TRANSCRIPT=%~1"
    if not defined TRANSCRIPT set "VIDEO=%~1"
)

REM --- Huong dan nhap (kem file vao cua so nay de duoc duong dan nhanh) ---
if not defined VIDEO      set /p "VIDEO=  [1/2] Duong dan VIDEO (.mp4/.mkv...): "
if not defined TRANSCRIPT set /p "TRANSCRIPT=  [2/2] Duong dan TRANSCRIPT (.srt/.vtt): "

:RUN
if not exist "%VIDEO%" (
    echo [LOI] Khong tim thay VIDEO: %VIDEO%
    pause & exit /b 1
)
if not exist "%TRANSCRIPT%" (
    echo [LOI] Khong tim thay TRANSCRIPT: %TRANSCRIPT%
    pause & exit /b 1
)

REM --- Thu muc chua ket qua (mac dinh: ten_video_clips gan video) ---
if not "%~3"=="" (
    set "OUT=%~3"
    goto DOIT
)
for %%F in ("%VIDEO%") do set "OUT_DEFAULT=%%~dpnF_clips"
set /p "OUT=  [3/3] Luu ket qua vao thu muc [Enter = %OUT_DEFAULT%]: "
if not defined OUT set "OUT=%OUT_DEFAULT%"

:DOIT
echo.
echo ============================================================
echo   BAT DAU XU LY
echo   Video     : %VIDEO%
echo   Transcript: %TRANSCRIPT%
echo   Output    : %OUT%
echo ============================================================
echo.

cd /d "%TOOL_DIR%"
python -m video_serializer "%VIDEO%" "%TRANSCRIPT%" -o "%OUT%"
set "RC=%errorlevel%"

echo.
if "%RC%"=="0" (
    echo ============================================================
    echo   [OK] HOAN TAT! Clip + metadata da luu vao:
    echo        %OUT%
    echo ============================================================
) else (
    echo ============================================================
    echo   [LOI] Xu ly that bai - ma loi %RC%. Xem loi chi tiet ben tren.
    echo ============================================================
)
echo.
pause
endlocal
