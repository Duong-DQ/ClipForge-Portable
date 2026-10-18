@echo off
REM =========================================================================
REM  ClipForge - OPTIONAL public sharing via a Cloudflare quick tunnel.
REM
REM  This exposes your local server (http://localhost:8000) on a temporary
REM  public https://*.trycloudflare.com URL that anyone can open - no account,
REM  no open inbound port needed.
REM
REM  REQUIREMENTS:
REM    1. The ClipForge server must already be running (run.bat in another
REM       window).
REM    2. cloudflared.exe must be present at backend\tools\cloudflared.exe
REM       Download it from the official repo (Windows amd64 build):
REM         https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe
REM       Save it as:  backend\tools\cloudflared.exe
REM
REM  IMPORTANT: many corporate firewalls block Cloudflare tunnels. If you get
REM  connection errors, this will NOT work on that network - use the LAN URL
REM  printed by setup_and_run.py instead (same Wi-Fi only).
REM =========================================================================

cd /d "%~dp0"

set CLOUDFLARED=backend\tools\cloudflared.exe
if not exist "%CLOUDFLARED%" (
    echo [ERROR] cloudflared.exe not found at %CLOUDFLARED%
    echo         See the comments at the top of this file for how to get it.
    pause
    exit /b 1
)

echo Starting Cloudflare quick tunnel to http://localhost:8000 ...
echo Look for a line like:  https://<random>.trycloudflare.com
echo Press Ctrl+C to stop the tunnel.
echo.
"%CLOUDFLARED%" tunnel --url http://localhost:8000
pause
