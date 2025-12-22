@echo off
REM Remove Windows Task Scheduler tasks for URL monitoring
REM Run this script as Administrator to remove scheduled tasks

echo ============================================================
echo TDZ C64 Knowledge Base - Remove URL Monitoring Tasks
echo ============================================================
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click the script and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo [*] Removing scheduled tasks...
echo.

REM Remove daily monitoring task
echo [1/2] Removing daily URL monitoring task...
schtasks /delete /tn "TDZ-C64-KB Daily URL Monitor" /f >nul 2>&1
if %errorLevel% equ 0 (
    echo [OK] Daily monitoring task removed
) else (
    echo [*] Daily monitoring task not found or already removed
)
echo.

REM Remove weekly monitoring task
echo [2/2] Removing weekly URL monitoring task...
schtasks /delete /tn "TDZ-C64-KB Weekly URL Monitor" /f >nul 2>&1
if %errorLevel% equ 0 (
    echo [OK] Weekly monitoring task removed
) else (
    echo [*] Weekly monitoring task not found or already removed
)
echo.

echo ============================================================
echo Cleanup complete!
echo ============================================================
echo.
pause
