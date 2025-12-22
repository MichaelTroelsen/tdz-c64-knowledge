@echo off
REM Setup Windows Task Scheduler tasks for URL monitoring
REM Run this script as Administrator to create scheduled tasks

echo ============================================================
echo TDZ C64 Knowledge Base - URL Monitoring Setup
echo ============================================================
echo.

REM Get the current directory
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click the script and select "Run as administrator"
    echo.
    pause
    exit /b 1
)

echo [*] Creating scheduled tasks...
echo [*] Script directory: %SCRIPT_DIR%
echo.

REM Create daily monitoring task
echo [1/2] Creating daily URL monitoring task...
schtasks /create /tn "TDZ-C64-KB Daily URL Monitor" /tr "%SCRIPT_DIR%\run_monitor_daily.bat" /sc daily /st 02:00 /ru SYSTEM /f
if %errorLevel% equ 0 (
    echo [OK] Daily monitoring task created successfully
) else (
    echo [ERROR] Failed to create daily monitoring task
)
echo.

REM Create weekly monitoring task
echo [2/2] Creating weekly URL monitoring task...
schtasks /create /tn "TDZ-C64-KB Weekly URL Monitor" /tr "%SCRIPT_DIR%\run_monitor_weekly.bat" /sc weekly /d SUN /st 03:00 /ru SYSTEM /f
if %errorLevel% equ 0 (
    echo [OK] Weekly monitoring task created successfully
) else (
    echo [ERROR] Failed to create weekly monitoring task
)
echo.

echo ============================================================
echo Setup complete!
echo ============================================================
echo.
echo Scheduled tasks created:
echo   - Daily URL Monitor:  Runs every day at 2:00 AM
echo   - Weekly URL Monitor: Runs every Sunday at 3:00 AM
echo.
echo To view/modify tasks:
echo   1. Open Task Scheduler (taskschd.msc)
echo   2. Navigate to Task Scheduler Library
echo   3. Look for "TDZ-C64-KB" tasks
echo.
echo To remove tasks, run: remove_monitoring_tasks.bat
echo.
pause
