@echo off
REM Wrapper script for daily URL monitoring
REM This script is called by Windows Task Scheduler

REM Get the script directory
cd /d %~dp0

REM Run the daily monitor script
echo [%date% %time%] Running daily URL monitor...
.venv\Scripts\python.exe monitor_daily.py --notify --output logs\url_check_daily_%date:~-4,4%%date:~-10,2%%date:~-7,2%.json

REM Log the exit code
if %errorLevel% equ 0 (
    echo [%date% %time%] Daily monitor completed successfully - no changes detected
) else if %errorLevel% equ 2 (
    echo [%date% %time%] Daily monitor completed - CHANGES DETECTED
) else if %errorLevel% equ 3 (
    echo [%date% %time%] Daily monitor completed - SOME CHECKS FAILED
) else (
    echo [%date% %time%] Daily monitor FAILED with error code %errorLevel%
)

exit /b %errorLevel%
