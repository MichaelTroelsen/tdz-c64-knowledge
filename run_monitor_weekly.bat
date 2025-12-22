@echo off
REM Wrapper script for weekly URL monitoring
REM This script is called by Windows Task Scheduler

REM Get the script directory
cd /d %~dp0

REM Run the weekly monitor script
echo [%date% %time%] Running weekly URL monitor (comprehensive check with structure discovery)...
.venv\Scripts\python.exe monitor_weekly.py --notify --output logs\url_check_weekly_%date:~-4,4%%date:~-10,2%%date:~-7,2%.json

REM Log the exit code
if %errorLevel% equ 0 (
    echo [%date% %time%] Weekly monitor completed successfully - no changes or new pages
) else if %errorLevel% equ 2 (
    echo [%date% %time%] Weekly monitor completed - CHANGES/NEW PAGES/MISSING PAGES DETECTED
) else if %errorLevel% equ 3 (
    echo [%date% %time%] Weekly monitor completed - SOME CHECKS FAILED
) else (
    echo [%date% %time%] Weekly monitor FAILED with error code %errorLevel%
)

exit /b %errorLevel%
