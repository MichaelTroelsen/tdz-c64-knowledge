@echo off
REM TDZ C64 Knowledge Base - Wiki Launcher
REM Starts local web server and opens wiki in browser

echo.
echo ============================================================
echo TDZ C64 Knowledge Base - Wiki Launcher
echo ============================================================
echo.

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo Current directory: %CD%
echo.

REM Check if wiki directory exists
if not exist "%SCRIPT_DIR%wiki\index.html" (
    echo ERROR: Wiki not found!
    echo.
    echo Looking for: %SCRIPT_DIR%wiki\index.html
    echo.
    echo Please export the wiki first by running:
    echo   .venv\Scripts\python.exe wiki_export.py --output wiki
    echo.
    pause
    exit /b 1
)

REM Get Python path from virtual environment or system
set "PYTHON_PATH=%SCRIPT_DIR%.venv\Scripts\python.exe"

REM Check if venv Python exists, otherwise use system Python
if not exist "%PYTHON_PATH%" (
    echo Virtual environment Python not found at: %PYTHON_PATH%
    echo Using system Python...
    set "PYTHON_PATH=python"
) else (
    echo Using Python from: %PYTHON_PATH%
)

echo.
echo Starting web server on port 8080...
echo.
echo Wiki will be available at:
echo   http://localhost:8080
echo.
echo Press Ctrl+C to stop the server when done.
echo.
echo Opening browser in 3 seconds...
echo.

REM Wait 3 seconds then open browser
timeout /t 3 /nobreak >nul
start http://localhost:8080

REM Start the web server (this will block until Ctrl+C)
cd /d "%SCRIPT_DIR%wiki"
if errorlevel 1 (
    echo ERROR: Failed to change to wiki directory
    pause
    exit /b 1
)

echo.
echo Server starting in: %CD%
echo.

"%PYTHON_PATH%" -m http.server 8080
