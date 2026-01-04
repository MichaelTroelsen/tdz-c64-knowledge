@echo off
REM Test version of start-wiki.bat - doesn't start server

echo.
echo ============================================================
echo TDZ C64 Knowledge Base - Wiki Launcher TEST
echo ============================================================
echo.

REM Get the directory where this batch file is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo [OK] Script directory: %SCRIPT_DIR%
echo [OK] Current directory: %CD%
echo.

REM Check if wiki directory exists
if not exist "%SCRIPT_DIR%wiki\index.html" (
    echo [FAIL] Wiki not found at: %SCRIPT_DIR%wiki\index.html
    pause
    exit /b 1
) else (
    echo [OK] Wiki found at: %SCRIPT_DIR%wiki\index.html
)

REM Get Python path from virtual environment or system
set "PYTHON_PATH=%SCRIPT_DIR%.venv\Scripts\python.exe"

REM Check if venv Python exists, otherwise use system Python
if not exist "%PYTHON_PATH%" (
    echo [WARN] Virtual environment Python not found at: %PYTHON_PATH%
    echo [INFO] Using system Python...
    set "PYTHON_PATH=python"
) else (
    echo [OK] Python found at: %PYTHON_PATH%
)

echo.
echo Testing Python version:
"%PYTHON_PATH%" --version

echo.
echo Testing wiki directory access:
cd /d "%SCRIPT_DIR%wiki"
if errorlevel 1 (
    echo [FAIL] Failed to change to wiki directory
    pause
    exit /b 1
) else (
    echo [OK] Changed to: %CD%
)

echo.
echo ============================================================
echo All checks passed! The real batch file should work fine.
echo ============================================================
echo.
pause
