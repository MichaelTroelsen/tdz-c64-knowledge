@echo off
echo ==========================================
echo TDZ C64 Knowledge - Setup Script
echo ==========================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.10+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found:
python --version
echo.

REM Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)
echo.

REM Activate and install dependencies
echo Installing dependencies...
call .venv\Scripts\activate.bat
pip install --upgrade pip >nul 2>&1
pip install mcp pypdf

if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo To use with Claude Code, add this to your MCP settings:
echo.
echo {
echo   "mcpServers": {
echo     "tdz-c64-knowledge": {
echo       "command": "%CD%\.venv\Scripts\python.exe",
echo       "args": ["%CD%\server.py"]
echo     }
echo   }
echo }
echo.
echo Or run: claude mcp add tdz-c64-knowledge -- "%CD%\.venv\Scripts\python.exe" "%CD%\server.py"
echo.
pause
