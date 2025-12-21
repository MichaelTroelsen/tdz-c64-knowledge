@echo off
REM TDZ C64 Knowledge Base - REST API Server Startup Script
REM
REM This script starts the REST API server using uvicorn.
REM
REM Environment variables (optional):
REM   TDZ_DATA_DIR - Data directory (default: ~/.tdz-c64-knowledge)
REM   TDZ_API_KEYS - Comma-separated API keys for authentication
REM   CORS_ORIGINS - Comma-separated CORS allowed origins
REM   API_HOST - Host to bind to (default: 0.0.0.0)
REM   API_PORT - Port to bind to (default: 8000)
REM
REM Usage:
REM   run_rest_api.bat

echo ========================================
echo TDZ C64 Knowledge Base - REST API Server
echo ========================================
echo.

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please run setup first: python -m venv .venv
    echo Then install dependencies: .venv\Scripts\pip install -e ".[rest]"
    pause
    exit /b 1
)

REM Activate virtual environment
call .venv\Scripts\activate

REM Check if uvicorn is installed
.venv\Scripts\python.exe -c "import uvicorn" 2>nul
if errorlevel 1 (
    echo ERROR: uvicorn not installed!
    echo Installing REST dependencies...
    .venv\Scripts\pip.exe install -e ".[rest]"
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies!
        pause
        exit /b 1
    )
)

REM Set default environment variables if not set
if not defined TDZ_DATA_DIR set TDZ_DATA_DIR=%USERPROFILE%\.tdz-c64-knowledge
if not defined API_HOST set API_HOST=0.0.0.0
if not defined API_PORT set API_PORT=8000

echo Data Directory: %TDZ_DATA_DIR%
echo Server: http://%API_HOST%:%API_PORT%
echo API Docs: http://localhost:%API_PORT%/api/docs
echo.

REM Check if API keys are configured
if not defined TDZ_API_KEYS (
    echo WARNING: No API keys configured - authentication disabled!
    echo Set TDZ_API_KEYS environment variable for production use.
    echo.
)

echo Starting REST API server...
echo Press Ctrl+C to stop the server.
echo.

REM Start uvicorn server
.venv\Scripts\python.exe -m uvicorn rest_server:app --host %API_HOST% --port %API_PORT% --reload

pause
