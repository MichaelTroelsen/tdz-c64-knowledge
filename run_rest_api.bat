@echo off
REM TDZ C64 Knowledge Base - REST API Server Startup Script
REM
REM This script activates the virtual environment and starts the REST API server
REM on port 8000 with auto-reload for development.
REM
REM Access the API at: http://localhost:8000
REM API Documentation: http://localhost:8000/api/docs

echo Starting TDZ C64 Knowledge Base REST API Server...
echo.

REM Activate virtual environment
call .venv\Scripts\activate

REM Start uvicorn server with auto-reload
python -m uvicorn rest_server:app --reload --host 0.0.0.0 --port 8000
