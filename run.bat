@echo off
echo Starting TDZ C64 Knowledge MCP Server...
echo Press Ctrl+C to stop
echo.
call .venv\Scripts\activate.bat
python server.py
