@echo off
REM Launch TDZ C64 Knowledge Base GUI
REM Double-click this file to open the admin interface

echo ========================================
echo  TDZ C64 Knowledge Base - Admin GUI
echo ========================================
echo.
echo Starting Streamlit server...
echo The GUI will open in your browser automatically.
echo.
echo To stop the server, close this window or press Ctrl+C
echo ========================================
echo.

REM Activate virtual environment and run Streamlit
call .venv\Scripts\activate.bat
python -m streamlit run admin_gui.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ========================================
    echo ERROR: Failed to start GUI
    echo ========================================
    echo.
    echo Possible fixes:
    echo 1. Make sure virtual environment is set up: .venv\Scripts\activate
    echo 2. Install Streamlit: pip install streamlit
    echo 3. Check server.py and admin_gui.py are in this directory
    echo.
    pause
)
