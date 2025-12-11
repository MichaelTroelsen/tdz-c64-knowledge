@echo off
call "%~dp0.venv\Scripts\activate.bat"
python "%~dp0cli.py" add-folder "%~dp0pdf" --tags reference --recursive
python "%~dp0cli.py" add-folder "%~dp0txt" --tags reference --recursive
python "%~dp0cli.py" add-folder "%~dp0codebase64_latest" --tags reference --recursive
