@echo off
rem Double-click to run AeroSculpt locally, then open it in your browser.
rem Close this window (or press Ctrl+C) to stop the server.
cd /d "%~dp0"
set "PY=C:\Users\hughe\anaconda3\python.exe"
if not exist "%PY%" set "PY=python"
echo Starting AeroSculpt on http://localhost:8000 ...
echo (If it says the port is in use, AeroSculpt is already running - just open the link.)
start "" cmd /c "timeout /t 5 /nobreak >nul & start http://localhost:8000/app"
"%PY%" run.py serve
pause
