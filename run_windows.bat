@echo off
rem Starts the VieNeu-TTS web app (http://127.0.0.1:7860) and opens it in the browser.
rem Set HF_HOME before running to keep downloaded models on another drive.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [!] Run setup_windows_gpu.bat first. & pause & exit /b 1)
set PYTHONIOENCODING=utf-8
start "" /b cmd /c "timeout /t 12 >nul & start http://127.0.0.1:7860"
.venv\Scripts\python.exe -m apps.gradio_main
pause
