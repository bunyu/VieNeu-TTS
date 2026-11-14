@echo off
rem Starts the VieNeu-TTS web app and opens it in the browser (http://127.0.0.1:7860, or the next free port).
rem Models are kept in .hf-home next to this file unless HF_HOME is already set.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [!] Run setup_windows_gpu.bat first. & pause & exit /b 1)
if not defined HF_HOME set HF_HOME=%~dp0.hf-home
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set VIENEU_OPEN_BROWSER=1
.venv\Scripts\python.exe -m apps.gradio_main
pause
