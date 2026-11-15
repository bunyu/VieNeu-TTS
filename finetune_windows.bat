@echo off
rem Fine-tunes VieNeu-TTS v3 Turbo on the voice recorded in the web app's "Fine-tune giong" tab.
rem Usage: finetune_windows.bat "Voice name"      (close the web app first: the GPU cannot hold both)
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (echo [!] Run setup_windows_gpu.bat first. & pause & exit /b 1)
if "%~1"=="" (echo Usage: finetune_windows.bat "Voice name" & pause & exit /b 1)
if not defined HF_HOME set HF_HOME=%~dp0.hf-home
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
rem The training extras, once: LoRA (peft), accelerate and the parquet writer.
.venv\Scripts\python.exe -c "import peft, accelerate, pyarrow" 2>nul || uv pip install --python .venv\Scripts\python.exe "peft>=0.13" accelerate pyarrow
if errorlevel 1 (echo [!] Could not install the training packages. & pause & exit /b 1)
.venv\Scripts\python.exe -m apps.finetune_studio train %*
pause
