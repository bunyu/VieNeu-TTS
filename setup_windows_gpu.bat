@echo off
setlocal
rem Installs VieNeu-TTS for Windows with an NVIDIA GPU (v3 Turbo on PyTorch/CUDA) into .venv.
rem Needs uv (https://docs.astral.sh/uv/) and a recent NVIDIA driver. No admin rights needed.
cd /d "%~dp0"

where uv >nul 2>nul || (echo [!] uv is not installed: https://docs.astral.sh/uv/getting-started/installation/ & pause & exit /b 1)

rem PyTorch's CUDA 12.8 wheels dropped GTX 9xx/10xx cards (compute capability below 7.0) and
rem CUDA 12.6 wheels cannot run RTX 50xx cards, so choose by the card's compute capability.
set CUDA_WHEELS=cu126
for /f "tokens=1 delims=." %%m in ('nvidia-smi --query-gpu=compute_cap --format=csv^,noheader 2^>nul') do (
    if %%m GEQ 12 set CUDA_WHEELS=cu128
)
echo [+] PyTorch wheels: %CUDA_WHEELS%

echo [+] Installing VieNeu-TTS (core) ...
uv sync --frozen || goto :fail
echo [+] Installing PyTorch 2.8 for CUDA ...
uv pip install --python .venv\Scripts\python.exe "torch==2.8.0" "torchaudio==2.8.0" --index-url https://download.pytorch.org/whl/%CUDA_WHEELS% || goto :fail
uv pip install --python .venv\Scripts\python.exe "transformers==4.57.6" safetensors || goto :fail

.venv\Scripts\python.exe -c "import torch; print('[+] GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none - will run on the CPU')"
echo [+] Done. Start the app with run_windows.bat
pause
exit /b 0

:fail
echo [!] Installation failed.
pause
exit /b 1
