@echo off
setlocal
REM ============================================================
REM  ComfyUI-MusicAnalyzer - Manual model download helper
REM  This is NOT automatic downloading: the script only runs
REM  when you double-click it.
REM
REM  Models go to the ComfyUI OFFICIAL shared folder:
REM      ComfyUI\models\audio_encoders\<model name>\
REM  (fallback: this plugin's local .\models\ folder)
REM ============================================================

REM China network: keep this line (hf-mirror). Remove it to use HuggingFace directly.
set HF_ENDPOINT=https://hf-mirror.com

REM Prefer the official ComfyUI models folder, fall back to the local plugin folder.
if exist "%~dp0..\..\models" (
  set "BASE=%~dp0..\..\models\audio_encoders"
) else (
  set "BASE=%~dp0models"
)
if not exist "%BASE%" mkdir "%BASE%"

echo.
echo Models will be saved to: %BASE%
echo.
echo [1/2] Downloading ACE-Step-Transcriber (default analyzer, ~4GB)...
huggingface-cli download ACE-Step/acestep-transcriber --local-dir "%BASE%\ACE-Step-Transcriber"
if errorlevel 1 (
  echo.
  echo [!] huggingface-cli not found.
  echo     Install it first: pip install huggingface_hub
  echo     (or run inside the ComfyUI venv: .venv\Scripts\activate)
  pause
  exit /b 1
)

echo.
echo [2/2] Downloading MiDaShengLM-7B GPTQ (best music captions, ~5GB)...
echo     Note: the GPTQ variant also needs: pip install auto-gptq
huggingface-cli download mispeech/midashenglm-7b-0804-w4a16-gptq --local-dir "%BASE%\MiDaShengLM-7B-GPTQ"

echo.
echo All done. Restart ComfyUI, then pick the models in the nodes.
pause
