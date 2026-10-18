@echo off
REM LabLens one-time setup for Windows. Double-click, or run from a terminal in this folder.
REM Creates .venv, installs the runtime (no PyTorch), downloads + converts the open-source models,
REM caches the offline speech model and builds the knowledge base.
setlocal
cd /d "%~dp0"

set "PY="
for %%V in (3.11 3.12 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo [error] LabLens needs Python 3.10, 3.11 or 3.12 ^(MediaPipe has no 3.13 wheels yet^).
  echo         Install Python 3.11 from https://www.python.org/downloads/ and run this again.
  pause
  exit /b 1
)
echo Using %PY%

if not exist .venv (
  %PY% -m venv .venv || goto :fail
)
call .venv\Scripts\activate.bat || goto :fail
python -m pip install --upgrade pip || goto :fail
pip install -r requirements.txt || goto :fail

if not exist .env copy .env.example .env >nul

python scripts\setup_models.py || goto :fail
python -m lablens.doctor

echo.
echo Setup complete. Start LabLens with run_lablens.bat
pause
exit /b 0

:fail
echo.
echo [error] Setup failed. Scroll up for the first error message.
pause
exit /b 1
