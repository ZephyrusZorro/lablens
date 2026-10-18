@echo off
REM Start LabLens.
REM   run_lablens.bat          -> desktop window (lowest latency, best for the live demo)
REM   run_lablens.bat web      -> browser UI (Streamlit)
REM   run_lablens.bat demo     -> no camera: runs the pipeline on the bundled sample photo
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo LabLens is not set up yet - running setup_windows.bat first.
  call setup_windows.bat || exit /b 1
)
call .venv\Scripts\activate.bat

if /i "%~1"=="web" (
  streamlit run app.py
) else if /i "%~1"=="demo" (
  python scripts\demo.py samples\bench_demo.png --question "What is this and which leg is positive?" --out demo_out
  start "" demo_out\demo_annotated.jpg
  pause
) else (
  python -m lablens.live %*
)
