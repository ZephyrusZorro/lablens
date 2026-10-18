@echo off
REM Start LabLens.
REM   run_lablens.bat          -> desktop window with the webcam (best for the live demo)
REM   run_lablens.bat web      -> browser UI (Streamlit)
REM   run_lablens.bat photo    -> desktop window on the sample photo; point with the mouse (no camera needed)
REM   run_lablens.bat demo     -> one-shot pipeline run on the sample photo, writes demo_out\
REM   run_lablens.bat doctor   -> pre-demo self-check
REM   run_lablens.bat eval     -> grounding numbers for the slides
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\activate.bat (
  echo LabLens is not set up yet - running setup_windows.bat first.
  call setup_windows.bat || exit /b 1
)
call .venv\Scripts\activate.bat

if /i "%~1"=="web" (
  streamlit run app.py
) else if /i "%~1"=="photo" (
  python -m lablens.live --source samples\bench_demo.png
) else if /i "%~1"=="demo" (
  python scripts\demo.py samples\bench_demo.png --question "What is this and which leg is positive?" --out demo_out
  start "" demo_out\demo_annotated.jpg
  pause
) else if /i "%~1"=="doctor" (
  python -m lablens.doctor
  pause
) else if /i "%~1"=="eval" (
  python scripts\evaluate.py grounding -v
  pause
) else (
  python -m lablens.live %*
)
