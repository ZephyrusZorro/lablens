# Windows validation - 19 October 2026

Setup and execution were performed on this Windows 11 machine with Python 3.11.9,
31.6 GB RAM and an RTX 4060 Laptop GPU. Changes are local and have not been pushed.

## Verified

- `setup_windows.bat` completed: runtime dependencies, both SHA-256-checked detector
  downloads and ONNX conversion, MediaPipe hand model, sample photo, cached offline
  Whisper `base.en`, manual PDF and search index.
- `pip check`: no broken requirements.
- All 58 tests pass, including the downloaded model test. Windows denied access to
  the default pytest temp root; a fresh directory inside `data/` worked:
  `.venv\Scripts\python -m pytest -q tests --basetemp=data\pytest_windows_final`.
  Use a new temp directory name for subsequent runs.
- Doctor reports READY. Camera 0 returns 640x480 frames, and MediaPipe Tasks loads.
  This proves device access and model initialization, not pointing accuracy.
- Both detector sessions now use up to four CPU threads and disable idle spinning.
  The self-check measured 665 ms/frame before and 124 ms/frame after this change.
  These are individual measurements, not a controlled performance benchmark.
- Ollama was already installed. Downloaded `llama3.2:3b` and exercised real answers.
  Its first model load/request took about 73 seconds; a warm short request with a
  manual excerpt took 2.35 seconds. Warm the model before demonstrating it.
- Photo pipeline: real detectors found an LED and breadboard; simulated pointing
  locked onto the LED; Ollama correctly explained anode/cathode, with manual page 3
  retrieved first (5.17 seconds). Guided Build passed breadboard presence and
  reported the absent resistor. See `demo_out/demo_report.md` and annotated image.
- The browser UI rendered the annotated photo. A typed NE555 ground question
  answered pin 1 through Ollama (7.2 seconds), using the local manual.
- Grounding evaluation reproduced 88% top-1, 100% top-4, and 81% with pointing
  versus 50% without on the 16 pointing questions. These are internal retrieval
  measurements on 40 authored questions, not a test of generated-answer accuracy.
- Cached offline Whisper loaded without downloading. A Windows-synthesized spoken
  question was correctly transcribed as "Which pin is ground on the 555 timer?".
  This is synthetic speech validation, not a real microphone trial.
- Windows SAPI completed spoken output to the configured default output device.
  A person still needs to confirm audibility and volume.

## Fixes made during this run

- Reduced ONNX CPU thread contention in `lablens/vision/yolo_onnx.py`.
- Made the browser source-page panel refresh independently while answers complete
  in the background, matching the existing chat refresh behavior.
- Ignored local `.log` files. The supplied Groq key is stored only in ignored `.env`.
- Tests still pass after the changes; Python compilation and `git diff --check` pass.

## Remaining limits

- Groq returned HTTP 404 with `{"error":{"message":"Not Found"}}` both through a
  direct API call and the installed Groq SDK. Key validity could not be established.
  LabLens's doctor labels Groq ready merely because a key and connectivity exist;
  that line does not establish successful authentication or inference. Actual
  answers successfully fell back to Ollama. No key is reproduced in this report.
- The default recording input is `CABLE Output (VB-Audio Virtual Cable)`, not a
  demonstrated physical microphone. Its input stream opened, but a short recording
  found no speech. Select a working physical microphone as the Windows default
  before testing SPACE/voice input. No Windows privacy/audio settings were changed.
- Camera opens and the native desktop window exists. Native screenshot capture
  timed out, so desktop interaction, real hand gestures, and live pointing precision
  were not visually verified. Browser photo mode was inspected successfully.
- The sample photo already contains annotations from its author and is a smoke
  test. It is not a detector benchmark or a substitute for photos of your own kit.
- The online NE555 datasheet search found no additional PDF; the local manual still
  supported the answer. Guided Build does not verify electrical connections.
- GitHub publication, physical camera mounting, kit fine-tuning and real pointing
  accuracy measurements were not performed.

## Run

Desktop camera: `run_lablens.bat`

Camera-free desktop: `run_lablens.bat photo`

Browser camera: `run_lablens.bat web`

Browser photo mode (used during verification):
`.venv\Scripts\python -m streamlit run app.py -- --source samples/bench_demo.png`

Self-check: `run_lablens.bat doctor`

Desktop controls: ENTER to type, SPACE to speak, Q to quit. Hold the left mouse
button over a component to point without hand tracking.
