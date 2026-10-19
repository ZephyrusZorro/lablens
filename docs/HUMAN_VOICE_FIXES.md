# Human recognition and voice fixes - 19 October 2026

The app is running locally at http://127.0.0.1:8501. These changes are local and unpushed.
This report supersedes the human/voice limitations in the earlier Windows validation snapshot.

## Camera behavior

- A separate, local YOLOv8n COCO model recognizes people, including distant or partly occluded people
  that a pose-only model can miss. It never feeds a person box into electronics pointing/RAG.
- MediaPipe Pose Lite provides visible body landmarks and segmentation; a face detector supports
  close webcam views. Overlays include human, head, torso, arms, legs, hands, feet and named joints
  where sufficiently visible. Existing hand tracking also labels fingers. Left/right are anatomical.
- Human regions and detected furniture reject overlapping electronics predictions. Old component
  tracks are filtered too, so a stale resistor box cannot linger over a newly recognized person.
- **Auto: bench and people** is the default. If people are visible, electronics are shown only on a
  detected breadboard outside human/furniture regions. **People and body parts** shows no electronics.
  **Mixed: loose components** permits separate loose lab parts without requiring a board.
- Pointing locks clear when their component disappears or is reclassified. Manual component selection
  remains available in the browser even when hand tracking works.
- Automatic gestures are disabled in people-only mode, during a request, and in Auto mode without a
  visible breadboard, so a hand elsewhere in the room does not issue lab commands.
- Camera failure messages and session resource cleanup replace silent stale video and abandoned
  camera/model workers. The desktop window's close button now performs normal shutdown.

## Voice behavior

- Browser **Record your question** captures browser microphone audio, rather than silently using a
  server-side Windows input. Stop recording, review/edit **Recognized text**, then **Ask this question**.
  The transcript is not automatically submitted. Recordings over 60 seconds are rejected with a message.
- Cached offline Whisper is reused across requests. Stereo and non-16-kHz recordings are converted
  to mono 16 kHz, digital silence produces no invented question, and overlapping requests are blocked.
- Desktop SPACE recording uses native microphone sample rate, keeps pre-roll so the first syllable
  is not cut off, tolerates quieter speech, and allows a longer question/trailing pause. Input can be
  explicitly selected by index or name: `run_lablens.bat --input-device 2`.
- Automatic Windows selection prefers a physical microphone and warns when only a virtual cable is
  available. A virtual cable can still be selected explicitly for an intentional routed-audio setup.
- Speech output stops before desktop recording. Listening/transcribing/errors are visible, and the
  component selected when the request begins is preserved while recognition/answering runs.
- Browser speech output starts off to reduce recording feedback. **Speak answers** can be enabled
  without rebuilding the camera pipeline or losing the conversation.
- Empty LLM responses now fail over; failed backends have a cooldown. A Groq API error no longer
  marks every unrelated network feature offline. Doctor now probes the Groq API rather than declaring
  it ready from the mere presence of a key.

## Verification

- **72 tests pass**: original 58 plus 14 regressions covering person/mask/board confusion, hidden limbs,
  human exclusion from pointing, stale locks, audio resampling, silence, virtual input selection,
  concurrent questions, review-before-submit, empty-backend failover, background gestures and recording length limits.
- The **live camera** recognized two people in the room and excluded chair/breadboard false labels in
  the inspected sequence. This is a live smoke test, not an accuracy benchmark.
- A public MediaPipe pose reference image produced head, torso, both arms/legs/hands/feet plus named
  joints. Only visible joints are rendered. This is reference-image validation, not a measurement on
  the user's body or a medical/anatomical assessment.
- A **synthesized recording** ran through the complete Runtime upload/transcription/review/answer path:
  "Which pin is ground on the 555 timer?" -> "Pin 1 is ground on the 555 timer." through Ollama, with
  manual page 7 retrieved first. The runtime then shut down cleanly. Evidence: `data/voice_validation.json`.
- The original sample electronics photo still produces LED + breadboard detections with the new
  models enabled. The browser rendered the live view, recognition modes and native recorder.
- A real browser recording completed and reached the transcriber, which correctly reported no speech
  from the current input. This validates capture/stop/upload/error handling, not intelligible speech
  from a physical microphone. An overlong recording also produced a clear length warning.
- `pip check`, Python compilation and whitespace validation pass. New model downloads have fixed
  source versions and SHA-256 verification in `scripts/setup_models.py`.

## Remaining limits

- Groq still returns HTTP 404 from this machine. Local Ollama answers and local Whisper work around it;
  the key has not been claimed valid. No credential is included in source or this report.
- Windows currently exposes a virtual-cable default input, with no verified physical microphone.
  A real spoken trial requires selecting/connecting the intended physical input. Synthetic speech
  validation does not establish microphone quality, accent/noise accuracy or speaker audibility.
- Body parts outside the view, hidden behind desks, poorly lit or too small may be omitted. Recognition
  is generic human/pose detection, not personal identity recognition. It is not infallible classification.
- Auto filtering may hide electronics held directly in front of the body; put parts on the bench.
  The existing lab detectors still need kit-specific data for a measured accuracy claim. Guided Build
  checks presence and approximate placement, not electrical connections.
- New Streamlit session cleanup uses the installed 1.66 API, now declared in requirements.

## Model sources and credits

- [Ultralytics YOLOv8n COCO model](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt),
  converted locally to ONNX using the existing exporter. The model is distributed under
  [Ultralytics' AGPL-3.0 terms](https://github.com/ultralytics/assets).
- [Google MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python)
  (Pose Lite version 1) and
  [Face Detector](https://developers.google.com/edge/mediapipe/solutions/vision/face_detector/python)
  (BlazeFace short-range version 1), executed locally through MediaPipe.
- [MediaPipe reference pose image](https://storage.googleapis.com/mediapipe-assets/pose.jpg), used only
  for the reference-image smoke test. Models and downloaded samples remain ignored by Git.
