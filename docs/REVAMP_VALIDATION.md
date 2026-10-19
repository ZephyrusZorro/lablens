# Workbench revamp — 19 October 2026

Saved locally; no commit or push performed. This replaces the earlier Premium UI layout.

## Workflow

The camera source and recognition mode are visible above the workspace. Inspect is for selecting a component and asking questions; Build contains project steps; Manual shows the cited source. Settings holds speech, skeleton overlays and optional gesture shortcuts. The interface uses warm neutral colours, a restrained rust accent, small corner radii and direct labels. Removed the marketing heading, metric cards, repeated panels and dense body-part box overlays.

Refresh detected components to populate a stable selection list. A selected ID follows that part rather than a stale screen coordinate. Selecting a part applies to a question immediately; no pointing dwell is needed. If it disappears, questions do not silently switch to a nearby part. Switching camera sources starts a new workspace and clears its audio draft and component selection.

## Recognition and stability

- Separate trackers for stationary electronics and moving people. People expire after one missed update; independent anatomy tracks no longer linger or jump between people.
- Pose estimation uses a crop for each detected person, up to six. Optional skeleton overlays avoid dense overlapping anatomy labels and unnecessary pose work during normal inspection.
- Multiple people or ambiguous hands disable automatic pointing. Gesture commands are opt-in and cannot advance a build in a crowd. Manual selection remains available.
- Failed recognition clears old boxes and exposes an error; results older than two seconds expire. Missing component models now produce an actionable error instead of silently displaying mock detections.
- Guided checks use a sequence counter rather than coarse Windows timestamps.
- Camera frames are JPEG data URLs, avoiding rapidly expired media URLs. Selection controls do not redraw on a timer. Only the live view and live status panels refresh.

## Verification

- 78 pytest tests passed in 2.79 seconds. Python compilation, dependency consistency and git whitespace checks passed locally.
- Live browser camera showed two and three people with zero electronics in the room view. This is an observation, not an accuracy benchmark.
- Synthetic three-person replay: three stable person tracks after the two-update confirmation, and three pose skeletons using per-person crops. The earlier full-frame pose pass found zero in the same replay.
- Synthetic cropped bench replay returned independent transistor, LED and breadboard tracks. This does not establish detection recall on the user's kit. The uncropped duplicated sample also includes a person behind the board and triggers Auto's conservative suppression; Mixed mode is intended for loose components.
- Browser: live-to-photo source switch; LED selection; question about positive leg answered correctly by local Ollama in 7.3 seconds; build start, visual board check, advance to step 2, end; Manual navigation.
- Doctor: camera 0 readable at 640x480; models, retrieval, local speech recognition and Ollama available. Physical Realtek microphone now enumerates. Six-second local recording returned no detected speech, so live word accuracy remains unverified.
- Groq still returns HTTP 404. Local Ollama is the working answer provider.

Evidence is saved in data/scene_validation.json, data/doctor_revamp.txt and data/microphone_revamp.json. These local diagnostics are gitignored.

## Remaining limits

No test suite can establish that the whole application is bug-free. Crossing/occluded people can change IDs: these are temporary tracks, not identity recognition. Body landmarks and component detections can be missed; train/evaluate on the actual bench before claiming accuracy. Automatic mode intentionally suppresses parts overlapping people or background furniture. Build checks do not verify electrical connectivity. A spoken microphone test with a known sentence is still needed.
