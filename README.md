# LabLens: a hands-free AI lab mentor you can point at

A camera watches your electronics bench. Point at a component and ask out loud,
*"What is this and where does it connect?"* LabLens works out what is under your fingertip,
retrieves the matching page of the lab manual or datasheet, and answers by voice with the cited
page on screen. In **Guided Build** mode it walks you through a circuit and uses the camera to check
that each part is actually on the board before moving on.

It keeps working when the lab Wi-Fi dies: answers switch from Groq to a local Ollama model
automatically, and if no LLM is available at all, LabLens quotes the relevant manual sentences with
page citations instead of failing.

## Run it (Windows)

1. Double-click **`setup_windows.bat`** (one time, needs internet: about 10 minutes).
   It creates a virtual environment, installs everything (no PyTorch), downloads and verifies the
   open-source component detectors, the hand model and the offline speech model, builds the
   knowledge base, then runs a self-check.
2. Double-click **`run_lablens.bat`** to start.

| Command | What it does |
|---|---|
| `run_lablens.bat` | Desktop window: lowest latency, best for the live demo |
| `run_lablens.bat web` | Browser UI (Streamlit) with the cited datasheet page beside each answer |
| `run_lablens.bat demo` | No camera needed: runs the full pipeline on a sample bench photo |
| `python -m lablens.doctor` | Pre-demo check of camera, mic, models, LLM, speech |

Linux / macOS: `./setup.sh`, then `source .venv/bin/activate && python -m lablens.live`.

Optional: put a free Groq key in `.env` (`GROQ_API_KEY=...`) for fast online answers and speech,
and install [Ollama](https://ollama.com) with `ollama pull llama3.2:3b` for offline answers.

### Desktop window controls

| Input | Action |
|---|---|
| Point with your index finger | Selects the part (locks after 0.5 s) |
| Hold left mouse button | Points with the cursor (works without hand tracking) |
| `SPACE` | Ask by voice |
| `ENTER` | Type a question |
| `1`-`4` | Start a Guided Build project |
| 👍 or `N` | Next step (a second 👍 skips a failed camera check) |
| ✋ or `R` | Repeat |
| ✊ or `S` | Stop talking |
| `Q` | Quit |

## How it works

```
camera ──► hands (MediaPipe, every frame) ──► pointing resolver ──► locked component
   │                                              ▲    (ray 6→8, 10-frame vote, 0.5 s dwell)
   └──► AsyncDetector (background thread) ────────┘
          ONNX ensemble of 2 YOLOv8 models + classical breadboard detector + IoU tracker

mic ─► STT (Groq Whisper / faster-whisper offline)
          ▼
query composer: "[POINTED COMPONENT: LED (0.87)] [GUIDED STEP 3 …] which leg is positive?"
          ▼
hybrid retrieval: BM25 (inverted index) + optional embeddings, RRF fusion, component/topic boost
          ▼  ◄── unknown part number? datasheet scraper downloads + indexes it
LLM router: Groq ⇄ Ollama (auto failover) ⇄ extractive answer with citations (no LLM)
          ▼
sentence-by-sentence TTS  +  answer and cited page in the UI
```

* **Pointing is the query.** The spoken words say *what* you want; the fingertip says *which part*.
  The composed query carries the detected class, so retrieval favours pages about that part.
* **Closed-loop Guided Build.** Each project (`lablens/guided/projects/*.json`) lists the parts,
  counts and approximate positions on the breadboard expected at each step. LabLens says
  "I don't see the resistor yet" until it does.
* **Resistor colour bands.** Point at a resistor and the band reader estimates its value; Guided Build
  uses it to catch a wrong resistor.
* **Instructor report.** `python -m lablens.cli report` ranks the steps students fail most often.

## Models

LabLens runs detectors with ONNX Runtime, so neither the runtime nor setup needs PyTorch. Ultralytics
`.pt` checkpoints are converted by a built-in, torch-free exporter (`lablens/vision/export.py`).
That exporter was checked against Ultralytics' official `yolov8n.pt` on their reference image
(4 people + bus at 0.84–0.90, matching Ultralytics' own output).

| Model | Classes | Reported by its author (their validation split) |
|---|---|---|
| `components_v8n_6cls` (YOLOv8n, 800 px) | capacitor, diode, LED, resistor, transistor, zener | mAP@0.5 0.715, precision 0.81, recall 0.65 |
| `components_v8s_4cls` (YOLOv8s, 640 px) | capacitor, IC, LED, resistor | mAP@0.5 0.783, precision 0.81, recall 0.72 |
| Breadboard | breadboard | classical CV (hole-grid detector), no model |

Their outputs are unified into one vocabulary and fused with Weighted Box Fusion.

**Credit and licensing.** The detector weights are third-party work by
[Hassan Mroueh](https://github.com/hassanmroueh2003/Electronic-Component-Detection-using-YOLOv8)
(trained on the Roboflow *Electronics Components* dataset by Jovine) and
[Ravi Ranjan](https://github.com/ravi-ranjan707089/ComponentDetector). Neither repository states a
licence, so the weights are **not** included in this repository: `setup_models.py` downloads them from
the original repositories at pinned commits and verifies their SHA-256. Ask the authors before any
commercial use, or train your own (below).

**Honest accuracy note.** The numbers above are the authors' own validation results. They have
not been measured on your bench. Accuracy depends heavily on camera angle, lighting and the exact
parts in your kit. For the best live results:

1. Mount the camera top-down, with a desk lamp and no glare.
2. Fine-tune on 150–300 photos of your own kit:
   `pip install -r requirements-train.txt`, then `python scripts/train_yolo.py --data <your data.yaml>`.
   Add the resulting `best.pt` to `vision.models` in `config.yaml`; it is converted to ONNX on first load.
3. Measure with `scripts/evaluate.py` and quote those numbers.

## Evaluation

```bash
python scripts/evaluate.py yolo --weights runs/lablens/weights/best.pt --data data.yaml  # mAP@0.5 (needs ultralytics)
python scripts/evaluate.py pointing data/eval/pointing.csv     # correct selections out of N trials
python scripts/evaluate.py latency -n 20                       # time to first token, Groq vs Ollama
python scripts/evaluate.py grounding data/eval/grounding.csv  # cited-page accuracy, RAG vs plain LLM
```

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

56 tests cover pointing geometry, gestures, query fusion and follow-ups, retrieval and the
extractive fallback, TTS sentence streaming, Guided Build verification, the ONNX writer and exporter,
NMS / letterbox / anchors, Weighted Box Fusion, the breadboard detector, and the resistor reader
(6 values × 4 angles). Model tests are skipped until `setup_models.py` has run. CI runs on
Python 3.10–3.12 (`.github/workflows/tests.yml`).

## Scope and limits

* Guided Build verifies **presence, count and approximate placement**, not wiring or electrical
  connectivity.
* The knowledge base ships with an original lab manual (`knowledge/lab_manual.md`: breadboards,
  resistors, LEDs, capacitors, diodes, BC547/2N2222, NE555, LM358, Arduino). Drop manufacturer
  datasheet PDFs into `data/datasheets/` and run `python scripts/build_manual.py` to add them.
* Hindi and Kannada answers use the LLM's translation plus gTTS (online). Check technical terms.
* The datasheet scraper relies on public search results. Use it for personal lab work and respect
  each site's terms.

## Layout

```
setup_windows.bat / run_lablens.bat / setup.sh    one-click setup and launch
app.py                     Streamlit UI
config.yaml                every tunable (models, thresholds, LLM, audio)
knowledge/lab_manual.md    source of the lab manual PDF
lablens/
  runtime.py               camera + detector + hands + guided threads; one API for every UI
  live.py                  desktop window UI
  doctor.py                pre-demo self-check
  vision/                  ONNX YOLO, ensemble (WBF), torch-free exporter, breadboard CV,
                           hands, pointing, gestures, resistor reader, overlay
  audio/                   STT (Groq / faster-whisper), TTS (pyttsx3 / Piper / gTTS)
  rag/                     ingest, hybrid store + extractive answers, datasheet scraper
  llm/                     Groq + Ollama backends, router, prompts
  core/                    query composer, answer cache, orchestrator
  guided/                  step verification engine + projects/*.json
  dashboard/               event log + stuck-step report
scripts/                   setup_models, build_manual, demo, train_yolo, evaluate
tests/                     unit + model tests
```
