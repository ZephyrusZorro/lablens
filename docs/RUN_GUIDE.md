# LabLens: run and operate

How to install, run, operate and demo LabLens. The jobs only you can do are in
[MANUAL_STEPS.md](MANUAL_STEPS.md).

---

## 1. What you need

| | Requirement |
|---|---|
| Computer | Windows 10/11 laptop (Linux and macOS work too). 8 GB RAM. No GPU needed. |
| Python | **3.10, 3.11 or 3.12** (3.11 recommended). MediaPipe does not support 3.13 yet. |
| Camera | Any USB webcam or the laptop camera. A top-down mount over the breadboard is best. |
| Microphone / speakers | Laptop built-ins are fine. A headset mic helps in a noisy hall. |
| Internet | Once, for setup (roughly 1 GB of downloads). Afterwards LabLens runs fully offline. |

---

## 2. One-time setup (about 10 minutes)

1. Open the `lablens` folder.
2. Double-click **`setup_windows.bat`**.

It will:

1. find Python 3.10–3.12 and create a private environment in `.venv\`
2. install the dependencies from `requirements.txt` (no PyTorch)
3. download the two open-source component detectors from GitHub, check their SHA-256 and convert
   them to ONNX (`models\`)
4. download the MediaPipe hand model and the offline speech-to-text model (Whisper `base.en`)
5. download the sample bench photo (`samples\bench_demo.png`)
6. build the lab-manual PDF and its search index (`data\`)
7. run the self-check (`lablens.doctor`) and print **READY** or what is missing

Re-running it is safe: anything already downloaded and verified is kept.

### Optional, but recommended before the event

* **Fast online answers:** paste a free Groq API key into `.env` as `GROQ_API_KEY=...`.
  See [MANUAL_STEPS.md](MANUAL_STEPS.md) §3.
* **Offline AI answers:** install Ollama and run `ollama pull llama3.2:3b`. See §4 there.

Without either, LabLens still answers, by quoting the most relevant manual sentences with their
page numbers.

---

## 3. Every time: start LabLens

| Double-click / run | What opens | When to use it |
|---|---|---|
| `run_lablens.bat` | Desktop window with the live webcam | **The live demo** (lowest latency) |
| `run_lablens.bat web` | Browser UI with the cited datasheet page beside each answer | Showing the citation page on a projector |
| `run_lablens.bat photo` | Desktop window on the sample photo; point with the mouse | **Backup demo** if the camera fails |
| `run_lablens.bat demo` | Runs once on the sample photo and opens `demo_out\demo_annotated.jpg` | Quick proof that everything works |
| `run_lablens.bat doctor` | Self-check | **Before every demo** |
| `run_lablens.bat eval` | Grounding numbers | Numbers for your slides |

If the wrong camera opens, change `camera_index` in `config.yaml` (0, 1, 2…) and run `doctor` again.

---

## 4. Operating the desktop window

### Asking about a part

1. **Point** at a component with your index finger. A ray is drawn from your finger. When the same
   part stays under it for half a second, its box turns green and shows **LOCKED**.
   You can lower your hand: the lock stays for about 4 seconds.
2. **Ask:** press `SPACE` and speak (it stops when you stop talking), or press `ENTER`, type, then
   `ENTER` again.
3. LabLens answers out loud. The text and the cited page (for example `[LabLens_Lab_Manual.pdf p.3]`)
   appear at the bottom.

Follow-ups keep the context: after "what is this 555?", "and pin 4?" still refers to that 555.

**Say the number printed on a chip.** The camera knows a part's *type* (IC, LED…), not its part number,
and several ICs share the "IC" class. "this 555", "LM358", "five five five" all work. Without a number,
LabLens asks which chip it is (with an LLM) or quotes both candidates (without one).

No hand tracking? **Hold the left mouse button** on a part: the cursor acts as your fingertip.

### Keys and gestures

| Key | Gesture | Action |
|---|---|---|
| `SPACE` | | Ask by voice |
| `ENTER` | | Type a question |
| `1` `2` `3` `4` | | Start Guided Build: 1 diode test · 2 NE555 blinker · 3 LED + resistor · 4 transistor switch |
| `N` | 👍 thumbs up (hold it still briefly) | Next step |
| `R` | ✋ open palm | Repeat the step / last answer |
| `S` | ✊ fist | Stop talking |
| `X` | | Stop the guided project |
| `Q` / `Esc` | | Quit |

### Guided Build

1. Press `3` (LED + resistor) for the quickest run.
2. LabLens speaks step 1. Do it on the bench.
3. When the camera sees the step is done, it says **"Looks good"**. Give 👍 or press `N`.
4. If something is missing it says so, for example *"I don't see the resistor yet."*
   A **second 👍 within 6 seconds** skips the check (for when the camera is wrong, not you).

What it checks: which parts are present, how many, and roughly where they sit on the board.
What it does **not** check: the wiring. Say this to the judges up front; it builds trust.

### The status line

The top banner shows `det` fps (detector, runs in the background, about 3–5 fps on a laptop CPU),
`cam` fps (should be 25–30), the hand-tracking backend, what you are pointing at, and the resistor
band reading when you point at a resistor.

---

## 5. Two-minute demo script

Run `run_lablens.bat doctor` 10 minutes before. Then start `run_lablens.bat`.

| # | You do / say | LabLens does | Point to make |
|---|---|---|---|
| 1 | Point at the **IC**: "What is this 555 and which pin is ground?" | Locks on the IC; answers pin 1 GND, citing manual p.7 | "The fingertip tells it *which* part; the voice says *what* I want." |
| 1b | Still pointing: "And what does pin 4 do?" | RESET, active low, p.7 | It remembers the context across follow-ups |
| 2 | Point at the **LED**: "Which leg is positive?" | Long leg = anode, citing p.3 | Same question about a different part gives a different, correct answer |
| 3 | Point at the **capacitor**: "Which leg is positive?" | Stripe marks negative, citing p.4 | Pointing raises retrieval accuracy from 50% to 81% (see §7) |
| 4 | Pull the Wi-Fi / switch to offline, ask again | Still answers (Ollama or cited extracts) | Works when lab Wi-Fi dies |
| 5 | Press `3`, place parts, **skip the resistor**, give 👍 | "I don't see the resistor yet." | Closed-loop: it checks your work before moving on |
| 6 | Place the resistor | "Looks good", then 👍 to finish | |
| 7 | Close | | "The same pipeline works for any manual-driven hands-on work: mechanics, lab instruments, appliance repair." |

**If something fails live:**

| Problem | Fallback |
|---|---|
| Camera dies | `run_lablens.bat photo`; point with the mouse |
| Hand tracking jittery | Hold the left mouse button to point |
| Mic doesn't catch you | `ENTER` and type the question |
| A part isn't detected | Say "the detector is pre-trained on public data; with photos of our own kit it improves", then point at a part it does see |
| No internet | Nothing to do: it falls back automatically |

---

## 6. Demo data in this project

| Path | What it is |
|---|---|
| `knowledge/lab_manual.md` | Source of the 14-page lab manual: breadboard, resistors, LEDs, capacitors, diodes, BC547/2N2222, NE555, LM358, Arduino, safety, 4 guided projects |
| `data/datasheets/LabLens_Lab_Manual.pdf` | The rendered manual (one topic per page, so citations are exact) |
| `lablens/guided/projects/*.json` | The 4 Guided Build projects |
| `samples/bench_demo.png` | Real bench photo used by `photo` and `demo` modes (downloaded at setup) |
| `eval/grounding.csv` | 40 hand-labelled questions with the page that answers each, 16 asked while pointing |
| `eval/pointing_template.csv` | Template for your own pointing-accuracy trials |

### Questions known to work well

Pointing at an **IC**: "Which pin is ground on this 555?", then "And what does pin 4 do?", "Which pin is the output?",
"How do I make it blink once a second?"
Pointing at an **LED**: "Which leg is positive?", "What resistor do I need for this?", "How much current can this take?"
Pointing at a **capacitor**: "Which leg is positive?", "What voltage rating should I pick?"
Pointing at a **resistor**: "How do I read the value?", "Does it matter which way round it goes?"
Pointing at a **transistor**: "What are the pins?", "What base resistor should I use?"
Without pointing: "Which PWM pins does the Arduino have?", "Something is getting hot, what do I do?",
"What does 104 mean on a ceramic capacitor?"

### Adding your own knowledge

* **Datasheets:** drop PDFs into `data\datasheets\`, then run
  `.venv\Scripts\python scripts\build_manual.py`.
* **Your college's lab manual:** same, or edit `knowledge/lab_manual.md` (each `## ` heading becomes one
  page) and rebuild.
* **New guided project:** copy a file in `lablens/guided/projects/` and edit the steps. `expect` lists
  parts and counts, `zone` is the area of the board (0–1 from top-left), and `near` says which parts
  must sit together.

---

## 7. Numbers you can show (measured, reproducible)

Run `run_lablens.bat eval`:

| Metric | Result | Notes |
|---|---|---|
| Correct page cited first | **88%** (35/40) | `eval/grounding.csv`, no LLM involved |
| Correct page in the top 4 | **100%** | |
| Pointing questions: correct page first | **81% with pointing vs 50% without** | 16 questions like "which leg is positive?" |
| Camera loop | ~29 fps | Detector runs in a background thread |
| Detector | ~3–4 fps (~280 ms per frame) | 2-core CPU, both models; faster on a laptop |
| Detector accuracy | mAP@0.5 0.71 / 0.78 | **Reported by the model authors on their own data**, not yet measured on your kit |

Be honest about the last row. Measuring your own mAP and pointing accuracy is in MANUAL_STEPS.md §6–7,
and makes the strongest slide.

The questions were written together with the manual, so these are internal benchmark numbers. Say so.

---

## 8. Instructor report

Every question and step check is logged in `data\lablens_events.sqlite`. To see which steps students
get stuck on:

```
.venv\Scripts\python -m lablens.cli report
```

Give each lab bench its own id before starting: `set LABLENS_BENCH=bench-07`.

---

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `setup_windows.bat` says Python not found | Install Python 3.11 from python.org (tick "Add to PATH"), then re-run |
| Setup fails downloading the detectors | GitHub is blocked on that network: use a phone hotspot for setup |
| Doctor: `camera … not readable` | Close other apps using the camera (Zoom, Teams); try `camera_index: 1` |
| Doctor: `hand tracking unavailable` | Re-run setup (downloads `models\hand_landmarker.task`); mouse pointing still works |
| Doctor: `LLM … extractive` | Fine for the demo; add a Groq key or start Ollama for full answers |
| Answers are robotic quotes | That is the no-LLM fallback: add a Groq key or run Ollama |
| No speech output | Windows Settings → Sound: check the output device; `audio.tts_backend: pyttsx3` in `config.yaml` |
| Parts flicker or aren't detected | More light, less glare, camera straight down; raise `vision.ensemble.slice_grid` to 2 for small parts (slower) |
| Box locks onto the wrong part | Point closer; increase `pointing.dwell_seconds` in `config.yaml` |
| Web UI looks stale | Use the desktop window for live demos; the web UI refreshes the video ~10×/s |

---

## 10. Developer commands

```
.venv\Scripts\activate
pip install -r requirements-dev.txt
python -m pytest -q tests                  # 58 tests
python -m lablens.cli ask "which pin is ground on the NE555?"
python scripts\evaluate.py grounding -v
python scripts\demo.py path\to\photo.jpg --question "what is this?"
```
