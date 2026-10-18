# LabLens: what needs a human

Everything here needs your accounts, your computer, your hardware or your judgement. For each item: **why**
it couldn't be automated, **exactly what to do**, and **how long** it takes. Work top to bottom.

**Already done for you:** all code, the knowledge base, 4 guided projects, one-click Windows setup and
launch scripts, the self-check, 58 passing tests, CI, retrieval evaluation (88% correct page first),
model download and conversion (verified), sample demo data, and a local git commit in this folder.

---

## Before the event: must do

### 1. Run setup on your laptop and check it works · 15 min · MUST

**Why:** I built and tested everything on Linux (Python 3.10 and 3.13). I could not run the Windows
`.bat` files, a real webcam, a microphone, speakers or MediaPipe hand tracking.

1. Double-click `setup_windows.bat` (needs internet). If it says Python is missing, see step 2.
2. Run `run_lablens.bat doctor`. Every line should be `OK` (`WARN` for LLM is fine until steps 3–4).
3. Run `run_lablens.bat demo`: an annotated photo opens and the answer cites the manual.
4. Run `run_lablens.bat`: point at a part, check it **LOCKS**, press `SPACE`, ask "which leg is positive?"
5. Test 👍 ✋ ✊ in front of the camera.

If anything fails, copy the red error text: it names the file and line.

### 2. Install Python 3.11 (only if setup says so) · 5 min

**Why:** installing software on Windows needs your approval.

Get it from <https://www.python.org/downloads/> and **tick "Add python.exe to PATH"**. Python 3.13 does
not work yet (MediaPipe has no 3.13 build).

### 3. Get a free Groq API key (fast online answers + speech) · 3 min · strongly recommended

**Why:** creating accounts and API keys needs you; I am not allowed to.

1. Sign in at <https://console.groq.com> → **API Keys** → **Create API Key**.
2. Open `.env` in this folder with Notepad and set `GROQ_API_KEY=gsk_...` (no quotes, no spaces).
3. Never commit `.env` or paste the key anywhere public. `.gitignore` already excludes it.

Without a key: answers come from Ollama (step 4) or as quoted manual sentences.

### 4. Install Ollama for offline answers · 10 min (≈2 GB download) · recommended

**Why:** installing desktop software needs you.

1. Install from <https://ollama.com/download>. Your PC has an `.ollama` folder, so it may already be there.
2. In a terminal: `ollama pull llama3.2:3b`
3. `run_lablens.bat doctor` should now show `ollama llama3.2:3b`.

This is what makes the "pull the Wi-Fi and it still answers properly" moment work.

### 5. Push to GitHub (`zephruszorro/lablens`) · 2 min · MUST if judges want a link

**Why I couldn't:** the GitHub token in my environment was invalid, my GitHub connection only covers the
`codingclub-aitm` organisation, and the browser on your PC isn't signed in to GitHub. I won't handle your
password. Everything is already committed locally (branch `main`).

In a terminal opened in this `lablens` folder, either:

```
gh auth login
gh repo create lablens --private --source . --push
```

or create an **empty** repo named `lablens` at <https://github.com/new> (no README, no .gitignore), then:

```
git remote add origin https://github.com/zephruszorro/lablens.git
git push -u origin main
```

Then, optionally:

* make it public: repo **Settings → General → Change visibility**
* add a description and the topics `computer-vision`, `rag`, `education`, `yolov8`
* CI runs automatically (tab **Actions**). It downloads the models from GitHub and runs the tests on
  Python 3.10–3.12.

The model weights are **not** committed, on purpose (see step 10). Setup downloads them.

### 6. Physical bench setup · 30 min · MUST

**Why:** hardware.

* Mount the camera **looking straight down** about 35–50 cm above the breadboard (phone stand, desk
  lamp arm, or tape it to a shelf). Side views roughly halve detection quality.
* Put a desk lamp on the bench and avoid glare from overhead lights on the shiny IC.
* Use a plain, matte background around the breadboard.
* Rehearse the 2-minute script in [RUN_GUIDE.md §5](RUN_GUIDE.md) **at least 3 times**, including
  the camera-failure fallback (`run_lablens.bat photo`).
* Bring a phone hotspot in case the venue Wi-Fi blocks Groq or GitHub.

---

## To raise accuracy (the biggest win for judging)

### 7. Photograph and label your own kit, then fine-tune · 3–4 h · high impact

**Why I couldn't:** it needs photos of *your* parts under *your* lighting, plus a GPU or Colab to train.
Dataset hosts (Roboflow, Hugging Face) were blocked from my environment, and PyTorch couldn't be
installed there.

The detectors are trained on other people's photos. They report mAP@0.5 of 0.71 and 0.78 on *their*
data, and on sample photos I saw both misses and wrong labels. Your own data fixes this.

1. Take **150–300 photos** from the mounted camera: different part placements, 1–8 parts per
   photo, hands in some shots, a few with the lamp off.
2. Label them in Roboflow (free) or CVAT with these exact class names:
   `resistor, led, capacitor, ic, diode, transistor`.
3. Export in **YOLOv8** format and download the zip.
4. Train, either:
   * **Google Colab (free GPU, ~30 min):** upload this repo and your dataset, then run
     `pip install ultralytics` and `python scripts/train_yolo.py --data <dataset>/data.yaml`
   * **Your laptop (CPU, a few hours):** `pip install -r requirements-train.txt`, then the same command.
5. Copy the resulting `models/my_kit.pt` into this folder's `models\`, and add it to `config.yaml` under
   `vision: models:` (the script prints the exact lines; give it `weight: 1.5`). It converts to ONNX
   by itself on first start.
6. Measure it: hold back ~30 labelled photos and run
   `python scripts/evaluate.py yolo --weights models/my_kit.pt --data <dataset>/data.yaml`.
   Put that mAP on your slide: "trained on our own lab's components".

### 8. Measure pointing accuracy · 30 min · high impact

**Why:** needs your hand, your camera and your parts.

1. Take 50 photos from the demo camera, each with your finger pointing at one part.
2. Save them in `eval\pointing\` and list them in `eval\pointing.csv` (copy `eval/pointing_template.csv`):
   `eval/pointing/ic_01.jpg,ic`
3. Run `python scripts/evaluate.py pointing eval/pointing.csv` → "Pointing accuracy: 46/50 = 92%".

### 9. Measure voice latency · 5 min

**Why:** needs your Groq key (step 3) and Ollama (step 4) on your machine.

Run `python scripts/evaluate.py latency -n 20`. It prints time-to-first-token for Groq and for Ollama.

---

## Decisions only you can make

### 10. Model licensing · decide before any public or commercial use

The two detector models come from GitHub users **hassanmroueh2003** and **ravi-ranjan707089**. Neither
repository states a licence, which by default means all rights reserved. What I did: the weights are
**not** in this repo. `setup_models.py` downloads them from the authors' repos at pinned commits (SHA-256
checked) for your own use, and the README credits both authors.

* For a college event this is normally fine. Credit them on a slide.
* For anything beyond that, open an issue on their repos asking for a licence, **or** use only your own
  model from step 7 (remove theirs from `config.yaml`).

### 11. Choose a licence for your code · 2 min

I did not pick one, because it's your decision. If you want others to reuse LabLens, add a `LICENSE`
file. On GitHub: **Add file → Create new file → name it LICENSE → Choose a license template** (MIT is
the common choice for student projects).

### 12. Check the lab manual against your lab · 20 min

`knowledge/lab_manual.md` holds typical values, which I checked against standard datasheet figures
(BC547 C-B-E, 2N2222 E-B-C, NE555 and LM358 pinouts, 555 timing, Arduino limits). Your lab's parts or
procedures may differ:

* Ask your instructor for the official lab manual or experiment sheets (PDF) and drop them into
  `data\datasheets\`, then run `python scripts/build_manual.py`.
* Add the datasheets for the exact part numbers in your kit (from the manufacturer, e.g. ti.com, onsemi.com).

### 13. Hindi / Kannada voice · 10 min · optional

Set `language: hi` or `kn` under `audio:` in `config.yaml`. This needs internet (gTTS) and the Groq
or Ollama LLM for translation. Have a native speaker check the technical terms before showing it.

---

## Things I couldn't verify (be ready for these)

| Area | Status | What to watch |
|---|---|---|
| Windows `.bat` files | Written for Windows cmd (CRLF), never executed | Step 1 |
| Live webcam + MediaPipe hands | Logic unit-tested and run on a photo with simulated pointing; not on a real camera | Pointing jitter: raise `pointing.dwell_seconds` |
| Microphone / Whisper / TTS voices | Code paths written; no audio hardware in my environment | `doctor`, then `SPACE` |
| Groq / Ollama answers | Router and fallbacks tested; no LLM reachable from my environment | Answers in LLM mode were not reviewed by me: try the demo questions |
| Detector on your bench | Ran on the authors' sample photos only | Step 7 |
| Resistor colour bands | Passes 24 synthetic tests; untested on real photos | Treat readings as a hint, not truth |
| Datasheet scraper | Code written; web search was blocked in my environment | Don't rely on it in the demo |
| Streamlit web UI | Made compatible with old and new Streamlit APIs; not launched | Use the desktop window for the live demo |
