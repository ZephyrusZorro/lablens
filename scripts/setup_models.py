"""One-time setup: download the open-source component detectors (pinned commit + SHA-256), convert
them to ONNX with the torch-free exporter, fetch the MediaPipe hand model, and build the knowledge base.

    python scripts/setup_models.py            # everything
    python scripts/setup_models.py --skip-hands

Nothing here needs PyTorch. Re-running is safe (existing verified files are kept).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"

DETECTORS = [
    {
        "name": "components_v8n_6cls",
        "url": "https://github.com/hassanmroueh2003/Electronic-Component-Detection-using-YOLOv8/raw/"
               "e7f506c55347072e6dd997d6e4aff7c636067367/best.pt",
        "sha256": "af49b3d6246947bfddbdb7013ba1786e45f868f8d13e01f17f969c0d9a81df71",
        "credit": "Hassan Mroueh - github.com/hassanmroueh2003/Electronic-Component-Detection-using-YOLOv8 "
                  "(YOLOv8n trained on the Roboflow 'Electronics components' dataset by Jovine)",
    },
    {
        "name": "components_v8s_4cls",
        "url": "https://github.com/ravi-ranjan707089/ComponentDetector/raw/"
               "65cdd37079cd97723e4f1f3a2b4fe6caecab4775/best.pt",
        "sha256": "7aaa40dde33b9cf166b1cdee0dcdb2a6c3825febff5d2e0cef807f504a98b8e8",
        "credit": "Ravi Ranjan - github.com/ravi-ranjan707089/ComponentDetector (YOLOv8s, Roboflow dataset)",
    },
]
HAND_TASK_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
                 "hand_landmarker.task")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "LabLens-setup"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            b = r.read(1 << 16)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total:
                print(f"\r  {dest.name}: {100 * done // total:3d}% of {total / 1e6:.1f} MB", end="", flush=True)
    print()
    tmp.replace(dest)


def setup_detectors() -> None:
    from lablens.vision.export import export

    card = ["# Detection models used by LabLens", "",
            "These weights are third-party work, downloaded at setup time from their original repositories",
            "(pinned commits, SHA-256 verified) and converted to ONNX locally. They are not redistributed",
            "in this repository. Credit:", ""]
    for m in DETECTORS:
        pt, onnx = MODELS / f"{m['name']}.pt", MODELS / f"{m['name']}.onnx"
        if onnx.exists() and Path(str(onnx) + ".json").exists():
            print(f"[ok] {onnx.name} already present")
        else:
            if not (pt.exists() and sha256(pt) == m["sha256"]):
                print(f"[download] {m['name']}")
                download(m["url"], pt)
                if sha256(pt) != m["sha256"]:
                    pt.unlink()
                    raise SystemExit(f"checksum mismatch for {m['name']} - refusing to use it")
            info = export(pt, onnx, {"source": m["url"], "credit": m["credit"]})
            tm = info.get("train_metrics") or {}
            print(f"[converted] {onnx.name}: classes={list(info['names'].values())} "
                  f"reported mAP50={tm.get('metrics/mAP50(B)')}")
        card.append(f"- **{m['name']}**: {m['credit']}")
    (MODELS / "MODEL_CARD.md").write_text("\n".join(card) + "\n", encoding="utf-8")


def setup_hands() -> None:
    dest = MODELS / "hand_landmarker.task"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print("[ok] hand_landmarker.task already present")
        return
    print("[download] MediaPipe hand landmarker")
    try:
        download(HAND_TASK_URL, dest)
    except Exception as e:
        print(f"[warn] could not download hand model ({e}); the legacy MediaPipe API or mouse pointing will be used")


SAMPLE = {
    # Real bench photo from the model author's repo (pinned commit, SHA-256), used by `run_lablens.bat demo`.
    # It is a phone-screen capture that already shows the author's own boxes; it is a smoke test, not a benchmark.
    "url": "https://github.com/hassanmroueh2003/Electronic-Component-Detection-using-YOLOv8/raw/"
           "e7f506c55347072e6dd997d6e4aff7c636067367/simulation-images/simulation-2-electronic-components.png",
    "sha256": "96e84686a8f4dc25eaa6fe4bcaf9fb449324beb66c475d01eecceaf32fc4ae94",
    "dest": ROOT / "samples" / "bench_demo.png",
}


def setup_sample() -> None:
    dest = SAMPLE["dest"]
    if dest.exists() and sha256(dest) == SAMPLE["sha256"]:
        print("[ok] sample photo already present")
        return
    try:
        print("[download] sample bench photo")
        download(SAMPLE["url"], dest)
        if sha256(dest) != SAMPLE["sha256"]:
            dest.unlink()
            print("[warn] sample photo checksum mismatch - skipped")
    except Exception as e:
        print(f"[warn] could not download the sample photo ({e}); `run_lablens.bat demo` needs an image path")


def setup_speech() -> None:
    """Cache the offline Whisper model now, so speech works later with no internet."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("[skip] faster-whisper not installed - offline speech-to-text disabled")
        return
    from lablens.config import get_config

    name = get_config().get("audio.whisper_local_model", "base.en")
    print(f"[download] offline speech model ({name}, ~150 MB, cached for offline use)")
    try:
        WhisperModel(name, device="cpu", compute_type="int8")
        print(f"[ok] {name} cached")
    except Exception as e:
        print(f"[warn] could not cache {name} ({e}); voice questions will need Groq (online) until it is")


def setup_knowledge() -> None:
    import runpy

    runpy.run_path(str(ROOT / "scripts" / "build_manual.py"), run_name="__main__")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-hands", action="store_true")
    ap.add_argument("--skip-knowledge", action="store_true")
    ap.add_argument("--skip-speech", action="store_true")
    a = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    try:
        setup_detectors()
    except (OSError, ValueError) as e:
        raise SystemExit(f"[error] could not download the component detectors: {e}\n"
                         "        GitHub must be reachable for this one-time step (college Wi-Fi sometimes blocks it;"
                         " try a phone hotspot), then run this script again.")
    if not a.skip_hands:
        setup_hands()
    setup_sample()
    if not a.skip_speech:
        setup_speech()
    if not a.skip_knowledge:
        setup_knowledge()
    print("\nSetup complete. Run:  python -m lablens.live   or   streamlit run app.py")


if __name__ == "__main__":
    main()
