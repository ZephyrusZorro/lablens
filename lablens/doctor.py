"""LabLens self-check: run before a demo.

    python -m lablens.doctor            # everything
    python -m lablens.doctor --no-camera

Prints OK / WARN / FAIL per subsystem. Exit code 1 only if a core part (detector, knowledge base) fails;
WARN means a feature degrades gracefully (e.g. no Groq key -> Ollama or offline extractive answers).
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from typing import Callable, List, Tuple

import numpy as np

RESULTS: List[Tuple[str, str, str]] = []


def check(name: str, core: bool = False):
    def deco(fn: Callable[[], Tuple[str, str]]):
        def run():
            try:
                status, detail = fn()
            except Exception as e:  # a check must never crash the doctor
                status, detail = ("FAIL" if core else "WARN"), f"{type(e).__name__}: {e}"
            RESULTS.append((status, name, detail))
            print(f"[{status:4s}] {name:18s} {detail}", flush=True)
        run.core = core
        return run
    return deco


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-camera", action="store_true")
    a = ap.parse_args(argv)

    from lablens.config import get_config

    cfg = get_config()

    @check("python")
    def _py():
        v = sys.version_info
        ok = (3, 10) <= (v.major, v.minor) <= (3, 12)
        return ("OK" if ok else "WARN"), f"{platform.python_version()} on {platform.system()}" + (
            "" if ok else " (MediaPipe needs 3.10-3.12)")

    @check("detector", core=True)
    def _det():
        from lablens.vision.detector import MockDetector, build_detector

        det = build_detector(cfg)
        if isinstance(det, MockDetector):
            return "FAIL", "no models found - run scripts/setup_models.py"
        frame = np.full((720, 1280, 3), 128, np.uint8)
        det.detect(frame)
        t = time.perf_counter()
        det.detect(frame)
        ms = (time.perf_counter() - t) * 1000
        classes = ", ".join(getattr(det, "classes", []))
        return ("OK" if ms < 600 else "WARN"), f"{ms:.0f} ms/frame on CPU; classes: {classes}"

    @check("hand tracking")
    def _hands():
        from lablens.vision.hands import HandTracker

        h = HandTracker(str(cfg.path("vision.hand_model", "models/hand_landmarker.task")))
        b = h.backend
        h.close()
        return ("OK" if b != "none" else "WARN"), b if b != "none" else "unavailable - mouse pointing still works"

    @check("knowledge base", core=True)
    def _kb():
        from lablens.rag.store import build_store

        st = build_store(cfg)
        if not st.chunks:
            return "FAIL", "empty - run scripts/build_manual.py"
        hits = st.search("which pin is ground on the NE555", 1, part_number="NE555")
        top = f"{hits[0].source} p.{hits[0].page}" if hits else "no hit"
        return "OK", f"{len(st.chunks)} chunks from {len(st.sources())} file(s); test query -> {top}"

    @check("LLM")
    def _llm():
        from lablens.llm.router import LLMRouter
        from lablens.net import is_online

        r = LLMRouter.from_config(cfg)
        g, o = r.backends["groq"], r.backends["ollama"]
        parts = []
        groq_ok = g.configured and is_online()
        parts.append("groq " + ("ready" if groq_ok else ("no GROQ_API_KEY" if not g.configured else "offline")))
        try:
            names = o.installed()
            parts.append(f"ollama {o.resolve_model()}" if names else "ollama running but no models")
            ollama_ok = bool(names)
        except Exception:
            parts.append("ollama not running")
            ollama_ok = False
        status = "OK" if (groq_ok or ollama_ok) else "WARN"
        if status == "WARN":
            parts.append("-> answers will be extractive quotes from the manual")
        return status, "; ".join(parts)

    @check("speech-to-text")
    def _stt():
        import importlib.util as u

        have_fw = u.find_spec("faster_whisper") is not None
        have_groq = u.find_spec("groq") is not None and bool(cfg.groq_api_key)
        if not (have_fw or have_groq):
            return "WARN", "no STT backend - type questions instead"
        return "OK", ", ".join(x for x, ok in (("faster-whisper (offline)", have_fw), ("groq whisper", have_groq)) if ok)

    @check("microphone")
    def _mic():
        import sounddevice as sd

        d = sd.query_devices(kind="input")
        return "OK", d["name"]

    @check("text-to-speech")
    def _tts():
        import importlib.util as u

        backs = [n for n in ("pyttsx3", "gtts") if u.find_spec(n)]
        return ("OK" if backs else "WARN"), ", ".join(backs) or "none - answers shown as text only"

    checks = [_py, _det, _hands, _kb, _llm, _stt, _mic, _tts]

    if not a.no_camera:
        @check("camera")
        def _cam():
            import cv2

            idx = cfg.get("vision.camera_index", 0)
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(idx, backend)
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            cap.release()
            if not ok:
                return "WARN", f"camera {idx} not readable - try another vision.camera_index in config.yaml"
            return "OK", f"camera {idx}: {frame.shape[1]}x{frame.shape[0]}"
        checks.append(_cam)

    print("LabLens doctor\n")
    for c in checks:
        c()
    fails = [n for s, n, _ in RESULTS if s == "FAIL"]
    warns = [n for s, n, _ in RESULTS if s == "WARN"]
    print()
    if fails:
        print("NOT READY: fix " + ", ".join(fails))
        return 1
    print("READY" + (f" (degraded: {', '.join(warns)})" if warns else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
