"""LabLens runtime: one object that owns the whole live pipeline.

    camera thread:  frame ─► hands (every frame) ─► pointing resolver ─► gestures ─► guided checks
                       └──► AsyncDetector (background thread, newest frame only)
    ask thread:     question + locked component ─► LabLens (RAG + LLM router) ─► streamed TTS

UIs (Streamlit app, desktop window, tests) only call the thread-safe methods below and read
`snapshot()`. Source can be a camera index, a video file, or a still image (looped) for demos.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Union

import cv2
import numpy as np

from lablens.types import Detection, HandObservation, Selection

log = logging.getLogger(__name__)
Source = Union[int, str]


@dataclass
class Snapshot:
    frame: Optional[np.ndarray] = None  # annotated BGR frame
    detections: List[Detection] = field(default_factory=list)
    selection: Optional[Selection] = None
    gesture: str = "none"
    hand_backend: str = "none"
    detector_fps: float = 0.0
    camera_fps: float = 0.0
    guided: Optional[dict] = None
    busy: bool = False
    resistor_reading: Optional[str] = None


class FrameSource:
    """Camera index, video file or image file, with a uniform read()."""

    def __init__(self, source: Source, width: int = 1280, height: int = 720):
        self.image: Optional[np.ndarray] = None
        self.cap = None
        if isinstance(source, str) and not source.isdigit() and Path(source).suffix.lower() in (
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            self.image = cv2.imread(source)
            if self.image is None:
                raise FileNotFoundError(source)
            return
        src = int(source) if isinstance(source, str) and source.isdigit() else source
        backend = cv2.CAP_DSHOW if isinstance(src, int) and hasattr(cv2, "CAP_DSHOW") and _is_windows() else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(src, backend)
        if isinstance(src, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source {source!r}")
        self.is_file = not isinstance(src, int)

    def read(self) -> Optional[np.ndarray]:
        if self.image is not None:
            time.sleep(1 / 30)
            return self.image.copy()
        ok, frame = self.cap.read()
        if not ok and getattr(self, "is_file", False):  # loop video files
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()


def _is_windows() -> bool:
    import sys

    return sys.platform == "win32"


class Runtime:
    def __init__(self, cfg, source: Optional[Source] = None, speak: bool = True, start: bool = True,
                 detector=None, lens=None):
        from lablens.core.assistant import LabLens
        from lablens.guided.engine import list_projects
        from lablens.vision.detector import AsyncDetector, build_detector
        from lablens.vision.gestures import GestureClassifier, GestureDebouncer
        from lablens.vision.hands import HandTracker
        from lablens.vision.pointing import PointingResolver

        self.cfg = cfg
        self.source = cfg.get("vision.camera_index", 0) if source is None else source
        self.detector = detector or build_detector(cfg)
        self.async_det = AsyncDetector(self.detector)
        self.hands = HandTracker(str(cfg.path("vision.hand_model", "models/hand_landmarker.task")))
        self.pointer = PointingResolver.from_config(cfg)
        self.gesture_clf = GestureClassifier(cfg.path("gestures.model", "models/gesture_knn.joblib"))
        self.debounce = GestureDebouncer(cfg.get("gestures.hold_frames", 6), cfg.get("gestures.cooldown_seconds", 1.5))
        self.lens = lens or LabLens.from_config(cfg)
        self.speaker = None
        if speak:
            from lablens.audio.tts import Speaker

            self.speaker = Speaker(cfg)
        self.projects = list_projects(cfg.path("guided.projects_dir", "lablens/guided/projects"))
        self.guided = None
        self.messages: Deque[Tuple[str, str]] = collections.deque(maxlen=200)
        self.last_citations = []
        self._lock = threading.Lock()
        self._snap = Snapshot(hand_backend=self.hands.backend)
        self._mouse: Optional[Tuple[float, float]] = None
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._last_det_update = 0.0
        self._last_answer = ""
        self._resistor_cache: dict = {}
        self._thread: Optional[threading.Thread] = None
        if start:
            self.start()

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._camera_loop, daemon=True, name="lablens-camera")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.async_det.close()
        self.hands.close()

    # ------------------------------------------------------------------ UI API (thread-safe)
    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._snap

    def set_mouse(self, x: Optional[float], y: Optional[float] = None) -> None:
        """Mouse pointing: a virtual fingertip at (x, y). Pass None to release."""
        self._mouse = None if x is None else (float(x), float(y))

    def say(self, text: str, role: str = "lablens") -> None:
        self.messages.append((role, text))
        if self.speaker:
            self.speaker.say(text)

    def stop_talking(self) -> None:
        if self.speaker:
            self.speaker.stop()

    def ask(self, question: str, block: bool = False):
        """Ask about whatever is locked under the finger. Non-blocking by default."""
        question = question.strip()
        if not question:
            return None
        self.messages.append(("you", question))
        t = threading.Thread(target=self._answer, args=(question,), daemon=True, name="lablens-ask")
        t.start()
        if block:
            t.join()
        return t

    def listen_and_ask(self) -> None:
        def run():
            from lablens.audio.stt import SpeechToText

            self.messages.append(("lablens", "🎤 listening…"))
            try:
                q = SpeechToText(self.cfg).listen()
            except Exception as e:
                self.messages.append(("lablens", f"Microphone/STT error: {e}"))
                return
            if q:
                self.ask(q)
            else:
                self.messages.append(("lablens", "I didn't catch that."))

        threading.Thread(target=run, daemon=True, name="lablens-listen").start()

    def start_project(self, project_id: str) -> str:
        from lablens.guided.engine import GuidedSession

        self.guided = GuidedSession(self.projects[project_id], events=self.lens.events,
                                    value_reader=self._read_resistor_value)
        msg = f"Starting {self.guided.project.title}. {self.guided.step_label}"
        self.say(msg)
        return msg

    def stop_project(self) -> None:
        self.guided = None

    def next_step(self) -> str:
        if not self.guided:
            return ""
        msg = self.guided.next()
        self.say(msg)
        if self.guided.finished:
            self.guided = None
        return msg

    def repeat(self) -> str:
        msg = self.guided.repeat() if self.guided else (self._last_answer or "Ask me about a part by pointing at it.")
        self.say(msg)
        return msg

    # ------------------------------------------------------------------ internals
    def _current_selection(self) -> Optional[Detection]:
        sel = self._snap.selection
        return sel.detection if sel and sel.locked else None

    def _read_resistor_value(self, d: Detection) -> Optional[float]:
        r = self._resistor_cache.get(d.track_id)
        return r["ohms"] if r else None

    def _answer(self, question: str) -> None:
        self._busy.set()
        try:
            pointed = self._current_selection()
            extra = []
            if pointed is not None and pointed.label == "resistor":
                r = self._resistor_cache.get(pointed.track_id)
                if r:
                    extra.append(f"COLOUR-BAND READER: bands {'-'.join(r['bands'])} = about {r['text']}")
            step = self.guided.step_label if self.guided else None
            stream = self.lens.ask_stream(question, pointed=pointed, guided_step=step, extra=extra,
                                          project=self.guided.project.id if self.guided else "",
                                          step=self.guided.idx + 1 if self.guided else -1)
            if self.speaker:
                self.speaker.say_stream(stream)
            else:
                for _ in stream:
                    pass
            ans = stream.answer
            text = ans.text if ans else stream.text
            self._last_answer = text
            self.last_citations = stream.citations
            meta = f"  ·  {ans.backend}{' (cached)' if ans.cached else ''} · {ans.latency_s:.1f}s" if ans else ""
            self.messages.append(("lablens", text + meta))
            for n in self.lens.pop_notices():
                self.messages.append(("lablens", "ℹ️ " + n))
        except Exception as e:
            log.exception("answer failed")
            self.messages.append(("lablens", f"Sorry, something went wrong: {e}"))
        finally:
            self._busy.clear()

    def _maybe_read_resistor(self, frame: np.ndarray, sel: Optional[Selection]) -> Optional[str]:
        if not sel or not sel.locked or sel.detection.label != "resistor":
            return None
        d = sel.detection
        cached = self._resistor_cache.get(d.track_id)
        if cached and time.monotonic() - cached["t"] < 2.0:
            return cached["text"]
        from lablens.guided.engine import format_ohms
        from lablens.vision.resistor import read_resistor

        x1, y1, x2, y2 = (int(max(0, v)) for v in (d.x1, d.y1, d.x2, d.y2))
        r = read_resistor(frame[y1:y2, x1:x2]) if x2 > x1 + 8 and y2 > y1 + 8 else None
        if r:
            r.update(t=time.monotonic(), text=format_ohms(r["ohms"]))
            self._resistor_cache[d.track_id] = r
            return r["text"]
        return None

    def _camera_loop(self) -> None:
        from lablens.vision.hands import virtual_hand
        from lablens.vision.overlay import draw
        from lablens.vision.pointing import pointing_ray

        try:
            src = FrameSource(self.source, self.cfg.get("vision.camera_width", 1280),
                              self.cfg.get("vision.camera_height", 720))
        except Exception as e:
            log.error("camera: %s", e)
            self.messages.append(("lablens", f"Camera error: {e}"))
            return
        t_prev = time.perf_counter()
        fps = 0.0
        try:
            while not self._stop.is_set():
                frame = src.read()
                if frame is None:
                    time.sleep(0.02)
                    continue
                self.async_det.submit(frame)
                dets = self.async_det.latest()
                hands = self.hands.process(frame) if self.hands.available else []
                hand: Optional[HandObservation] = hands[0] if hands else None
                gesture = self.gesture_clf(hand) if hand is not None else "none"
                if hand is None and self._mouse is not None:
                    hand = virtual_hand(self._mouse)
                sel = self.pointer.update(hand, dets)
                cmd = self.debounce.update(gesture)
                if cmd == "next":
                    self.next_step() if self.guided else self.say("Thumbs up received.")
                elif cmd == "repeat":
                    self.repeat()
                elif cmd == "stop":
                    self.stop_talking()

                guided_state = None
                if self.guided is not None:
                    if self.async_det.updated_at != self._last_det_update:  # one vote per detector result
                        self._last_det_update = self.async_det.updated_at
                        self.guided.observe(dets)
                        if self.guided.just_became_ok():
                            self.say("Looks good. Give me a thumbs up for the next step.")
                    st = self.guided.status
                    guided_state = {"project": self.guided.project.title, "step": self.guided.idx + 1,
                                    "total": len(self.guided.project.steps),
                                    "instruction": self.guided.step["instruction"], "ok": st.ok,
                                    "messages": st.messages[:2]}
                reading = self._maybe_read_resistor(frame, sel)
                now = time.perf_counter()
                fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
                t_prev = now
                ray = pointing_ray(hand, self.pointer.length_factor) if hand is not None else None
                banner = f"det {self.async_det.fps:4.1f} fps | cam {fps:4.1f} fps | hands: {self.hands.backend}"
                if sel:
                    banner += f" | pointing: {sel.detection.label}{' (locked)' if sel.locked else ''}"
                if reading:
                    banner += f" | bands ~ {reading}"
                vis = draw(frame.copy(), dets, hand, sel, ray, banner)
                with self._lock:
                    self._snap = Snapshot(vis, dets, sel, gesture, self.hands.backend, self.async_det.fps, fps,
                                          guided_state, self._busy.is_set(), reading)
        finally:
            src.release()
