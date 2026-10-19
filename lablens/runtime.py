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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Deque, List, Optional, Tuple, Union

import cv2
import numpy as np

from lablens.types import Detection, HandObservation, Selection

log = logging.getLogger(__name__)
Source = Union[int, str]


def gesture_controls_allowed(dets, scene_mode, busy=False):
    return (not busy and sum(d.label == "person" for d in dets) <= 1 and scene_mode != "people" and
            (scene_mode == "mixed" or any(d.label == "breadboard" for d in dets)))


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
    listening: bool = False
    transcribing: bool = False
    camera_error: str = ""
    last_transcript: str = ""
    audio_error: str = ""
    transcript_id: int = 0


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
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
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
        self.show_body = cfg.get("vision.humans.show_body", False)
        self.gestures_enabled = cfg.get("gestures.enabled", False)
        self._manual_track = None
        if getattr(self.detector, "humans", None) is not None:
            self.detector.humans.pose_enabled = self.show_body
        self._mouse: Optional[Tuple[float, float]] = None
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._request_lock = threading.Lock()
        self._listening = False
        self._transcribing = False
        self.last_transcript = ""
        self.transcript_id = 0
        self.audio_error = ""
        self.camera_error = ""
        self.input_device = cfg.get("audio.input_device")
        from lablens.audio.stt import SpeechToText
        self.stt = SpeechToText(cfg)
        self._request_thread = None
        self._closed = False
        self._shutdown_lock = threading.Lock()
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
        with self._shutdown_lock:
            if self._closed:
                return
            self._closed = True
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        self.async_det.close()
        if self._thread is None or not self._thread.is_alive():
            self.hands.close()
        if self.speaker:
            self.speaker.close()

    # ------------------------------------------------------------------ UI API (thread-safe)
    def snapshot(self) -> Snapshot:
        with self._lock:
            return replace(self._snap, busy=self._busy.is_set(), listening=self._listening,
                           transcribing=self._transcribing, camera_error=self.camera_error,
                           last_transcript=self.last_transcript, audio_error=self.audio_error,
                           transcript_id=self.transcript_id)

    def set_mouse(self, x: Optional[float], y: Optional[float] = None) -> None:
        """Mouse pointing: a virtual fingertip at (x, y). Pass None to release."""
        self._mouse = None if x is None else (float(x), float(y))

    def select_component(self, track_id):
        """Resolve the selected ID each frame, never a stale coordinate."""
        self._manual_track = track_id

    def say(self, text: str, role: str = "lablens") -> None:
        self.messages.append((role, text))
        speaker = self.speaker
        if speaker:
            speaker.say(text)

    def set_speech_enabled(self, enabled):
        if enabled and self.speaker is None:
            from lablens.audio.tts import Speaker
            self.speaker = Speaker(self.cfg)
        elif not enabled and self.speaker is not None:
            speaker, self.speaker = self.speaker, None
            speaker.close()

    def stop_talking(self) -> None:
        speaker = self.speaker
        if speaker:
            speaker.stop()

    def ask(self, question: str, block: bool = False):
        """Ask about whatever is locked under the finger. Non-blocking by default."""
        question = question.strip()
        if not question or self._stop.is_set():
            return None
        if not self._request_lock.acquire(blocking=False):
            self.messages.append(("lablens", "Please wait for the current question to finish."))
            return None
        pointed = self._current_selection()
        self._busy.set()
        self.messages.append(("you", question))
        def run():
            try:
                self._answer(question, pointed)
            finally:
                self._busy.clear()
                self._request_lock.release()
        t = threading.Thread(target=run, daemon=True, name="lablens-ask")
        self._request_thread = t
        t.start()
        if block:
            t.join()
        return t

    def _start_voice(self, wav=None, ask=True):
        if self._stop.is_set():
            return None
        if not self._request_lock.acquire(blocking=False):
            return None
        self._busy.set()
        self.audio_error = ""
        self.last_transcript = ""
        pointed = self._current_selection()
        self.stop_talking()  # prevent recording the assistant's own speech
        def run():
            from lablens.audio.stt import record, decode_audio
            try:
                if wav is None:
                    self._listening = True
                    audio = record(self.cfg.get("audio.record_seconds", 12),
                                   self.cfg.get("audio.silence_seconds", 1.2),
                                   self.cfg.get("audio.speech_threshold", 0.004), self.input_device)
                else:
                    audio = decode_audio(wav)
                self._listening = False
                self._transcribing = True
                q = self.stt.transcribe(audio)
                self.last_transcript = q
                self.transcript_id += 1
                self._transcribing = False
                if q:
                    if ask and not self._stop.is_set():
                        self.messages.append(("you", q))
                        self._answer(q, pointed)
                else:
                    self.audio_error = "No speech detected. Check the selected microphone, speak clearly, and try again."
            except Exception as e:
                self.audio_error = f"Microphone/STT error: {e}"
                log.exception("Voice input failed")
            finally:
                self._listening = self._transcribing = False
                self._busy.clear()
                self._request_lock.release()
        t = threading.Thread(target=run, daemon=True, name="lablens-listen")
        self._request_thread = t
        t.start()
        return t

    def listen_and_ask(self):
        return self._start_voice()

    def transcribe_recording(self, wav: bytes):
        """Browser recording -> editable transcript, without submitting it as a question."""
        return self._start_voice(wav, ask=False)

    def register_project(self, project) -> None:
        self.projects[project.id] = project

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
        manual = getattr(self, "_manual_track", None)
        if manual is not None:
            return next((d for d in self._snap.detections if d.track_id == manual), None)
        sel = self._snap.selection
        return sel.detection if sel and sel.locked else None

    def _read_resistor_value(self, d: Detection) -> Optional[float]:
        r = self._resistor_cache.get(d.track_id)
        return r["ohms"] if r else None

    def _answer(self, question: str, pointed=None) -> None:
        self._busy.set()
        try:
            extra = []
            if pointed is not None and pointed.label == "resistor":
                r = self._resistor_cache.get(pointed.track_id)
                if r:
                    extra.append(f"COLOUR-BAND READER: bands {'-'.join(r['bands'])} = about {r['text']}")
            step = self.guided.step_label if self.guided else None
            stream = self.lens.ask_stream(question, pointed=pointed, guided_step=step, extra=extra,
                                          project=self.guided.project.id if self.guided else "",
                                          step=self.guided.idx + 1 if self.guided else -1)
            speaker = self.speaker
            if speaker:
                speaker.say_stream(stream)
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
            self.camera_error = str(e)
            return
        t_prev = time.perf_counter()
        fps = 0.0
        failed_reads = 0
        try:
            while not self._stop.is_set():
                frame = src.read()
                if frame is None:
                    failed_reads += 1
                    if failed_reads >= 60:
                        self.camera_error = "Camera stopped delivering frames. Reconnect it or choose another camera source."
                    time.sleep(0.02)
                    continue
                failed_reads = 0
                self.camera_error = ""
                self.async_det.submit(frame)
                dets = self.async_det.latest()
                hands = self.hands.process(frame) if self.hands.available else []
                people_count = sum(d.label == "person" for d in dets)
                hand: Optional[HandObservation] = hands[0] if len(hands) == 1 and people_count <= 1 else None
                gesture = self.gesture_clf(hand) if hand is not None else "none"
                if self._mouse is not None:
                    hand = virtual_hand(self._mouse)
                if self._manual_track is not None:
                    target = next((d for d in dets if d.track_id == self._manual_track), None)
                    hand = virtual_hand(target.center) if target is not None else None
                sel = self.pointer.update(hand, dets)
                humans = getattr(self.detector, "humans", None)
                scene_mode = humans.scene_mode if humans is not None else "mixed"
                allowed = self.gestures_enabled and gesture_controls_allowed(dets, scene_mode, self._busy.is_set())
                cmd = self.debounce.update(gesture if allowed else "none")
                if cmd == "next":
                    self.next_step() if self.guided else self.say("Thumbs up received.")
                elif cmd == "repeat":
                    self.repeat()
                elif cmd == "stop":
                    self.stop_talking()

                guided_state = None
                guided = self.guided
                if guided is not None:
                    if self.async_det.sequence != self._last_det_update:  # one vote per detector result
                        self._last_det_update = self.async_det.sequence
                        guided.observe(dets)
                        if guided.just_became_ok():
                            self.say("Visual check passed. Select Next to continue.")
                    st = guided.status
                    guided_state = {"project": guided.project.title, "step": guided.idx + 1,
                                    "total": len(guided.project.steps),
                                    "instruction": guided.step["instruction"], "ok": st.ok,
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
                vis = draw(frame.copy(), dets, hand, sel, ray)
                humans = getattr(self.detector, "humans", None)
                if humans is not None and self.show_body:
                    from lablens.vision.humans import draw_pose
                    draw_pose(vis, humans.last)
                with self._lock:
                    self._snap = Snapshot(vis, dets, sel, gesture, self.hands.backend, self.async_det.fps, fps,
                                          guided_state, self._busy.is_set(), reading)
        except Exception as e:
            self.camera_error = f"Camera processing failed: {e}"
            log.exception("Camera processing failed")
        finally:
            src.release()
