"""Regressions for person/electronics confusion and the complete voice input contract."""
import io
import threading
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

from lablens.audio.stt import SpeechToText, decode_audio, resolve_input_device
from lablens.config import Config
from lablens.runtime import Runtime, Snapshot, gesture_controls_allowed
from lablens.types import Detection, Selection
from lablens.vision.humans import HumanResult, filter_components, pose_detections
from lablens.vision.hands import virtual_hand
from lablens.vision.pointing import PointingResolver, resolve_pointed


def test_person_mask_removes_fake_parts_but_keeps_separate_bench():
    mask = np.zeros((100, 200), np.float32)
    mask[:, :80] = 1
    person = HumanResult([Detection("person", .95, 0, 0, 80, 100)], mask=mask)
    fake = Detection("resistor", .9, 10, 10, 40, 40)
    real = Detection("led", .7, 120, 60, 130, 70)
    assert filter_components([fake, real], person) == [real]


def test_auto_person_view_is_not_treated_as_a_lab_bench():
    human = HumanResult([Detection("person", .9, 0, 0, 60, 100)])
    stray = Detection("led", .4, 150, 20, 160, 30)
    assert filter_components([stray], human, scene_mode="auto") == []
    assert filter_components([stray], human, scene_mode="mixed") == [stray]
    assert filter_components([stray], HumanResult(), scene_mode="people") == []
    board = Detection("breadboard", .95, 100, 0, 190, 90)
    assert filter_components([board, stray], human, scene_mode="auto") == [board, stray]


def test_chair_is_not_a_breadboard():
    chair = Detection("chair", .9, 30, 30, 150, 100)
    board = Detection("breadboard", .65, 40, 40, 140, 90)
    assert not filter_components([board], HumanResult(exclusions=[chair]))


def test_hidden_legs_are_not_reported_as_visible():
    points = [(50, 50, 0)] * 33
    for i, xy in {0: (50, 10), 1: (48, 9), 2: (47, 8), 3: (46, 9),
                  11: (30, 30), 12: (70, 30), 13: (20, 50), 15: (10, 70)}.items():
        points[i] = (*xy, .9)
    labels = {d.label for d in pose_detections(points, 100, 100)}
    assert {"person", "head", "left arm"} <= labels
    assert not {"torso", "left leg", "right leg", "right arm"} & labels


def test_human_boxes_do_not_capture_component_pointing():
    human = Detection("person", .95, 0, 0, 300, 300)
    head = Detection("head", .9, 20, 20, 100, 100)
    led = Detection("led", .7, 100, 100, 120, 120)
    hand = virtual_hand((110, 130))
    assert resolve_pointed(hand, [human, head, led]) is led
    assert resolve_pointed(hand, [human, head]) is None


def test_lock_clears_when_component_is_reclassified_as_human():
    r = PointingResolver(dwell_seconds=0)
    part = Detection("resistor", .8, 30, 30, 70, 70, 1)
    assert r.update(virtual_hand((50, 70)), [part]).locked
    assert r.update(None, [Detection("person", .9, 0, 0, 100, 100, 2)]) is None


def test_browser_stereo_48khz_audio_is_decoded_to_mono_16khz():
    import soundfile as sf
    t = np.arange(48000) / 48000
    stereo = np.column_stack([.2 * np.sin(2 * np.pi * 440 * t)] * 2)
    buf = io.BytesIO()
    sf.write(buf, stereo, 48000, format="WAV")
    audio = decode_audio(buf.getvalue())
    assert audio.shape == (16000,) and audio.dtype == np.float32
    assert np.max(np.abs(audio)) > .15


def test_silence_never_generates_a_whisper_question(monkeypatch):
    stt = SpeechToText(Config())
    monkeypatch.setattr(stt, '_faster_whisper', lambda x: pytest.fail("silence reached Whisper"))
    assert stt.transcribe(np.zeros(16000, np.float32)) == ""


def test_virtual_default_requires_explicit_selection(monkeypatch):
    import sounddevice as sd
    devices = [{"name": "CABLE Output", "max_input_channels": 2}]
    monkeypatch.setattr(sd, "default", SimpleNamespace(device=[0, 1]))
    monkeypatch.setattr(sd, "query_devices", lambda index=None, kind=None: devices if index is None else devices[index])
    with pytest.raises(RuntimeError, match="physical microphone"):
        resolve_input_device()
    assert resolve_input_device(0) == 0
    devices.append({"name": "Microphone (USB)", "max_input_channels": 1})
    assert resolve_input_device() == 1


def runtime_stub():
    rt = Runtime.__new__(Runtime)
    rt._request_lock = threading.Lock()
    rt._busy = threading.Event()
    rt._lock = threading.Lock()
    rt._snap = Snapshot()
    rt._listening = rt._transcribing = False
    rt.camera_error = rt.audio_error = rt.last_transcript = ""
    rt.transcript_id = 0
    rt.messages = deque()
    rt.speaker = None
    rt.cfg = Config()
    rt._stop = threading.Event()
    return rt


def test_concurrent_questions_are_rejected_and_pointing_is_captured():
    rt = runtime_stub()
    led = Detection("led", .8, 1, 1, 10, 10)
    rt._snap.selection = Selection(led, True, 1)
    release, entered = threading.Event(), threading.Event()
    calls = []
    def answer(q, pointed):
        entered.set()
        release.wait(2)
        calls.append((q, pointed))
    rt._answer = answer
    t = rt.ask("first")
    try:
        assert entered.wait(1)
        rt._snap.selection = None
        assert rt.ask("second") is None
        assert rt.snapshot().busy
    finally:
        release.set()
        t.join(3)
    assert calls == [("first", led)]
    assert not rt.snapshot().busy


def test_browser_audio_returns_a_draft_without_sending_it(monkeypatch):
    rt = runtime_stub()
    monkeypatch.setattr("lablens.audio.stt.decode_audio", lambda wav: np.ones(16000, np.float32))
    rt.stt = SimpleNamespace(transcribe=lambda audio: "Which pin is ground on the 555?")
    rt._answer = lambda *a: pytest.fail("recording was submitted without reviewing")
    t = rt.transcribe_recording(b"recorded audio")
    t.join(3)
    snap = rt.snapshot()
    assert "555" in snap.last_transcript and snap.transcript_id == 1
    assert not snap.busy and not snap.audio_error


def test_empty_llm_response_falls_back_and_failure_is_not_retried(monkeypatch):
    from lablens.llm.router import LLMRouter
    monkeypatch.setattr("lablens.llm.router.is_online", lambda *a: True)
    class Backend:
        configured = True
        def __init__(self, name, tokens):
            self.name, self.tokens, self.calls = name, tokens, 0
        def stream(self, messages):
            self.calls += 1
            yield from self.tokens
    groq, ollama = Backend("groq", ["", " "]), Backend("ollama", ["Pin 1"])
    router = LLMRouter([groq, ollama])
    assert router.complete([]) == "Pin 1"
    assert router.complete([]) == "Pin 1"
    assert groq.calls == 1 and ollama.calls == 2


def test_background_body_gestures_do_not_control_the_lab():
    person = Detection("person", .9, 0, 0, 100, 100)
    board = Detection("breadboard", .9, 120, 120, 250, 200)
    assert not gesture_controls_allowed([person], "auto")
    assert not gesture_controls_allowed([person, board], "people")
    assert gesture_controls_allowed([person, board], "auto")
    assert not gesture_controls_allowed([board], "auto", busy=True)


def test_browser_rejects_an_overlong_recording():
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.zeros(16000 * 61, np.float32), 16000, format="WAV")
    with pytest.raises(ValueError, match="under 60 seconds"):
        decode_audio(buf.getvalue())
