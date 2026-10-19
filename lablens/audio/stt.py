"""Speech-to-text: Groq Whisper online, faster-whisper offline.

Recording uses a simple energy VAD: starts on speech, stops after ~0.8 s of silence or max seconds.
"""
from __future__ import annotations

import io
import logging
import math
import threading
import time
import wave
from collections import deque
from typing import Optional

import numpy as np

from lablens.net import is_online

log = logging.getLogger(__name__)
SR = 16000


def input_devices():
    import sounddevice as sd
    return [(i, d) for i, d in enumerate(sd.query_devices()) if d['max_input_channels'] > 0]


def is_virtual_input(name):
    return any(s in name.lower() for s in ("cable", "loopback", "stereo mix", "sound mapper", "primary sound"))


def resolve_input_device(requested=None):
    """Prefer a real microphone; never silently substitute a virtual cable."""
    import sounddevice as sd
    if requested not in (None, "", "auto"):
        sd.query_devices(requested, kind="input")  # validate configured id/name
        return requested
    default = sd.default.device[0]
    if default >= 0 and not is_virtual_input(sd.query_devices(default)['name']):
        return default
    physical = [(i, d) for i, d in input_devices() if not is_virtual_input(d['name'])
                and any(s in d['name'].lower() for s in ("mic", "headset", "usb", "webcam"))]
    if physical:
        return physical[0][0]
    raise RuntimeError("No physical microphone is available. Use the browser recorder and select its microphone, "
                       "or connect a microphone and choose it in Audio input. The Windows default is a virtual cable.")


def resample_mono(audio, source_rate):
    from scipy.signal import resample_poly
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if audio.ndim != 1 or not np.isfinite(audio).all() or source_rate <= 0:
        raise ValueError("Invalid audio samples or sample rate")
    rate = int(source_rate)
    if rate != SR:
        gcd = math.gcd(rate, SR)
        audio = resample_poly(audio, SR // gcd, rate // gcd)
    return np.ascontiguousarray(audio, dtype=np.float32)


def decode_audio(wav: bytes):
    import soundfile as sf
    # Reject unbounded uploads before decoding into memory.
    if len(wav) > 20 * 1024 * 1024:
        raise ValueError("Recording is too large; record a shorter question")
    with sf.SoundFile(io.BytesIO(wav)) as f:
        if len(f) > f.samplerate * 60:
            raise ValueError("Please keep voice questions under 60 seconds")
        return resample_mono(f.read(dtype="float32"), f.samplerate)


def record(max_seconds: float = 12.0, silence_s: float = 1.2, threshold: float = 0.004,
           device=None) -> np.ndarray:
    """Record at the device's native rate, preserving the start of quiet speech."""
    import sounddevice as sd
    device = resolve_input_device(device)
    rate = int(sd.query_devices(device, kind="input")['default_samplerate'])
    block = max(1, int(rate * 0.05))
    frames, started, silent_for = [], False, 0.0
    pre_roll = deque(maxlen=5)
    t0 = time.monotonic()
    with sd.InputStream(device=device, samplerate=rate, channels=1, dtype="float32", blocksize=block) as stream:
        while time.monotonic() - t0 < max_seconds:
            data, _ = stream.read(block)
            data = data[:, 0].copy()
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > threshold:
                if not started:
                    frames.extend(pre_roll)
                started, silent_for = True, 0.0
            elif started:
                silent_for += 0.05
            if started:
                frames.append(data)
                if silent_for >= silence_s:
                    break
            else:
                pre_roll.append(data)
    return resample_mono(np.concatenate(frames), rate) if frames else np.zeros(0, dtype=np.float32)


def to_wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class SpeechToText:
    def __init__(self, cfg):
        self.cfg = cfg
        self.backend = cfg.get("audio.stt_backend", "auto")
        self.language = cfg.get("audio.language", "en")
        self._local = None
        self.last_backend = ""
        self._lock = threading.Lock()
        self._groq_failed_until = 0.0

    def _groq(self, wav: bytes) -> str:
        from groq import Groq

        client = Groq(api_key=self.cfg.groq_api_key, timeout=10, max_retries=0)
        r = client.audio.transcriptions.create(file=("q.wav", wav), model="whisper-large-v3-turbo",
                                               language=self.language)
        return r.text.strip()

    def _faster_whisper(self, audio: np.ndarray) -> str:
        if self._local is None:
            from faster_whisper import WhisperModel

            name = self.cfg.get("audio.whisper_local_model", "base.en")
            if self.language != "en" and name.endswith(".en"):
                name = name[:-3]
            self._local = WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=4)
        segs, _ = self._local.transcribe(audio, language=self.language, vad_filter=True,
                                         beam_size=5, condition_on_previous_text=False,
                                         initial_prompt="Electronics lab: NE555 timer, LM358, LED, resistor, capacitor, ground pin.")
        return " ".join(s.text for s in segs).strip()

    def transcribe(self, audio: np.ndarray) -> str:
        with self._lock:
            return self._transcribe(audio)

    def _transcribe(self, audio: np.ndarray) -> str:
        audio = resample_mono(audio, SR)
        if audio.size == 0:
            return ""
        if np.max(np.abs(audio)) < 0.0001:
            return ""  # do not hallucinate words from digital silence
        use_groq = self.backend == "groq" or (
            self.backend == "auto" and self.cfg.groq_api_key and is_online()
            and time.monotonic() >= self._groq_failed_until)
        if use_groq:
            try:
                self.last_backend = "groq-whisper"
                return self._groq(to_wav_bytes(audio))
            except Exception as e:
                self._groq_failed_until = time.monotonic() + 300
                log.warning("Groq STT failed (%s), falling back to faster-whisper", e)
        self.last_backend = "faster-whisper"
        return self._faster_whisper(audio)

    def listen(self, max_seconds: Optional[float] = None) -> str:
        return self.transcribe(record(max_seconds or self.cfg.get("audio.record_seconds", 12),
                                      silence_s=self.cfg.get("audio.silence_seconds", 1.2),
                                      threshold=self.cfg.get("audio.speech_threshold", 0.004),
                                      device=self.cfg.get("audio.input_device")))
