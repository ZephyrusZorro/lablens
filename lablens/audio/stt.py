"""Speech-to-text: Groq Whisper online, faster-whisper offline.

Recording uses a simple energy VAD: starts on speech, stops after ~0.8 s of silence or max seconds.
"""
from __future__ import annotations

import io
import logging
import time
import wave
from typing import Optional

import numpy as np

from lablens.net import is_online

log = logging.getLogger(__name__)
SR = 16000


def record(max_seconds: float = 6.0, silence_s: float = 0.8, threshold: float = 0.015) -> np.ndarray:
    """Record mono 16 kHz float32 from the default mic, auto-stopping on trailing silence."""
    import sounddevice as sd

    block = int(SR * 0.05)
    frames, started, silent_for = [], False, 0.0
    t0 = time.time()
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=block) as stream:
        while time.time() - t0 < max_seconds:
            data, _ = stream.read(block)
            data = data[:, 0]
            rms = float(np.sqrt(np.mean(data ** 2)))
            if rms > threshold:
                started, silent_for = True, 0.0
            elif started:
                silent_for += 0.05
            if started:
                frames.append(data.copy())
                if silent_for >= silence_s:
                    break
    return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)


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

    def _groq(self, wav: bytes) -> str:
        from groq import Groq

        client = Groq(api_key=self.cfg.groq_api_key)
        r = client.audio.transcriptions.create(file=("q.wav", wav), model="whisper-large-v3-turbo",
                                               language=self.language)
        return r.text.strip()

    def _faster_whisper(self, audio: np.ndarray) -> str:
        if self._local is None:
            from faster_whisper import WhisperModel

            name = self.cfg.get("audio.whisper_local_model", "base.en")
            if self.language != "en" and name.endswith(".en"):
                name = name[:-3]
            self._local = WhisperModel(name, device="cpu", compute_type="int8")
        segs, _ = self._local.transcribe(audio, language=self.language, vad_filter=True)
        return " ".join(s.text for s in segs).strip()

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        use_groq = self.backend == "groq" or (
            self.backend == "auto" and self.cfg.groq_api_key and is_online())
        if use_groq:
            try:
                self.last_backend = "groq-whisper"
                return self._groq(to_wav_bytes(audio))
            except Exception as e:
                log.warning("Groq STT failed (%s), falling back to faster-whisper", e)
        self.last_backend = "faster-whisper"
        return self._faster_whisper(audio)

    def listen(self, max_seconds: Optional[float] = None) -> str:
        return self.transcribe(record(max_seconds or self.cfg.get("audio.record_seconds", 6)))
