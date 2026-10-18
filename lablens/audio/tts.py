"""Text-to-speech with sentence-by-sentence streaming and instant stop (fist gesture).

Backends: pyttsx3 (offline, default), piper (offline, better voice), gTTS (online; needed for hi/kn
unless you install a Piper voice for that language), none (silent, for tests).
"""
from __future__ import annotations

import logging
import queue
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Iterable, Iterator

from lablens.net import is_online

log = logging.getLogger(__name__)

_SENT_END = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(stream: Iterable[str], min_chars: int = 25) -> Iterator[str]:
    """Group a token stream into speakable sentences as early as possible."""
    buf = ""
    for tok in stream:
        buf += tok
        parts = _SENT_END.split(buf)
        if len(parts) > 1:
            ready, buf = parts[:-1], parts[-1]
            pending = ""
            for s in ready:
                pending = f"{pending} {s}".strip()
                if len(pending) >= min_chars:
                    yield pending
                    pending = ""
            buf = f"{pending} {buf}".strip() if pending else buf
    if buf.strip():
        yield buf.strip()


def clean_for_speech(text: str) -> str:
    # drop citation tags like "[NE555.pdf p.3]" or "[p. 3]" (no need to read them aloud)
    text = re.sub(r"\s*\[[^\]]*\bp(?:age)?\.?\s*\d+[^\]]*\]", "", text, flags=re.I)
    text = re.sub(r"[*_`#>]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class Speaker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.language = cfg.get("audio.language", "en")
        backend = cfg.get("audio.tts_backend", "auto")
        if backend == "auto":
            if self.language != "en":
                backend = "gtts" if is_online() else "system"
            elif Path(cfg.path("audio.piper_model", "models/piper/x.onnx")).exists():
                backend = "piper"
            else:
                backend = "system"  # OS voice: Windows SAPI / macOS say / espeak, else pyttsx3
        self.backend = backend
        self._q: "queue.Queue[str | None]" = queue.Queue()
        self._stop = threading.Event()
        self._engine = None
        self._proc: "subprocess.Popen | None" = None
        self.speaking = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    # public -------------------------------------------------------------
    def say(self, text: str) -> None:
        self._stop.clear()
        self._q.put(clean_for_speech(text))

    def say_stream(self, tokens: Iterable[str]) -> str:
        """Speak while the LLM is still generating. Returns the full text."""
        self._stop.clear()
        full = []
        for sent in split_sentences(tokens):
            full.append(sent)
            if not self._stop.is_set():
                self._q.put(clean_for_speech(sent))
        return " ".join(full)

    def stop(self) -> None:
        self._stop.set()
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    # worker --------------------------------------------------------------
    def _run(self) -> None:
        while True:
            text = self._q.get()
            if text is None:
                return
            if self._stop.is_set() or not text:
                continue
            self.speaking.set()
            try:
                getattr(self, f"_speak_{self.backend}", self._speak_none)(text)
            except Exception as e:
                log.warning("TTS (%s) failed: %s", self.backend, e)
            finally:
                if self._q.empty():
                    self.speaking.clear()

    def _speak_none(self, text: str) -> None:
        log.info("[TTS] %s", text)

    def _run_proc(self, cmd, stdin_text: "str | None" = None) -> None:
        flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin_text is not None else None,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        if stdin_text is not None:
            try:
                self._proc.communicate(stdin_text.encode("utf-8"), timeout=120)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        else:
            self._proc.wait()

    def _speak_system(self, text: str) -> None:
        """OS speech in a killable subprocess (instant stop on the fist gesture, no thread hangs)."""
        import shutil

        if sys.platform == "win32":
            ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                  "$s.Rate = 1; $s.Speak([Console]::In.ReadToEnd())")
            self._run_proc(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps], text)
        elif sys.platform == "darwin":
            self._run_proc(["say", text])
        elif shutil.which("espeak-ng") or shutil.which("espeak"):
            self._run_proc([shutil.which("espeak-ng") or shutil.which("espeak"), "-s", "165", text])
        else:
            self._speak_pyttsx3(text)

    def _speak_pyttsx3(self, text: str) -> None:
        if self._engine is None:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 175)
        self._engine.say(text)
        self._engine.runAndWait()

    def _play_file(self, path: str) -> None:
        import sounddevice as sd
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32")
        sd.play(data, sr)
        sd.wait()

    def _speak_piper(self, text: str) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            out = f.name
        model = str(self.cfg.path("audio.piper_model", ""))
        subprocess.run(["piper", "--model", model, "--output_file", out], input=text.encode("utf-8"),
                       check=True, capture_output=True)
        self._play_file(out)

    def _speak_gtts(self, text: str) -> None:
        from gtts import gTTS

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            out = f.name
        gTTS(text=text, lang=self.language).save(out)
        try:
            self._play_file(out)  # soundfile >= 0.12 reads mp3
        except Exception:
            import pygame

            pygame.mixer.init()
            pygame.mixer.music.load(out)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy() and not self._stop.is_set():
                pygame.time.wait(50)
            pygame.mixer.music.stop()
