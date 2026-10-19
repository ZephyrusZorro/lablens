"""Hybrid LLM router: Groq when online (fast), Ollama locally when the lab Wi-Fi dies.

Switching is automatic and happens *before the first token*: if Groq errors or times out we mark
the network offline and replay the same messages to Ollama, so the user just hears an answer.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Dict, Iterator, List, Optional

from lablens.net import is_online, mark_offline

log = logging.getLogger(__name__)
Messages = List[Dict[str, str]]


class BackendError(RuntimeError):
    pass


class GroqBackend:
    name = "groq"

    def __init__(self, api_key: str, model: str, temperature: float = 0.2, timeout: float = 20):
        self.api_key, self.model, self.temperature, self.timeout = api_key, model, temperature, timeout
        self._client = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def stream(self, messages: Messages) -> Iterator[str]:
        if self._client is None:
            from groq import Groq

            self._client = Groq(api_key=self.api_key, timeout=self.timeout, max_retries=0)
        resp = self._client.chat.completions.create(model=self.model, messages=messages, stream=True,
                                                    temperature=self.temperature, max_tokens=500)
        try:
            for ev in resp:
                delta = ev.choices[0].delta.content if ev.choices else None
                if delta:
                    yield delta
        finally:
            resp.close()


class OllamaBackend:
    name = "ollama"
    # preference order when the configured model isn't installed
    FAMILIES = ("llama3.2", "llama3.1", "llama3", "qwen2.5", "qwen3", "gemma3", "gemma2", "mistral", "phi4", "phi3")

    def __init__(self, host: str, model: str, temperature: float = 0.2, timeout: float = 60):
        self.host, self.model, self.temperature, self.timeout = host.rstrip("/"), model, temperature, timeout
        self._resolved: Optional[str] = None

    def installed(self) -> List[str]:
        import requests

        r = requests.get(f"{self.host}/api/tags", timeout=1.5)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]

    @property
    def configured(self) -> bool:
        try:
            return bool(self.installed())
        except Exception:
            return False

    def resolve_model(self) -> str:
        """Use the configured model if installed, else the best installed chat model."""
        if self._resolved:
            return self._resolved
        names = self.installed()
        if not names:
            raise BackendError("Ollama is running but has no models (try: ollama pull llama3.2:3b)")
        base = lambda n: n.split(":")[0]
        if self.model in names or any(base(n) == self.model for n in names):
            self._resolved = self.model if self.model in names else next(n for n in names if base(n) == self.model)
        else:
            chat = [n for n in names if "embed" not in n]
            ranked = sorted(chat, key=lambda n: next((i for i, f in enumerate(self.FAMILIES) if n.startswith(f)), 99))
            self._resolved = (ranked or names)[0]
            log.info("Ollama model %s not installed; using %s", self.model, self._resolved)
        return self._resolved

    def stream(self, messages: Messages) -> Iterator[str]:
        import requests

        body = {"model": self.resolve_model(), "messages": messages, "stream": True,
                "options": {"temperature": self.temperature, "num_predict": 500}}
        with requests.post(f"{self.host}/api/chat", json=body, stream=True, timeout=(3, self.timeout)) as r:
            if not r.ok:
                raise BackendError(f"ollama HTTP {r.status_code}: {r.text[:200]}")
            for line in r.iter_lines():
                if not line:
                    continue
                msg = json.loads(line)
                tok = msg.get("message", {}).get("content")
                if tok:
                    yield tok
                if msg.get("done"):
                    break


class LLMRouter:
    def __init__(self, backends: list, prefer: str = "groq", online_ttl: float = 10.0):
        self.backends = {b.name: b for b in backends}
        self.prefer = prefer
        self.online_ttl = online_ttl
        self.last_backend = ""
        self.last_ttft = 0.0  # time to first token, seconds
        self.forced: Optional[str] = None  # "groq" | "ollama" | None (demo toggle)
        self._failed_until = {}

    @classmethod
    def from_config(cls, cfg) -> "LLMRouter":
        t = cfg.get("llm.temperature", 0.2)
        return cls([
            GroqBackend(cfg.groq_api_key, cfg.get("llm.groq_model", "llama-3.3-70b-versatile"), t,
                        cfg.get("llm.request_timeout", 20)),
            OllamaBackend(cfg.ollama_host, cfg.get("llm.ollama_model", "llama3.2:3b"), t),
        ], cfg.get("llm.prefer", "groq"), cfg.get("llm.online_check_seconds", 10))

    def order(self) -> List[str]:
        if self.forced:
            return [self.forced]
        names = ["groq", "ollama"] if self.prefer == "groq" else ["ollama", "groq"]
        g = self.backends.get("groq")
        if "groq" in names and (g is None or not g.configured or not is_online(self.online_ttl)):
            names.remove("groq")
            names.append("groq")  # last resort only
        return [n for n in names if n in self.backends
                and (n != "groq" or self.backends[n].configured)
                and time.monotonic() >= self._failed_until.get(n, 0)]

    def stream(self, messages: Messages) -> Iterator[str]:
        errors = []
        self.last_backend = ""
        self.last_ttft = 0.0
        for name in self.order():
            b = self.backends[name]
            t0 = time.perf_counter()
            gen = b.stream(messages)
            try:
                first = next(t for t in gen if t and t.strip())
            except StopIteration:
                errors.append(f"{name}: empty response")
                self._failed_until[name] = time.monotonic() + 30
                continue
            except Exception as e:
                errors.append(f"{name}: {e}")
                log.warning("LLM backend %s failed before first token: %s", name, e)
                # An invalid key / HTTP 404 is a service failure, not evidence
                # that every network feature is offline. Avoid retrying it on
                # every question while allowing the local model immediately.
                self._failed_until[name] = time.monotonic() + 60
                continue
            self.last_backend, self.last_ttft = name, time.perf_counter() - t0
            yield first
            try:
                yield from gen
            except Exception as e:  # mid-answer drop: say so instead of hanging
                log.warning("%s dropped mid-stream: %s", name, e)
                yield " … (connection lost, please ask again)"
            return
        raise BackendError("No LLM backend available. " + "; ".join(errors) +
                           " - set GROQ_API_KEY or start Ollama (`ollama serve`).")

    def complete(self, messages: Messages) -> str:
        return "".join(self.stream(messages))
