"""Cheap, cached connectivity check used by the STT / TTS / LLM routers."""
from __future__ import annotations

import socket
import time

_cache = {"t": 0.0, "v": False}


def is_online(ttl: float = 10.0, host: str = "api.groq.com", port: int = 443, timeout: float = 1.5) -> bool:
    now = time.monotonic()
    if now - _cache["t"] < ttl:
        return _cache["v"]
    try:
        with socket.create_connection((host, port), timeout=timeout):
            ok = True
    except OSError:
        ok = False
    _cache.update(t=now, v=ok)
    return ok


def mark_offline() -> None:
    """Call after a failed request so the next call skips the network immediately."""
    _cache.update(t=time.monotonic(), v=False)


def force_mode(online: bool | None) -> None:
    """Demo helper: force online/offline for ~1 hour (None to reset)."""
    if online is None:
        _cache.update(t=0.0)
    else:
        _cache.update(t=time.monotonic() + 3600, v=online)
