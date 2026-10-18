"""Hand gestures -> app commands.

    👍 thumbs_up  -> next step
    ✋ open_palm  -> repeat last answer / instruction
    ✊ fist       -> stop talking
    ☝ point      -> (not a command) pointing mode

Default classifier is rule-based on landmark geometry (rotation/scale invariant, no training).
If models/gesture_knn.joblib exists (scripts/train_gestures.py, RPS-workshop style), it is used instead.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from lablens.types import HandObservation, Point

WRIST = 0
TIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIPS = {"thumb": 3, "index": 6, "middle": 10, "ring": 14, "pinky": 18}
MCPS = {"thumb": 2, "index": 5, "middle": 9, "ring": 13, "pinky": 17}

COMMANDS = {"thumbs_up": "next", "open_palm": "repeat", "fist": "stop"}


def _d(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def finger_states(lm: List[Point]) -> dict:
    """True = extended. A finger is extended if its tip is clearly farther from the wrist than its PIP."""
    w = lm[WRIST]
    st = {f: _d(lm[TIPS[f]], w) > _d(lm[PIPS[f]], w) * 1.15 for f in ("index", "middle", "ring", "pinky")}
    palm = _d(lm[WRIST], lm[9]) or 1.0
    # thumb: tip far from the index MCP relative to palm size
    st["thumb"] = _d(lm[4], lm[5]) > 0.55 * palm and _d(lm[4], lm[WRIST]) > _d(lm[3], lm[WRIST])
    return st


def features(lm: List[Point]) -> np.ndarray:
    """Normalised landmark vector for the optional learned classifier (wrist-centred, scale-free)."""
    a = np.asarray(lm, dtype=np.float32)
    a = a - a[0]
    s = np.linalg.norm(a[9]) or 1.0
    return (a / s).flatten()


def classify_rules(hand: HandObservation) -> str:
    lm = hand.landmarks
    st = finger_states(lm)
    fingers = [st[f] for f in ("index", "middle", "ring", "pinky")]
    palm = _d(lm[WRIST], lm[9]) or 1.0
    if not any(fingers):
        # thumbs up: thumb extended and pointing up (image y grows downward)
        if st["thumb"] and (lm[2][1] - lm[4][1]) > 0.6 * palm and lm[4][1] < min(p[1] for p in lm[5:]):
            return "thumbs_up"
        return "fist"
    if all(fingers):
        return "open_palm"
    if st["index"] and not any(fingers[1:]):
        return "point"
    return "none"


class GestureClassifier:
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        if model_path and Path(model_path).exists():
            import joblib

            self.model = joblib.load(model_path)

    def __call__(self, hand: Optional[HandObservation]) -> str:
        if hand is None:
            return "none"
        if self.model is not None:
            return str(self.model.predict(features(hand.landmarks)[None])[0])
        return classify_rules(hand)


class GestureDebouncer:
    """Emit a command once the same gesture is held N frames, then wait out a cooldown."""

    def __init__(self, hold_frames: int = 6, cooldown_s: float = 1.5, clock=time.monotonic):
        self.hold, self.cooldown, self.clock = hold_frames, cooldown_s, clock
        self._cur, self._n, self._last_fire = "none", 0, -1e9
        self._armed = True

    def update(self, gesture: str) -> Optional[str]:
        if gesture == self._cur:
            self._n += 1
        else:
            self._cur, self._n, self._armed = gesture, 1, True
        if (self._armed and gesture in COMMANDS and self._n >= self.hold
                and self.clock() - self._last_fire >= self.cooldown):
            self._last_fire = self.clock()
            self._armed = False  # must change gesture before firing again
            return COMMANDS[gesture]
        return None
