"""Hand landmarks (21 points, MediaPipe topology) in pixel coordinates.

Backends, tried in order:
  1. MediaPipe Tasks HandLandmarker (current API; needs models/hand_landmarker.task,
     downloaded by scripts/setup_models.py)
  2. MediaPipe legacy `mp.solutions.hands` (older mediapipe versions)
If neither is available `HandTracker.available` is False and the apps fall back to mouse pointing.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from lablens.types import HandObservation

log = logging.getLogger(__name__)

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
TASK_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/"
                  "hand_landmarker.task")


class HandTracker:
    def __init__(self, model_path: str = "models/hand_landmarker.task", max_hands: int = 1,
                 det_conf: float = 0.5, track_conf: float = 0.5):
        self.backend = "none"
        self._last: List[HandObservation] = []
        self._t0 = time.monotonic()
        try:
            import mediapipe as mp
        except ImportError:
            log.warning("mediapipe not installed - hand pointing disabled (mouse pointing still works)")
            return
        self._mp = mp
        mp_path = Path(model_path)
        if mp_path.exists():
            try:
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

                opts = HandLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(mp_path)),
                                             running_mode=RunningMode.VIDEO, num_hands=max_hands,
                                             min_hand_detection_confidence=det_conf,
                                             min_tracking_confidence=track_conf)
                self._tasks = HandLandmarker.create_from_options(opts)
                self.backend = "mediapipe-tasks"
                return
            except Exception as e:
                log.warning("MediaPipe Tasks init failed (%s); trying legacy API", e)
        try:
            self._legacy = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=max_hands,
                                                    min_detection_confidence=det_conf,
                                                    min_tracking_confidence=track_conf)
            self.backend = "mediapipe-legacy"
        except Exception as e:
            log.warning("No usable MediaPipe hand backend (%s). Run scripts/setup_models.py to fetch "
                        "hand_landmarker.task", e)

    @property
    def available(self) -> bool:
        return self.backend != "none"

    def process(self, frame_bgr: np.ndarray) -> List[HandObservation]:
        if not self.available:
            return []
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        out: List[HandObservation] = []
        if self.backend == "mediapipe-tasks":
            img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            ts = int((time.monotonic() - self._t0) * 1000)
            res = self._tasks.detect_for_video(img, ts)
            for lms, hd in zip(res.hand_landmarks, res.handedness):
                out.append(HandObservation([(p.x * w, p.y * h) for p in lms], hd[0].category_name))
        else:
            res = self._legacy.process(rgb)
            if res.multi_hand_landmarks:
                for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
                    out.append(HandObservation([(p.x * w, p.y * h) for p in lm.landmark],
                                               hd.classification[0].label))
        self._last = out
        return out

    def close(self) -> None:
        for attr in ("_tasks", "_legacy"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass


def virtual_hand(tip: tuple, direction: tuple = (0.0, -1.0), length: float = 40.0) -> HandObservation:
    """A synthetic pointing hand whose index tip is at `tip` (mouse pointing / tests / demos).

    Landmark 6 (index PIP) is placed `length` px behind the tip along `direction`.
    """
    dx, dy = direction
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    dx, dy = dx / n, dy / n
    x, y = tip
    lm = [(x - dx * length * 3, y - dy * length * 3)] * 21  # wrist & others far behind
    lm = list(lm)
    lm[5] = (x - dx * length * 1.6, y - dy * length * 1.6)
    lm[6] = (x - dx * length, y - dy * length)
    lm[7] = (x - dx * length * 0.5, y - dy * length * 0.5)
    lm[8] = (x, y)
    return HandObservation(lm, "Virtual")
