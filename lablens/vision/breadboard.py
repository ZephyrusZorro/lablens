"""Classical breadboard detector (no training data needed).

A breadboard seen from above is a large, light, low-saturation rectangle densely covered in a regular
grid of small dark holes. We find light low-saturation regions, keep rectangular ones of plausible
aspect ratio, then verify the hole texture by counting small dark blobs per unit area.
The box is smoothed over time so Guided Build zone checks don't jitter.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from lablens.types import Detection


class BreadboardDetector:
    def __init__(self, work_width: int = 480, min_area_frac: float = 0.06, min_hole_density: float = 1.5,
                 smooth: float = 0.6):
        self.work_width = work_width
        self.min_area_frac = min_area_frac
        self.min_hole_density = min_hole_density  # holes per 1000 px^2 at work resolution
        self.smooth = smooth
        self._ema: Optional[np.ndarray] = None
        self._miss = 0

    def _holes_density(self, gray: np.ndarray, rect) -> float:
        x, y, w, h = rect
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0
        # holes are darker than their immediate neighbourhood
        bg = cv2.blur(roi, (9, 9))
        dark = ((bg.astype(np.int16) - roi.astype(np.int16)) > 18).astype(np.uint8)
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        holes = int(((areas >= 2) & (areas <= 40)).sum())
        return 1000.0 * holes / float(w * h)

    def detect_once(self, img_bgr: np.ndarray) -> Optional[Detection]:
        H, W = img_bgr.shape[:2]
        s = self.work_width / float(W)
        small = cv2.resize(img_bgr, (self.work_width, max(1, int(H * s))), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        v = hsv[..., 2]
        thr_v = max(120, int(np.percentile(v, 55)))
        mask = ((hsv[..., 1] < 70) & (v > thr_v)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_img = float(small.shape[0] * small.shape[1])
        best, best_score = None, 0.0
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.min_area_frac * area_img:
                continue
            x, y, w, h = cv2.boundingRect(c)
            rectangularity = a / float(w * h)
            aspect = max(w, h) / max(1.0, min(w, h))
            if rectangularity < 0.6 or not (1.1 <= aspect <= 5.0):
                continue
            density = self._holes_density(gray, (x, y, w, h))
            if density < self.min_hole_density:
                continue
            score = rectangularity * min(1.0, density / (3 * self.min_hole_density)) * (a / area_img) ** 0.25
            if score > best_score:
                best, best_score = (x, y, w, h), score
        if best is None:
            return None
        x, y, w, h = best
        conf = float(min(0.99, 0.5 + best_score))
        return Detection("breadboard", conf, x / s, y / s, (x + w) / s, (y + h) / s)

    def __call__(self, img_bgr: np.ndarray) -> Optional[Detection]:
        d = self.detect_once(img_bgr)
        if d is None:
            self._miss += 1
            if self._ema is not None and self._miss <= 5:  # brief occlusion by a hand
                x1, y1, x2, y2 = self._ema
                return Detection("breadboard", 0.5, x1, y1, x2, y2)
            self._ema = None
            return None
        self._miss = 0
        box = np.array([d.x1, d.y1, d.x2, d.y2], np.float32)
        self._ema = box if self._ema is None else self.smooth * self._ema + (1 - self.smooth) * box
        x1, y1, x2, y2 = self._ema
        return Detection("breadboard", d.conf, float(x1), float(y1), float(x2), float(y2))


def synth_breadboard(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    """Synthetic top-down bench image with a breadboard (used by tests and the offline demo)."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), (40, 70, 110), np.uint8)  # wooden desk (BGR)
    img = cv2.add(img, rng.integers(0, 25, (h, w, 3), dtype=np.uint8))
    x1, y1, x2, y2 = int(w * 0.12), int(h * 0.25), int(w * 0.88), int(h * 0.82)
    cv2.rectangle(img, (x1, y1), (x2, y2), (225, 230, 232), -1)
    cv2.rectangle(img, (x1, (y1 + y2) // 2 - 4), (x2, (y1 + y2) // 2 + 4), (200, 205, 208), -1)  # channel
    for yy in range(y1 + 12, y2 - 8, 9):
        if abs(yy - (y1 + y2) // 2) < 8:
            continue
        for xx in range(x1 + 10, x2 - 8, 9):
            cv2.rectangle(img, (xx, yy), (xx + 2, yy + 2), (60, 60, 60), -1)
    return img
