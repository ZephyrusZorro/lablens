"""Resolve which detected component the index finger is pointing at.

Geometry: a ray from landmark 6 (index PIP) through landmark 8 (index tip), extended forward.
Selection prefers the first non-container box the ray hits; containers (breadboard) only win
if nothing else is under the finger. Results are smoothed over N frames and must dwell before
locking. A lock is "sticky" for a grace period so you can lower your hand and then ask.
"""
from __future__ import annotations

import math
import time
from collections import Counter, deque
from typing import Deque, Iterable, List, Optional, Tuple

from lablens.types import Detection, HandObservation, Point, Selection
from lablens.vision.humans import HUMAN_CLASSES

CONTAINER_CLASSES = {"breadboard"}


def segment_box_t(p0: Point, p1: Point, d: Detection) -> Optional[float]:
    """Liang-Barsky clip. Returns entry parameter t in [0,1] where segment p0->p1 enters the box."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, p0[0] - d.x1), (dx, d.x2 - p0[0]), (-dy, p0[1] - d.y1), (dy, d.y2 - p0[1])):
        if p == 0:
            if q < 0:
                return None
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return None
    return t0


def _dist_to_box(p: Point, d: Detection) -> float:
    dx = max(d.x1 - p[0], 0, p[0] - d.x2)
    dy = max(d.y1 - p[1], 0, p[1] - d.y2)
    return math.hypot(dx, dy)


def pointing_ray(hand: HandObservation, length_factor: float = 2.5) -> Tuple[Point, Point]:
    (x6, y6), (x8, y8) = hand.index_pip, hand.index_tip
    return (x8, y8), (x8 + (x8 - x6) * length_factor, y8 + (y8 - y6) * length_factor)


def resolve_pointed(hand: HandObservation, dets: Iterable[Detection], length_factor: float = 2.5,
                    nearest_max_px: float = 120.0) -> Optional[Detection]:
    """Single-frame decision."""
    dets = [d for d in dets if d.label not in HUMAN_CLASSES]
    if not dets:
        return None
    start, end = pointing_ray(hand, length_factor)
    hits: List[Tuple[int, float, float, Detection]] = []
    for d in dets:
        t = segment_box_t(start, end, d)
        if t is not None:
            is_container = 1 if d.label in CONTAINER_CLASSES else 0
            hits.append((is_container, t, d.area, d))
    parts = [h for h in hits if h[0] == 0]
    if parts:
        return min(parts, key=lambda h: (h[1], h[2]))[3]
    # Ray missed every part: nearest non-container box to the fingertip
    near = [(d, _dist_to_box(hand.index_tip, d)) for d in dets if d.label not in CONTAINER_CLASSES]
    near = [x for x in near if x[1] <= nearest_max_px]
    if near:
        return min(near, key=lambda x: x[1])[0]
    if hits:
        return min(hits, key=lambda h: (h[1], h[2]))[3]
    return None


def _key(d: Detection) -> str:
    return f"{d.label}#{d.track_id}" if d.track_id is not None else d.label


class PointingResolver:
    """Temporal smoothing + dwell lock + sticky hold."""

    def __init__(self, smoothing_frames: int = 10, dwell_seconds: float = 0.5, min_agreement: float = 0.6,
                 length_factor: float = 2.5, nearest_max_px: float = 120.0, sticky_seconds: float = 4.0,
                 clock=time.monotonic):
        self.window: Deque[Optional[str]] = deque(maxlen=smoothing_frames)
        self.dwell = dwell_seconds
        self.min_agreement = min_agreement
        self.length_factor = length_factor
        self.nearest_max_px = nearest_max_px
        self.sticky = sticky_seconds
        self.clock = clock
        self._candidate: Optional[str] = None
        self._candidate_since = 0.0
        self._latest: dict[str, Detection] = {}
        self.locked: Optional[Detection] = None
        self._locked_at = 0.0
        self._last_seen_hand = 0.0

    @classmethod
    def from_config(cls, cfg) -> "PointingResolver":
        return cls(cfg.get("pointing.smoothing_frames", 10), cfg.get("pointing.dwell_seconds", 0.5),
                   cfg.get("pointing.min_agreement", 0.6), cfg.get("pointing.ray_length_factor", 2.5),
                   cfg.get("pointing.nearest_max_px", 120))

    def update(self, hand: Optional[HandObservation], dets: List[Detection]) -> Optional[Selection]:
        now = self.clock()
        dets = [d for d in dets if d.label not in HUMAN_CLASSES]
        self._latest = {_key(d): d for d in dets}
        if self.locked and _key(self.locked) not in self._latest:
            self.clear()
        if hand is None:
            self.window.append(None)
            if self.locked and now - self._last_seen_hand <= self.sticky:
                # keep the lock alive, refreshed with the latest box of the same track if visible
                self._refresh_locked(dets)
                return Selection(self.locked, True, 1.0)
            self.locked = None
            return None
        self._last_seen_hand = now
        d = resolve_pointed(hand, dets, self.length_factor, self.nearest_max_px)
        k = _key(d) if d else None
        if d:
            self._latest[k] = d
        self.window.append(k)
        votes = Counter(x for x in self.window if x is not None)
        if not votes:
            self._candidate = None
            return Selection(self.locked, True, 1.0) if self.locked else None
        mode, n = votes.most_common(1)[0]
        stability = n / len(self.window)
        if mode != self._candidate:
            self._candidate, self._candidate_since = mode, now
        best = self._latest.get(mode)
        if stability >= self.min_agreement and now - self._candidate_since >= self.dwell and best is not None:
            if self.locked is None or _key(self.locked) != mode:
                self._locked_at = now
            self.locked = best
            return Selection(best, True, stability)
        if best is None:
            return None
        return Selection(best, False, stability)

    def _refresh_locked(self, dets: List[Detection]) -> None:
        if not self.locked:
            return
        k = _key(self.locked)
        for d in dets:
            if _key(d) == k:
                self.locked = d
                return

    def clear(self) -> None:
        self.window.clear()
        self.locked = None
        self._candidate = None
