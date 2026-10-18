"""Shared data types used across vision, RAG and the app."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Point = Tuple[float, float]


@dataclass
class Detection:
    """One YOLO box in pixel coordinates."""
    label: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: Optional[int] = None

    @property
    def center(self) -> Point:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    def contains(self, p: Point, pad: float = 0.0) -> bool:
        return (self.x1 - pad) <= p[0] <= (self.x2 + pad) and (self.y1 - pad) <= p[1] <= (self.y2 + pad)


@dataclass
class HandObservation:
    """21 MediaPipe landmarks in pixel coordinates (x, y)."""
    landmarks: List[Point]
    handedness: str = "Right"

    @property
    def index_tip(self) -> Point:
        return self.landmarks[8]

    @property
    def index_pip(self) -> Point:
        return self.landmarks[6]


@dataclass
class Selection:
    """The component the user is currently pointing at."""
    detection: Detection
    locked: bool
    stability: float  # 0..1, share of recent frames agreeing


@dataclass
class Citation:
    source: str
    page: int
    snippet: str
    score: float = 0.0


@dataclass
class Answer:
    text: str
    citations: List[Citation] = field(default_factory=list)
    backend: str = ""
    latency_s: float = 0.0
    cached: bool = False
