"""Component detection front-end used by the apps.

    frame ─► ComponentEnsemble (ONNX models, WBF) ─┐
          └► BreadboardDetector (classical CV) ────┴─► IouTracker (stable ids, smoothed boxes)

`AsyncDetector` runs this in a background thread so the camera / hand-tracking loop never waits for
the (slower) detector: the UI always shows the latest finished result, and boxes are carried by the
tracker between detector updates.

Models: every `*.onnx` listed in config `vision.models` (or found in models/). A `.pt` from
Ultralytics is converted on first use with the torch-free exporter.
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from lablens.types import Detection

log = logging.getLogger(__name__)


def iou(a: Detection, b: Detection) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    """Greedy IoU tracker with box smoothing and short-term memory.

    Gives each physical part a stable id (the pointing smoother votes on ids) and keeps a part alive
    for a few frames when a hand briefly occludes it.
    """

    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 8, smooth: float = 0.5, min_hits: int = 1):
        self.iou_thresh, self.max_missed, self.smooth, self.min_hits = iou_thresh, max_missed, smooth, min_hits
        self._tracks: dict = {}  # id -> [Detection, missed, hits]
        self._ids = itertools.count(1)

    def update(self, dets: List[Detection]) -> List[Detection]:
        unmatched = set(self._tracks)
        out: List[Detection] = []
        pairs = sorted(((iou(self._tracks[t][0], d), t, i) for t in self._tracks for i, d in enumerate(dets)
                        if self._tracks[t][0].label == d.label), reverse=True)
        used_t, used_d = set(), set()
        for v, t, i in pairs:
            if v < self.iou_thresh or t in used_t or i in used_d:
                continue
            used_t.add(t), used_d.add(i)
            prev, _, hits = self._tracks[t]
            d, a = dets[i], self.smooth
            d = Detection(d.label, d.conf, a * prev.x1 + (1 - a) * d.x1, a * prev.y1 + (1 - a) * d.y1,
                          a * prev.x2 + (1 - a) * d.x2, a * prev.y2 + (1 - a) * d.y2, t)
            self._tracks[t] = [d, 0, hits + 1]
            unmatched.discard(t)
        for i, d in enumerate(dets):
            if i not in used_d:
                t = next(self._ids)
                self._tracks[t] = [Detection(d.label, d.conf, d.x1, d.y1, d.x2, d.y2, t), 0, 1]
        for t in list(unmatched):
            self._tracks[t][1] += 1
            if self._tracks[t][1] > self.max_missed:
                del self._tracks[t]
        for t, (d, missed, hits) in self._tracks.items():
            if hits >= self.min_hits:
                out.append(d if missed == 0 else Detection(d.label, d.conf * 0.9 ** missed, d.x1, d.y1, d.x2, d.y2, t))
        return out


class BaseDetector:
    name = "base"

    def detect(self, frame: np.ndarray) -> List[Detection]:  # pragma: no cover - interface
        raise NotImplementedError


class ComponentDetector(BaseDetector):
    name = "ensemble"

    def __init__(self, ensemble, breadboard=None, tracker: Optional[IouTracker] = None):
        self.ensemble = ensemble
        self.breadboard = breadboard
        self.tracker = tracker or IouTracker()
        self.last_ms = 0.0

    @property
    def classes(self) -> List[str]:
        return sorted(set(self.ensemble.classes) | ({"breadboard"} if self.breadboard else set()))

    def detect_raw(self, frame: np.ndarray) -> List[Detection]:
        t0 = time.perf_counter()
        dets = self.ensemble(frame)
        if self.breadboard is not None:
            bb = self.breadboard(frame)
            if bb is not None:
                dets.append(bb)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return dets

    def detect(self, frame: np.ndarray) -> List[Detection]:
        return self.tracker.update(self.detect_raw(frame))


class MockDetector(BaseDetector):
    """Fixed fake layout - lets you develop the UI without any model (`vision.mock: true`)."""

    name = "mock"
    classes = ["breadboard", "ic", "resistor", "led", "capacitor"]

    def __init__(self, layout: Optional[List[Detection]] = None):
        self.layout = layout
        self.last_ms = 0.0

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        if self.layout is not None:
            return [Detection(**d.__dict__) for d in self.layout]
        return [
            Detection("breadboard", 0.95, 0.10 * w, 0.30 * h, 0.90 * w, 0.90 * h, 1),
            Detection("ic", 0.90, 0.40 * w, 0.50 * h, 0.52 * w, 0.62 * h, 2),
            Detection("resistor", 0.85, 0.20 * w, 0.40 * h, 0.32 * w, 0.46 * h, 3),
            Detection("led", 0.88, 0.65 * w, 0.42 * h, 0.70 * w, 0.52 * h, 4),
            Detection("capacitor", 0.80, 0.75 * w, 0.62 * h, 0.81 * w, 0.72 * h, 5),
        ]


class AsyncDetector:
    """Run a detector on the newest frame in a background thread (drops stale frames)."""

    def __init__(self, detector: BaseDetector):
        self.detector = detector
        self._frame: Optional[np.ndarray] = None
        self._result: List[Detection] = []
        self._cv = threading.Condition()
        self._stop = False
        self.fps = 0.0
        self.updated_at = 0.0
        self._t = threading.Thread(target=self._loop, daemon=True, name="lablens-detector")
        self._t.start()

    def submit(self, frame: np.ndarray) -> None:
        with self._cv:
            self._frame = frame
            self._cv.notify()

    def latest(self) -> List[Detection]:
        return list(self._result)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._frame is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                frame, self._frame = self._frame, None
            t0 = time.perf_counter()
            try:
                self._result = self.detector.detect(frame)
            except Exception as e:  # never kill the UI because of one bad frame
                log.exception("detector failed: %s", e)
            dt = time.perf_counter() - t0
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / max(dt, 1e-3))
            self.updated_at = time.monotonic()

    def close(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()


def _resolve_model(path: Path) -> Optional[Path]:
    """Return an ONNX path for `path`, converting a .pt next to it if needed."""
    if path.suffix == ".onnx" and path.exists():
        return path
    pt = path if path.suffix == ".pt" else path.with_suffix(".pt")
    onnx = pt.with_suffix(".onnx")
    if onnx.exists():
        return onnx
    if pt.exists():
        from lablens.vision.export import export

        log.info("converting %s -> %s (torch-free exporter)", pt.name, onnx.name)
        export(pt, onnx)
        return onnx
    return None


def build_detector(cfg) -> BaseDetector:
    if cfg.get("vision.mock", False):
        return MockDetector()
    from lablens.vision.breadboard import BreadboardDetector
    from lablens.vision.ensemble import ComponentEnsemble, EnsembleConfig, ModelSpec

    specs = []
    for m in cfg.get("vision.models", []) or []:
        p = _resolve_model(_abs(cfg, m["path"]))
        if p is None:
            log.warning("model %s not found (run scripts/setup_models.py)", m["path"])
            continue
        specs.append(ModelSpec(str(p), int(m.get("imgsz", 640)), float(m.get("weight", 1.0)),
                               float(m.get("conf", 0.2))))
    if not specs:
        log.warning("No detection models found - using MockDetector. Run: python scripts/setup_models.py")
        return MockDetector()
    e = cfg.get("vision.ensemble", {}) or {}
    ens = ComponentEnsemble(specs, EnsembleConfig(
        iou_fuse=e.get("iou_fuse", 0.5), conf_out=e.get("conf", 0.30), max_area_frac=e.get("max_area_frac", 0.25),
        slice_grid=e.get("slice_grid", 0), single_model_penalty=e.get("single_model_penalty", 0.8)))
    bb = BreadboardDetector() if cfg.get("vision.breadboard_cv", True) else None
    return ComponentDetector(ens, bb)


def _abs(cfg, p: str) -> Path:
    from lablens.config import ROOT

    q = Path(p)
    return q if q.is_absolute() else ROOT / q
