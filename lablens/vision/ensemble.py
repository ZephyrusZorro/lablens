"""Multi-model component detector: class unification, slicing, Weighted Box Fusion, sanity priors.

Why an ensemble: the open-source models we ship each know a different subset of parts
(model A: capacitor/diode/LED/resistor/transistor/zener; model B: capacitor/IC/LED/resistor).
Fusing them gives wider coverage, and agreement between models is strong evidence.

Algorithms
* Class unification: every model's label is mapped to one canonical vocabulary.
* SAHI-style slicing (optional): overlapping tiles + full frame, so small parts on a 1080p bench
  view get more pixels.
* Weighted Box Fusion (Solovyev et al., 2021): boxes of the same class with IoU > thr are clustered;
  coordinates are confidence-weighted averages. Fused confidence is the mean score scaled by
  support / (number of models that *can* predict that class). A class only one model knows is not
  penalised for the other model's silence.
* Size prior: boxes covering > max_area_frac of the frame are dropped (top-down bench parts are
  small; huge boxes were the dominant false-positive mode in testing).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lablens.types import Detection
from lablens.vision.yolo_onnx import RawDetections, YoloOnnx, box_iou, nms

CANONICAL = {
    "capacitor": "capacitor", "electrolytic capacitor": "capacitor", "ceramic capacitor": "capacitor",
    "ic": "ic", "chip": "ic", "integrated circuit": "ic",
    "led": "led", "resistor": "resistor",
    "diode": "diode", "zener diode": "diode", "zener": "diode",
    "transistor": "transistor", "potentiometer": "potentiometer", "pot": "potentiometer",
    "button": "button", "push button": "button", "switch": "button",
    "breadboard": "breadboard", "arduino": "arduino", "arduino nano": "arduino", "arduino uno": "arduino",
    "battery": "battery", "buzzer": "buzzer", "inductor": "inductor",
}


def canonical(name: str) -> str:
    return CANONICAL.get(name.strip().lower(), name.strip().lower())


@dataclass
class ModelSpec:
    path: str
    imgsz: int = 640
    weight: float = 1.0
    conf: float = 0.2


@dataclass
class EnsembleConfig:
    iou_fuse: float = 0.5
    conf_out: float = 0.35
    max_area_frac: float = 0.25
    min_area_px: float = 64.0
    slice_grid: int = 0          # 0 = off, 2 = 2x2 tiles + full frame
    slice_overlap: float = 0.2
    single_model_penalty: float = 0.7  # applied when a class known by >1 model is seen by only one


def weighted_box_fusion(dets: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]], n_known: Dict[int, int],
                        iou_thr: float = 0.5, single_penalty: float = 0.8
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """dets: list of (boxes xyxy, scores, canonical class ids, model weight) per model.
    Returns fused boxes, scores, classes and the support count for each fused box."""
    allb, alls, allc, allw, allm = [], [], [], [], []
    for mi, (b, s, c, w) in enumerate(dets):
        if len(s):
            allb.append(b), alls.append(s), allc.append(c)
            allw.append(np.full(len(s), w, np.float32)), allm.append(np.full(len(s), mi))
    if not allb:
        z = np.zeros(0, np.float32)
        return np.zeros((0, 4), np.float32), z, np.zeros(0, np.int64), np.zeros(0, np.int64)
    B, S, C = np.concatenate(allb), np.concatenate(alls), np.concatenate(allc)
    W, M = np.concatenate(allw), np.concatenate(allm)
    out_b, out_s, out_c, out_n = [], [], [], []
    for cls in np.unique(C):
        idx = np.where(C == cls)[0]
        idx = idx[np.argsort(-S[idx] * W[idx])]
        clusters: List[List[int]] = []
        fused: List[np.ndarray] = []
        for i in idx:
            if fused:
                ious = box_iou(B[i:i + 1], np.stack(fused))[0]
                j = int(ious.argmax())
                if ious[j] > iou_thr:
                    clusters[j].append(i)
                    k = np.array(clusters[j])
                    wk = (S[k] * W[k])[:, None]
                    fused[j] = (B[k] * wk).sum(0) / wk.sum()
                    continue
            clusters.append([i])
            fused.append(B[i].copy())
        known = max(1, n_known.get(int(cls), 1))
        for cl, fb in zip(clusters, fused):
            k = np.array(cl)
            models = len(set(M[k].tolist()))
            # best score per model, then average over the models that know this class
            per_model = {}
            for i in k:
                per_model[M[i]] = max(per_model.get(M[i], 0.0), float(S[i] * W[i]))
            score = sum(per_model.values()) / known
            if known > 1 and models == 1:
                score = max(score, max(per_model.values()) * single_penalty)
            out_b.append(fb), out_s.append(min(score, 1.0)), out_c.append(int(cls)), out_n.append(models)
    return (np.array(out_b, np.float32), np.array(out_s, np.float32), np.array(out_c, np.int64),
            np.array(out_n, np.int64))


def _tiles(h: int, w: int, grid: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    th, tw = int(np.ceil(h / (grid - (grid - 1) * overlap))), int(np.ceil(w / (grid - (grid - 1) * overlap)))
    ys = np.linspace(0, h - th, grid).astype(int)
    xs = np.linspace(0, w - tw, grid).astype(int)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


class ComponentEnsemble:
    def __init__(self, specs: Sequence[ModelSpec], cfg: Optional[EnsembleConfig] = None):
        self.cfg = cfg or EnsembleConfig()
        self.models = [YoloOnnx(s.path, s.imgsz, s.conf) for s in specs]
        self.specs = list(specs)
        # canonical vocabulary + per-model id maps
        vocab: Dict[str, int] = {}
        self.maps: List[Dict[int, int]] = []
        for m in self.models:
            mp = {}
            for k, v in m.names.items():
                c = canonical(v)
                vocab.setdefault(c, len(vocab))
                mp[int(k)] = vocab[c]
            self.maps.append(mp)
        self.vocab = vocab
        self.names = {i: n for n, i in vocab.items()}
        self.n_known = {cid: sum(cid in mp.values() for mp in self.maps) for cid in self.names}
        self.last_ms = 0.0
        self.timing: Dict[str, float] = {}

    @property
    def classes(self) -> List[str]:
        return list(self.vocab)

    def _run_model(self, mi: int, img: np.ndarray) -> RawDetections:
        m, cfg = self.models[mi], self.cfg
        if cfg.slice_grid < 2:
            return m(img)
        h, w = img.shape[:2]
        parts = [m(img)]
        for x1, y1, x2, y2 in _tiles(h, w, cfg.slice_grid, cfg.slice_overlap):
            r = m(img[y1:y2, x1:x2])
            if len(r):
                r.boxes += np.array([x1, y1, x1, y1], np.float32)
            parts.append(r)
        boxes = np.concatenate([p.boxes for p in parts])
        scores = np.concatenate([p.scores for p in parts])
        classes = np.concatenate([p.classes for p in parts])
        keep = nms(boxes, scores, classes, 0.5)
        return RawDetections(boxes[keep], scores[keep], classes[keep], m.names)

    def detect_arrays(self, img: np.ndarray):
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        frame_area = float(h * w)
        per = []
        for mi in range(len(self.models)):
            t = time.perf_counter()
            r = self._run_model(mi, img)
            self.timing[f"model{mi}"] = (time.perf_counter() - t) * 1000
            area = (r.boxes[:, 2] - r.boxes[:, 0]) * (r.boxes[:, 3] - r.boxes[:, 1]) if len(r) else np.zeros(0)
            ok = (area <= self.cfg.max_area_frac * frame_area) & (area >= self.cfg.min_area_px)
            cls = np.array([self.maps[mi][int(c)] for c in r.classes[ok]], dtype=np.int64)
            per.append((r.boxes[ok], r.scores[ok], cls, self.specs[mi].weight))
        b, s, c, n = weighted_box_fusion(per, self.n_known, self.cfg.iou_fuse, self.cfg.single_model_penalty)
        keep = s >= self.cfg.conf_out
        b, s, c, n = b[keep], s[keep], c[keep], n[keep]
        # cross-class suppression: one physical part should carry one label
        if len(s):
            k = nms(b, s, c, 0.7, agnostic=True)
            b, s, c, n = b[k], s[k], c[k], n[k]
        self.last_ms = (time.perf_counter() - t0) * 1000
        return b, s, c, n

    def __call__(self, img: np.ndarray) -> List[Detection]:
        b, s, c, _ = self.detect_arrays(img)
        return [Detection(self.names[int(ci)], float(si), *map(float, bi)) for bi, si, ci in zip(b, s, c)]
