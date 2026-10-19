"""YOLO inference on ONNX Runtime (or OpenCV DNN) with numpy post-processing.

Pipeline: letterbox -> ONNX forward -> early score filter -> DFL decode (only surviving anchors)
-> class-aware vectorised NMS -> boxes in original image coordinates.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- geometry helpers
def letterbox(img: np.ndarray, size: int, stride: int = 32, color=(114, 114, 114)):
    """Resize keeping aspect ratio and pad to `size` x `size`. Returns (image, scale, (pad_x, pad_y))."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA)
    ph, pw = size - nh, size - nw
    top, left = ph // 2, pw // 2
    img = cv2.copyMakeBorder(img, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


def make_anchors(h: int, w: int, strides: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    pts, st = [], []
    for s in strides:
        gh, gw = int(np.ceil(h / s)), int(np.ceil(w / s))
        sy, sx = np.meshgrid(np.arange(gh, dtype=np.float32) + 0.5, np.arange(gw, dtype=np.float32) + 0.5,
                             indexing="ij")
        pts.append(np.stack([sx.ravel(), sy.ravel()], 1))
        st.append(np.full(gh * gw, s, dtype=np.float32))
    return np.concatenate(pts), np.concatenate(st)


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4) and (M,4) xyxy boxes."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, iou_thr: float = 0.45,
        max_det: int = 300, agnostic: bool = False) -> np.ndarray:
    """Class-aware greedy NMS via the coordinate-offset trick (one pass for all classes)."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    off = 0 if agnostic else classes.astype(np.float32)[:, None] * (boxes.max() + 1)
    b = boxes + off
    order = scores.argsort()[::-1]
    keep = []
    x1, y1, x2, y2 = b.T
    area = (x2 - x1) * (y2 - y1)
    while order.size and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou <= iou_thr]
    return np.asarray(keep, dtype=np.int64)


@dataclass
class RawDetections:
    boxes: np.ndarray    # (N,4) xyxy in original image pixels
    scores: np.ndarray   # (N,)
    classes: np.ndarray  # (N,) int
    names: Dict[int, str]

    def __len__(self):
        return len(self.scores)


# --------------------------------------------------------------------------- model
class YoloOnnx:
    """One YOLO detection model exported by lablens.vision.export (or by Ultralytics, see below)."""

    def __init__(self, path, imgsz: int = 640, conf: float = 0.25, iou: float = 0.45,
                 backend: str = "auto", threads: Optional[int] = None):
        self.path = Path(path)
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        meta = self._read_meta()
        self.names: Dict[int, str] = {int(k): v for k, v in meta.get("names", {}).items()}
        self.strides = meta.get("stride", [8.0, 16.0, 32.0])
        self.reg_max = int(meta.get("reg_max", 16))
        self.decoded_output = False  # True for stock Ultralytics exports (xywh + scores already decoded)
        self.backend = self._open(backend, threads)
        self._anchor_cache: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = {}
        self._proj = np.arange(self.reg_max, dtype=np.float32)
        self.last_ms = 0.0

    def _read_meta(self) -> Dict:
        side = Path(str(self.path) + ".json")
        meta: Dict = {}
        try:
            import onnxruntime as ort

            so = ort.SessionOptions()
            so.log_severity_level = 3
            m = ort.InferenceSession(str(self.path), so, providers=["CPUExecutionProvider"]).get_modelmeta()
            cm = m.custom_metadata_map
            if "names" in cm:
                names = cm["names"]
                try:
                    meta["names"] = json.loads(names)
                except json.JSONDecodeError:  # Ultralytics writes a python-literal dict
                    import ast

                    meta["names"] = ast.literal_eval(names)
            if "stride" in cm:
                s = json.loads(cm["stride"]) if cm["stride"].startswith("[") else [float(cm["stride"])]
                meta["stride"] = s if len(s) > 1 else [8.0, 16.0, 32.0]
            if "reg_max" in cm:
                meta["reg_max"] = int(cm["reg_max"])
        except Exception:
            pass
        if not meta.get("names") and side.exists():
            meta["names"] = json.loads(side.read_text(encoding="utf-8")).get("names", {})
        return meta

    def _open(self, backend: str, threads: Optional[int]) -> str:
        if backend in ("auto", "onnxruntime"):
            try:
                import onnxruntime as ort

                so = ort.SessionOptions()
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                # Multiple detectors run alongside hands, audio and the UI. Giving
                # each session every CPU core causes contention on Windows laptops.
                so.intra_op_num_threads = threads or min(4, max(1, os.cpu_count() or 2))
                so.add_session_config_entry("session.intra_op.allow_spinning", "0")
                so.log_severity_level = 3
                avail = ort.get_available_providers()
                prov = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider")
                        if p in avail] + ["CPUExecutionProvider"]
                self.sess = ort.InferenceSession(str(self.path), so, providers=prov)
                self.input_name = self.sess.get_inputs()[0].name
                out_shape = self.sess.get_outputs()[0].shape
                ch = out_shape[1] if len(out_shape) == 3 else None
                if isinstance(ch, int) and self.names and ch == 4 + len(self.names):
                    self.decoded_output = True
                return "onnxruntime:" + self.sess.get_providers()[0]
            except ImportError:
                if backend == "onnxruntime":
                    raise
        self.net = cv2.dnn.readNetFromONNX(str(self.path))
        return "opencv-dnn"

    # ------------------------------------------------------------------ inference
    def _forward(self, blob: np.ndarray) -> np.ndarray:
        if self.backend.startswith("onnxruntime"):
            return self.sess.run(None, {self.input_name: blob})[0]
        self.net.setInput(blob)
        return self.net.forward()

    def _anchors(self, h: int, w: int):
        key = (h, w)
        if key not in self._anchor_cache:
            self._anchor_cache[key] = make_anchors(h, w, self.strides)
        return self._anchor_cache[key]

    def preprocess(self, img_bgr: np.ndarray, size: Optional[int] = None):
        size = size or self.imgsz
        lb, r, pad = letterbox(img_bgr, size)
        blob = cv2.dnn.blobFromImage(lb, 1 / 255.0, swapRB=True)  # NCHW float32 RGB
        return blob, r, pad

    def decode(self, out: np.ndarray, h: int, w: int, conf: float):
        """out: (no, A). Returns xyxy (letterboxed coords), scores, classes for anchors above conf."""
        nc = len(self.names) if self.names else out.shape[0] - 4 * self.reg_max
        if self.decoded_output:  # Ultralytics-exported ONNX: rows = cx,cy,w,h, class scores
            cls = out[4:4 + nc]
            scores = cls.max(0)
            m = scores > conf
            xywh = out[:4, m].T
            classes = cls[:, m].argmax(0)
            xyxy = np.concatenate([xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2], 1)
            return xyxy, scores[m], classes
        nreg = 4 * self.reg_max
        logits = out[nreg:nreg + nc]
        # sigmoid is monotonic: filter on logits first, apply sigmoid only to survivors
        thr = np.log(conf / (1 - conf))
        best = logits.max(0)
        m = best > thr
        if not m.any():
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)
        classes = logits[:, m].argmax(0)
        scores = 1.0 / (1.0 + np.exp(-best[m]))
        dist = out[:nreg, m].reshape(4, self.reg_max, -1)
        dist = np.exp(dist - dist.max(1, keepdims=True))
        dist = (dist * self._proj[None, :, None]).sum(1) / dist.sum(1)  # (4, n) l,t,r,b in grid units
        anchors, strides = self._anchors(h, w)
        a, s = anchors[m], strides[m]
        xyxy = np.stack([a[:, 0] - dist[0], a[:, 1] - dist[1], a[:, 0] + dist[2], a[:, 1] + dist[3]], 1)
        return xyxy * s[:, None], scores.astype(np.float32), classes

    def __call__(self, img_bgr: np.ndarray, conf: Optional[float] = None, size: Optional[int] = None,
                 agnostic: bool = False) -> RawDetections:
        t0 = time.perf_counter()
        conf = self.conf if conf is None else conf
        blob, r, (px, py) = self.preprocess(img_bgr, size)
        out = self._forward(blob)[0]
        boxes, scores, classes = self.decode(out, blob.shape[2], blob.shape[3], conf)
        if len(scores):
            keep = nms(boxes, scores, classes, self.iou, agnostic=agnostic)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
            boxes = (boxes - np.array([px, py, px, py], dtype=np.float32)) / r
            h, w = img_bgr.shape[:2]
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
        self.last_ms = (time.perf_counter() - t0) * 1000
        return RawDetections(boxes, scores, classes.astype(np.int64), self.names)
