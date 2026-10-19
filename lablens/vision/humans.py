"""Local person/face/pose recognition, separate from the electronics vocabulary.

Only visible landmarks are displayed. Person segmentation rejects component boxes
on a person; it is never used as evidence that a circuit is assembled correctly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import cv2
import numpy as np

from lablens.types import Detection

log = logging.getLogger(__name__)
HUMAN_CLASSES = {"person", "head", "torso", "left arm", "right arm", "left leg", "right leg",
                 "left hand", "right hand", "left foot", "right foot"}
JOINT_NAMES = {0: "nose", 2: "left eye", 5: "right eye", 7: "left ear", 8: "right ear",
               11: "left shoulder", 12: "right shoulder", 13: "left elbow", 14: "right elbow",
               15: "left wrist", 16: "right wrist", 23: "left hip", 24: "right hip",
               25: "left knee", 26: "right knee", 27: "left ankle", 28: "right ankle"}
CONNECTIONS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24),
               (23, 24), (23, 25), (25, 27), (24, 26), (26, 28), (27, 31), (28, 32)]
GROUPS = {"head": list(range(11)), "torso": [11, 12, 23, 24],
          "left arm": [11, 13, 15], "right arm": [12, 14, 16],
          "left leg": [23, 25, 27], "right leg": [24, 26, 28],
          "left hand": [15, 17, 19, 21], "right hand": [16, 18, 20, 22],
          "left foot": [27, 29, 31], "right foot": [28, 30, 32]}


@dataclass
class HumanResult:
    detections: List[Detection] = field(default_factory=list)
    poses: list = field(default_factory=list)  # lists of (x_px, y_px, confidence)
    mask: np.ndarray | None = None
    exclusions: List[Detection] = field(default_factory=list)


def pose_detections(points, width, height, threshold=0.55):
    """Never draw off-screen or low-visibility body parts as detected anatomy."""
    visible = {i: p for i, p in enumerate(points)
               if p[2] >= threshold and 0 <= p[0] < width and 0 <= p[1] < height}
    if len(visible) < 4:
        return []

    def box(name, group, pad):
        pts = [visible[i] for i in group if i in visible]
        x, y, score = np.array(pts).T
        return Detection(name, float(np.mean(score)), max(0, float(x.min()) - pad),
                         max(0, float(y.min()) - pad), min(width, float(x.max()) + pad),
                         min(height, float(y.max()) + pad))

    out = [box("person", list(visible), max(width, height) * 0.03)]
    for name, indices in GROUPS.items():
        # A full limb needs both endpoints and the intermediate joint; do not
        # invent legs that are hidden below a desk or outside the camera.
        if all(i in visible for i in indices) or (name == "head" and sum(i in visible for i in indices) >= 4):
            out.append(box(name, indices, max(6, min(width, height) * 0.012)))
    return out


def boxes_match(a, b):
    """Associate nested face/pose boxes without merging adjacent people's boxes."""
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return intersection / max(1, min(a.area, b.area)) >= 0.65


def filter_components(dets, human: HumanResult, overlap=0.55, scene_mode="mixed"):
    """Reject electronics over a detected human silhouette or head.

    Person and furniture boxes reject overlapping components; parts beside them survive.
    A component held directly against the body can be suppressed; use the bench.
    """
    if scene_mode == "people":
        return []
    out = []
    for d in dets:
        covered = 0.0
        if d.area <= 0:
            continue
        for head in [h for h in human.detections if h.label in ("head", "person")] + human.exclusions:
            inter = max(0, min(d.x2, head.x2) - max(d.x1, head.x1)) * max(
                0, min(d.y2, head.y2) - max(d.y1, head.y1))
            covered = max(covered, inter / d.area)
        if human.mask is not None:
            h, w = human.mask.shape
            x1, y1 = max(0, int(d.x1)), max(0, int(d.y1))
            x2, y2 = min(w, int(np.ceil(d.x2))), min(h, int(np.ceil(d.y2)))
            roi = human.mask[y1:y2, x1:x2]
            if roi.size:
                covered = max(covered, float(np.mean(roi > 0.6)))
        if covered < overlap:
            out.append(d)
    if scene_mode == "auto" and any(h.label == "person" for h in human.detections):
        # A room/person-facing webcam is not a verified electronics bench.
        # Keep lab parts only on a breadboard outside the human/background boxes.
        boards = [d for d in out if d.label == "breadboard"]
        out = [d for d in out if d.label == "breadboard" or any(b.contains(d.center) for b in boards)]
    return out


class HumanDetector:
    def __init__(self, cfg):
        self.pose = self.face = None
        self.pose_enabled = True
        self.max_poses = int(cfg.get("vision.humans.max_poses", 6))
        self.general = None
        self.scene_mode = cfg.get("vision.humans.scene_mode", "auto")
        self.last = HumanResult()
        self.error = ""
        self._closed = False
        self.threshold = cfg.get("vision.humans.visibility", 0.55)
        if not cfg.get("vision.humans.enabled", True):
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (PoseLandmarker, PoseLandmarkerOptions,
                                                       FaceDetector, FaceDetectorOptions, RunningMode)
            self.mp = mp
            general = cfg.path("vision.humans.person_model", "models/person_yolov8n.onnx")
            if Path(general).exists():
                from lablens.vision.yolo_onnx import YoloOnnx
                self.general = YoloOnnx(general, imgsz=640, conf=0.4, threads=2)
            pose = cfg.path("vision.humans.pose_model", "models/pose_landmarker_lite.task")
            face = cfg.path("vision.humans.face_model", "models/blaze_face_short_range.tflite")
            if Path(pose).exists():
                self.pose = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(pose)), running_mode=RunningMode.IMAGE,
                    num_poses=1, output_segmentation_masks=True))
            if Path(face).exists():
                self.face = FaceDetector.create_from_options(FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=str(face)), min_detection_confidence=0.6))
            if self.pose is None or self.face is None or self.general is None:
                self.error = "Human models missing; run scripts/setup_models.py"
        except Exception as e:
            self.error = f"Human model initialization failed: {e}"
            log.warning(self.error)

    @property
    def available(self):
        return self.pose is not None or self.face is not None or self.general is not None

    def detect(self, frame):
        result = HumanResult()
        if not self.available:
            self.last = result
            return result
        h, w = frame.shape[:2]
        if self.general:
            raw = self.general(frame)
            for box, score, cls in zip(raw.boxes, raw.scores, raw.classes):
                label = raw.names[int(cls)].lower()
                detection = Detection(label, float(score), *map(float, box))
                if label == "person":
                    result.detections.append(detection)
                elif label in ("chair", "couch", "tv", "laptop", "keyboard", "bed", "refrigerator"):
                    result.exclusions.append(detection)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                              data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if self.pose and self.pose_enabled:
            # Estimate each detected person's anatomy in their own crop. A single
            # full-frame pose pass loses small people and can mix limbs in crowds.
            people = sorted([d for d in result.detections if d.label == "person"],
                            key=lambda d: d.conf, reverse=True)[:self.max_poses]
            crops = []
            for person in people:
                pad = max(person.x2-person.x1, person.y2-person.y1) * .12
                crops.append((max(0, int(person.x1-pad)), max(0, int(person.y1-pad)),
                              min(w, int(person.x2+pad)), min(h, int(person.y2+pad))))
            if not crops:
                crops = [(0, 0, w, h)]
            for x1, y1, x2, y2 in crops:
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)))
                pose = self.pose.detect(crop)
                for landmarks in pose.pose_landmarks:
                    points = [(x1+p.x*(x2-x1), y1+p.y*(y2-y1), min(p.visibility or 0, p.presence or 0))
                              for p in landmarks]
                    result.poses.append(points)
                    for d in pose_detections(points, w, h, self.threshold):
                        if d.label != "person" or not people:
                            result.detections.append(d)
                if pose.segmentation_masks:
                    if result.mask is None:
                        result.mask = np.zeros((h, w), np.float32)
                    mask = np.maximum.reduce([m.numpy_view().copy() for m in pose.segmentation_masks])
                    result.mask[y1:y2, x1:x2] = np.maximum(result.mask[y1:y2, x1:x2],
                                                         cv2.resize(mask, (x2-x1, y2-y1)))
        if self.face:
            for face in self.face.detect(image).detections:
                b = face.bounding_box
                head = Detection("head", face.categories[0].score, max(0, b.origin_x), max(0, b.origin_y),
                                 min(w, b.origin_x + b.width), min(h, b.origin_y + b.height))
                # Face fallback handles a close-up webcam view with no full pose.
                if not any(d.label == "head" and boxes_match(d, head) for d in result.detections):
                    result.detections.append(head)
                if not any(d.label == "person" and boxes_match(d, head) for d in result.detections):
                    result.detections.append(Detection("person", head.conf, head.x1, head.y1, head.x2, head.y2))
        self.last = result
        return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        for model in (self.pose, self.face):
            if model is not None:
                model.close()


def draw_pose(frame, result, labels=False):
    color = (230, 210, 70)
    for points in result.poses:
        visible = {i: (int(x), int(y)) for i, (x, y, v) in enumerate(points)
                   if v >= 0.55 and 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]}
        for a, b in CONNECTIONS:
            if a in visible and b in visible:
                cv2.line(frame, visible[a], visible[b], color, 2, cv2.LINE_AA)
        for i, xy in visible.items():
            if i in JOINT_NAMES:
                cv2.circle(frame, xy, 4, color, -1, cv2.LINE_AA)
                if labels:
                    cv2.putText(frame, JOINT_NAMES[i], (xy[0] + 5, xy[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return frame
