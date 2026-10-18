"""Drawing helpers shared by the Streamlit app, the desktop app and the offline demo."""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from lablens.types import Detection, HandObservation, Selection
from lablens.vision.hands import HAND_CONNECTIONS

PALETTE = {
    "breadboard": (180, 180, 180), "ic": (255, 140, 0), "resistor": (0, 200, 255), "led": (60, 60, 255),
    "capacitor": (255, 80, 200), "diode": (120, 255, 120), "transistor": (255, 255, 0),
}


def _label(img, text, x, y, color, scale=0.55):
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw(frame: np.ndarray, dets: Iterable[Detection], hand: Optional[HandObservation] = None,
         sel: Optional[Selection] = None, ray: Optional[tuple] = None, banner: str = "") -> np.ndarray:
    out = frame
    sel_key = (sel.detection.label, sel.detection.track_id) if sel else None
    for d in dets:
        color = PALETTE.get(d.label, (200, 200, 200))
        is_sel = sel_key == (d.label, d.track_id)
        thick = 4 if is_sel else (1 if d.label == "breadboard" else 2)
        if is_sel:
            color = (0, 255, 0) if sel.locked else (0, 255, 255)
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        cv2.rectangle(out, p1, p2, color, thick)
        _label(out, f"{d.label} {d.conf:.2f}" + (" [LOCKED]" if is_sel and sel.locked else ""), p1[0], p1[1], color)
    if hand is not None:
        pts = [(int(x), int(y)) for x, y in hand.landmarks]
        if hand.handedness != "Virtual":
            for a, b in HAND_CONNECTIONS:
                cv2.line(out, pts[a], pts[b], (230, 230, 230), 1, cv2.LINE_AA)
        if ray is not None:
            (x0, y0), (x1, y1) = ray
            cv2.arrowedLine(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.15)
        cv2.circle(out, pts[8], 7, (0, 0, 255), -1, cv2.LINE_AA)
    if banner:
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (25, 25, 25), -1)
        cv2.putText(out, banner, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out
