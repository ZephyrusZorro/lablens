"""Resistor colour-band reader.

Algorithm (on the crop of a detected resistor):
 1. Background model = median Lab colour of the crop border. Body = largest blob that differs from it.
 2. Orientation from the body's minimum-area rectangle; the body is warped to a horizontal strip.
 3. 1-D colour profile = per-column median Lab over the central rows (robust to glare & legs).
 4. Body colour = dominant profile colour; bands = runs of columns far (ΔE) from the body colour.
 5. Each band is classified to the nearest reference colour in Lab (chroma-weighted).
 6. Both reading directions are decoded; each is scored by code validity, tolerance-band position
    (gold/silver last, larger gap before it) and whether the value is in the E12/E24 series.
Returns {"ohms", "bands", "tolerance_pct", "confidence"} or None.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

DIGIT = {"black": 0, "brown": 1, "red": 2, "orange": 3, "yellow": 4, "green": 5, "blue": 6, "violet": 7,
         "grey": 8, "white": 9}
MULT = {**{k: 10.0 ** v for k, v in DIGIT.items() if v <= 7}, "gold": 0.1, "silver": 0.01}
TOL = {"brown": 1.0, "red": 2.0, "green": 0.5, "blue": 0.25, "violet": 0.1, "grey": 0.05, "gold": 5.0, "silver": 10.0}
# Reference colours (BGR as seen under a neutral desk lamp).
REF_BGR = {
    "black": (30, 30, 30), "brown": (20, 60, 120), "red": (30, 30, 190), "orange": (30, 130, 240),
    "yellow": (40, 215, 235), "green": (60, 140, 30), "blue": (190, 70, 30), "violet": (160, 60, 120),
    "grey": (125, 125, 125), "white": (230, 235, 235), "gold": (60, 150, 190), "silver": (195, 190, 190),
}
E24 = [1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5,
       8.2, 9.1]


def _lab(bgr) -> np.ndarray:
    a = np.asarray(bgr, np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(a, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)


_REF_NAMES = list(REF_BGR)
_REF_LAB = _lab([REF_BGR[n] for n in _REF_NAMES])


def classify_lab(lab: np.ndarray, exclude: Tuple[str, ...] = ()) -> Tuple[str, float]:
    """Nearest reference colour; chroma (a,b) weighted above lightness to tolerate lamp brightness."""
    d = _REF_LAB - lab[None]
    dist = np.sqrt(0.5 * d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)
    for i, n in enumerate(_REF_NAMES):
        if n in exclude:
            dist[i] = np.inf
    i = int(dist.argmin())
    return _REF_NAMES[i], float(dist[i])


def classify_rgb(rgb) -> str:
    """Convenience for tests: classify an (R,G,B) triple."""
    r, g, b = rgb
    return classify_lab(_lab([(b, g, r)])[0])[0]


def in_series(ohms: float) -> bool:
    if ohms <= 0:
        return False
    mant = ohms / 10 ** np.floor(np.log10(ohms))
    return any(abs(mant - e) < 0.02 for e in E24)


def _body_strip(crop: np.ndarray) -> Optional[np.ndarray]:
    h, w = crop.shape[:2]
    if h < 8 or w < 8:
        return None
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (3, 3), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    bg = np.median(border, axis=0)
    diff = np.linalg.norm(lab - bg, axis=2)
    mask = (diff > max(18.0, np.percentile(diff, 40))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    blob = (labels == k).astype(np.uint8)
    # Remove the thin leads: body thickness ~ 2x the max distance-to-edge; open with a disc
    # about half that size so wires vanish but the body survives.
    thick = 2.0 * float(cv2.distanceTransform(blob, cv2.DIST_L2, 3).max())
    ks = max(3, int(0.55 * thick) | 1)
    body = cv2.morphologyEx(blob, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks)))
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(body)
    if n2 >= 2:
        blob = (lab2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    pts = np.column_stack(np.where(blob > 0))[:, ::-1].astype(np.float32)
    if len(pts) < 30:
        return None
    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect).astype(np.float32)  # 4 corners, any order/orientation
    # order corners so the long side becomes the strip's x axis
    e01 = np.linalg.norm(box[1] - box[0])
    e12 = np.linalg.norm(box[2] - box[1])
    if e01 >= e12:
        src, L, S = np.array([box[0], box[1], box[2]]), e01, e12
    else:
        src, L, S = np.array([box[1], box[2], box[3]]), e12, e01
    L, S = int(round(L)), int(round(S))
    if L < 12 or S < 4:
        return None
    dst = np.array([[0, 0], [L - 1, 0], [L - 1, S - 1]], np.float32)
    strip = cv2.warpAffine(crop, cv2.getAffineTransform(src.astype(np.float32), dst), (L, S),
                           flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    t = int(0.04 * L)  # skip end caps / anti-aliased edges
    strip = strip[int(S * 0.25): int(np.ceil(S * 0.75)), t: L - t]  # central half of the body height
    return strip if strip.shape[0] >= 2 and strip.shape[1] >= 12 else None


def _bands(strip: np.ndarray, min_width: int = 2) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    lab = cv2.cvtColor(strip, cv2.COLOR_BGR2LAB).astype(np.float32)
    prof = np.median(lab, axis=0)  # (W, 3)
    # body colour: the most populated cluster of the profile (coarse 3-D histogram mode)
    q = (prof // 12).astype(int)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    body = prof[(q == keys[counts.argmax()]).all(1)].mean(0)
    d = prof - body
    dist = np.sqrt(0.5 * d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 2] ** 2)  # chroma-weighted ΔE
    is_band = dist > 16.0
    runs, start = [], None
    for i, f in enumerate(np.append(is_band, False)):
        if f and start is None:
            start = i
        elif not f and start is not None:
            if i - start >= min_width:
                runs.append((start, i))
            start = None
    # ignore runs touching the strip ends (body end caps / leg shadows) and slivers much narrower
    # than a typical band (edge artefacts)
    W = len(prof)
    runs = [(a, b) for a, b in runs if a > 1 and b < W - 1] or runs
    if len(runs) >= 3:
        med = float(np.median([b - a for a, b in runs]))
        runs = [(a, b) for a, b in runs if (b - a) >= 0.45 * med]
    colours = [np.median(prof[a:b], axis=0) for a, b in runs]
    return colours, runs


def _decode(names: List[str], runs: List[Tuple[int, int]]) -> Optional[Dict]:
    n = len(names)
    if n not in (3, 4, 5, 6):
        return None
    if n == 3:  # 3-band: two digits + multiplier, ±20 %
        dig, mul, tol = names[:2], names[2], None
    elif n == 4:
        dig, mul, tol = names[:2], names[2], names[3]
    else:  # 5-band (6th = tempco, ignored)
        dig, mul, tol = names[:3], names[3], names[4]
    if any(d not in DIGIT for d in dig) or mul not in MULT or (tol is not None and tol not in TOL):
        return None
    if DIGIT[dig[0]] == 0:  # leading black is almost always a mis-read direction
        return None
    value = int("".join(str(DIGIT[d]) for d in dig)) * MULT[mul]
    score = 1.0
    if tol in ("gold", "silver"):
        score += 1.0
    if n >= 4:
        gaps = [runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)]
        if gaps and gaps[-1] >= max(gaps[:-1] or [0]):
            score += 0.5  # tolerance band separated by the widest gap
    if in_series(value):
        score += 1.0
    if not (0.1 <= value <= 10e6):
        return None
    return {"ohms": float(value), "bands": names, "tolerance_pct": TOL.get(tol, 20.0) if tol else 20.0,
            "score": score}


def read_resistor(crop_bgr: np.ndarray) -> Optional[Dict]:
    strip = _body_strip(crop_bgr)
    if strip is None:
        return None
    cols, runs = _bands(strip)
    if len(cols) < 3:
        return None
    names = [classify_lab(c)[0] for c in cols]
    best = None
    for nm, rn in ((names, runs), (names[::-1], [(-b, -a) for a, b in runs[::-1]])):
        r = _decode(nm, rn)
        if r and (best is None or r["score"] > best["score"]):
            best = r
    if best is None:
        return None
    best["confidence"] = round(min(1.0, best.pop("score") / 3.5), 2)
    return best


def synth_resistor(bands: List[str], angle: float = 0.0, size=(240, 90), noise: float = 6.0, seed: int = 0,
                   body=(150, 200, 225), background=(235, 235, 235)) -> np.ndarray:
    """Synthetic resistor photo for tests: beige body, given bands, last band after a wider gap."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.full((h, w, 3), background, np.uint8)
    cv2.line(img, (0, h // 2), (w, h // 2), (170, 170, 170), 3)  # legs
    bx1, bx2, by1, by2 = int(w * 0.2), int(w * 0.8), int(h * 0.3), int(h * 0.7)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), body, -1)
    n = len(bands)
    bw = max(4, (bx2 - bx1) // 14)
    xs = [bx1 + int((bx2 - bx1) * f) for f in np.linspace(0.12, 0.55, n - 1)] + [bx1 + int((bx2 - bx1) * 0.82)]
    for x, b in zip(xs, bands):
        cv2.rectangle(img, (x, by1), (x + bw, by2), REF_BGR[b], -1)
    if angle:  # rotate on an expanded canvas so nothing is clipped
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        c, s_ = abs(M[0, 0]), abs(M[0, 1])
        nw, nh = int(h * s_ + w * c), int(h * c + w * s_)
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        img = cv2.warpAffine(img, M, (nw, nh), borderValue=background)
    if noise:
        img = np.clip(img.astype(np.int16) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img
