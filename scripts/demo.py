"""Offline end-to-end demo on a photo (no camera / microphone needed).

    python scripts/demo.py path/to/bench.jpg --question "what is this and which leg is positive?"
    python scripts/demo.py bench.jpg --point 0.62,0.48        # fingertip at 62% x, 48% y of the image
    python scripts/demo.py bench.jpg --project led_simple      # also run a Guided Build check

Writes <out>/demo_annotated.jpg and <out>/demo_report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    import cv2

    from lablens.config import get_config
    from lablens.core.assistant import LabLens
    from lablens.guided.engine import check_step, list_projects
    from lablens.vision.detector import build_detector
    from lablens.vision.hands import virtual_hand
    from lablens.vision.overlay import draw
    from lablens.vision.pointing import PointingResolver, pointing_ray

    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--question", default="What is this and how do I connect it?")
    ap.add_argument("--point", default=None, help="fingertip as fx,fy fractions; default = largest part")
    ap.add_argument("--project", default=None)
    ap.add_argument("--out", default="demo_out")
    a = ap.parse_args()

    cfg = get_config()
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    h, w = img.shape[:2]
    det = build_detector(cfg)
    t0 = time.perf_counter()
    dets = det.detect(img)
    t_det = (time.perf_counter() - t0) * 1000
    parts = [d for d in dets if d.label != "breadboard"]

    # fingertip: given, or just below-left of the most confident part, pointing up-right at it
    if a.point:
        fx, fy = (float(v) for v in a.point.split(","))
        tip, direction = (fx * w, fy * h), (0.0, -1.0)
    elif parts:
        target = max(parts, key=lambda d: d.conf)
        cx, cy = target.center
        tip = (cx - 0.6 * target.width - 10, cy + 0.6 * target.height + 10)
        direction = (cx - tip[0], cy - tip[1])
    else:
        tip, direction = (w / 2, h / 2), (0.0, -1.0)
    hand = virtual_hand(tip, direction)
    resolver = PointingResolver.from_config(cfg)
    resolver.dwell = 0.0
    sel = None
    for _ in range(resolver.window.maxlen):
        sel = resolver.update(hand, dets)
    pointed = sel.detection if sel and sel.locked else None

    extra, reading = [], None
    if pointed is not None and pointed.label == "resistor":
        from lablens.guided.engine import format_ohms
        from lablens.vision.resistor import read_resistor

        x1, y1, x2, y2 = (int(max(0, v)) for v in (pointed.x1, pointed.y1, pointed.x2, pointed.y2))
        reading = read_resistor(img[y1:y2, x1:x2])
        if reading:
            extra.append(f"COLOUR-BAND READER: bands {'-'.join(reading['bands'])} = about "
                         f"{format_ohms(reading['ohms'])} (confidence {reading['confidence']})")

    lens = LabLens.from_config(cfg)
    if not lens.store.chunks:
        lens.store.build_from_dir(cfg.path("rag.pdf_dir", "data/datasheets"))
    ans = lens.ask(a.question, pointed=pointed, extra=extra)

    guided = None
    if a.project:
        proj = list_projects(cfg.path("guided.projects_dir", "lablens/guided/projects"))[a.project]
        guided = [(i + 1, s["instruction"], check_step(s, dets)) for i, s in enumerate(proj.steps)]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vis = draw(img.copy(), dets, hand, sel, pointing_ray(hand, resolver.length_factor),
               f"LabLens demo | {len(parts)} parts | pointing: {pointed.label if pointed else '-'}")
    cv2.imwrite(str(out / "demo_annotated.jpg"), vis)

    lines = [f"# LabLens demo - {Path(a.image).name}", "",
             f"Detector: {type(det).__name__} ({t_det:.0f} ms) - classes: {', '.join(getattr(det, 'classes', []))}",
             "", "## Detections", ""]
    lines += [f"- {d.label} {d.conf:.2f} at ({d.x1:.0f},{d.y1:.0f})-({d.x2:.0f},{d.y2:.0f})" for d in dets] or ["- none"]
    lines += ["", "## Pointing", "", f"Fingertip at ({tip[0]:.0f},{tip[1]:.0f}) -> "
              f"{pointed.label + ' (locked)' if pointed else 'nothing locked'}", "",
              f"Colour-band reader: {reading if reading else 'n/a'}", "",
              "## Question", "", f"> {a.question}", "", "## Answer", "", ans.text, "",
              f"*backend: {ans.backend} - {ans.latency_s:.2f}s - cached: {ans.cached}*", "", "Sources:"]
    lines += [f"- {c.source} p.{c.page} (score {c.score:.2f})" for c in ans.citations]
    if guided:
        lines += ["", f"## Guided Build check: {a.project}", ""]
        lines += [f"- step {i}: {'PASS' if r.ok else 'WAIT'} - {instr}" + ("" if r.ok else f"  ({' '.join(r.messages)})")
                  for i, instr, r in guided]
    (out / "demo_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out / 'demo_annotated.jpg'} and {out / 'demo_report.md'}")


if __name__ == "__main__":
    main()
