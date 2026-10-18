"""LabLens desktop app (OpenCV window) - lowest latency, best for live demos.

    python -m lablens.live                 # webcam
    python -m lablens.live --source x.mp4  # video / image file

Keys:  SPACE ask by voice   ENTER type a question   1-4 start guided project   N next step
       R repeat   S stop talking   X stop project   Q / ESC quit
Mouse: hold the left button to point with the cursor (works without a hand tracker).
"""
from __future__ import annotations

import argparse
import textwrap
import time

import cv2
import numpy as np

from lablens.config import get_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--no-voice", action="store_true")
    a = ap.parse_args()
    cfg = get_config()
    from lablens.runtime import Runtime

    rt = Runtime(cfg, source=a.source if a.source is not None else cfg.get("vision.camera_index", 0),
                 speak=not a.no_voice)
    win = "LabLens"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    mouse_down = {"v": False}

    def on_mouse(ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            mouse_down["v"] = True
        elif ev == cv2.EVENT_LBUTTONUP:
            mouse_down["v"] = False
            rt.set_mouse(None)
        if mouse_down["v"]:
            rt.set_mouse(x, y)

    cv2.setMouseCallback(win, on_mouse)
    typing, buf = False, ""
    project_ids = list(rt.projects)
    rt.say("LabLens ready. Point at a part and ask.")
    try:
        while True:
            snap = rt.snapshot()
            if snap.frame is None:
                frame = np.zeros((480, 800, 3), np.uint8)
                cv2.putText(frame, "Waiting for camera...", (20, 240), 0, 0.9, (255, 255, 255), 2)
            else:
                frame = snap.frame.copy()
            h, w = frame.shape[:2]
            # transcript panel
            lines = []
            for role, text in list(rt.messages)[-4:]:
                for ln in textwrap.wrap(("You: " if role == "you" else "LabLens: ") + text, 95)[:3]:
                    lines.append((role, ln))
            if snap.guided:
                g = snap.guided
                lines.insert(0, ("guided", f"[{g['step']}/{g['total']}] {g['instruction'][:80]} -> "
                                           f"{'OK' if g['ok'] else ' '.join(g['messages'])[:60]}"))
            if typing:
                lines.append(("you", "> " + buf + "_"))
            panel_h = 22 * len(lines) + 10
            if lines:
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - panel_h), (w, h), (20, 20, 20), -1)
                frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
                for i, (role, ln) in enumerate(lines):
                    col = {"you": (180, 255, 180), "guided": (0, 220, 255)}.get(role, (255, 255, 255))
                    cv2.putText(frame, ln, (10, h - panel_h + 22 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1,
                                cv2.LINE_AA)
            cv2.imshow(win, frame)
            k = cv2.waitKey(15) & 0xFF
            if k == 255:
                continue
            if typing:
                if k in (13, 10):
                    typing = False
                    rt.ask(buf)
                    buf = ""
                elif k == 27:
                    typing, buf = False, ""
                elif k == 8:
                    buf = buf[:-1]
                elif 32 <= k < 127:
                    buf += chr(k)
                continue
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                rt.listen_and_ask()
            elif k in (13, 10):
                typing, buf = True, ""
            elif k == ord("n"):
                rt.next_step()
            elif k == ord("r"):
                rt.repeat()
            elif k == ord("s"):
                rt.stop_talking()
            elif k == ord("x"):
                rt.stop_project()
            elif ord("1") <= k <= ord("9") and k - ord("1") < len(project_ids):
                rt.start_project(project_ids[k - ord("1")])
    finally:
        rt.stop()
        cv2.destroyAllWindows()
        time.sleep(0.1)


if __name__ == "__main__":
    main()
