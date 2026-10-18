"""Guided Build: closed-loop, camera-verified step-by-step instructions.

Scope (be honest with judges): we verify PRESENCE, COUNT and APPROXIMATE PLACEMENT of parts relative
to the breadboard box. We do NOT verify electrical connectivity / wiring correctness.

Step schema (see projects/*.json):
  expect: [{label, min=1, max=None, on_board=true, zone=[x1,y1,x2,y2] (breadboard-normalised)}]
  near:   [[label_a, label_b, max_dist_fraction_of_board_diagonal]]
  value:  {label: "resistor", ohms: 220, tolerance: 0.1}       (uses the colour-band reader if available)
"""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional

from lablens.types import Detection

BOARD = "breadboard"


@dataclass
class StepResult:
    ok: bool
    missing: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.ok:
            return "Looks good. Give me a thumbs up for the next step."
        return " ".join(self.messages[:2])


def _plural(label: str, n: int) -> str:
    return label if n == 1 else (label + "es" if label.endswith("s") else label + "s")


def _board(dets: List[Detection]) -> Optional[Detection]:
    boards = [d for d in dets if d.label == BOARD]
    return max(boards, key=lambda d: d.area) if boards else None


def _norm(d: Detection, board: Detection):
    cx, cy = d.center
    return ((cx - board.x1) / max(board.width, 1), (cy - board.y1) / max(board.height, 1))


def check_step(step: dict, dets: List[Detection], value_reader: Optional[Callable[[Detection], Optional[float]]] = None
               ) -> StepResult:
    res = StepResult(True)
    board = _board(dets)
    by_label: Dict[str, List[Detection]] = {}
    for d in dets:
        by_label.setdefault(d.label, []).append(d)

    for e in step.get("expect", []):
        label, lo, hi = e["label"], e.get("min", 1), e.get("max")
        found = by_label.get(label, [])
        if e.get("on_board", label != BOARD) and board is not None and label != BOARD:
            found = [d for d in found if board.contains(d.center, pad=0.03 * board.width)]
        if len(found) < lo:
            res.ok = False
            res.missing.append(label)
            if not found:
                res.messages.append(f"I don't see the {label} yet.")
            else:
                res.messages.append(f"I see {len(found)} {_plural(label, len(found))}, but this step needs {lo}.")
            continue
        if hi is not None and len(found) > hi:
            res.ok = False
            res.messages.append(f"I see {len(found)} {_plural(label, len(found))}; this step expects at most {hi}.")
        if "zone" in e and board is not None:
            x1, y1, x2, y2 = e["zone"]
            inside = [d for d in found if x1 <= _norm(d, board)[0] <= x2 and y1 <= _norm(d, board)[1] <= y2]
            if len(inside) < lo:
                res.ok = False
                res.messages.append(e.get("zone_hint", f"The {label} isn't where I expect it on the breadboard."))

    if board is not None:
        diag = math.hypot(board.width, board.height)
        for a, b, frac in step.get("near", []):
            A, B = by_label.get(a, []), by_label.get(b, [])
            if A and B:
                dmin = min(math.dist(x.center, y.center) for x in A for y in B)
                if dmin > frac * diag:
                    res.ok = False
                    res.messages.append(f"The {a} should be right next to the {b}.")

    v = step.get("value")
    if v and value_reader is not None:
        for d in by_label.get(v["label"], []):
            ohms = value_reader(d)
            if ohms and abs(ohms - v["ohms"]) / v["ohms"] > v.get("tolerance", 0.1):
                res.ok = False
                res.messages.append(f"That resistor reads about {format_ohms(ohms)}, but this step needs "
                                    f"{format_ohms(v['ohms'])}.")
    return res


def format_ohms(ohms: float) -> str:
    for div, suf in ((1e6, "M"), (1e3, "k")):
        if ohms >= div:
            v = ohms / div
            return f"{v:g}{suf} ohm"
    return f"{ohms:g} ohm"


class Project:
    def __init__(self, data: dict):
        self.data = data
        self.id = data["id"]
        self.title = data["title"]
        self.steps: List[dict] = data["steps"]

    @classmethod
    def load(cls, path: Path) -> "Project":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def list_projects(directory: Path) -> Dict[str, Project]:
    return {p.id: p for p in (Project.load(f) for f in sorted(Path(directory).glob("*.json")))}


class GuidedSession:
    """State machine. Call observe(dets) every frame; call next()/repeat() from gestures or buttons."""

    def __init__(self, project: Project, events=None, window: int = 15, required: float = 0.7,
                 value_reader=None, clock=time.monotonic):
        self.project = project
        self.idx = 0
        self.events = events
        self.clock = clock
        self.value_reader = value_reader
        self._hist: Deque[StepResult] = deque(maxlen=window)
        self.required = required
        self.step_started = clock()
        self.finished = False
        self._announced_ok = False
        self._last_refusal = -1e9
        self.override_window = 6.0  # a second thumbs-up within this many seconds skips verification
        if events:
            events.log("project_start", project.id, 0)

    @property
    def step(self) -> dict:
        return self.project.steps[self.idx]

    @property
    def step_label(self) -> str:
        return f"Step {self.idx + 1} of {len(self.project.steps)}: {self.step['instruction']}"

    def observe(self, dets: List[Detection]) -> StepResult:
        r = check_step(self.step, dets, self.value_reader)
        self._hist.append(r)
        return self.status

    @property
    def status(self) -> StepResult:
        if not self._hist:
            return StepResult(False, [], ["Waiting for the camera..."])
        oks = sum(r.ok for r in self._hist)
        if oks / len(self._hist) >= self.required:
            return StepResult(True)
        # most recent failing result explains why
        return next(r for r in reversed(self._hist) if not r.ok)

    def just_became_ok(self) -> bool:
        """True once per step when verification first passes (for a spoken 'looks good')."""
        if self.status.ok and not self._announced_ok:
            self._announced_ok = True
            return True
        return False

    def next(self, force: bool = False) -> str:
        st = self.status
        now = self.clock()
        secs = now - self.step_started
        overridden = not st.ok and not force and now - self._last_refusal <= self.override_window
        if self.events:
            self.events.log("step_check", self.project.id, self.idx + 1, ok=st.ok, missing=st.missing,
                            seconds_on_step=round(secs, 1), forced=force or overridden)
        if not st.ok and not force and not overridden:
            self._last_refusal = now
            return st.summary + " Give a second thumbs up to skip the check."
        prefix = "Okay, skipping the camera check. " if overridden else ""
        if self.events:
            self.events.log("step_done", self.project.id, self.idx + 1, ok=True, seconds_on_step=round(secs, 1))
        if self.idx + 1 >= len(self.project.steps):
            self.finished = True
            if self.events:
                self.events.log("project_done", self.project.id, self.idx + 1)
            return prefix + self.project.data.get("finish", "That's the last step. Nice work! Power it up and test it.")
        self.idx += 1
        self._hist.clear()
        self._announced_ok = False
        self._last_refusal = -1e9
        self.step_started = self.clock()
        return prefix + self.step_label

    def repeat(self) -> str:
        tip = self.step.get("tip")
        return self.step_label + (f" Tip: {tip}" if tip else "")
