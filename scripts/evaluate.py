"""Reproduce the four numbers that go on the resume / slide deck.

  python scripts/evaluate.py --yolo-test data/eval/images --yolo-labels data/eval/labels
  python scripts/evaluate.py --pointing data/eval/pointing.csv
  python scripts/evaluate.py --latency 20
  python scripts/evaluate.py --grounding data/eval/grounding.csv

Each sub-command prints a number you can cite. Keep the raw CSVs in the repo for reproducibility.

pointing.csv : columns image,target_label   (you hand-label which box the finger is on, 50 trials)
grounding.csv: columns question,expected_page,source   (hand-labelled correct citation)
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def yolo_map(args) -> None:
    from ultralytics import YOLO

    model = YOLO(args.weights)
    m = model.val(data=args.data, imgsz=640, verbose=False)
    print(f"YOLO mAP@0.5 = {m.box.map50:.3f}   mAP@0.5:0.95 = {m.box.map:.3f}")


def pointing(args) -> None:
    import cv2
    from lablens.config import get_config
    from lablens.vision.detector import build_detector
    from lablens.vision.hands import HandTracker
    from lablens.vision.pointing import resolve_pointed

    cfg = get_config()
    det, hands = build_detector(cfg), HandTracker(max_hands=1)
    ok = total = 0
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            frame = cv2.imread(row["image"])
            if frame is None:
                continue
            total += 1
            h = hands.process(frame)
            d = resolve_pointed(h[0], det.detect(frame), cfg.get("pointing.ray_length_factor", 2.5)) if h else None
            ok += int(d is not None and d.label == row["target_label"])
    print(f"Pointing accuracy: {ok}/{total} = {100 * ok / max(total, 1):.0f}%")


def latency(args) -> None:
    from lablens.config import get_config
    from lablens.llm.router import GroqBackend, OllamaBackend
    from lablens.llm.prompts import system_prompt

    cfg = get_config()
    q = [{"role": "system", "content": system_prompt("en")},
         {"role": "user", "content": "What is the function of pin 1 on a 555 timer?"}]
    for name, b in (("groq", GroqBackend(cfg.groq_api_key, cfg.get("llm.groq_model"))),
                    ("ollama", OllamaBackend(cfg.ollama_host, cfg.get("llm.ollama_model")))):
        ttft = []
        for _ in range(args.n):
            t0 = time.perf_counter()
            try:
                next(b.stream(q))
                ttft.append(time.perf_counter() - t0)
            except Exception as e:
                print(f"{name}: skipped ({e})")
                break
        if ttft:
            print(f"{name}: time-to-first-token median {statistics.median(ttft):.2f}s "
                  f"(p90 {sorted(ttft)[int(0.9 * (len(ttft) - 1))]:.2f}s, n={len(ttft)})")


def grounding(args) -> None:
    from lablens.config import get_config
    from lablens.core.assistant import LabLens

    lens = LabLens.from_config(get_config())
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    rag_ok = plain_ok = 0
    for r in rows:
        a = lens.ask(r["question"])
        rag_ok += int(any(c.page == int(r["expected_page"]) and c.source == r["source"] for c in a.citations))
        lens.convo.clear()
        # Baseline: plain LLM, no retrieval - it cannot cite a page, so it only passes if it names the right
        # source file in its text. This keeps the comparison honest.
        plain = lens.ask_without_rag(r["question"])
        plain_ok += int(r["source"].split(".")[0].lower() in plain.lower())
    n = max(len(rows), 1)
    print(f"RAG grounding (correct cited page): {100 * rag_ok / n:.0f}%  |  plain LLM: {100 * plain_ok / n:.0f}%  (n={len(rows)})")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    y = sub.add_parser("yolo")
    y.add_argument("--weights", default="models/lablens_yolov5s.pt")
    y.add_argument("--data", default="data/eval/data.yaml")
    y.set_defaults(fn=yolo_map)
    pt = sub.add_parser("pointing")
    pt.add_argument("csv")
    pt.set_defaults(fn=pointing)
    la = sub.add_parser("latency")
    la.add_argument("-n", type=int, default=10)
    la.set_defaults(fn=latency)
    g = sub.add_parser("grounding")
    g.add_argument("csv")
    g.set_defaults(fn=grounding)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
