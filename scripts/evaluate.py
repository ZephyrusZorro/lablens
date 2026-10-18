"""Reproduce the four numbers that go on the resume / slide deck.

  python scripts/evaluate.py grounding               # eval/grounding.csv, no LLM needed
  python scripts/evaluate.py latency -n 20           # time to first token, Groq vs Ollama
  python scripts/evaluate.py pointing eval/pointing.csv
  python scripts/evaluate.py yolo --weights models/my_kit.pt --data path/to/data.yaml   # needs ultralytics

Each sub-command prints a number you can cite. Keep the raw CSVs in the repo for reproducibility.

eval/grounding.csv : question,pointed,expected_pages,source   (hand-labelled; pages may be "7|11")
eval/pointing.csv  : image,target_label   (photos of your hand pointing at a part, one row per trial)
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
        for row in csv.DictReader(ln for ln in f if not ln.startswith("#")):
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


def _retrieval_ranks(rows, store, k, label_boost, use_pointing=True):
    from lablens.core.query import QueryContext, compose
    from lablens.types import Detection

    ranks = []
    for r in rows:
        pointed = Detection(r["pointed"], 0.9, 0, 0, 1, 1) if use_pointing and r.get("pointed") else None
        cq = compose(r["question"], QueryContext(component=pointed))
        hits = store.search(cq.retrieval, k, cq.component_label, label_boost, cq.part_number)
        ok_pages = {int(x) for x in r["expected_pages"].split("|")}
        rank = next((i + 1 for i, c in enumerate(hits) if c.page in ok_pages and c.source == r["source"]), None)
        ranks.append(rank)
    return ranks


def _summary(ranks, k):
    n = max(len(ranks), 1)
    top1 = sum(1 for x in ranks if x == 1) / n
    topk = sum(1 for x in ranks if x is not None) / n
    mrr = sum(1.0 / x for x in ranks if x) / n
    return top1, topk, mrr


def grounding(args) -> None:
    """Retrieval grounding: is the page that answers the question the one LabLens cites?

    Runs with no LLM (retrieval is what decides the citation). Also runs an ablation with the pointed
    component removed, which measures what pointing adds. (A plain LLM without retrieval has no source to
    cite at all, so its citation accuracy is 0% by construction; that is the baseline.)
    """
    from lablens.config import get_config
    from lablens.rag.store import build_store

    cfg = get_config()
    store = build_store(cfg)
    if not store.chunks:
        raise SystemExit("knowledge base is empty - run scripts/build_manual.py first")
    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    k = cfg.get("rag.top_k", 4)
    boost = cfg.get("rag.label_boost", 0.15)
    with_p = _retrieval_ranks(rows, store, k, boost, True)
    without = _retrieval_ranks(rows, store, k, boost, False)
    a1, ak, am = _summary(with_p, k)
    b1, bk, bm = _summary(without, k)
    pointed_rows = [i for i, r in enumerate(rows) if r.get("pointed")]
    c1 = _summary([with_p[i] for i in pointed_rows], k)[0] if pointed_rows else 0
    d1 = _summary([without[i] for i in pointed_rows], k)[0] if pointed_rows else 0
    print(f"Grounding on {len(rows)} hand-labelled questions ({len(pointed_rows)} asked while pointing):")
    print(f"  cited page correct @1 : {100 * a1:.0f}%   @{k}: {100 * ak:.0f}%   MRR {am:.2f}")
    print(f"  same, ignoring the pointed part: @1 {100 * b1:.0f}%   @{k}: {100 * bk:.0f}%   MRR {bm:.2f}")
    print(f"  pointing questions only: @1 {100 * c1:.0f}% with pointing vs {100 * d1:.0f}% without")
    misses = [(r["question"], r.get("pointed") or "-", r["expected_pages"], x) for r, x in zip(rows, with_p) if x != 1]
    if misses and args.verbose:
        print("  not first:")
        for q, pt, exp, x in misses:
            print(f"    [{pt}] {q!r} expected p.{exp}, rank {x}")


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
    g.add_argument("csv", nargs="?", default=str(Path(__file__).resolve().parent.parent / "eval" / "grounding.csv"))
    g.add_argument("-v", "--verbose", action="store_true")
    g.set_defaults(fn=grounding)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
