"""Regression test: retrieval must keep citing the right lab-manual page for the labelled questions in
eval/grounding.csv, and pointing must keep helping. Builds the manual PDF from knowledge/lab_manual.md
into a temp dir, so it needs no setup step.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grounding_regression(tmp_path):
    pytest.importorskip("reportlab")
    build = _load("build_manual", ROOT / "scripts" / "build_manual.py")
    ev = _load("evaluate", ROOT / "scripts" / "evaluate.py")
    from lablens.rag.store import HybridStore

    pdf = tmp_path / "LabLens_Lab_Manual.pdf"
    build.render_pdf(ROOT / "knowledge" / "lab_manual.md", pdf)
    store = HybridStore(tmp_path / "index", use_dense=False)
    store.add_pdf(pdf)
    rows = list(csv.DictReader(open(ROOT / "eval" / "grounding.csv", newline="", encoding="utf-8")))
    with_p = ev._retrieval_ranks(rows, store, 4, 0.15, True)
    without = ev._retrieval_ranks(rows, store, 4, 0.15, False)
    top1, top4, _ = ev._summary(with_p, 4)
    assert top4 == 1.0, [r["question"] for r, x in zip(rows, with_p) if x is None]
    assert top1 >= 0.85
    pointed = [i for i, r in enumerate(rows) if r["pointed"]]
    assert ev._summary([with_p[i] for i in pointed], 4)[0] > ev._summary([without[i] for i in pointed], 4)[0]
