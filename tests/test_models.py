"""Tests for the model stack: ONNX writer, exporter, inference post-processing, ensemble fusion,
breadboard detector, resistor reader, extractive answering. No PyTorch, camera or network needed.
Tests that need downloaded weights are skipped when the files are absent.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent


# ---- ONNX writer -------------------------------------------------------------

def test_onnx_writer_conv_silu_matches_numpy(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from lablens.vision import onnx_writer as ow

    rng = np.random.default_rng(0)
    x = rng.normal(size=(1, 2, 5, 5)).astype(np.float32)
    w = rng.normal(size=(3, 2, 1, 1)).astype(np.float32)
    b = rng.normal(size=(3,)).astype(np.float32)
    g = ow.GraphBuilder()
    y = g.op("Conv", ["images", g.init(w), g.init(b)], kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0])
    y = g.op("Mul", [y, g.op("Sigmoid", [y])])
    model = ow.model(g.nodes, g.inits, [ow.value_info("images", ow.FLOAT, [1, 2, 5, 5])],
                     [ow.value_info(y, ow.FLOAT, [1, 3, 5, 5])])
    p = tmp_path / "t.onnx"
    p.write_bytes(model)
    out = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"images": x})[0]
    ref = np.einsum("oc,bchw->bohw", w[:, :, 0, 0], x) + b[None, :, None, None]
    ref = ref / (1 + np.exp(-ref))
    assert np.allclose(out, ref, atol=1e-5)


def test_varint_negative_roundtrip():
    from lablens.vision.onnx_writer import _varint

    assert _varint(1) == b"\x01"
    assert _varint(300) == b"\xac\x02"
    assert len(_varint(-1)) == 10  # int64 two's complement


# ---- post-processing ------------------------------------------------------------

def test_nms_class_aware():
    from lablens.vision.yolo_onnx import nms

    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [0, 0, 10, 10]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    classes = np.array([0, 0, 1])
    keep = nms(boxes, scores, classes, 0.5)
    assert sorted(keep.tolist()) == [0, 2]  # same-class duplicate removed, other class kept


def test_letterbox_and_anchors():
    from lablens.vision.yolo_onnx import letterbox, make_anchors

    img = np.zeros((480, 640, 3), np.uint8)
    lb, r, (px, py) = letterbox(img, 640)
    assert lb.shape == (640, 640, 3) and r == 1.0 and px == 0 and py == 80
    pts, st = make_anchors(640, 640, [8, 16, 32])
    assert len(pts) == 80 * 80 + 40 * 40 + 20 * 20 == 8400


def test_wbf_fuses_agreeing_models_and_respects_class_coverage():
    from lablens.vision.ensemble import weighted_box_fusion

    a = (np.array([[10, 10, 50, 50]], np.float32), np.array([0.8], np.float32), np.array([0]), 1.0)
    b = (np.array([[12, 12, 52, 52], [100, 100, 140, 140]], np.float32), np.array([0.6, 0.7], np.float32),
         np.array([0, 1]), 1.0)
    # class 0 known by both models, class 1 known only by model b
    boxes, scores, classes, support = weighted_box_fusion([a, b], {0: 2, 1: 1}, 0.5, 0.7)
    by_cls = {int(c): (bx, s, n) for bx, s, c, n in zip(boxes, scores, classes, support)}
    assert by_cls[0][2] == 2 and abs(by_cls[0][1] - 0.7) < 1e-6           # mean of 0.8 and 0.6
    assert 10 < by_cls[0][0][0] < 12                                       # weighted average box
    assert by_cls[1][2] == 1 and abs(by_cls[1][1] - 0.7) < 1e-6           # not penalised: only b knows it


# ---- exporter + real weights (skipped if not set up) --------------------------------

@pytest.mark.skipif(not (ROOT / "models" / "components_v8n_6cls.onnx").exists(), reason="run setup_models.py")
def test_component_model_runs_and_has_expected_classes():
    from lablens.vision.yolo_onnx import YoloOnnx

    m = YoloOnnx(ROOT / "models" / "components_v8n_6cls.onnx", 800)
    assert {"Resistor", "LED", "Capacitor"} <= set(m.names.values())
    r = m(np.full((480, 640, 3), 128, np.uint8))
    assert r.boxes.shape[1] == 4


# ---- breadboard + resistor (classical CV) -------------------------------------------

def test_breadboard_detector_synthetic_and_negative():
    import cv2
    from lablens.vision.breadboard import BreadboardDetector, synth_breadboard

    d = BreadboardDetector().detect_once(synth_breadboard())
    assert d is not None and d.label == "breadboard" and d.width > 300
    paper = np.full((480, 640, 3), (40, 70, 110), np.uint8)
    cv2.rectangle(paper, (80, 120), (560, 400), (230, 230, 230), -1)
    assert BreadboardDetector().detect_once(paper) is None


@pytest.mark.parametrize("bands,ohms", [
    (["red", "red", "brown", "gold"], 220), (["orange", "orange", "brown", "gold"], 330),
    (["brown", "black", "red", "gold"], 1000), (["yellow", "violet", "red", "gold"], 4700),
    (["brown", "black", "yellow", "gold"], 100000), (["brown", "black", "black", "red", "brown"], 10000),
])
@pytest.mark.parametrize("angle", [0, 30, 90, 160])
def test_resistor_reader_synthetic(bands, ohms, angle):
    from lablens.vision.resistor import read_resistor, synth_resistor

    r = read_resistor(synth_resistor(bands, angle=angle, seed=1))
    assert r is not None and r["ohms"] == pytest.approx(ohms)


def test_colour_classifier():
    from lablens.vision.resistor import classify_rgb

    assert classify_rgb((190, 35, 30)) == "red"
    assert classify_rgb((120, 60, 25)) == "brown"


# ---- answering without an LLM ---------------------------------------------------------

def test_extractive_answer_cites_and_picks_fact():
    from lablens.rag.store import extractive_answer
    from lablens.types import Citation

    cites = [Citation("m.pdf", 7, "NE555 timer\nPin 1 GND (ground). Pin 8 VCC (supply). The 555 is a timer.", 1.0),
             Citation("m.pdf", 3, "LEDs\nThe longer leg is the anode.", 0.5)]
    a = extractive_answer("which pin is ground", cites)
    assert "Pin 1 GND" in a and "[m.pdf p.7]" in a


def test_topic_boost_prefers_page_about_pointed_part(tmp_path):
    from lablens.rag.store import HybridStore

    st = HybridStore(tmp_path, use_dense=False)
    st.add_text("Guided project: diode with LED indicator\nUse a 330 ohm resistor before the LED.", "m.pdf", page=14)
    st.add_text("LEDs: polarity and current limiting\nSeries resistor R = (Vs - Vf) / I.", "m.pdf", page=3)
    hits = st.search("what resistor do I need", 2, component="led", label_boost=0.5)
    assert hits[0].page == 3
