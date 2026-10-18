"""Unit tests for the parts that don't need a camera, microphone, model or network.
Run:  python -m pytest -q tests
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from lablens.types import Detection, HandObservation
from lablens.vision.pointing import resolve_pointed, segment_box_t, PointingResolver
from lablens.vision.gestures import classify_rules, GestureDebouncer
from lablens.core.query import compose, QueryContext, Conversation, Turn, extract_part_number, normalise_spoken_numbers, cache_key
from lablens.rag.ingest import chunk_text, tags_for, DEFAULT_PART_KEYWORDS
from lablens.rag.store import BM25, tokenize, HybridStore
from lablens.llm.prompts import format_context
from lablens.audio.tts import split_sentences, clean_for_speech
from lablens.guided.engine import check_step, GuidedSession, Project
from lablens.types import Citation


def hand_pointing_at(x, y, dx=1.0, dy=0.0):
    """21 landmarks; index PIP at (x-dx*k, y-dy*k), tip at (x, y)."""
    lm = [(0.0, 0.0)] * 21
    lm[0] = (x - 200 * dx, y + 200)  # wrist, far
    lm[6] = (x - 20 * dx, y - 20 * dy)
    lm[8] = (x, y)
    return HandObservation(lm)


# ---- pointing --------------------------------------------------------------

def test_segment_hits_box():
    d = Detection("ic", 0.9, 100, 100, 200, 200)
    assert segment_box_t((0, 150), (300, 150), d) is not None
    assert segment_box_t((0, 50), (300, 50), d) is None


def test_pointing_prefers_part_over_breadboard():
    board = Detection("breadboard", 0.9, 0, 0, 640, 480)
    ic = Detection("ic", 0.9, 300, 200, 360, 260)
    hand = hand_pointing_at(280, 230)
    got = resolve_pointed(hand, [board, ic])
    assert got is ic


def test_pointing_nearest_fallback():
    ic = Detection("ic", 0.9, 300, 200, 360, 260)
    hand = hand_pointing_at(250, 100, dx=0, dy=-1)  # ray points away
    assert resolve_pointed(hand, [ic], nearest_max_px=400) is ic


def test_resolver_dwell_and_lock():
    t = [0.0]
    r = PointingResolver(smoothing_frames=5, dwell_seconds=0.5, min_agreement=0.6, clock=lambda: t[0])
    ic = Detection("ic", 0.9, 300, 200, 360, 260, track_id=1)
    hand = hand_pointing_at(320, 230)
    s = None
    for _ in range(5):
        s = r.update(hand, [ic])
        assert s is None or not s.locked
        t[0] += 0.1
    t[0] += 0.5
    s = r.update(hand, [ic])
    assert s and s.locked and s.detection.label == "ic"


# ---- gestures ----------------------------------------------------------------

def _curled(thumb_up=False):
    lm = [(0.0, 0.0)] * 21
    lm[0] = (0, 100)
    for i in range(1, 21):
        lm[i] = (0, 100 - 5)  # tips close to wrist = curled
    lm[9] = (0, 50)
    if thumb_up:
        lm[2], lm[3], lm[4] = (-10, 90), (-10, 75), (-10, 40)
        lm[5] = (-5, 60)
    return HandObservation(lm)


def test_fist_and_debounce():
    h = _curled()
    assert classify_rules(h) == "fist"
    now = [0.0]
    d = GestureDebouncer(hold_frames=3, cooldown_s=1.0, clock=lambda: now[0])
    out = [d.update("fist") for _ in range(4)]
    assert out.count("stop") == 1


# ---- query composer ---------------------------------------------------------

def test_spoken_numbers_and_part_extraction():
    assert normalise_spoken_numbers("pin five five five") == "pin 555"
    assert extract_part_number("what is the ne five five five pinout") == "NE555"


def test_pointing_is_fused_into_retrieval():
    d = Detection("ic", 0.91, 0, 0, 1, 1)
    cq = compose("what is this and where does it connect", QueryContext(component=d))
    assert "ic" in cq.retrieval and "pinout" in cq.retrieval
    assert "POINTED COMPONENT: IC" in cq.header


def test_followup_inherits_component():
    conv = Conversation()
    d = Detection("ic", 0.9, 0, 0, 1, 1)
    first = compose("what is this", QueryContext(component=d), conv)
    conv.add(Turn("what is this", "a timer", first.component_label, first.part_number))
    cq = compose("and pin four?", QueryContext(), conv)
    assert cq.component_label == "ic"


def test_cache_key_normalises():
    cq = compose("which pin is ground", QueryContext())
    assert cache_key(cq, "Which pin is GND?") == cache_key(cq, "which pin is ground")


# ---- RAG ---------------------------------------------------------------------

def test_chunking_overlaps_and_covers():
    text = "A" * 500 + ". " + "B" * 500
    chunks = chunk_text(text, 400, 50)
    assert len(chunks) >= 2
    assert "".join(c[:1] for c in chunks).startswith("A")


def test_tags_from_filename():
    tags = tags_for("NE555-timer-datasheet.pdf", DEFAULT_PART_KEYWORDS)
    assert "ic" in tags


def test_bm25_finds_exact_part_number():
    docs = [tokenize("resistor colour code bands"), tokenize("ne555 timer pin 1 ground"), tokenize("led anode")]
    s = BM25(docs).scores(tokenize("ne555"))
    assert int(np.argmax(s)) == 1


def test_store_search_and_boost(tmp_path):
    st = HybridStore(tmp_path, use_dense=False)
    st.add_text("NE555 pin 1 is ground. Pin 8 is VCC.", "ne555.pdf", tags=["ic"])
    st.add_text("Electrolytic capacitor polarity marking on the can.", "cap.pdf", tags=["capacitor"])
    # The component boost must never promote a passage that doesn't match the question at all
    hits = st.search("which pin is ground", k=2, component="capacitor", label_boost=5.0)
    assert hits[0].source == "ne555.pdf" and all(h.source != "cap.pdf" for h in hits)
    # ...but among matching passages it decides the winner
    hits = st.search("polarity pin marking", k=2, component="capacitor", label_boost=5.0)
    assert hits[0].source == "cap.pdf"
    hits = st.search("which pin is ground", k=1)
    assert hits[0].source == "ne555.pdf" and hits[0].page == 1


def test_store_persists(tmp_path):
    st = HybridStore(tmp_path, use_dense=False)
    st.add_text("LED forward voltage around 2 V", "led.pdf")
    st2 = HybridStore(tmp_path, use_dense=False)
    assert len(st2.chunks) == 1


def test_format_context_empty():
    assert "no matching" in format_context([])
    assert "[x.pdf p.3]" in format_context([Citation("x.pdf", 3, "hi")])


# ---- audio text -------------------------------------------------------------

def test_sentence_split_streams():
    toks = ["Pin 1 is ", "ground. ", "Pin 8 is VCC, ", "connect it to +5 V. ", "Done."]
    sents = list(split_sentences(iter(toks), min_chars=5))
    assert sents[0].startswith("Pin 1 is ground.")
    assert sents[-1].endswith("Done.")


def test_clean_for_speech_drops_citations():
    assert clean_for_speech("Pin 1 is ground [NE555.pdf p.3].") == "Pin 1 is ground."


# ---- guided build -----------------------------------------------------------

def _det(label, x1, y1, x2, y2):
    return Detection(label, 0.9, x1, y1, x2, y2)


def test_step_reports_missing_part():
    dets = [_det("breadboard", 0, 0, 640, 480)]
    step = {"instruction": "x", "expect": [{"label": "resistor", "min": 1}]}
    r = check_step(step, dets)
    assert not r.ok and "I don't see the resistor yet." in r.messages


def test_step_zone_and_near():
    board = _det("breadboard", 0, 0, 640, 480)
    res = _det("resistor", 100, 200, 140, 230)
    led = _det("led", 150, 200, 180, 230)
    step = {"instruction": "x", "expect": [{"label": "resistor", "zone": [0, 0, 0.5, 1]}],
            "near": [["resistor", "led", 0.3]]}
    assert check_step(step, [board, res, led]).ok


def test_guided_session_advances_on_thumbs_up():
    p = Project({"id": "t", "title": "t", "steps": [
        {"instruction": "board", "expect": [{"label": "breadboard"}]},
        {"instruction": "led", "expect": [{"label": "led"}]},
    ]})
    g = GuidedSession(p)
    board = _det("breadboard", 0, 0, 640, 480)
    for _ in range(3):
        g.observe([board])
    assert g.next().startswith("Step 2")
    # LED not on the bench yet: thumbs-up must be refused with a spoken reason
    for _ in range(3):
        g.observe([board])
    msg = g.next()
    assert msg.startswith("I don't see the led yet.") and "second thumbs up" in msg
    # once the LED appears, the step passes and the project finishes
    led = _det("led", 100, 100, 130, 130)
    for _ in range(12):  # window is 15 frames; need >=70% passing
        g.observe([board, led])
    g.next()
    assert g.finished


def test_projects_load_and_validate():
    import json
    from lablens.guided.engine import list_projects
    d = Path(__file__).resolve().parent.parent / "lablens" / "guided" / "projects"
    projects = list_projects(d)
    assert len(projects) >= 3
    for p in projects.values():
        assert p.steps, p.id
        for s in p.steps:
            assert "instruction" in s


def test_double_thumbs_up_overrides_verification():
    t = [0.0]
    p = Project({"id": "t", "title": "t", "steps": [{"instruction": "led", "expect": [{"label": "led"}]},
                                                    {"instruction": "x", "expect": []}]})
    g = GuidedSession(p, clock=lambda: t[0])
    g.observe([])
    assert "second thumbs up" in g.next()
    t[0] += 2.0
    assert g.next().startswith("Okay, skipping the camera check.") and g.idx == 1
