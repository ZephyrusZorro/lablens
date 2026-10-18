"""LabLens web UI (Streamlit).

    streamlit run app.py                      # webcam 0
    streamlit run app.py -- --source demo.mp4 # video file or image instead of a camera

The camera, detector and hand tracker run continuously in the LabLens runtime (background threads).
This page only polls snapshots, so the webcam is opened once and the UI stays responsive.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import streamlit as st

from lablens.config import get_config

st.set_page_config(page_title="LabLens", page_icon="🔬", layout="wide")
cfg = get_config()


def wide(fn, *args, **kw):
    """Full-width widget on any Streamlit version. The argument changed across releases:
    use_column_width (old st.image) -> use_container_width -> width="stretch" (current)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        params = {}
    w = params.get("width")
    if w is not None and isinstance(w.default, str):  # current API: width="stretch" | "content" | px
        kw["width"] = "stretch"
    elif "use_container_width" in params:
        kw["use_container_width"] = True
    elif "use_column_width" in params:
        kw["use_column_width"] = True
    return fn(*args, **kw)


def _cli_source():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    ap.add_argument("--no-voice", action="store_true")
    a, _ = ap.parse_known_args(sys.argv[1:])
    return a


@st.cache_resource(show_spinner="Loading models (first start can take ~20 s)…")
def get_runtime(source, speak: bool):
    from lablens.runtime import Runtime

    return Runtime(cfg, source=source, speak=speak)


args = _cli_source()
source = args.source if args.source is not None else cfg.get("vision.camera_index", 0)
rt = get_runtime(source, not args.no_voice)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🔬 LabLens")
    st.caption("Point at a part and ask. Gestures: 👍 next step · ✋ repeat · ✊ stop talking")
    net = st.radio("Answer engine", ["Auto", "Groq (online)", "Ollama (offline)"], horizontal=False,
                   help="Auto uses Groq when online and falls back to Ollama, then to cited extracts.")
    rt.lens.router.forced = {"Groq (online)": "groq", "Ollama (offline)": "ollama"}.get(net)

    st.subheader("Guided Build")
    pid = st.selectbox("Project", list(rt.projects), format_func=lambda k: rt.projects[k].title)
    c1, c2 = st.columns(2)
    if wide(c1.button, "▶ Start"):
        rt.start_project(pid)
    if wide(c2.button, "■ Stop"):
        rt.stop_project()
    c3, c4, c5 = st.columns(3)
    if wide(c3.button, "👍 Next"):
        rt.next_step()
    if wide(c4.button, "✋ Repeat"):
        rt.repeat()
    if wide(c5.button, "✊ Quiet"):
        rt.stop_talking()

    st.subheader("System")
    snap0 = rt.snapshot()
    st.caption(f"Detector: {type(rt.detector).__name__} · classes: {', '.join(getattr(rt.detector, 'classes', []))}")
    st.caption(f"Hand tracking: {snap0.hand_backend} · TTS: {rt.speaker.backend if rt.speaker else 'off'}")
    st.caption(f"Knowledge base: {len(rt.lens.store.sources())} documents, {len(rt.lens.store.chunks)} passages")

left, right = st.columns([3, 2], gap="medium")


@st.fragment(run_every=0.1)
def video_panel():
    snap = rt.snapshot()
    if snap.frame is None:
        st.info("Waiting for camera… (check vision.camera_index in config.yaml)")
        return
    wide(st.image, cv2.cvtColor(snap.frame, cv2.COLOR_BGR2RGB))
    g = snap.guided
    if g:
        st.markdown(f"**{g['project']}** — step {g['step']}/{g['total']}: {g['instruction']}")
        (st.success if g["ok"] else st.warning)("✅ Verified — thumbs up for next step" if g["ok"]
                                                 else " ".join(g["messages"]) or "Checking…")


@st.fragment(run_every=0.5)
def chat_panel():
    snap = rt.snapshot()
    sel = snap.selection
    if sel:
        st.markdown(f"**Pointing at:** `{sel.detection.label}` {'🔒' if sel.locked else '…'}"
                    + (f" · bands ≈ {snap.resistor_reading}" if snap.resistor_reading else ""))
    elif snap.hand_backend == "none":
        parts = [f"{d.label} #{d.track_id}" for d in snap.detections if d.label != "breadboard"]
        choice = st.selectbox("No hand tracker - pick the part to ask about", ["(none)"] + parts, key="pick")
        if choice != "(none)":
            d = next(d for d in snap.detections if f"{d.label} #{d.track_id}" == choice)
            rt.set_mouse(*d.center)
        else:
            rt.set_mouse(None)
    if snap.busy:
        st.caption("thinking…")
    box = st.container(height=420)
    for role, text in list(rt.messages)[-30:]:
        box.chat_message("user" if role == "you" else "assistant").write(text)


with left:
    video_panel()
with right:
    chat_panel()
    q = st.chat_input("Ask about the part you're pointing at (e.g. 'which pin is ground?')")
    if q:
        rt.ask(q)
    if wide(st.button, "🎤 Ask by voice"):
        rt.listen_and_ask()
    with st.expander("📄 Source page for the last answer", expanded=True):
        cites = rt.last_citations
        if cites:
            c = cites[0]
            pdf_dir = cfg.path("rag.pdf_dir", "data/datasheets")
            pdf = next(Path(pdf_dir).rglob(c.source), None)
            st.caption(f"{c.source} — page {c.page}")
            if pdf is not None:
                try:
                    import fitz  # PyMuPDF

                    page = fitz.open(str(pdf))[c.page - 1]
                    wide(st.image, page.get_pixmap(dpi=100).tobytes("png"))
                except Exception:
                    st.text(c.snippet[:800])
        else:
            st.caption("Answers show the datasheet / manual page they came from here.")
