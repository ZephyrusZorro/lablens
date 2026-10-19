"""LabLens workspace: local vision, reviewed voice input, circuit synthesizer, and cited answers."""
from __future__ import annotations

import argparse
import base64
import hashlib
import sys
from pathlib import Path

import cv2
import streamlit as st

from lablens.config import ROOT, get_config
from lablens.vision.humans import HUMAN_CLASSES

st.set_page_config(page_title="LabLens | Workbench", page_icon=":material/center_focus_strong:",
                   layout="wide", initial_sidebar_state="collapsed")

cfg = get_config()

# Premium UI styling
st.markdown("""
<style>
/* Refined card, button, and expander styling */
div[data-testid="stExpander"] {
    border: 1px solid #D5D7D1 !important;
    border-radius: 8px !important;
    background-color: #FFFFFF !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03) !important;
    margin-bottom: 0.6rem !important;
}
div[data-testid="stChatMessage"] {
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
}
div[data-testid="stImage"] img {
    border-radius: 8px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05) !important;
}
.lablens-header-status {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    background: #E8F3EE;
    border: 1px solid #B4D7C6;
    border-radius: 14px;
    font-size: 12px;
    font-weight: 600;
    color: #16372D;
    letter-spacing: 0.02em;
}
.lablens-pill-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: 500;
    background: #ECEBE5;
    color: #3B4243;
    margin-right: 4px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=None)
    parser.add_argument("--no-voice", action="store_true")
    return parser.parse_known_args(sys.argv[1:])[0]


args = cli_args()
MODES = {"Auto: bench and people": "auto", "People and body parts": "people", "Mixed: loose components": "mixed"}


@st.cache_resource(scope="session", max_entries=1, on_release=lambda runtime: runtime.stop())
def get_runtime(source):
    from lablens.runtime import Runtime
    return Runtime(cfg, source=source, speak=False)


@st.cache_data(max_entries=16, show_spinner=False)
def source_page_png(path, page, modified):
    import fitz
    with fitz.open(path) as doc:
        return doc[page - 1].get_pixmap(dpi=100).tobytes("png")


with st.sidebar:
    st.caption("LABLENS SETTINGS")
    st.space("small")
    st.subheader("Workspace", icon=":material/tune:")
    show_body = st.toggle("Show body skeletons", value=False)
    gestures = st.toggle("Enable gesture shortcuts", value=False, help="Disabled automatically when multiple people are visible.")
    st.space("small")
    st.subheader("Responses", icon=":material/graphic_eq:")
    engine = st.selectbox("Answer engine", ["Auto", "Ollama (offline)", "Groq (online)"], key="answer_engine",
                          help="Auto falls back from Groq to Ollama, then to cited manual extracts.")
    speak = st.toggle("Speak answers", value=False, disabled=args.no_voice, key="spoken_answers",
                       help="Keep off while recording to avoid feedback. Turn on to hear answers.")
    with st.expander("Devices & diagnostics", icon=":material/settings_input_component:"):
        camera_index = st.number_input("Camera index", 0, 9, int(cfg.get("vision.camera_index", 0)), step=1,
                                       key="camera_index")
        from lablens.audio.stt import input_devices, is_virtual_input
        try:
            devices = input_devices()
        except Exception:
            devices = []
        device_names = {i: f"{i}: {d['name']}" + (" (virtual)" if is_virtual_input(d['name']) else "")
                        for i, d in devices}
        input_device = st.selectbox("Desktop audio input", [None] + list(device_names),
                                    format_func=lambda i: device_names.get(i, "Auto: physical microphone"),
                                    key="desktop_audio_input")
        st.caption("The browser recorder uses your browser's microphone selection.")
        diagnostics_slot = st.container()
    st.space("small")
    with st.expander("Quick controls", icon=":material/gesture:"):
        st.markdown("**Point** to focus a component.\n\n**Thumbs up** advances a build.\n\n"
                    "**Open palm** repeats. **Fist** stops speech.")
        st.caption("Automatic gestures activate in a bench context. Desktop: SPACE to speak, ENTER to type.")


# Top header bar
wordmark = ROOT / "assets" / "lablens-wordmark.svg"
if wordmark.exists():
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.image(str(wordmark), width=190)
        st.markdown('<div class="lablens-header-status">● System Ready · Multimodal AI Bench</div>', unsafe_allow_html=True)
else:
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.title("LabLens")
        st.caption("ELECTRONICS WORKBENCH & MENTOR")

with st.container(horizontal=True, vertical_alignment="bottom"):
    view = st.selectbox("Camera source", ["Live camera", "Sample photo"],
                        index=1 if args.source is not None else 0, key="camera_source")
    mode = st.selectbox("Recognition mode", list(MODES), key="recognition_mode",
                        help="Auto keeps electronics on a detected board when people are visible. Mixed allows loose parts.")

section = st.segmented_control("Workspace", ["Inspect", "Build", "Circuit Synthesizer", "Manual"],
                               default="Inspect", key="workspace_section", label_visibility="collapsed")

source = (args.source or "samples/bench_demo.png") if view == "Sample photo" else int(camera_index)
if st.session_state.get("runtime_key") != source:
    for key in ("draft_transcript_id", "voice_draft", "processed_recording", "voice_recording", "pick", "focus_options"):
        st.session_state.pop(key, None)
    st.session_state.runtime_key = source
try:
    rt = get_runtime(source)
except Exception as exc:
    st.error(f"Could not start this workspace: {exc}")
    st.caption("Check Devices & diagnostics, or run run_lablens.bat setup.")
    st.stop()
rt.set_speech_enabled(speak)
rt.show_body = show_body
rt.gestures_enabled = gestures
rt.input_device = input_device
rt.lens.router.forced = {"Groq (online)": "groq", "Ollama (offline)": "ollama"}.get(engine)
humans = getattr(rt.detector, "humans", None)
if humans is not None:
    humans.scene_mode = MODES[mode]
    humans.pose_enabled = show_body
with diagnostics_slot:
    st.caption(f"Hand tracking: {rt.snapshot().hand_backend}")
    st.caption(f"Speech output: {rt.speaker.backend if rt.speaker else 'off'}")
    st.caption(f"Knowledge: {len(rt.lens.store.sources())} document(s), {len(rt.lens.store.chunks)} passages")
    if humans and humans.error:
        st.warning(humans.error)


left, right = st.columns([1.7, 1], gap="large")


@st.fragment(run_every=0.2)
def video_panel():
    snap = rt.snapshot()
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center"):
        st.subheader("Camera" if view == "Live camera" else "Sample bench")
        if snap.frame is not None and not snap.camera_error:
            st.badge("Live" if view == "Live camera" else "Photo", color="green", icon=":material/fiber_manual_record:")
        else:
            st.badge("Unavailable" if snap.camera_error else "Starting", color="orange")
    if snap.camera_error:
        st.warning(f"{snap.camera_error} Try another camera index or Sample photo.")
    if snap.frame is None:
        with st.container(height=280, vertical_alignment="center", horizontal_alignment="center"):
            st.markdown(":material/videocam_off:")
            st.caption("Waiting for the camera. Your workspace will appear here.")
    else:
        ok, jpg = cv2.imencode(".jpg", snap.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        st.image("data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii"), width="stretch",
                  alt="Camera annotated with detected people, visible body parts and lab components")
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            feed = "camera" if view == "Live camera" else "preview"
            st.caption(f"{snap.camera_fps:.0f} {feed} fps · {snap.detector_fps:.1f} detector fps")
            people = sum(d.label == "person" for d in snap.detections)
            parts = sum(d.label not in HUMAN_CLASSES | {"breadboard"} for d in snap.detections)
            st.caption(f"{people} people · {parts} components")
        if snap.selection and snap.selection.locked:
            st.caption(f"Selected: {snap.selection.detection.label}")
        if rt.async_det.error:
            st.error(rt.async_det.error)
        if sum(d.label == "person" for d in snap.detections) > 1:
            st.caption("Multiple people: choose a component below. Hand shortcuts are paused.")


@st.fragment
def focus_panel():
    snap = rt.snapshot()
    parts = [d for d in snap.detections if d.label not in HUMAN_CLASSES | {"breadboard"}]
    if "focus_options" not in st.session_state:
        st.session_state.focus_options = {}
    if st.button("Refresh detected components", icon=":material/refresh:", key="refresh_parts"):
        st.session_state.focus_options = {d.track_id: f"{d.label.capitalize()} {d.track_id}" for d in parts}
    labels = st.session_state.focus_options
    selected = st.selectbox("Component", [None] + list(labels),
                            format_func=lambda i: labels.get(i, "Automatic pointing"), key="pick")
    rt.select_component(selected)
    if selected is not None and not any(d.track_id == selected for d in parts):
        st.caption("This component left the view. Refresh to choose a visible component.")
    elif selected is None:
        st.caption("Point with one hand, or refresh the list and choose a part.")


@st.fragment(run_every=0.5)
def conversation():
    snap = rt.snapshot()
    messages = list(rt.messages)[-30:]
    with st.container(height=290, border=False, autoscroll=True):
        if not messages:
            st.space("medium")
            st.markdown(":material/question_answer:")
            st.markdown("**Ask about a component**")
            st.caption("Select a part in the camera view, then ask about its pins, polarity or use.")
            if st.button("Try: Which pin is ground on the NE555?", type="tertiary", key="sample_question",
                          icon=":material/north_east:", disabled=snap.busy, wrap=True):
                rt.ask("Which pin is ground on the NE555?")
        for role, text in messages:
            avatar = ":material/person:" if role == "you" else ":material/center_focus_strong:"
            with st.chat_message("user" if role == "you" else "assistant", avatar=avatar):
                st.markdown(text)
    if snap.listening:
        st.badge("Listening — speak now", color="orange", icon=":material/mic:")
    elif snap.transcribing:
        st.badge("Transcribing your recording", color="orange", icon=":material/graphic_eq:")
    elif snap.busy:
        st.badge("Preparing your answer", color="orange", icon=":material/hourglass_top:")


def question_input():
    question = st.chat_input("Ask about what you see…", key="question")
    if question:
        rt.ask(question)


@st.fragment(run_every=0.5)
def voice_panel():
    st.markdown("**Record a question**")
    st.caption("Speak naturally. Review your words before asking.")
    recording = st.audio_input("Record your question", sample_rate=16000, key="voice_recording")
    if recording is not None:
        wav = recording.getvalue()
        digest = hashlib.sha256(wav).hexdigest()
        if digest != st.session_state.get("processed_recording") and rt.transcribe_recording(wav) is not None:
            st.session_state.processed_recording = digest
    snap = rt.snapshot()
    if snap.transcribing:
        st.badge("Transcribing locally", color="orange", icon=":material/graphic_eq:")
    if snap.audio_error:
        st.warning(snap.audio_error)
    if snap.last_transcript and snap.transcript_id != st.session_state.get("draft_transcript_id"):
        st.session_state.voice_draft = snap.last_transcript
        st.session_state.draft_transcript_id = snap.transcript_id
    if st.session_state.get("voice_draft"):
        draft = st.text_area("Recognized text — edit if needed", key="voice_draft", height=100)
        if st.button("Ask this question", icon=":material/arrow_upward:", type="primary", width="stretch",
                      disabled=snap.busy or not draft.strip(), key="ask_transcript"):
            rt.ask(draft)
    st.caption("Record → stop → review → ask. Up to 60 seconds. Choose your physical mic in the browser.")


@st.fragment(run_every=0.5)
def source_panel():
    st.subheader("Manual references")
    cites = rt.last_citations
    if not cites:
        st.caption("Ask a question to see its cited manual page here.")
        return
    citation = cites[0]
    st.badge(f"Manual · page {citation.page}", color="green", icon=":material/verified:")
    st.caption(citation.source)
    pdf = next(Path(cfg.path("rag.pdf_dir", "data/datasheets")).rglob(citation.source), None)
    with st.expander(f"Read cited page {citation.page}", icon=":material/description:"):
        if pdf is not None:
            try:
                st.image(source_page_png(str(pdf), citation.page, pdf.stat().st_mtime_ns), width="stretch",
                          alt=f"Page {citation.page} of {citation.source}")
            except Exception:
                st.text(citation.snippet[:800])
        else:
            st.text(citation.snippet[:800])


@st.fragment(run_every=1)
def build_panel():
    st.subheader("Build instructions")
    st.caption("One step at a time. Visual checks along the way.")
    project = st.selectbox("Choose a project", list(rt.projects), format_func=lambda key: rt.projects[key].title,
                           key="guided_project", label_visibility="collapsed")
    with st.container(horizontal=True):
        if st.button("Start build", icon=":material/play_arrow:", type="primary", key="start_build"):
            rt.start_project(project)
        if st.button("End build", icon=":material/stop:", disabled=rt.guided is None, key="stop_build"):
            rt.stop_project()
    guided = rt.snapshot().guided if rt.guided is not None else None
    if guided:
        st.progress(guided["step"] / max(1, guided["total"]), text=f"Step {guided['step']} of {guided['total']}")
        st.markdown(guided["instruction"])
        if guided["ok"]:
            st.badge("Visual check passed", color="green", icon=":material/check:")
        elif guided["messages"]:
            st.caption(" · ".join(guided["messages"]))
    with st.container(horizontal=True):
        if st.button("Next", icon=":material/arrow_forward:", disabled=rt.guided is None, key="next_build"):
            rt.next_step()
        if st.button("Repeat", icon=":material/replay:", key="repeat_answer"):
            rt.repeat()
        if st.button("Quiet", icon=":material/volume_off:", key="quiet"):
            rt.stop_talking()
    st.caption("Checks parts and approximate placement. Electrical wiring needs your own verification.")


def synthesizer_panel():
    from lablens.core.synthesizer import CircuitSynthesizer

    st.subheader("Circuit & Project Synthesizer", icon=":material/auto_awesome:")
    st.caption("Input what components and materials you have on hand, or scan them directly from the bench camera. "
               "LabLens synthesizes working circuits, calculations, schematics, and camera-verified builds.")

    synth = CircuitSynthesizer()

    if "student_materials" not in st.session_state:
        st.session_state.student_materials = ["Breadboard", "Red LED", "330Ω Resistor", "9V Battery / DC Supply"]
    if "synthesized_results" not in st.session_state:
        st.session_state.synthesized_results = None

    with st.container(border=True):
        st.markdown("**1. What parts do you have on your bench?**")
        col1, col2 = st.columns([1, 1], gap="medium")
        with col1:
            if st.button("Scan bench parts from camera", icon=":material/center_focus_strong:",
                         use_container_width=True, help="Detects components on the camera and adds them to your inventory."):
                snap = rt.snapshot()
                scanned = [d.label.capitalize() for d in snap.detections
                           if d.label not in HUMAN_CLASSES]
                if scanned:
                    existing = set(st.session_state.student_materials)
                    for item in scanned:
                        if item not in existing:
                            st.session_state.student_materials.append(item)
                    st.toast(f"Detected {len(scanned)} parts from bench camera!", icon="🔍")
                else:
                    st.info("No components detected on camera yet. Point the camera at your breadboard or parts.")

        with col2:
            if st.button("Reset parts list", icon=":material/restart_alt:", use_container_width=True):
                st.session_state.student_materials = []
                st.session_state.synthesized_results = None

        common_palette = [
            "Breadboard", "Red LED", "Green LED", "330Ω Resistor", "1kΩ Resistor",
            "10kΩ Resistor", "100kΩ Resistor", "NE555 Timer IC", "10µF Capacitor",
            "100µF Capacitor", "BC547 NPN Transistor", "1N4007 Diode", "Zener Diode",
            "9V Battery / DC Supply", "LDR (Photoresistor)", "Pushbutton"
        ]

        selected_materials = st.multiselect(
            "Select available components:",
            options=common_palette,
            default=[item for item in st.session_state.student_materials if item in common_palette],
            key="materials_multiselect"
        )

        custom_input = st.text_input("Add other specialized parts (comma-separated):",
                                     placeholder="e.g. buzzer, LM358, 47k potentiometer, ceramic capacitor",
                                     key="custom_materials_input")

        all_materials = list(selected_materials)
        if custom_input:
            all_materials.extend([p.strip() for p in custom_input.split(",") if p.strip()])
        st.session_state.student_materials = all_materials

        if all_materials:
            st.caption("Active inventory: " + " · ".join(f"`{m}`" for m in all_materials))
        else:
            st.warning("Please select or add at least one component to synthesize projects.")

        c_act1, c_act2 = st.columns([1.5, 1])
        with c_act1:
            do_synthesize = st.button("Synthesize Circuits & Projects", icon=":material/biotech:",
                                      type="primary", use_container_width=True,
                                      disabled=not all_materials)
        with c_act2:
            use_ai_synth = st.checkbox("AI Custom Synthesis", value=False,
                                       help="Uses local/cloud LLM to creatively design custom circuits from your exact kit.")

        if do_synthesize and all_materials:
            with st.spinner("Synthesizing viable circuits from your bench components..."):
                if use_ai_synth:
                    custom_proj = synth.synthesize_with_llm(all_materials, rt.lens.router, rt.lens.store)
                    standard = synth.match_projects(all_materials)
                    st.session_state.synthesized_results = [custom_proj] + standard
                else:
                    st.session_state.synthesized_results = synth.match_projects(all_materials)

    results = st.session_state.synthesized_results
    if results:
        st.markdown(f"**Synthesized {len(results)} viable projects from your components:**")
        for idx, p in enumerate(results):
            with st.expander(f"{p.title}  ·  [{p.difficulty}]", expanded=(idx == 0)):
                if not p.missing_materials:
                    st.badge("100% Complete — All Materials Available", color="green", icon=":material/check_circle:")
                else:
                    st.badge("Partially Matched", color="orange", icon=":material/info:")
                    st.caption("Missing parts: " + ", ".join(p.missing_materials))

                st.markdown(f"**Overview:** {p.summary}")
                st.markdown(f"**Circuit Theory:** {p.working_principle}")

                if p.formula:
                    st.markdown("**Key Calculation:**")
                    st.latex(p.formula)

                st.markdown("**Step-by-Step Breadboard Wiring Guide:**")
                for s_idx, s in enumerate(p.breadboard_steps, start=1):
                    st.markdown(f"**{s_idx}.** {s}")

                if p.safety_tips:
                    st.warning("**Safety & Verification:**\n" + "\n".join(f"- {tip}" for tip in p.safety_tips))

                if p.citations:
                    st.caption("Lab Manual Citations: " + " · ".join(p.citations))

                b_col1, b_col2 = st.columns([1, 1], gap="small")
                with b_col1:
                    if st.button("Launch in Camera Guided Build", icon=":material/play_circle:",
                                 key=f"launch_{p.id}_{idx}", use_container_width=True, type="primary"):
                        guided_proj = p.to_guided_project()
                        rt.register_project(guided_proj)
                        rt.start_project(guided_proj.id)
                        st.session_state.workspace_section = "Build"
                        st.rerun()
                with b_col2:
                    if st.button("Ask Assistant About This", icon=":material/chat:",
                                 key=f"ask_assistant_{p.id}_{idx}", use_container_width=True):
                        rt.ask(f"How do I wire and test the {p.title}?")
                        st.toast(f"Consulted LabLens on {p.title}", icon="💡")
                        st.rerun()


with left:
    if section in ("Inspect", "Build"):
        video_panel()
        if section == "Inspect":
            focus_panel()
        else:
            build_panel()
    elif section == "Circuit Synthesizer":
        synthesizer_panel()
    else:
        source_panel()
        st.divider()
        st.caption("Available documents")
        for document in rt.lens.store.sources():
            st.text(str(document))
with right:
    st.subheader("Questions")
    conversation()
    question_input()
    with st.expander("Use microphone", icon=":material/mic:", on_change="rerun") as voice:
        if voice.open:
            voice_panel()
    with st.container(horizontal=True):
        if st.button("Stop speech", icon=":material/volume_off:"):
            rt.stop_talking()
        if st.button("Repeat answer", icon=":material/replay:"):
            rt.repeat()
    st.caption("Answers cite your manual. Check wiring before applying power.")
