"""Page 2 — interactive Hansard viewer for diarized transcripts in data/."""

from __future__ import annotations

import html
import json
import time

import pandas as pd
import streamlit as st

from core.config import CHAMBER, RUN_PAGE
from core.jobs import get_job_manager
from core.transcript import (
    SEGMENT_KEYS, UNIDENTIFIED, audio_mime, build_docx, build_txt, classify, fmt_ts,
    list_transcripts, load_audio_bytes, load_transcript, meta_row, pill, render_card, resolve_audio,
)

PAGE_SIZE = 25
HIDDEN_META = {"audio_file", "video_id", "status", "max_duration_min"}

if flash := st.session_state.pop("flash", None):
    st.toast(flash, icon="✅")

# ------------------------------------------------------------ session select
transcripts = list_transcripts()
if not transcripts:
    st.title("🏛️ Transcript Viewer")
    st.info("No transcripts in `data/` yet. Generate one on the Run Pipeline page.")
    st.page_link(RUN_PAGE, label="Go to Run Pipeline", icon="🚀")
    st.stop()

names = [p.name for p in transcripts]
current = st.session_state.get("viewer_transcript")
if current not in names:
    current = names[0]

with st.sidebar:
    st.header("📂 Session")
    selected = st.selectbox("Transcript dataset", names, index=names.index(current),
                            help="Newest first. New pipeline output appears here automatically.")
if selected != current:  # a new session resets playback and paging
    st.session_state.update(seek=None, active_turn=None, page=1)
st.session_state["viewer_transcript"] = selected
path = transcripts[names.index(selected)]
mtime = path.stat().st_mtime

try:
    segments, file_meta = load_transcript(path, mtime)
except (json.JSONDecodeError, TypeError, ValueError) as e:
    st.error(f"Could not parse `{path.name}`: {e}")
    st.stop()

if not segments:
    st.warning("No transcript turns found. Expected a list under one of: "
               + ", ".join(f"`{k}`" for k in SEGMENT_KEYS) + ".")
    st.stop()

speakers = sorted({s["speaker"] for s in segments})
talk_time = {sp: 0.0 for sp in speakers}
for s in segments:
    talk_time[s["speaker"]] += s["end"] - s["start"]
by_talk_time = sorted(speakers, key=lambda sp: -talk_time[sp])
audio_path = resolve_audio(path, file_meta)


def play_from(seg: dict) -> None:
    st.session_state["seek"] = seg["start"]
    st.session_state["active_turn"] = seg["id"]


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🔊 Playback")
    if audio_path:
        seek = st.session_state.get("seek")
        st.audio(load_audio_bytes(audio_path, audio_path.stat().st_mtime), format=audio_mime(audio_path),
                 start_time=int(seek or 0), autoplay=seek is not None)
        st.caption(f"▶ Playing from **{fmt_ts(seek)}**" if seek is not None
                   else "Click ▶ on any turn to jump there.")
    else:
        st.caption("Audio not found for this transcript — text only.")

    st.header("🏛️ Chamber")
    shown_meta = {**CHAMBER, **{k.replace("_", " ").title(): str(v)
                                for k, v in file_meta.items() if k not in HIDDEN_META}}
    st.markdown("".join(meta_row(k, v) for k, v in shown_meta.items()), unsafe_allow_html=True)

    st.header(f"🗣️ Speakers ({len(speakers)})")
    for sp in by_talk_time:
        icon, role, _ = classify(sp)
        status = pill("Unidentified", "warn") if role == UNIDENTIFIED else pill("Identified", "ok")
        st.markdown(f"{icon} **{html.escape(sp)}** {status}<br><small>{role} · {fmt_ts(talk_time[sp])} talk time</small>",
                    unsafe_allow_html=True)

    st.header("📥 Export")
    st.download_button("Hansard (.txt)", build_txt(segments), file_name=f"{path.stem}_hansard.txt",
                       mime="text/plain", width="stretch", icon="📄")
    st.download_button("Hansard (.docx)", build_docx(path, mtime), file_name=f"{path.stem}_hansard.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       width="stretch", icon="📝")
    st.download_button("Raw JSON (.json)", path.read_bytes(), file_name=path.name,
                       mime="application/json", width="stretch", icon="🧾")

# ----------------------------------------------------------------- main view
st.title("🏛️ Transcript Viewer")
st.caption(f"{file_meta.get('title') or path.stem} · speaker diarization & Malay-English transcription")

job = get_job_manager().current
if job and job.running:
    st.info(f"A pipeline run is in progress: **{job.label}**", icon="⏳")
if file_meta.get("status") == "in_progress":
    st.warning(f"This transcript is still being written (last update {time.strftime('%H:%M:%S', time.localtime(mtime))}). "
               "Rerun the page to load new turns.", icon="✍️")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total turns", len(segments))
m2.metric("Distinct speakers", len(speakers))
m3.metric("Spoken time", fmt_ts(sum(talk_time.values())))
m4.metric("Words", f"{sum(len(s['text'].split()) for s in segments):,}")

with st.expander("📊 Speaker distribution & analytics", expanded=False):
    df = pd.DataFrame([
        {
            "Speaker": sp,
            "Talk time (min)": round(talk_time[sp] / 60, 1),
            "Share (%)": round(100 * talk_time[sp] / max(sum(talk_time.values()), 1), 1),
            "Turns": sum(1 for s in segments if s["speaker"] == sp),
            "Words": sum(len(s["text"].split()) for s in segments if s["speaker"] == sp),
        }
        for sp in by_talk_time
    ])
    chart_col, table_col = st.columns(2)
    with chart_col:
        st.bar_chart(df, x="Speaker", y="Talk time (min)", horizontal=True, sort="-Talk time (min)")
    with table_col:
        st.dataframe(df, width="stretch", hide_index=True)

st.divider()

c1, c2 = st.columns([3, 2])
query = c1.text_input("🔍 Search transcript", placeholder="e.g. belanjawan, subsidi, peruntukan…").strip()
chosen = c2.multiselect("Filter by speaker", speakers, default=speakers)

shown = [s for s in segments
         if s["speaker"] in chosen and (not query or query.lower() in s["text"].lower())]
if not shown:
    st.info("No turns match the current search / filter.")
    st.stop()

n_pages = (len(shown) - 1) // PAGE_SIZE + 1
info_col, page_col = st.columns([3, 1], vertical_alignment="bottom")
page = page_col.number_input("Page", 1, n_pages, min(st.session_state.get("page", 1), n_pages))
st.session_state["page"] = page
lo = (page - 1) * PAGE_SIZE
info_col.caption(f"Showing turns **{lo + 1}–{min(lo + PAGE_SIZE, len(shown))}** of **{len(shown)}** "
                 f"matching ({len(segments)} total)")

active = st.session_state.get("active_turn")
for seg in shown[lo:lo + PAGE_SIZE]:
    btn_col, card_col = st.columns([1, 24], vertical_alignment="center", gap="small")
    btn_col.button("▶", key=f"play_{seg['id']}", help=f"Play from {fmt_ts(seg['start'])}",
                   on_click=play_from, args=(seg,), disabled=audio_path is None, type="tertiary")
    card_col.markdown(render_card(seg, query, active=seg["id"] == active), unsafe_allow_html=True)
