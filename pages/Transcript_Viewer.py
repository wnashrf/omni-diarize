"""Page 2 — interactive Hansard viewer for diarized transcripts in data/."""

from __future__ import annotations

import json
import time

import pandas as pd
import streamlit as st

from core.config import CHAMBER, MODELS, RUN_PAGE
from core.jobs import get_job_manager
from core.transcript import (
    SEGMENT_KEYS, UNIDENTIFIED, audio_mime, build_docx, build_txt, classify, fmt_ts,
    list_transcripts, load_audio_bytes, load_transcript, meta_row, render_card, resolve_audio,
)

PAGE_SIZE = 25
HIDDEN_META = {"audio_file", "video_id", "status", "max_duration_min", "title"}

# ------------------------------------------------------------ session select
transcripts = list_transcripts()
if not transcripts:
    st.title("Transcript Viewer")
    st.info("No transcripts in `data/` yet. Generate one on the Run Pipeline page.")
    st.page_link(RUN_PAGE, label="Go to Run Pipeline", icon=":material/play_circle:")
    st.stop()

names = [p.name for p in transcripts]
current = st.session_state.get("viewer_transcript")
if current not in names:
    current = names[0]

# Header row: title (filled in once the transcript is loaded) · session picker · export
title_col, session_col, export_col = st.columns([4, 2, 1], vertical_alignment="bottom")
selected = session_col.selectbox("Session", names, index=names.index(current),
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


def reset_page() -> None:
    st.session_state["page"] = 1


def step_page(delta: int) -> None:
    st.session_state["page"] += delta


# ------------------------------------------------------------------- header
with title_col:
    st.title(str(file_meta.get("title") or path.stem.removesuffix("_transcript").replace("_", " ").title()))
    st.caption(" · ".join([CHAMBER["Chamber"], CHAMBER["Parliament"], path.name]))
with export_col, st.popover("Export", icon=":material/download:", width="stretch"):
    st.download_button("Hansard (.txt)", build_txt(segments), file_name=f"{path.stem}_hansard.txt",
                       mime="text/plain", width="stretch", icon=":material/description:")
    st.download_button("Hansard (.docx)", build_docx(path, mtime), file_name=f"{path.stem}_hansard.docx",
                       mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                       width="stretch", icon=":material/article:")
    st.download_button("Raw JSON", path.read_bytes(), file_name=path.name,
                       mime="application/json", width="stretch", icon=":material/data_object:")

job = get_job_manager().current
if job and job.running:
    st.info(f"A pipeline run is in progress: **{job.label}**", icon=":material/hourglass_top:")
if file_meta.get("status") == "in_progress":
    st.warning(f"This transcript is still being written (last update {time.strftime('%H:%M:%S', time.localtime(mtime))}). "
               "Rerun the page to load new turns.", icon=":material/edit_note:")

words = {sp: 0 for sp in speakers}
turns = {sp: 0 for sp in speakers}
for s in segments:
    words[s["speaker"]] += len(s["text"].split())
    turns[s["speaker"]] += 1
total_talk = sum(talk_time.values())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Turns", len(segments), border=True)
m2.metric("Speakers", len(speakers), border=True)
m3.metric("Spoken time", fmt_ts(total_talk), border=True)
m4.metric("Words", f"{sum(words.values()):,}", border=True)

# ----------------------------------------------------------------- playback
with st.container(border=True):
    if audio_path:
        seek = st.session_state.get("seek")
        st.audio(load_audio_bytes(audio_path, audio_path.stat().st_mtime), format=audio_mime(audio_path),
                 start_time=int(seek or 0), autoplay=seek is not None)
        st.caption(f"Playing from **{fmt_ts(seek)}**" if seek is not None
                   else "Press ▶ beside any turn to jump there.")
    else:
        st.caption("No audio found for this transcript — text only.")

tab_transcript, tab_speakers, tab_session = st.tabs(["Transcript", "Speakers", "Session details"], key="viewer_tab")

# --------------------------------------------------------------- transcript
with tab_transcript:
    c1, c2 = st.columns([3, 2])
    query = c1.text_input("Search", placeholder="Search transcript — e.g. belanjawan, subsidi, peruntukan…",
                          label_visibility="collapsed", icon=":material/search:", on_change=reset_page).strip()
    chosen = c2.multiselect("Speakers", by_talk_time, placeholder="All speakers", label_visibility="collapsed",
                            on_change=reset_page) or speakers

    shown = [s for s in segments
             if s["speaker"] in chosen and (not query or query.lower() in s["text"].lower())]
    if not shown:
        st.info("No turns match the current search / filter.", icon=":material/search_off:")
    else:
        n_pages = (len(shown) - 1) // PAGE_SIZE + 1
        st.session_state["page"] = min(max(st.session_state.get("page", 1), 1), n_pages)
        lo = (st.session_state["page"] - 1) * PAGE_SIZE
        st.caption(f"Turns **{lo + 1}–{min(lo + PAGE_SIZE, len(shown))}** of **{len(shown)}**"
                   + (f" matching · {len(segments)} total" if len(shown) != len(segments) else ""))

        active = st.session_state.get("active_turn")
        for seg in shown[lo:lo + PAGE_SIZE]:
            btn_col, card_col = st.columns([1, 24], vertical_alignment="center", gap="small")
            btn_col.button("", key=f"play_{seg['id']}", help=f"Play from {fmt_ts(seg['start'])}",
                           icon=":material/play_arrow:", on_click=play_from, args=(seg,),
                           disabled=audio_path is None, type="tertiary")
            card_col.markdown(render_card(seg, query, active=seg["id"] == active), unsafe_allow_html=True)

        if n_pages > 1:
            page = st.session_state["page"]
            with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="center"):
                st.button("Previous", icon=":material/chevron_left:", on_click=step_page, args=(-1,),
                          disabled=page <= 1, type="tertiary")
                st.caption(f"Page {page} of {n_pages}")
                st.button("Next", icon=":material/chevron_right:", on_click=step_page, args=(1,),
                          disabled=page >= n_pages, type="tertiary")

# ----------------------------------------------------------------- speakers
with tab_speakers:
    rows = []
    for sp in by_talk_time:
        icon, role, _ = classify(sp)
        rows.append({
            "Speaker": f"{icon} {sp}",
            "Role": role,
            "Identified": role != UNIDENTIFIED,
            "Talk time": fmt_ts(talk_time[sp]),
            "Share": 100 * talk_time[sp] / max(total_talk, 1),
            "Turns": turns[sp],
            "Words": words[sp],
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_config={
            "Speaker": st.column_config.TextColumn(width="large"),
            "Identified": st.column_config.CheckboxColumn(),
            "Share": st.column_config.ProgressColumn("Share of talk time", format="%.0f%%",
                                                     min_value=0, max_value=100),
        },
    )

# ------------------------------------------------------------------ session
with tab_session:
    session_meta = {**CHAMBER, **{k.replace("_", " ").title(): str(v)
                                  for k, v in file_meta.items() if k not in HIDDEN_META}}
    session_meta["Transcript file"] = path.name
    session_meta["Audio file"] = audio_path.name if audio_path else "Not found"
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("##### Session")
        st.markdown("".join(meta_row(k, v) for k, v in session_meta.items()), unsafe_allow_html=True)
    with right:
        st.markdown("##### Model stack")
        st.markdown("".join(meta_row(k, v) for k, v in MODELS.items()), unsafe_allow_html=True)
