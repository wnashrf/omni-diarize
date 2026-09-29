"""omni-diarize — Streamlit viewer for diarized Dewan Rakyat transcripts.

Reads datasets from data/ and plays corresponding audio tracks.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

MODELS = {
    "Diarization": "pyannote/speaker-diarization-3.1",
    "ASR": "faster-whisper large-v3",
    "Speaker ID": "Cosine similarity · voice embeddings",
}
CHAMBER = {
    "Chamber": "Dewan Rakyat",
    "Parliament": "Parlimen Malaysia",
    "Language": "Bahasa Melayu / English",
}

# (keywords matched against the lower-cased speaker label, icon, role, accent colour)
ROLES = [
    (("prime minister", "perdana menteri", "pm"), "🏛️", "Prime Minister", "#b8860b"),
    (("sri aman",), "🎤", "MP Sri Aman", "#1f77b4"),
    (("speaker", "yang dipertua", "tuan yang di-pertua"), "👤", "Speaker", "#6c757d"),
]
UNKNOWN_ROLE = ("👤", "Speaker", "#4f8bf9")


# ---------------------------------------------------------------- data loading
def _first(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


SEGMENT_KEYS = ("turns", "segments", "transcript", "utterances", "results")


def _parse_clock(value: str) -> float:
    """'MM:SS.ss' or 'HH:MM:SS.ss' -> seconds."""
    secs = 0.0
    for part in value.strip().split(":"):
        secs = secs * 60 + float(part)
    return secs


def _times(seg: dict) -> tuple[float, float]:
    start = _first(seg, "start", "start_time", "begin")
    end = _first(seg, "end", "end_time", "stop")
    ts = seg.get("timestamp")
    if (start is None or end is None) and isinstance(ts, str) and "-" in ts:
        try:
            ts_start, ts_end = (_parse_clock(p) for p in ts.split("-", 1))
            start = ts_start if start is None else start
            end = ts_end if end is None else end
        except ValueError:
            pass
    return float(start or 0.0), float(end or 0.0)


def _find_segment_list(raw: dict) -> list:
    for key in SEGMENT_KEYS:
        if isinstance(raw.get(key), list):
            return raw[key]
    for value in raw.values():
        if isinstance(value, list) and value and isinstance(value[0], dict) and "text" in value[0]:
            return value
    return []


@st.cache_data
def load_transcript(path: Path, mtime: float) -> tuple[list[dict], dict]:
    """Load segments from JSON: a bare list, or a dict holding them under 'turns'/'segments'/…"""
    raw = json.loads(path.read_text(encoding="utf-8"))
    meta: dict = {}
    if isinstance(raw, dict):
        meta = {k: v for k, v in raw.items() if not isinstance(v, (list, dict))}
        raw = _find_segment_list(raw)

    segments = []
    for i, seg in enumerate(raw):
        if not isinstance(seg, dict):
            continue
        start, end = _times(seg)
        segments.append(
            {
                "id": i,
                "start": start,
                "end": end,
                "speaker": str(_first(seg, "speaker_name", "speaker", "label", default="Unknown")),
                "text": str(_first(seg, "text", "transcript", "content", default="")).strip(),
                "confidence": _first(seg, "confidence", "similarity", "score"),
            }
        )
    return segments, meta


def classify(speaker: str) -> tuple[str, str, str]:
    """Return (icon, role, colour) for a speaker label."""
    s = speaker.lower()
    for keywords, icon, role, colour in ROLES:
        if any(k == s or k in s.split() or (" " in k and k in s) for k in keywords):
            return icon, role, colour
    return UNKNOWN_ROLE


def fmt_ts(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def highlight(text: str, query: str) -> str:
    safe = html.escape(text)
    if not query:
        return safe
    q = html.escape(query)
    lower = safe.lower()
    out, i, ql = [], 0, len(q)
    while (j := lower.find(q.lower(), i)) != -1:
        out.append(safe[i:j])
        out.append(f"<mark>{safe[j:j + ql]}</mark>")
        i = j + ql
    out.append(safe[i:])
    return "".join(out)


# ---------------------------------------------------------------------- styles
CSS = """
<style>
.pill {display:inline-block;padding:2px 10px;margin:2px 4px 2px 0;border-radius:999px;
       font-size:0.78rem;font-weight:600;border:1px solid transparent;white-space:nowrap}
.pill-ok   {background:rgba(40,167,69,.15);color:#28a745;border-color:rgba(40,167,69,.4)}
.pill-warn {background:rgba(255,193,7,.15);color:#c69500;border-color:rgba(255,193,7,.5)}
.pill-info {background:rgba(31,119,180,.12);color:#1f77b4;border-color:rgba(31,119,180,.4)}
.meta-row {display:flex;justify-content:space-between;gap:8px;font-size:0.85rem;padding:2px 0}
.meta-row span:first-child {opacity:.65}
.card {border-left:4px solid var(--accent);border-radius:10px;padding:10px 14px;margin:8px 0;
       background:rgba(127,127,127,.07)}
.card-head {display:flex;align-items:center;gap:8px;margin-bottom:4px;flex-wrap:wrap}
.card-icon {font-size:1.35rem;line-height:1}
.card-speaker {font-weight:700;color:var(--accent)}
.card-role {font-size:.72rem;opacity:.7}
.card-ts {margin-left:auto;font-family:ui-monospace,monospace;font-size:.78rem;opacity:.7}
.card-text {line-height:1.55}
.card-text mark {background:#ffe066;color:#000;padding:0 2px;border-radius:3px}
</style>
"""


def pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill pill-{kind}">{html.escape(text)}</span>'


def render_card(seg: dict, query: str) -> str:
    icon, role, colour = classify(seg["speaker"])
    conf = ""
    if isinstance(seg["confidence"], (int, float)):
        conf = f' · sim {seg["confidence"]:.2f}'
    return f"""
<div class="card" style="--accent:{colour}">
  <div class="card-head">
    <span class="card-icon">{icon}</span>
    <span class="card-speaker">{html.escape(seg["speaker"])}</span>
    <span class="card-role">{role}{conf}</span>
    <span class="card-ts">{fmt_ts(seg["start"])} → {fmt_ts(seg["end"])}</span>
  </div>
  <div class="card-text">{highlight(seg["text"], query)}</div>
</div>"""


# -------------------------------------------------------------------------- UI
st.set_page_config(page_title="omni-diarize · Dewan Rakyat", page_icon="🏛️", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
# 1. Dynamic Session Selector across data/ and root
available_json = sorted(list(DATA_DIR.glob("*.json")) + list(Path(__file__).parent.glob("*.json")))
json_names = [p.name for p in available_json]

with st.sidebar:
    st.header("📂 Session Select")
    if json_names:
        selected_name = st.selectbox("Transcript Dataset", json_names, index=0)
        active_transcript_path = next(p for p in available_json if p.name == selected_name)
    else:
        st.warning("No transcript JSON found in data/ or root.")
        st.stop()

    st.header("⚙️ Pipeline")
    for k, v in MODELS.items():
        st.markdown(f'<div class="meta-row"><span>{k}</span><span>{html.escape(v)}</span></div>',
                    unsafe_allow_html=True)
    st.markdown(pill("Diarized", "ok") + pill("Transcribed", "ok") + pill("MP matching", "info"),
                unsafe_allow_html=True)

try:
    segments, file_meta = load_transcript(active_transcript_path, active_transcript_path.stat().st_mtime)
except (json.JSONDecodeError, TypeError, ValueError) as e:
    st.error(f"Could not parse transcript: {e}")
    st.stop()

if not segments:
    st.warning("No transcript turns found. Expected a list under one of: "
               + ", ".join(f"`{k}`" for k in SEGMENT_KEYS) + ".")
    st.stop()

speakers = sorted({s["speaker"] for s in segments})
talk_time = {sp: sum(s["end"] - s["start"] for s in segments if s["speaker"] == sp) for sp in speakers}

with st.sidebar:
    st.header("🏛️ Chamber")
    for k, v in {**CHAMBER, **{k.title(): str(v) for k, v in file_meta.items()}}.items():
        st.markdown(f'<div class="meta-row"><span>{html.escape(k)}</span>'
                    f'<span>{html.escape(v)}</span></div>', unsafe_allow_html=True)

    st.header(f"🗣️ Speakers ({len(speakers)})")
    for sp in sorted(speakers, key=lambda s: -talk_time[s]):
        icon, role, _ = classify(sp)
        status = pill("Unidentified", "warn") if role == "Speaker" else pill("Identified", "ok")
        st.markdown(f"{icon} **{html.escape(sp)}** {status}<br>"
                    f"<small>{role} · {fmt_ts(talk_time[sp])} talk time</small>",
                    unsafe_allow_html=True)

    # 2. Export Hansard and JSON Actions
    st.header("📥 Export")
    hansard_txt = "\n\n".join(
        [f"[{fmt_ts(s['start'])} → {fmt_ts(s['end'])}] {s['speaker']}:\n{s['text']}" for s in segments]
    )
    st.download_button(
        "Download Hansard (.txt)",
        data=hansard_txt,
        file_name=f"{active_transcript_path.stem}_hansard.txt",
        mime="text/plain",
        use_container_width=True
    )
    st.download_button(
        "Download JSON (.json)",
        data=active_transcript_path.read_text(encoding="utf-8"),
        file_name=active_transcript_path.name,
        mime="application/json",
        use_container_width=True
    )

# ----------------- MAIN VIEW -----------------
st.title("🏛️ omni-diarize")
st.caption("Speaker diarization & Malay-English transcription for the Dewan Rakyat")

# Dynamic audio path finder
audio_target = DATA_DIR / f"{active_transcript_path.stem.replace('_transcript', '')}.wav"
fallback_audio = DATA_DIR / "parlimen_test.wav"
active_audio = audio_target if audio_target.exists() else (fallback_audio if fallback_audio.exists() else None)

if active_audio:
    st.audio(str(active_audio), format="audio/wav")
else:
    st.info(f"Audio not found for this transcript — displaying Hansard text.")

# Top Metrics Row
m1, m2, m3 = st.columns(3)
total_speech_time = sum(talk_time.values())
m1.metric("Total Turns", len(segments))
m2.metric("Distinct Speakers", len(speakers))
m3.metric("Total Spoken Time", fmt_ts(total_speech_time))

# 3. Speaker Activity Analytics Expander
with st.expander("📊 Speaker Distribution & Analytics", expanded=False):
    chart_col, table_col = st.columns([1, 1])
    speaker_data = []
    for sp in sorted(speakers, key=lambda s: -talk_time[s]):
        sp_segments = [s for s in segments if s["speaker"] == sp]
        words = sum(len(s["text"].split()) for s in sp_segments)
        speaker_data.append({
            "Speaker": sp,
            "Talk Time (s)": round(talk_time[sp], 1),
            "Turns": len(sp_segments),
            "Words": words
        })
    df_speakers = pd.DataFrame(speaker_data)

    with chart_col:
        st.bar_chart(df_speakers.set_index("Speaker")["Talk Time (s)"])
    with table_col:
        st.dataframe(df_speakers, use_container_width=True, hide_index=True)

st.divider()

# Search / Filter
c1, c2 = st.columns([3, 2])
query = c1.text_input("🔍 Search transcript", placeholder="e.g. belanjawan, subsidi, peruntukan…").strip()
chosen = c2.multiselect("Filter by speaker", speakers, default=speakers)

shown = [
    s for s in segments
    if s["speaker"] in chosen and (not query or query.lower() in s["text"].lower())
]

st.caption(f"Showing **{len(shown)}** of **{len(segments)}** turns")

if not shown:
    st.info("No segments match the current search / filter.")
else:
    st.markdown("".join(render_card(s, query) for s in shown), unsafe_allow_html=True)