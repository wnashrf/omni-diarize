"""Page 1 — ingest a YouTube URL or local file and run the pipeline with live logs."""

from __future__ import annotations

import html
import os
import re
import shlex
import shutil
from importlib.util import find_spec
from pathlib import Path

import streamlit as st

from core.config import MODELS, ROOT, UPLOAD_DIR, UPLOAD_TYPES, VIEWER_PAGE
from core.jobs import Job, RunState, get_job_manager
from core.transcript import fmt_ts, load_transcript, meta_row
from run_pipeline import PipelineError, fetch_video_info, sanitize

DEMO_URL = "https://www.youtube.com/watch?v=kY8wAT2QO5Y"
HF_LINKS = {
    "pyannote/speaker-diarization-3.1": "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0": "https://huggingface.co/pyannote/segmentation-3.0",
}

# (title, technology) for each pipeline stage, in order — indices match RunState.stage
STAGES = [
    ("Ingesting audio", "yt-dlp · ffmpeg"),
    ("Speaker diarization", "pyannote VAD & clustering"),
    ("Turn-by-turn transcription", "faster-whisper large-v3"),
]
STAGE_DETAILS = [
    "Downloads the sitting (or reads your file) and converts it to 16 kHz mono audio.",
    "Detects speech and groups it into turns by voice — who spoke when.",
    "Transcribes each turn in Malay-English; dialogue appears live as it is produced.",
]
# Header status badge: (label, icon) per running stage, (label, icon, colour) per final outcome
STAGE_BADGES = [("Downloading", ":material/download:"), ("Diarizing", ":material/groups:"),
                ("Transcribing", ":material/subtitles:")]
OUTCOME_BADGES = {"done": ("Complete", ":material/check_circle:", "green"),
                  "failed": ("Failed", ":material/error:", "red"),
                  "cancelled": ("Cancelled", ":material/stop_circle:", "orange")}
FEED_SIZE = 5  # latest transcribed turns shown in the live dialogue feed
SPEAKER_COLOURS = ["#6ea8fe", "#f6c344", "#5fd3a0", "#f28b82", "#c39bf5", "#4fd1e0", "#ffa94d", "#9aa5b1"]

manager = get_job_manager()


# ----------------------------------------------------------------- helpers
def _has_module(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _hf_token_set() -> bool:
    if os.environ.get("HF_TOKEN", "").strip():
        return True
    try:  # read .env fresh each time so a newly added token shows up on the next rerun
        from dotenv import dotenv_values
    except ImportError:
        return False
    return bool((dotenv_values(ROOT / ".env").get("HF_TOKEN") or "").strip())


def _cuda_available() -> bool | None:
    """Best guess at CUDA support without importing torch (that takes ~5 s).

    torch/version.py records whether the installed wheel is a CUDA build; a CUDA build is only
    usable with an NVIDIA driver, which ships nvidia-smi. The pipeline makes the real check itself.
    """
    try:
        spec = find_spec("torch")
    except ModuleNotFoundError:
        spec = None
    if spec is None or not spec.submodule_search_locations:
        return None
    try:
        version = (Path(spec.submodule_search_locations[0]) / "version.py").read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"^cuda\b[^=]*=\s*['\"]([^'\"]+)['\"]", version, re.MULTILINE)
    return bool(m) and shutil.which("nvidia-smi") is not None


def environment_status() -> dict:
    """Every check is a file or PATH lookup (milliseconds), so it runs fresh on each rerun."""
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "hf_token": _hf_token_set(),
        "ml": {m: _has_module(m) for m in ("torch", "pyannote.audio", "faster_whisper")},
        "cuda": _cuda_available(),
    }


@st.cache_data(ttl=3600, show_spinner="Fetching video metadata…")
def video_info(url: str) -> dict:
    return fetch_video_info(url)


def save_upload(upload) -> Path:
    stem, ext = Path(upload.name).stem, Path(upload.name).suffix.lower()
    dest = UPLOAD_DIR / f"{sanitize(stem)}{ext}"
    if not (dest.exists() and dest.stat().st_size == upload.size):
        upload.seek(0)
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload, f, length=8 * 1024 * 1024)
    return dest


def cli_equivalent(job: Job) -> str:
    args = []
    for a in job.cmd[2:]:  # drop interpreter + -u
        p = Path(a)
        if p.is_absolute() and p.is_relative_to(ROOT):
            a = p.relative_to(ROOT).as_posix()
        args.append(a)
    return "python " + " ".join(shlex.quote(a) for a in args)


def fmt_duration(seconds: float | None) -> str:
    return fmt_ts(seconds) if seconds else "unknown"


def badge(label: str, ok: bool) -> None:
    st.badge(label, icon=":material/check:" if ok else ":material/close:", color="green" if ok else "red")


# ------------------------------------------------------------- run monitor
def outcome_of(job: Job) -> str:
    rc = job.poll()
    if rc is None:
        return "running"
    if job.cancelled:
        return "cancelled"
    return "done" if rc == 0 else "failed"


def stepper_html(state: RunState, outcome: str) -> str:
    steps = []
    for i, (title, sub) in enumerate(STAGES):
        if i < state.stage:
            cls, dot = "done", "✓"
        elif i == state.stage:
            cls, dot = {"running": ("active", str(i + 1)), "failed": ("failed", "✕"),
                        "cancelled": ("stopped", "■"), "done": ("done", "✓")}[outcome]
        else:
            cls, dot = "pending", str(i + 1)
        note = state.notes.get(i, "Waiting" if cls == "pending" else "")
        if cls == "done" and i == 0:
            note = "Audio ready"
        elif cls == "done" and i == 2:
            note = f"Transcribed {state.current} turns"
        elif cls == "failed":
            note = f"Failed · {note}" if note else "Failed"
        elif cls == "stopped":
            note = "Cancelled"
        steps.append(
            f'<div class="step step-{cls}"><div class="step-dot">{dot}</div><div>'
            f'<div class="step-title">{title}</div><div class="step-sub">{sub}</div>'
            f'<div class="step-note">{html.escape(note)}</div></div></div>'
        )
    return f'<div class="stepper">{"".join(steps)}</div>'


def speaker_colour(speaker: str) -> str:
    m = re.search(r"(\d+)$", speaker)
    return SPEAKER_COLOURS[(int(m.group(1)) if m else sum(map(ord, speaker))) % len(SPEAKER_COLOURS)]


def bubble_html(turn: dict, latest: bool) -> str:
    m = re.search(r"(\d+)$", turn["speaker"])
    avatar = m.group(1)[-2:] if m else turn["speaker"][:2].upper()
    return (
        f'<div class="bubble {"bubble-latest" if latest else "bubble-old"}" '
        f'style="--accent:{speaker_colour(turn["speaker"])}">'
        f'<div class="bubble-avatar">{html.escape(avatar)}</div><div class="bubble-body">'
        f'<div class="bubble-head"><span class="bubble-speaker">{html.escape(turn["speaker"])}</span>'
        f'<span class="bubble-ts">{fmt_ts(turn["start"])} – {fmt_ts(turn["end"])}</span></div>'
        f'<div class="bubble-text">{html.escape(turn["text"])}</div></div></div>'
    )


def completion_banner(job: Job, state: RunState, outcome: str) -> None:
    if outcome == "cancelled":
        st.warning(f"Run cancelled after {fmt_ts(job.elapsed)}.", icon=":material/stop_circle:")
    elif outcome == "failed":
        st.error(f"**The pipeline stopped with an error.** {state.error}", icon=":material/error:")
        st.caption("Open **Detailed Terminal Logs** below for the full output.")
    elif not (out := job.output_json()):
        st.error("The run finished but its transcript file could not be found.", icon=":material/error:")
    else:
        segments, _ = load_transcript(out, out.stat().st_mtime)
        if not segments:
            st.warning("**Finished, but no speech was found.** Diarization detected no speakers in this audio — "
                       "the clip may be silent or music only (e.g. a live stream's intro). "
                       "Try a longer duration limit.", icon=":material/voice_over_off:")
            return
        n_speakers = len({s["speaker"] for s in segments})
        st.success(f"**Transcription complete** in {fmt_ts(job.elapsed)} — **{len(segments)} turns** "
                   f"from **{n_speakers} speakers**.", icon=":material/check_circle:")
        if st.button("Open in Transcript Viewer 🏛️", type="primary", width="stretch"):
            st.session_state["viewer_transcript"] = out.name
            st.switch_page(VIEWER_PAGE)


def terminal_logs(job: Job) -> None:
    log_text = "\n".join(job.lines())
    with st.expander("🛠️ Detailed Terminal Logs", expanded=False):
        st.code(log_text or "Starting pipeline…", language=None, wrap_lines=True, height=360)
        st.caption("Equivalent command")
        st.code(cli_equivalent(job), language="bash", wrap_lines=True)
        st.download_button("Download log", log_text, file_name=job.log_path.name, mime="text/plain",
                           icon=":material/download:", type="tertiary")


def monitor(job: Job) -> None:
    state, outcome = job.state(), outcome_of(job)

    if outcome == "running":
        badge_label, badge_icon, badge_colour = (*STAGE_BADGES[min(state.stage, 2)], "blue")
    else:
        badge_label, badge_icon, badge_colour = OUTCOME_BADGES[outcome]

    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(badge_label, icon=badge_icon, color=badge_colour)
        st.markdown(f"**{html.escape(job.label)}**")
        st.space("stretch")
        st.caption(f"{fmt_ts(job.elapsed)} elapsed" if outcome == "running" else f"Ran for {fmt_ts(job.elapsed)}")
        if outcome == "running":
            st.button("Cancel", key=f"cancel_{job.id}", on_click=job.cancel, icon=":material/stop_circle:",
                      type="tertiary")

    if outcome != "running":
        completion_banner(job, state, outcome)
    st.markdown(stepper_html(state, outcome), unsafe_allow_html=True)

    if outcome == "running":
        if state.total:
            label = state.notes.get(2) if state.stage == 2 else "Waiting for transcription to start"
            st.progress(state.current / state.total, text=label)
        audio = (f"{fmt_ts(state.processed)} / {fmt_ts(state.audio_total)}" if state.audio_total
                 else fmt_ts(state.processed) if state.current else "—")
        m1, m2, m3 = st.columns(3)
        m1.metric("Current turn", f"{state.current} / {state.total}" if state.current else "—", border=True)
        m2.metric("Audio processed", audio, border=True)
        m3.metric("ETA", fmt_ts(state.eta) if state.eta is not None else "—", border=True)

    if state.turns or outcome == "running":
        st.markdown("##### Live transcript" if outcome == "running" else "##### Last transcribed turns")
        recent = state.turns[-FEED_SIZE:]
        if recent:
            st.markdown("".join(bubble_html(t, latest=i == len(recent) - 1) for i, t in enumerate(recent)),
                        unsafe_allow_html=True)
        else:
            st.caption("Spoken dialogue appears here turn by turn once transcription starts.")

    terminal_logs(job)


@st.fragment(run_every=1.0)
def live_monitor() -> None:
    job = manager.current
    if job is None:
        return
    if not job.running:
        st.rerun(scope="app")  # switch to the static summary
    monitor(job)


def setup_form(*, running: bool) -> None:
    st.markdown("##### Source")
    source = st.segmented_control("Input type", ["YouTube URL", "Upload file"], default="YouTube URL",
                                  required=True, disabled=running, label_visibility="collapsed")
    url, upload, info, preview_slot = None, None, None, None

    if source == "YouTube URL":
        url = st.text_input("YouTube URL", value=DEMO_URL, disabled=running, label_visibility="collapsed",
                            placeholder="https://www.youtube.com/watch?v=…").strip() or None
        if url and not url.startswith(("http://", "https://")):
            st.error("Enter a full URL starting with https://")
            url = None
        preview_slot = st.container()  # filled at the end of the form, see below
    else:
        upload = st.file_uploader("Audio or video file", type=UPLOAD_TYPES, disabled=running,
                                  label_visibility="collapsed",
                                  help="Converted to 16 kHz mono WAV with ffmpeg before processing.")

    st.divider()
    st.markdown("##### Settings")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        full_audio = st.toggle("Process full audio", value=False, disabled=running)
        max_minutes = st.slider("Duration limit (minutes)", 1, 60, 2, disabled=running or full_audio,
                                help="Keep this short for local CPU demos to avoid long processing times.")
        span_slot = st.empty()
    with c2:
        force_cpu = st.toggle("Force CPU (int8)", value=not env["cuda"], disabled=running,
                              help="Off = auto-detect CUDA (float16 on GPU, int8 on CPU fallback).")
        st.caption("CPU · int8 quantised Whisper" if force_cpu else f"Auto-detect · {cuda_txt}")

    has_source = bool(url) or upload is not None
    blocked = running or not has_source or bool(problems)
    start = st.button("Start processing", type="primary", width="stretch", disabled=blocked,
                      icon=":material/play_arrow:")

    # The YouTube preview is the slow part (a yt-dlp call, ~4 s until cached), so it is fetched
    # last: everything above is already on screen while it loads into its reserved slot.
    if url and preview_slot is not None:
        with preview_slot:
            try:
                info = video_info(url)
            except (PipelineError, FileNotFoundError, ValueError) as e:
                st.error(f"Could not fetch metadata: {e}")
                url = None
            else:
                thumb, details = st.columns([1, 3], vertical_alignment="center")
                if info.get("thumbnail"):
                    thumb.image(info["thumbnail"], width="stretch")
                d = info.get("upload_date") or ""
                facts = [info.get("channel") or "Unknown channel", fmt_duration(info.get("duration"))]
                if len(d) == 8:
                    facts.append(f"{d[:4]}-{d[4:6]}-{d[6:]}")
                details.markdown(f"**[{info['title']}]({info['webpage_url']})**")
                details.caption(" · ".join(facts))
        if info and info.get("duration"):
            span = info["duration"] if full_audio else min(info["duration"], max_minutes * 60)
            span_slot.caption(f"Will process **{fmt_ts(span)}** of {fmt_ts(info['duration'])}.")

    if start:
        if not url and upload is None:  # the URL turned out to be unreadable
            st.error("Fix the YouTube URL above (or upload a file) before starting.")
            return
        audio_path = save_upload(upload) if upload is not None else None
        label = info["title"] if info else (upload.name if upload else url)
        try:
            manager.start(label=label, url=url if audio_path is None else None, audio=audio_path,
                          max_duration=None if full_audio else max_minutes, force_cpu=force_cpu)
        except RuntimeError as e:
            st.error(str(e))
        else:
            st.rerun()


# -------------------------------------------------------------------- page
job = manager.current

st.title("Run Pipeline")
st.caption("Ingest a Dewan Rakyat sitting → diarize speakers → transcribe Malay-English speech → Hansard JSON")

# ------------------------------------------------------------- environment
env = environment_status()
ml_ok = all(env["ml"].values())
cuda_txt = {True: "CUDA GPU", False: "CPU only", None: "CUDA unknown"}[env["cuda"]]

with st.container(horizontal=True, vertical_alignment="center", gap="small"):
    badge("ffmpeg", env["ffmpeg"])
    badge("HF_TOKEN", env["hf_token"])
    badge("ML stack", ml_ok)
    st.badge(cuda_txt, icon=":material/memory:", color="green" if env["cuda"] else "gray")
    st.space("stretch")
    with st.popover("Model stack", icon=":material/info:", type="tertiary"):
        st.markdown("".join(meta_row(k, v) for k, v in MODELS.items()), unsafe_allow_html=True)

problems = []
if not env["ffmpeg"]:
    problems.append("Install **ffmpeg** and restart the terminal (see README → Prerequisites).")
if not env["hf_token"]:
    problems.append("Add `HF_TOKEN=…` to `.env` and accept the model licences: "
                    + ", ".join(f"[{k}]({v})" for k, v in HF_LINKS.items()) + ".")
if not ml_ok:
    missing = ", ".join(f"`{m}`" for m, ok in env["ml"].items() if not ok)
    problems.append(f"Missing Python packages {missing} — run `pip install -r requirements.txt`.")
if problems:
    with st.container(border=True):
        st.markdown("**Setup needed before running**\n" + "\n".join(f"- {p}" for p in problems))
        if st.button("Re-check environment", icon=":material/refresh:", type="tertiary"):
            st.rerun()

st.space("small")
running = bool(job and job.running)

if job is None:
    # First visit: setup form beside an overview of what a run does
    form_col, info_col = st.columns([3, 2], gap="large")
    with form_col, st.container(border=True):
        setup_form(running=False)
    with info_col:
        st.markdown("##### How a run works")
        for (title, sub), detail in zip(STAGES, STAGE_DETAILS):
            with st.container(border=True):
                st.markdown(f"**{title}**  \n{detail}")
                st.caption(sub)
else:
    st.markdown("##### Current run" if running else "##### Last run")
    with st.container(border=True):
        if running:
            live_monitor()
        else:
            monitor(job)
    with st.expander("Start a new run", icon=":material/add_circle:", expanded=False):
        if running:
            st.caption("A run is in progress — the pipeline runs one job at a time.")
        setup_form(running=running)
