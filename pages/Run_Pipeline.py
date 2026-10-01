"""Page 1 — ingest a YouTube URL or local file and run the pipeline with live logs."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import streamlit as st

from core.config import ROOT, UPLOAD_DIR, UPLOAD_TYPES, VIEWER_PAGE
from core.jobs import Job, get_job_manager
from core.transcript import fmt_ts, pill
from run_pipeline import PipelineError, fetch_video_info, sanitize

DEMO_URL = "https://www.youtube.com/watch?v=kY8wAT2QO5Y"
HF_LINKS = {
    "pyannote/speaker-diarization-3.1": "https://huggingface.co/pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0": "https://huggingface.co/pyannote/segmentation-3.0",
}

manager = get_job_manager()


# ----------------------------------------------------------------- helpers
def _has_module(name: str) -> bool:
    try:
        return find_spec(name) is not None
    except ModuleNotFoundError:
        return False


@st.cache_data(ttl=60, show_spinner=False)
def environment_status() -> dict:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ml = {m: _has_module(m) for m in ("torch", "pyannote.audio", "faster_whisper")}
    cuda = None
    if ml["torch"]:  # probe in a subprocess so the UI process never imports torch
        probe = subprocess.run([sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
                               capture_output=True, text=True)
        cuda = probe.stdout.strip() == "True"
    return {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "hf_token": bool(os.environ.get("HF_TOKEN", "").strip()),
        "ml": ml,
        "cuda": cuda,
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


# ------------------------------------------------------------- live panel
@st.fragment(run_every=1.0)
def live_panel() -> None:
    job = manager.current
    if job is None:
        return
    if not job.running:
        if st.session_state.get("watch_job") == job.id:
            st.session_state["auto_open"] = job.id
        st.rerun(scope="app")

    frac, label = job.progress()
    st.progress(min(frac, 1.0), text=f"**{label}** · elapsed {fmt_ts(job.elapsed)}")
    with st.expander("📟 Live terminal output", expanded=True):
        st.code("\n".join(job.lines(tail=40)) or "Starting pipeline…", language=None, wrap_lines=True)
    st.button("⏹ Cancel run", key=f"cancel_{job.id}", on_click=job.cancel)


def finished_panel(job: Job) -> None:
    rc, out = job.poll(), job.output_json()
    if rc == 0 and out:
        st.success(f"Completed in {fmt_ts(job.elapsed)} → `{out.name}`", icon="✅")
        if st.button("🏛️ Open in Transcript Viewer", type="primary"):
            st.session_state["viewer_transcript"] = out.name
            st.switch_page(VIEWER_PAGE)
    elif job.cancelled:
        st.warning(f"Run cancelled after {fmt_ts(job.elapsed)}.", icon="⏹")
    else:
        st.error(f"Pipeline failed (exit code {rc}) after {fmt_ts(job.elapsed)}. See the log below.", icon="🚨")

    log_text = "\n".join(job.lines())
    with st.expander("📟 Full terminal output", expanded=rc != 0 and not job.cancelled):
        st.code(log_text or "(empty log)", language=None, wrap_lines=True, height=420)
    st.download_button("Download log", log_text, file_name=job.log_path.name, mime="text/plain")


# -------------------------------------------------------------------- page
job = manager.current

# Auto-open the viewer when a run this session was watching completes successfully
if job and not job.running and st.session_state.get("auto_open") == job.id:
    del st.session_state["auto_open"]
    if job.poll() == 0 and (out := job.output_json()):
        st.session_state["viewer_transcript"] = out.name
        st.session_state["flash"] = f"New transcript ready: {out.name}"
        st.switch_page(VIEWER_PAGE)

st.title("🚀 Run Pipeline")
st.caption("Ingest a Dewan Rakyat sitting → diarize speakers → transcribe Malay-English speech → Hansard JSON")

env = environment_status()
ml_ok = all(env["ml"].values())
cuda_txt = {True: "CUDA GPU detected", False: "No CUDA GPU (CPU mode)", None: "CUDA: unknown"}[env["cuda"]]
st.markdown(
    pill("ffmpeg ✓" if env["ffmpeg"] else "ffmpeg missing", "ok" if env["ffmpeg"] else "err")
    + pill("HF_TOKEN ✓" if env["hf_token"] else "HF_TOKEN missing", "ok" if env["hf_token"] else "err")
    + pill("ML stack ✓" if ml_ok else "ML stack missing", "ok" if ml_ok else "err")
    + pill(cuda_txt, "ok" if env["cuda"] else "info"),
    unsafe_allow_html=True,
)
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
        st.markdown("**Setup needed before running:**\n" + "\n".join(f"- {p}" for p in problems))
        if st.button("↻ Re-check environment"):
            environment_status.clear()
            st.rerun()

running = bool(job and job.running)

# ---------------------------------------------------------------- 1. source
st.subheader("1 · Source")
source = st.segmented_control("Input type", ["▶️ YouTube URL", "📁 Upload file"], default="▶️ YouTube URL",
                              required=True, disabled=running)
url, upload, info = None, None, None

if source == "▶️ YouTube URL":
    url = st.text_input("YouTube URL", value=DEMO_URL, disabled=running).strip() or None
    if url and not url.startswith(("http://", "https://")):
        st.error("Enter a full URL starting with https://")
        url = None
    if url:
        try:
            info = video_info(url)
        except (PipelineError, FileNotFoundError, ValueError) as e:
            st.error(f"Could not fetch metadata: {e}")
            url = None
        if info:
            with st.container(border=True):
                thumb, details = st.columns([1, 2], vertical_alignment="center")
                if info.get("thumbnail"):
                    thumb.image(info["thumbnail"], width="stretch")
                details.markdown(f"#### {info['title']}")
                d = info.get("upload_date") or ""
                details.markdown(
                    f"📺 **Channel:** {info.get('channel') or '—'}  \n"
                    f"⏱️ **Duration:** {fmt_duration(info.get('duration'))}  \n"
                    + (f"📅 **Uploaded:** {d[:4]}-{d[4:6]}-{d[6:]}  \n" if len(d) == 8 else "")
                    + f"🔗 [{info['webpage_url']}]({info['webpage_url']})"
                )
else:
    upload = st.file_uploader("Audio or video file", type=UPLOAD_TYPES, disabled=running,
                              help="Converted to 16 kHz mono WAV with ffmpeg before processing.")
    if upload:
        st.caption(f"📎 **{upload.name}** · {upload.size / 1_048_576:.1f} MB")

# -------------------------------------------------------------- 2. settings
st.subheader("2 · Execution settings")
c1, c2 = st.columns(2, gap="large")
with c1:
    full_audio = st.checkbox("Full audio", value=False, disabled=running)
    max_minutes = st.slider("Duration limit (minutes)", 1, 60, 2, disabled=running or full_audio)
    st.caption("💡 Recommended for local CPU demos to prevent long processing times.")
    if info and info.get("duration"):
        span = info["duration"] if full_audio else min(info["duration"], max_minutes * 60)
        st.caption(f"Will process **{fmt_ts(span)}** of {fmt_ts(info['duration'])}.")
with c2:
    force_cpu = st.checkbox("Force CPU execution (int8)", value=not env["cuda"], disabled=running,
                            help="Unchecked = auto-detect CUDA (float16 on GPU, int8 on CPU fallback).")
    st.caption("🖥️ CPU · int8 quantised Whisper" if force_cpu else f"⚡ Auto-detect CUDA — {cuda_txt.lower()}")

# ----------------------------------------------------------------- 3. start
has_source = bool(url) or upload is not None
blocked = running or not has_source or bool(problems)
if st.button("▶  Start Processing", type="primary", width="stretch", disabled=blocked):
    audio_path = save_upload(upload) if upload is not None else None
    label = info["title"] if info else (upload.name if upload else url)
    try:
        new_job = manager.start(label=label, url=url if audio_path is None else None, audio=audio_path,
                                max_duration=None if full_audio else max_minutes, force_cpu=force_cpu)
    except RuntimeError as e:
        st.error(str(e))
    else:
        st.session_state["watch_job"] = new_job.id
        st.rerun()
elif not has_source and not running:
    st.caption("Provide a YouTube URL or upload a file to enable processing.")

# ------------------------------------------------------------------ 4. run
if job:
    st.subheader("3 · Run status")
    st.markdown(f"**{job.label}**")
    st.code(cli_equivalent(job), language="bash")
    if job.running:
        live_panel()
    else:
        finished_panel(job)
