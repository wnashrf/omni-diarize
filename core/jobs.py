"""Background execution of run_pipeline.py for the Streamlit UI.

The pipeline runs as a separate process writing to data/logs/<job>.log. The UI never
blocks on it: a polling fragment tails the log file and checks the exit code, so the job
survives reruns, page switches and browser refreshes.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from core.config import DATA_DIR, LOG_DIR, PIPELINE_SCRIPT, ROOT
from core.transcript import parse_clock

OUTPUT_MARKER = "OUTPUT_JSON="  # keep in sync with run_pipeline.OUTPUT_MARKER

# Pipeline messages are "[HH:MM:SS] <msg>"; everything else (tracebacks, warnings) is noise.
LOG_LINE_RE = re.compile(r"^\[\d{2}:\d{2}:\d{2}\] (.*)$")
TURN_RE = re.compile(r"^\[(\d+)/(\d+)\] (\S+) ([\d:.]+)-([\d:.]+) \(ETA ([\d.]+) min\) (.*)$")
STITCH_RE = re.compile(r"^Stitched into (\d+) turns across (\d+) speakers")
DURATION_RE = re.compile(r"^Audio duration: .*\(([\d.]+)s\)$")
EXCEPTION_RE = re.compile(r"^[\w.]*(?:Error|Exception)\b")
NO_SPEECH = "<no speech>"

# Stages: 0 ingest audio · 1 diarize · 2 transcribe · 3 complete.
# (message prefix, stage it starts or belongs to, plain-language note)
MILESTONES = [
    ("Execution device", 0, "Starting up"),
    ("Using local file", 0, "Reading local file"),
    ("Video:", 0, "Fetched video details"),
    ("Downloading audio", 0, "Downloading audio from YouTube"),
    ("Retrying download", 0, "YouTube blocked the first attempt — retrying"),
    ("Reusing existing audio", 0, "Using previously downloaded audio"),
    ("Reusing prepared audio", 0, "Using previously prepared audio"),
    ("Converting audio", 0, "Converting audio to 16 kHz mono"),
    ("Loading pyannote", 1, "Loading diarization model"),
    ("Diarizing", 1, "Detecting who speaks when"),
    ("Loading faster-whisper", 2, "Loading speech-recognition model"),
    ("Done:", 3, "Complete"),
]


@dataclass
class RunState:
    """What the pipeline log says so far, in UI terms."""

    stage: int = 0
    notes: dict[int, str] = field(default_factory=lambda: {0: "Starting up"})
    turns: list[dict] = field(default_factory=list)  # transcribed turns that contain speech
    current: int = 0  # turns processed (including silent ones)
    total: int = 0  # turns to transcribe, known once diarization finishes
    processed: float = 0.0  # seconds of audio covered by processed turns
    audio_total: float | None = None
    eta: float | None = None  # seconds
    speakers_found: int | None = None
    error: str | None = None


def parse_log(lines: list[str]) -> RunState:
    state = RunState()
    for line in lines:
        if not (m := LOG_LINE_RE.match(line)):
            continue
        msg = m.group(1)
        if t := TURN_RE.match(msg):
            i, n, speaker, start, end, eta, text = t.groups()
            state.stage, state.current, state.total = 2, int(i), int(n)
            state.processed, state.eta = parse_clock(end), float(eta) * 60
            state.notes[2] = f"Transcribing turn {i} of {n}"
            if text != NO_SPEECH:
                state.turns.append({"n": int(i), "speaker": speaker, "start": parse_clock(start),
                                    "end": state.processed, "text": text})
        elif s := STITCH_RE.match(msg):
            state.stage = max(state.stage, 1)
            state.total, state.speakers_found = int(s.group(1)), int(s.group(2))
            state.notes[1] = f"Found {s.group(1)} turns from {s.group(2)} speakers"
        elif d := DURATION_RE.match(msg):
            state.audio_total = float(d.group(1))
        elif msg.startswith("ERROR:"):
            state.error = msg.removeprefix("ERROR:").strip()
        else:
            for prefix, stage, note in MILESTONES:
                if msg.startswith(prefix):
                    state.stage = max(state.stage, stage)
                    state.notes[stage] = note
                    break
    return state


@dataclass
class Job:
    id: str
    label: str
    cmd: list[str]
    log_path: Path
    proc: subprocess.Popen
    started: float = field(default_factory=time.time)
    finished: float | None = None
    cancelled: bool = False
    _log_handle: object | None = None

    # -------------------------------------------------------------- state
    def poll(self) -> int | None:
        rc = self.proc.poll()
        if rc is not None and self.finished is None:
            self.finished = time.time()
            if self._log_handle:
                self._log_handle.close()
                self._log_handle = None
        return rc

    @property
    def running(self) -> bool:
        return self.poll() is None

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def lines(self, tail: int | None = None) -> list[str]:
        """Log lines, with carriage-return progress bars collapsed to their last state."""
        try:
            raw = self.log_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            return []
        out = []
        for line in raw.splitlines():
            parts = [p for p in line.split("\r") if p.strip()]
            if parts:
                out.append(parts[-1])
        return out[-tail:] if tail else out

    def state(self) -> RunState:
        state = parse_log(lines := self.lines())
        rc = self.poll()
        if rc not in (None, 0) and not self.cancelled and not state.error:
            # Uncaught exception: surface its final "SomethingError: message" line
            state.error = next((ln.strip() for ln in reversed(lines) if EXCEPTION_RE.match(ln.strip())),
                               f"exit code {rc}")
        return state

    def output_json(self) -> Path | None:
        for line in reversed(self.lines(tail=20)):
            if line.startswith(OUTPUT_MARKER):
                path = Path(line[len(OUTPUT_MARKER):].strip())
                return path if path.exists() else None
        return None

    def cancel(self) -> None:
        if not self.running:
            return
        self.cancelled = True
        if os.name == "nt":  # kill the whole tree (yt-dlp / ffmpeg children included)
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
        else:
            os.killpg(self.proc.pid, signal.SIGTERM)


class JobManager:
    """One pipeline at a time, shared by all browser sessions (it is one GPU/CPU)."""

    def __init__(self) -> None:
        self.current: Job | None = None
        self._lock = threading.Lock()

    def start(self, *, label: str, url: str | None = None, audio: Path | None = None,
              max_duration: int | None = None, force_cpu: bool = False) -> Job:
        with self._lock:
            if self.current and self.current.running:
                raise RuntimeError("A pipeline run is already in progress.")

            cmd = [sys.executable, "-u", str(PIPELINE_SCRIPT), "--output-dir", str(DATA_DIR),
                   "--device", "cpu" if force_cpu else "auto"]
            cmd += ["--url", url] if url else ["--audio", str(audio)]
            if max_duration:
                cmd += ["--max-duration", str(max_duration)]

            job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}"
            log_path = LOG_DIR / f"pipeline_{job_id}.log"
            handle = open(log_path, "wb")
            env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
            kwargs: dict = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                kwargs["start_new_session"] = True
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, **kwargs)
            self.current = Job(id=job_id, label=label, cmd=cmd, log_path=log_path,
                               proc=proc, _log_handle=handle)
            return self.current


@st.cache_resource
def get_job_manager() -> JobManager:
    return JobManager()
