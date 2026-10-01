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

OUTPUT_MARKER = "OUTPUT_JSON="  # keep in sync with run_pipeline.OUTPUT_MARKER
TURN_RE = re.compile(r"\[(\d+)/(\d+)\]")

# (substring in a log line, progress fraction, human label) — checked newest line first
STAGES = [
    ("Done:", 1.00, "Complete"),
    ("Writing transcript", 0.45, "Transcribing turns"),
    ("Loading faster-whisper", 0.42, "Loading ASR model"),
    ("Stitched into", 0.40, "Diarization complete"),
    ("Diarizing", 0.15, "Diarizing speakers"),
    ("Loading pyannote", 0.10, "Loading diarization model"),
    ("Downloading audio", 0.05, "Downloading audio"),
    ("Converting audio", 0.05, "Preparing audio"),
    ("Execution device", 0.02, "Starting"),
]


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

    def progress(self) -> tuple[float, str]:
        lines = self.lines()
        for line in reversed(lines):
            if m := TURN_RE.search(line):
                i, n = int(m.group(1)), int(m.group(2))
                return 0.45 + 0.55 * i / max(n, 1), f"Transcribing turn {i}/{n}"
            for needle, frac, label in STAGES:
                if needle in line:
                    return frac, label
        return 0.0, "Starting"

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
