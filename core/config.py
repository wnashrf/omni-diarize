"""Paths and static metadata shared by every page."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
LOG_DIR = DATA_DIR / "logs"
PIPELINE_SCRIPT = ROOT / "run_pipeline.py"

for _d in (DATA_DIR, UPLOAD_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RUN_PAGE = "pages/Run_Pipeline.py"
VIEWER_PAGE = "pages/Transcript_Viewer.py"

UPLOAD_TYPES = ["wav", "mp3", "mp4", "m4a"]

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
