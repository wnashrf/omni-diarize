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
# Whisper vocabulary prompt: comma-separated terms curated from Dewan Rakyat Hansard records,
# most important first. run_pipeline.py reads data/parliament_vocab.txt instead when it exists
# (tune the vocabulary there without code changes) and trims terms from the END to fit
# Whisper's 223-token prompt budget.
DEFAULT_PARLIAMENT_PROMPT = (
    # Procedural & greetings
    "Dewan Rakyat, Yang di-Pertua, Tuan Yang di-Pertua, Yang Berhormat, Tuan Pengerusi, "
    "Perdana Menteri, Ketua Pembangkang, Rang Undang-Undang, Peraturan Mesyuarat, Titah Diraja, "
    "Pertanyaan-Pertanyaan Bagi Jawab Lisan, Soalan Tambahan, Usul, Belanjawan, Peruntukan, Hansard, "
    "Jawatankuasa Seluruh Majlis, Bacaan Kali Yang Kedua, Bacaan Kali Yang Ketiga, "
    "Bismillahirrahmanirrahim, Assalamualaikum warahmatullahi wabarakatuh, Insya-Allah, "
    # Honorifics & titles
    "Dato' Seri, Datuk Seri, Tan Sri, Dato' Sri, "
    # Hansard acronyms
    "TVET, STEM, SOSMA, PADU, BUDI, ASEAN, NADMA, COVID, MOF, PAC, SST, SOP, KPI, ECRL, MAHB, PAAB, OKU, REE, "
    # Recently misheard place and agency names
    "Kulim-Bandar Baharu, Jerlun, Geting, SkillsLab, MyMahir"
)
CHAMBER = {
    "Chamber": "Dewan Rakyat",
    "Parliament": "Parlimen Malaysia",
    "Language": "Bahasa Melayu / English",
}
