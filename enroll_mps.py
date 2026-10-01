"""enroll_mps.py — Register MP reference voiceprints into an acoustic database.

Usage:
    python enroll_mps.py [--audio data/parlimen_full.wav] [--db mp_database.json]

Extracts a pyannote/embedding voiceprint for each MP from a known clean window of the
full sitting and stores it in mp_database.json, for cosine-similarity speaker matching.
Requires HF_TOKEN (env or .env) and the gated pyannote/embedding licence.
"""

import argparse
import json
import os
from pathlib import Path

import soundfile as sf
import torch
from pyannote.audio import Inference, Model

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
DEFAULT_AUDIO = ROOT / "data" / "parlimen_full.wav"
DEFAULT_DB = ROOT / "mp_database.json"

# Define target MPs with known clean windows from this session
INITIAL_ENROLLMENT = [
    {
        "name": "Tan Sri Johari Abdul",
        "title": "Yang di-Pertua Dewan Rakyat",
        "constituency": "Speaker",
        "start": 354.0,  # 05:54
        "end": 390.0,    # 06:30
    },
    {
        "name": "Dato' Seri Anwar Ibrahim",
        "title": "Perdana Menteri",
        "constituency": "Tambun",
        "start": 480.0,  # 08:00
        "end": 515.0,    # 08:35
    },
    {
        "name": "Datuk Seri Dr. Shahidan bin Kassim",
        "title": "Ahli Parlimen",
        "constituency": "Arau",
        "start": 1706.0, # 28:26
        "end": 1731.0,   # 28:51
    },
    {
        "name": "Datuk Ewon Benedick",
        "title": "Menteri Pembangunan Usahawan dan Koperasi",
        "constituency": "Penampang",
        "start": 2893.0, # 48:13
        "end": 2925.0,   # 48:45
    },
    {
        "name": "Puan Hannah Yeoh Tseow Suan",
        "title": "Menteri Belia dan Sukan",
        "constituency": "Segambut",
        "start": 3506.0, # 58:26
        "end": 3540.0,   # 59:00
    },
    {
        "name": "Datuk Seri Amir Hamzah Azizan",
        "title": "Menteri Kewangan II",
        "constituency": "Senator",
        "start": 4262.0, # 01:11:02
        "end": 4305.0,   # 01:11:45
    },
    {
        "name": "Tuan Syed Saddiq bin Syed Abdul Rahman",
        "title": "Ahli Parlimen",
        "constituency": "Muar",
        "start": 6283.0, # 01:44:43
        "end": 6320.0,   # 01:45:20
    },
    {
        "name": "Datuk Haji Awang bin Hashim",
        "title": "Ahli Parlimen",
        "constituency": "Pendang",
        "start": 6754.0, # 01:52:34
        "end": 6795.0,   # 01:53:15
    },
]


def load_audio_slice(wav_path: Path, start: float, end: float) -> torch.Tensor:
    """Reads [start, end) as (channel, time) float32 tensor."""
    offset = int(start * SAMPLE_RATE)
    frames = int((end - start) * SAMPLE_RATE)
    data, sr = sf.read(str(wav_path), start=offset, frames=frames, dtype="float32", always_2d=True)
    tensor = torch.from_numpy(data.T)
    if sr != SAMPLE_RATE:
        import torchaudio
        tensor = torchaudio.functional.resample(tensor, sr, SAMPLE_RATE)
    return tensor.mean(dim=0, keepdim=True)  # ensure mono (1, frames)


def main():
    p = argparse.ArgumentParser(description="Enrol MP voiceprints from a full sitting recording.")
    p.add_argument("--audio", type=Path, default=DEFAULT_AUDIO, help="16 kHz WAV of the full sitting")
    p.add_argument("--db", type=Path, default=DEFAULT_DB, help="Voiceprint database JSON to create/update")
    args = p.parse_args()
    wav_path, db_file = args.audio, args.db
    if not wav_path.exists():
        raise SystemExit(f"Audio not found: {wav_path} (download the full sitting first)")

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (env var or .env).")

    print("Loading PyAnnote embedding model...")
    try:
        model = Model.from_pretrained("pyannote/embedding", token=token)
    except TypeError:  # pyannote.audio < 4 uses the older kwarg
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=token)
    inference = Inference(model, window="whole")

    db = {}
    if db_file.exists():
        with open(db_file, "r", encoding="utf-8") as f:
            db = json.load(f)

    for item in INITIAL_ENROLLMENT:
        key = f"{item['name']} ({item['constituency']})"
        print(f"Extracting voiceprint for: {key} [{item['start']}s - {item['end']}s]...")

        waveform = load_audio_slice(wav_path, item["start"], item["end"])
        embedding = inference({"waveform": waveform, "sample_rate": SAMPLE_RATE})

        db[key] = {
            "name": item["name"],
            "title": item["title"],
            "constituency": item["constituency"],
            "embedding": embedding.tolist(),  # 512 float values
        }

    with open(db_file, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2, ensure_ascii=False)

    print(f"\nEnrollment complete! {len(db)} MP voiceprints saved in {db_file.resolve()}")


if __name__ == "__main__":
    main()