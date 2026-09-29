"""omni-diarize — end-to-end pipeline for a Dewan Rakyat sitting.

Usage:
    # 1. Direct from local audio (Bypasses YouTube datacenter blocks entirely):
    python run_pipeline.py --audio data/parlimen_full.wav

    # 2. Or from YouTube URL directly:
    python run_pipeline.py --url "https://www.youtube.com/watch?v=XXXX" [--max-duration 120]

Steps: (yt-dlp if URL) -> pyannote 3.1 diarization + turn stitching
-> faster-whisper large-v3 per turn -> <output-dir>/<stem>_transcript.json

The JSON matches the schema read by app.py (`turns[]` of speaker/start/end/timestamp/text)
and sits next to its WAV so the viewer can find the audio. Requires HF_TOKEN (env or .env).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
ASR_MODEL = "large-v3"
ASR_LANGUAGE = "ms"
SAMPLE_RATE = 16000
MERGE_GAP_S = 1.5       # stitch same-speaker segments separated by <= this gap
MIN_TURN_S = 0.3        # turns shorter than this are too short to transcribe
SAVE_EVERY = 10         # checkpoint the JSON every N transcribed turns
INITIAL_PROMPT = (
    "Dewan Rakyat, Yang di-Pertua, Perdana Menteri, Yang Berhormat, Ahli Parlimen, "
    "Rang Undang-Undang, Belanjawan, Peruntukan, Hansard."
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_clock(seconds: float) -> str:
    """Seconds -> 'MM:SS.ss' (or 'HH:MM:SS.ss'), the format app.py parses."""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def sanitize(title: str, max_len: int = 80) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    clean = re.sub(r"[^A-Za-z0-9]+", "_", ascii_title).strip("_")
    return clean[:max_len].rstrip("_") or "video"


# ------------------------------------------------------------------ download
def ytdlp_cmd() -> list[str]:
    if exe := shutil.which("yt-dlp"):
        return [exe]
    return [sys.executable, "-m", "yt_dlp"]


def fetch_metadata(url: str) -> tuple[str, str]:
    """Return (video_id, title) without downloading."""
    out = subprocess.run(
        [*ytdlp_cmd(), "--no-playlist", "--skip-download", "--encoding", "utf-8",
         "--print", "id", "--print", "title", url],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout.strip().splitlines()
    if len(out) < 2:
        raise RuntimeError(f"Unexpected yt-dlp metadata output: {out!r}")
    return out[0].strip(), out[1].strip()


def download_audio(url: str, stem: str, output_dir: Path, max_minutes: int | None) -> Path:
    wav = output_dir / f"{stem}.wav"
    if wav.exists():
        log(f"Reusing existing audio {wav} (delete it to re-download)")
        return wav

    cmd = [
        *ytdlp_cmd(), "--no-playlist", "-f", "bestaudio/best",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ExtractAudio+ffmpeg_o:-ar {SAMPLE_RATE} -ac 1",
        "-o", str(output_dir / f"{stem}.%(ext)s"),
    ]
    if max_minutes:
        end_h, end_rem = divmod(max_minutes * 60, 3600)
        end_m, end_s = divmod(end_rem, 60)
        section_str = f"*00:00:00-{int(end_h):02d}:{int(end_m):02d}:{int(end_s):02d}"
        cmd += ["--download-sections", section_str]
    log(f"Downloading audio{f' (first {max_minutes} min)' if max_minutes else ''} -> {wav}")
    subprocess.run([*cmd, url], check=True)
    if not wav.exists():
        raise FileNotFoundError(f"yt-dlp finished but {wav} was not produced (is ffmpeg on PATH?)")
    return wav


# --------------------------------------------------------------- diarization
def diarize(wav: Path, device) -> list[dict]:
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set (env var or .env). It is required for pyannote models.")

    log(f"Loading {DIARIZATION_MODEL} on {device}")
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    except TypeError:  # pyannote.audio < 4 uses the older kwarg
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    if pipeline is None:
        sys.exit(f"Could not load {DIARIZATION_MODEL}; accept its licence on Hugging Face first.")
    pipeline.to(device)

    log("Diarizing (this is the slow step for long sittings)…")
    t0 = time.time()
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            output = pipeline(str(wav), hook=hook)
    except ImportError:
        output = pipeline(str(wav))
    annotation = getattr(output, "speaker_diarization", output)  # pyannote 4 wraps the Annotation

    segments = [
        {"speaker": label, "start": float(seg.start), "end": float(seg.end)}
        for seg, _, label in annotation.itertracks(yield_label=True)
    ]
    log(f"Diarization done in {time.time() - t0:.0f}s: {len(segments)} raw segments")
    return segments


def stitch_turns(segments: list[dict], max_gap: float = MERGE_GAP_S) -> list[dict]:
    """Merge consecutive segments of the same speaker separated by <= max_gap seconds."""
    turns: list[dict] = []
    for seg in sorted(segments, key=lambda s: s["start"]):
        last = turns[-1] if turns else None
        if last and last["speaker"] == seg["speaker"] and seg["start"] - last["end"] <= max_gap:
            last["end"] = max(last["end"], seg["end"])
        else:
            turns.append(dict(seg))
    return turns


# ------------------------------------------------------------- transcription
def load_slice(wav: Path, start: float, end: float):
    """Read only [start, end) of the WAV as mono float32 numpy, so memory stays flat."""
    offset, frames = int(start * SAMPLE_RATE), max(1, int((end - start) * SAMPLE_RATE))
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(wav), frame_offset=offset, num_frames=frames)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        return waveform.mean(dim=0).numpy().astype("float32")
    except (ImportError, RuntimeError):  # torchaudio backend unavailable -> seek with soundfile
        import soundfile as sf
        audio, _ = sf.read(str(wav), start=offset, frames=frames, dtype="float32", always_2d=True)
        return audio.mean(axis=1)


def save_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def transcribe_turns(wav: Path, turns: list[dict], cuda: bool, out_path: Path, payload: dict) -> None:
    from faster_whisper import WhisperModel

    device, compute_type = ("cuda", "float16") if cuda else ("cpu", "int8")
    log(f"Loading faster-whisper {ASR_MODEL} ({device}/{compute_type})")
    model = WhisperModel(ASR_MODEL, device=device, compute_type=compute_type)

    results = payload["turns"]
    total, t0 = len(turns), time.time()
    for i, turn in enumerate(turns, 1):
        start, end = turn["start"], turn["end"]
        if end - start < MIN_TURN_S:
            continue
        segments, _ = model.transcribe(
            load_slice(wav, start, end),
            language=ASR_LANGUAGE,
            initial_prompt=INITIAL_PROMPT,
            condition_on_previous_text=False,
            vad_filter=True,
            beam_size=5,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            results.append({
                "speaker": turn["speaker"],
                "start": round(start, 2),
                "end": round(end, 2),
                "timestamp": f"{fmt_clock(start)} - {fmt_clock(end)}",
                "text": text,
            })

        elapsed = time.time() - t0
        eta = elapsed / i * (total - i)
        preview = (text[:70] + "…") if len(text) > 70 else text
        log(f"[{i}/{total}] {turn['speaker']} {fmt_clock(start)}-{fmt_clock(end)} "
            f"(ETA {eta / 60:.1f} min) {preview or '<no speech>'}")
        if i % SAVE_EVERY == 0:
            payload["speakers_detected"] = sorted({t["speaker"] for t in results})
            save_json(out_path, payload)


# ----------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download, diarize and transcribe a Dewan Rakyat sitting.")
    p.add_argument("--url", default=None, help="YouTube URL of the sitting")
    p.add_argument("--audio", default=None, help="Path to an existing local WAV audio file")
    p.add_argument("--max-duration", type=int, default=None, metavar="MINUTES",
                   help="Only process the first N minutes (when using --url)")
    p.add_argument("--output-dir", default="data", help="Folder for the WAV and transcript JSON")
    args = p.parse_args()
    if not args.url and not args.audio:
        p.error("You must provide either --url or --audio")
    return args


def main() -> None:
    args = parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.audio:
        wav = Path(args.audio)
        if not wav.exists():
            sys.exit(f"Audio file not found: {wav}")
        stem = wav.stem
        title = stem.replace("_", " ").title()
        video_id = stem
        source_url = "local"
        log(f"Using local audio file: {wav}")
    else:
        video_id, title = fetch_metadata(args.url)
        stem = f"{sanitize(title)}_{video_id}"
        source_url = args.url
        log(f"Video: {title} [{video_id}]")
        wav = download_audio(args.url, stem, output_dir, args.max_duration)

    out_path = output_dir / f"{stem}_transcript.json"

    import torch
    cuda = torch.cuda.is_available()

    turns = stitch_turns(diarize(wav, torch.device("cuda" if cuda else "cpu")))
    speakers = sorted({t["speaker"] for t in turns})
    log(f"Stitched into {len(turns)} turns across {len(speakers)} speakers (gap <= {MERGE_GAP_S}s)")
    if cuda:
        torch.cuda.empty_cache()

    payload = {
        "title": title,
        "video_id": video_id,
        "source_url": source_url,
        "audio_file": wav.name,
        "status": "in_progress",
        "speakers_detected": [],
        "turns": [],
    }
    save_json(out_path, payload)
    log(f"Writing transcript to {out_path}")

    transcribe_turns(wav, turns, cuda, out_path, payload)

    payload["speakers_detected"] = sorted({t["speaker"] for t in payload["turns"]})
    payload["status"] = "complete"
    save_json(out_path, payload)
    log(f"Done: {len(payload['turns'])} turns -> {out_path}")


if __name__ == "__main__":
    main()