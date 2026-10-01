"""omni-diarize — end-to-end pipeline for a Dewan Rakyat sitting.

Usage:
    # Fast CPU demo: first 2 minutes of a YouTube sitting
    python run_pipeline.py --url "https://www.youtube.com/watch?v=XXXX" --max-duration 2

    # Local audio or video (.wav / .mp3 / .mp4 / .m4a …) — bypasses YouTube entirely
    python run_pipeline.py --audio data/parlimen_full.wav [--device cpu]

Programmatic:
    from run_pipeline import run
    transcript_path = run(url="https://…", max_duration=2, output_dir="data")

Steps: (yt-dlp if URL | ffmpeg normalise if local) -> 16 kHz mono WAV
-> pyannote 3.1 diarization + turn stitching -> faster-whisper large-v3 per turn
-> <output-dir>/<stem>_transcript.json

The JSON matches the schema read by the Streamlit viewer (`turns[]` of
speaker/start/end/timestamp/text) and sits next to its WAV so the viewer can find
the audio. Requires HF_TOKEN (env or .env) and ffmpeg on PATH.
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
try:
    from core.config import DEFAULT_PARLIAMENT_PROMPT
except ImportError:  # run outside the repo: fall back to a minimal built-in vocabulary
    DEFAULT_PARLIAMENT_PROMPT = (
        "Dewan Rakyat, Yang di-Pertua, Perdana Menteri, Yang Berhormat, Ahli Parlimen, "
        "Rang Undang-Undang, Belanjawan, Peruntukan, Hansard"
    )
VOCAB_FILE = Path(__file__).resolve().parent / "data" / "parliament_vocab.txt"
# faster-whisper keeps only the LAST max_length // 2 - 1 = 223 prompt tokens, silently dropping
# the start of a longer prompt; fit_prompt() trims from the end instead, keeping the top terms.
PROMPT_TOKEN_BUDGET = 223


def load_vocab_prompt() -> str:
    """Comma-separated vocabulary from data/parliament_vocab.txt, else DEFAULT_PARLIAMENT_PROMPT."""
    if VOCAB_FILE.is_file():
        text = " ".join(VOCAB_FILE.read_text(encoding="utf-8").split())
        if text:
            return text
    return DEFAULT_PARLIAMENT_PROMPT


def vocab_terms(prompt: str) -> list[str]:
    return [t.strip() for t in prompt.rstrip(". ").split(",") if t.strip()]


INITIAL_PROMPT = load_vocab_prompt()
OUTPUT_MARKER = "OUTPUT_JSON="  # final stdout line, parsed by the Streamlit runner
# yt-dlp download attempts, in order: (name for the log, extra yt-dlp args)
YTDLP_CLIENTS = [
    ("android", ["--extractor-args", "youtube:player_client=android"]),
    ("default", []),
]


class PipelineError(RuntimeError):
    """A user-facing failure (missing token, missing ffmpeg, bad input …)."""


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fmt_clock(seconds: float) -> str:
    """Seconds -> 'MM:SS.ss' (or 'HH:MM:SS.ss'), the format the viewer parses."""
    m, s = divmod(seconds, 60)
    h, m = divmod(int(m), 60)
    return f"{h:02d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"


def sanitize(title: str, max_len: int = 80) -> str:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    clean = re.sub(r"[^A-Za-z0-9]+", "_", ascii_title).strip("_")
    return clean[:max_len].rstrip("_") or "video"


def duration_suffix(max_minutes: int | None) -> str:
    """Clipped runs get their own files so a 2-min demo never shadows a full run."""
    return f"_{max_minutes}min" if max_minutes else ""


# ------------------------------------------------------------------ download
def ytdlp_cmd() -> list[str]:
    if exe := shutil.which("yt-dlp"):
        return [exe]
    return [sys.executable, "-m", "yt_dlp"]


def require_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise PipelineError("ffmpeg was not found on PATH. Install it (see README) and reopen the terminal.")
    return exe


def fetch_video_info(url: str) -> dict:
    """Return YouTube metadata (id, title, channel, duration, thumbnail) without downloading."""
    proc = subprocess.run(
        [*ytdlp_cmd(), "--no-playlist", "--skip-download", "--dump-single-json", "--no-warnings", url],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise PipelineError(f"yt-dlp could not read {url}: {detail[-1] if detail else 'unknown error'}")
    info = json.loads(proc.stdout)
    return {
        "id": info.get("id", ""),
        "title": info.get("title") or info.get("id", "video"),
        "channel": info.get("channel") or info.get("uploader") or "",
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
        "upload_date": info.get("upload_date"),
        "webpage_url": info.get("webpage_url") or url,
    }


def download_audio(url: str, stem: str, output_dir: Path, max_minutes: int | None) -> Path:
    wav = output_dir / f"{stem}.wav"
    if wav.exists():
        log(f"Reusing existing audio {wav} (delete it to re-download)")
        return wav

    require_ffmpeg()
    cmd = [
        # "ba/b": take the best audio-only stream, else the best combined one. Some clients only
        # expose combined formats, and a strict audio-only selector would fail there.
        *ytdlp_cmd(), "--no-playlist", "--newline", "-f", "ba/b",
        "-x", "--audio-format", "wav",
        "--postprocessor-args", f"ExtractAudio+ffmpeg_o:-ar {SAMPLE_RATE} -ac 1",
        "-o", str(output_dir / f"{stem}.%(ext)s"),
    ]
    if max_minutes:
        end_h, end_rem = divmod(max_minutes * 60, 3600)
        end_m, end_s = divmod(end_rem, 60)
        cmd += ["--download-sections", f"*00:00:00-{end_h:02d}:{end_m:02d}:{end_s:02d}"]
    log(f"Downloading audio{f' (first {max_minutes} min)' if max_minutes else ''} -> {wav}")

    # YouTube intermittently answers 403 Forbidden for one player client's stream URLs
    # (signature / PO-token blocks), so try the android client first, then yt-dlp's defaults.
    for i, (client, extra) in enumerate(YTDLP_CLIENTS):
        if i:
            log(f"Retrying download with the {client} player client")
            for partial in output_dir.glob(f"{stem}.*"):  # never resume from a failed attempt's file
                partial.unlink(missing_ok=True)
        if subprocess.run([*cmd, *extra, url]).returncode == 0 and wav.exists():
            return wav
        log(f"Download with the {client} player client failed")
    raise PipelineError(
        "YouTube refused the audio download (often a temporary 403 Forbidden block). "
        "Try again in a few minutes, update yt-dlp (`pip install -U yt-dlp`), or download the "
        "video yourself and use the Upload file tab instead."
    )


def _is_16k_mono_wav(path: Path) -> bool:
    if path.suffix.lower() != ".wav":
        return False
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return info.samplerate == SAMPLE_RATE and info.channels == 1
    except Exception:
        return False


def prepare_local_audio(src: Path, output_dir: Path, max_minutes: int | None) -> Path:
    """Return a 16 kHz mono WAV for `src` (any audio/video ffmpeg can read), clipped if asked.

    Turn slicing seeks by sample offset at SAMPLE_RATE, so every input is normalised first.
    """
    if not max_minutes and _is_16k_mono_wav(src):
        return src

    dst = output_dir / f"{src.stem}{duration_suffix(max_minutes) or '_16k'}.wav"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        log(f"Reusing prepared audio {dst}")
        return dst

    cmd = [require_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]
    if max_minutes:
        cmd += ["-t", str(max_minutes * 60)]
    cmd += ["-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), str(dst)]
    log(f"Converting audio{f' (first {max_minutes} min)' if max_minutes else ''} to 16 kHz mono -> {dst}")
    subprocess.run(cmd, check=True)
    return dst


# --------------------------------------------------------------- diarization
def diarize(wav: Path, device) -> list[dict]:
    from pyannote.audio import Pipeline
    import torch

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise PipelineError("HF_TOKEN is not set (env var or .env). It is required for pyannote models.")

    log(f"Loading {DIARIZATION_MODEL} on {device}")
    try:
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, token=token)
    except TypeError:  # pyannote.audio < 4 uses the older kwarg
        pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=token)
    if pipeline is None:
        raise PipelineError(f"Could not load {DIARIZATION_MODEL}; accept its licence on Hugging Face first.")
    pipeline.to(device)

    # Pre-load the waveform so pyannote does not need its own (torchcodec) decoder
    try:
        import torchaudio
        waveform, sr = torchaudio.load(str(wav))
    except Exception:
        import soundfile as sf
        data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        waveform = torch.from_numpy(data.T)
    audio_input = {"waveform": waveform, "sample_rate": sr}

    seconds = waveform.shape[-1] / sr
    log(f"Audio duration: {fmt_clock(seconds)} ({seconds:.1f}s)")
    log("Diarizing (this is the slow step for long sittings)…")
    t0 = time.time()
    try:
        from pyannote.audio.pipelines.utils.hook import ProgressHook
        with ProgressHook() as hook:
            output = pipeline(audio_input, hook=hook)
    except ImportError:
        output = pipeline(audio_input)
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
    except Exception:  # torchaudio missing / no backend -> seek with soundfile
        import soundfile as sf
        audio, _ = sf.read(str(wav), start=offset, frames=frames, dtype="float32", always_2d=True)
        return audio.mean(axis=1)


def save_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def fit_prompt(prompt: str, tokenizer) -> tuple[str, list[str]]:
    """Keep vocabulary terms in order until the next one would exceed PROMPT_TOKEN_BUDGET.

    Returns the fitted prompt and the dropped terms. Tokens are counted exactly as
    faster-whisper encodes initial_prompt (with a leading space).
    """
    terms, kept = vocab_terms(prompt), []
    for term in terms:
        candidate = ", ".join([*kept, term]) + "."
        if len(tokenizer.encode(" " + candidate, add_special_tokens=False).ids) > PROMPT_TOKEN_BUDGET:
            break
        kept.append(term)
    return ", ".join(kept) + ".", terms[len(kept):]


def transcribe_turns(wav: Path, turns: list[dict], cuda: bool, out_path: Path, payload: dict) -> None:
    from faster_whisper import WhisperModel

    device, compute_type = ("cuda", "float16") if cuda else ("cpu", "int8")
    log(f"Loading faster-whisper {ASR_MODEL} ({device}/{compute_type})")
    model = WhisperModel(ASR_MODEL, device=device, compute_type=compute_type)
    prompt, dropped = fit_prompt(INITIAL_PROMPT, model.hf_tokenizer)
    if dropped:
        log(f"WARNING: vocabulary prompt exceeds Whisper's {PROMPT_TOKEN_BUDGET}-token budget; "
            f"dropped {len(dropped)} terms from the end: {', '.join(dropped)}")

    results = payload["turns"]
    total, t0 = len(turns), time.time()
    for i, turn in enumerate(turns, 1):
        start, end = turn["start"], turn["end"]
        if end - start < MIN_TURN_S:
            continue
        segments, _ = model.transcribe(
            load_slice(wav, start, end),
            language=ASR_LANGUAGE,
            initial_prompt=prompt,  # INITIAL_PROMPT, fitted to the token budget
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
        # Full text on one line: the Streamlit runner parses these into its live dialogue feed
        log(f"[{i}/{total}] {turn['speaker']} {fmt_clock(start)}-{fmt_clock(end)} "
            f"(ETA {eta / 60:.1f} min) {' '.join(text.split()) or '<no speech>'}")
        if i % SAVE_EVERY == 0:
            payload["speakers_detected"] = sorted({t["speaker"] for t in results})
            save_json(out_path, payload)


# ----------------------------------------------------------------------- main
def resolve_cuda(device: str) -> bool:
    try:
        import torch
    except ImportError as e:
        raise PipelineError("ML dependencies are missing — run `pip install -r requirements.txt`.") from e
    available = torch.cuda.is_available()
    if device == "cuda" and not available:
        raise PipelineError("--device cuda was requested but torch cannot see a CUDA GPU.")
    return device != "cpu" and available


def run(
    url: str | None = None,
    audio: str | Path | None = None,
    max_duration: int | None = None,
    output_dir: str | Path = "data",
    device: str = "auto",
) -> Path:
    """Run the full pipeline and return the path of the transcript JSON."""
    if bool(url) == bool(audio):
        raise PipelineError("Provide exactly one of `url` or `audio`.")
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).with_name(".env"))
    except ImportError:
        pass

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cuda = resolve_cuda(device)  # fail fast before any download
    log(f"Execution device: {'cuda (float16)' if cuda else 'cpu (int8)'}")
    log(f"Loaded parliamentary vocabulary prompt ({len(vocab_terms(INITIAL_PROMPT))} terms)")

    if audio:
        src = Path(audio)
        if not src.exists():
            raise PipelineError(f"Audio file not found: {src}")
        log(f"Using local file: {src}")
        wav = prepare_local_audio(src, output_dir, max_duration)
        stem, title, video_id, source_url = wav.stem, src.stem.replace("_", " ").title(), src.stem, "local"
    else:
        info = fetch_video_info(url)
        video_id, title = info["id"], info["title"]
        stem = f"{sanitize(title)}_{video_id}{duration_suffix(max_duration)}"
        source_url = url
        log(f"Video: {title} [{video_id}]")
        wav = download_audio(url, stem, output_dir, max_duration)

    out_path = output_dir / f"{stem}_transcript.json"

    import torch
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
        "max_duration_min": max_duration,
        "device": "cuda" if cuda else "cpu",
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
    return out_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download, diarize and transcribe a Dewan Rakyat sitting.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube URL of the sitting")
    src.add_argument("--audio", help="Path to a local audio/video file (.wav, .mp3, .mp4, .m4a, …)")
    p.add_argument("--max-duration", type=int, default=None, metavar="MINUTES",
                   help="Only process the first N minutes (recommended: 2 for CPU demos)")
    p.add_argument("--output-dir", default="data", help="Folder for the WAV and transcript JSON (default: data)")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                   help="auto = CUDA if available; cpu forces int8 CPU inference")
    args = p.parse_args(argv)
    if args.max_duration is not None and args.max_duration <= 0:
        p.error("--max-duration must be a positive number of minutes")
    return args


def main(argv: list[str] | None = None) -> int:
    # Windows consoles/pipes default to cp1252; Malay text and "…" must not crash logging
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args(argv)
    try:
        out_path = run(url=args.url, audio=args.audio, max_duration=args.max_duration,
                       output_dir=args.output_dir, device=args.device)
    except PipelineError as e:
        log(f"ERROR: {e}")
        return 1
    except subprocess.CalledProcessError as e:
        log(f"ERROR: external command failed (exit {e.returncode}): {' '.join(map(str, e.cmd[:3]))} …")
        return 1
    print(f"{OUTPUT_MARKER}{out_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
