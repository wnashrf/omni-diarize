# omni-diarize

Speaker diarization and Malay–English transcription for **Malaysian Parliament (Dewan Rakyat)** sittings.

Given a YouTube livestream or a local recording, omni-diarize answers two questions: **who spoke when** and **what they said**. It produces a Hansard-style transcript you can search, replay turn by turn, and export to `.txt`, `.docx` or `.json`.

| | |
|---|---|
| **Diarization** | [`pyannote/speaker-diarization-3.1`](https://huggingface.co/pyannote/speaker-diarization-3.1), overlap-aware neural segmentation and clustering |
| **ASR** | [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) `large-v3`, Malay with English code-switching, primed with parliamentary vocabulary |
| **UI** | Streamlit multi-page app: run the pipeline, then explore the transcript |
| **Runs on** | NVIDIA GPU (float16) or CPU (int8) on Windows, macOS and Linux |

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Setup (step by step)](#2-setup-step-by-step)
3. [How to run](#3-how-to-run)
4. [Architecture](#4-architecture)
5. [Performance](#5-performance)
6. [Project structure](#6-project-structure)
7. [Transcript format](#7-transcript-format)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | Use 3.11 or 3.12. They have the widest coverage of prebuilt `torch` and `ctranslate2` wheels. |
| **FFmpeg** | Used to download, convert and trim audio. It must be on your `PATH`. |
| **Git** | Used to clone the repository. |
| **Hugging Face account** | Free. Needed for the gated pyannote models (see [step 4](#step-4--configure-your-hugging-face-token)). |
| *(Optional)* **NVIDIA GPU + CUDA driver** | Strongly recommended for full-length sittings. |

### Install FFmpeg

**Windows (PowerShell):**

```powershell
winget install Gyan.FFmpeg
```

**macOS (Homebrew):**

```bash
brew install ffmpeg
```

**Ubuntu / Debian:**

```bash
sudo apt update && sudo apt install ffmpeg
```

Close and reopen your terminal, then check the install:

```bash
ffmpeg -version
```

---

## 2. Setup (step by step)

### Step 1 — Clone the repository

```bash
git clone https://github.com/wnashrf/omni-diarize.git
cd omni-diarize
```

### Step 2 — Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> If PowerShell blocks the activation script, run this once and try again:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

Your prompt should now start with `(venv)`.

### Step 3 — Install dependencies

**CPU only (any OS, including Apple Silicon):**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**NVIDIA GPU (Windows / Linux):** install the CUDA build of PyTorch **first**, then the rest:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Check that the GPU is visible (the command should print `True`):

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

> Pick the `cu1xx` index that matches your driver. See [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

### Step 4 — Configure your Hugging Face token

The pyannote diarization models are **gated**. Without these steps the pipeline cannot download them.

1. Sign in at [huggingface.co](https://huggingface.co/join), or create a free account.
2. Open each model page below while signed in. Fill in the short form and click **"Agree and access repository"**:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
   - *(only needed for `enroll_mps.py`)* <https://huggingface.co/pyannote/embedding>
3. Create an access token at <https://huggingface.co/settings/tokens>. **Read** permission is enough.
4. Copy the example env file and paste in your token:

   **Windows (PowerShell):**
   ```powershell
   Copy-Item .env.example .env
   notepad .env
   ```

   **macOS / Linux:**
   ```bash
   cp .env.example .env
   nano .env
   ```

   The file should contain one line:

   ```env
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

`.env` is git-ignored, so your token is never committed.

> **First run downloads the models** (~3 GB for Whisper large-v3, plus a few hundred MB for pyannote) into your Hugging Face cache. Later runs reuse that cache.

---

## 3. How to run

### Web UI

```bash
streamlit run app.py
```

Then open <http://localhost:8501>. The app has two pages, listed in the sidebar.

#### 🚀 Run Pipeline

1. **Environment check.** Status pills show whether ffmpeg, `HF_TOKEN`, the ML packages and a CUDA GPU are available. If anything is missing, a checklist tells you how to fix it, and **Start Processing** stays disabled until it is fixed.
2. **Source.** Choose either:
   - **YouTube URL.** The video's thumbnail, title, channel, duration and upload date load before you run anything.
   - **Upload file.** Accepts `.wav`, `.mp3`, `.mp4` or `.m4a`, up to 2 GB. The file is saved to `data/uploads/`.
3. **Execution settings.**
   - **Duration limit (minutes).** Defaults to **2**. Tick **Full audio** to process the whole recording. Keep the limit for local CPU demos to avoid long processing times.
   - **Force CPU execution (int8).** Leave it unticked to auto-detect CUDA (float16 on GPU).
4. **▶ Start Processing.** Runs `run_pipeline.py` as a background process. The page shows:
   - a progress bar with the current stage (download → diarization → `Transcribing turn i/N`)
   - the live terminal output
   - the equivalent CLI command, so you can reproduce the run
   - a **Cancel** button

   The job keeps running if you switch pages or refresh the browser. When it finishes, the app **opens the new transcript in the viewer automatically**.

#### 🏛️ Transcript Viewer

- **Session selector.** Lists every transcript JSON in `data/`, newest first, so fresh pipeline output is selected by default.
- **Click-to-play.** Each turn has a **▶** button that jumps the sidebar audio player to that moment and highlights the turn.
- **Metrics and analytics.** Shows turns, speakers, spoken time and word count, plus a speaker-distribution chart and table (talk time, share, turns, words).
- **Search and filter.** Full-text search highlights matches. You can also filter by speaker. Results are paginated in pages of 25 turns.
- **Export.** Download the Hansard as `.txt` or `.docx`, or the raw `.json`.
- **Offline demo.** Two small mock transcripts ship in `data/`, so the viewer works right after cloning, before any pipeline run.

### CLI mode

**Fast 2-minute CPU demo:**

```bash
python run_pipeline.py --url "https://www.youtube.com/watch?v=kY8wAT2QO5Y" --max-duration 2
```

**Full run on GPU, from a local file:**

```bash
python run_pipeline.py --audio data/parlimen_full.wav
```

All flags:

| Flag | Description |
|---|---|
| `--url URL` | YouTube URL to download. Cannot be combined with `--audio`. |
| `--audio PATH` | Local audio or video file (`.wav`, `.mp3`, `.mp4`, `.m4a`, or anything else ffmpeg reads). |
| `--max-duration N` | Process only the first *N* minutes. |
| `--output-dir DIR` | Folder for the WAV and the transcript JSON. Default: `data`. |
| `--device {auto,cpu,cuda}` | `auto` uses CUDA when available. `cpu` forces int8. `cuda` fails if no GPU is found. |

The output is `data/<stem>_transcript.json`, written next to its 16 kHz WAV. Clipped runs get a `_<N>min` suffix, so a 2-minute demo never overwrites a full run. The JSON is checkpointed every 10 turns, so you can open a long run in the viewer while it is still in progress.

### Programmatic use

```python
from run_pipeline import run

transcript = run(url="https://www.youtube.com/watch?v=kY8wAT2QO5Y", max_duration=2, device="cpu")
print(transcript)  # Path to the transcript JSON
```

### MP voiceprint enrolment (experimental)

```bash
python enroll_mps.py --audio data/parlimen_full.wav
```

This extracts `pyannote/embedding` voiceprints for a list of MPs, using known clean time windows from a full sitting. It writes them to `mp_database.json`, which is git-ignored. The database is the basis for cosine-similarity speaker naming (see [Architecture](#4-architecture)).

---

## 4. Architecture

```
 YouTube URL ──► yt-dlp ──┐
                          ├──► 16 kHz mono WAV (ffmpeg; trimmed to --max-duration)
 Local file ──► ffmpeg ───┘                │
                                           ▼
                        ┌─────────────────────────────────────┐
                        │ 1. Diarization: pyannote 3.1        │  SPEAKER_00 [t0,t1], SPEAKER_01 …
                        └──────────────────┬──────────────────┘
                                           ▼
                        ┌─────────────────────────────────────┐
                        │ 2. Turn stitching                   │  merge same-speaker gaps ≤ 1.5 s
                        └──────────────────┬──────────────────┘
                                           ▼
                        ┌─────────────────────────────────────┐
                        │ 3. ASR per turn: faster-whisper     │  large-v3, lang=ms, VAD,
                        │    (sample-accurate WAV slices)     │  Hansard vocabulary prompt
                        └──────────────────┬──────────────────┘
                                           ▼
                           data/<stem>_transcript.json  ──►  Streamlit viewer
                                           ▲
             enroll_mps.py ─► mp_database.json (voiceprints, for speaker naming)
```

**Code layout**

- `run_pipeline.py` contains the whole pipeline. You can run it as a CLI or import it as a library through `run()`. Heavy ML imports are lazy, so the UI can import its helpers cheaply.
- `app.py` is the Streamlit router built on `st.navigation`. Each page lives in `pages/`.
- `core/jobs.py` holds the `JobManager`. It starts `run_pipeline.py` as a separate process that logs to `data/logs/`, and the UI tails that log from a polling fragment. This keeps the UI responsive, lets a run survive reruns, and runs one job at a time.
- `core/transcript.py` handles transcript loading (it tolerates several schema variants), speaker roles, card rendering and the exports.

**Design notes**

- **Diarize first, then transcribe each turn.** Every ASR call sees exactly one speaker. Text is attributed by construction, so no word-to-speaker alignment heuristics are needed.
- **Normalised audio.** Every input is converted to 16 kHz mono before processing. Turn slices are read by sample offset, which keeps memory use flat even for a 3-hour sitting.
- **Speaker naming.** Diarization labels are anonymous (`SPEAKER_00`, …). `enroll_mps.py` builds an enrolment gallery of MP voiceprints. A speaker takes an MP's name when the cosine similarity `(a·b)/(‖a‖‖b‖)` clears a threshold (e.g. `0.70`). Wiring this matching into `run_pipeline.py` is the next milestone. The viewer already shows role icons and *Identified* / *Unidentified* badges based on the speaker label.

---

## 5. Performance

| Scenario | Hardware | Wall-clock time |
|---|---|---|
| **Full sitting, ~3 h audio** | NVIDIA GPU (CUDA, float16) | **~6.5 min** (measured reference) |
| 2-min clip (`--max-duration 2`) | NVIDIA GPU | under 1 min after models are cached |
| 2-min clip (`--max-duration 2`) | Modern laptop CPU (int8) | roughly 3–8 min after models are cached |
| Full sitting, ~3 h audio | CPU only | many hours; not recommended |

CPU figures are rough estimates. They depend heavily on core count and on how much speech the clip contains. The first run on any machine also includes the one-time model download.

**Tips:**

- On CPU, keep the default **2-minute** limit for live demos.
- Run full sittings on a CUDA machine.

---

## 6. Project structure

```
omni-diarize/
├── app.py                          # Streamlit entry point (router + shared sidebar)
├── pages/
│   ├── Run_Pipeline.py         # ingestion, settings, live run logs
│   └── Transcript_Viewer.py    # Hansard viewer, playback, analytics, export
├── core/
│   ├── config.py                   # paths, page routes, static metadata
│   ├── jobs.py                     # background pipeline runner (subprocess + log tailing)
│   └── transcript.py               # loading, roles, rendering, .txt/.docx export
├── run_pipeline.py                 # CLI + library: download → diarize → transcribe
├── enroll_mps.py                   # MP voiceprint enrolment (experimental)
├── data/
│   ├── parlimen_transcript.json            # mock transcript (tracked, for offline UI)
│   ├── session_mock_pmqt_transcript.json   # mock transcript (tracked)
│   ├── uploads/   logs/                    # created at runtime, git-ignored
│   └── *.wav, *_transcript.json            # pipeline output, git-ignored
├── .streamlit/config.toml          # 2 GB upload limit
├── .env.example                    # copy to .env and add HF_TOKEN
├── requirements.txt
└── README.md
```

---

## 7. Transcript format

`run_pipeline.py` writes the following. The viewer reads any JSON in `data/` with this shape:

```json
{
  "title": "LANGSUNG: Persidangan Dewan Rakyat | 30 Jun 2026",
  "video_id": "kY8wAT2QO5Y",
  "source_url": "https://www.youtube.com/watch?v=kY8wAT2QO5Y",
  "audio_file": "LANGSUNG_Persidangan_Dewan_Rakyat_kY8wAT2QO5Y_2min.wav",
  "status": "complete",
  "speakers_detected": ["SPEAKER_00", "SPEAKER_01"],
  "turns": [
    { "speaker": "SPEAKER_00", "start": 1.87, "end": 12.79,
      "timestamp": "00:01.87 - 00:12.79", "text": "Bismillahirrahmanirrahim. …" }
  ]
}
```

The loader is lenient about key names:

- The list of turns can sit under `turns`, `segments`, `transcript`, `utterances` or `results`, or the file can be a bare list.
- Speaker can be `speaker`, `speaker_name` or `label`.
- Start and end times can be `start`/`end`, `start_time`/`end_time`, or a parsed `timestamp` string.
- An optional score can be `confidence`, `similarity` or `score`.
- Top-level scalar fields appear in the sidebar's *Chamber* panel.

Audio is found through `audio_file`, or failing that as `<stem without _transcript>.wav` in the same folder.

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `ffmpeg was not found on PATH` | Install FFmpeg (see [Prerequisites](#1-prerequisites)) and **open a new terminal**. |
| `HF_TOKEN is not set` | Create `.env` from `.env.example` (see [step 4](#step-4--configure-your-hugging-face-token)). |
| `Could not load pyannote/...` / 401 / 403 | Accept the licence on **both** pyannote model pages while signed in to the account that owns the token. |
| `ML dependencies are missing` | Activate the venv, then run `pip install -r requirements.txt`. |
| `torch.cuda.is_available()` is `False` on a GPU machine | Reinstall torch from the CUDA index (see [step 3](#step-3--install-dependencies)). |
| YouTube download fails or `Sign in to confirm you're not a bot` | Run `pip install -U yt-dlp`. On cloud or datacenter IPs, download the audio elsewhere and use `--audio` or the upload option instead. |
| Upload rejected as too large | Raise `server.maxUploadSize` in `.streamlit/config.toml`. |
| Viewer shows *Audio not found* | Put the WAV in `data/` next to the JSON, using the name given in its `audio_file` field. |
