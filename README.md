# omni-diarize

Proof-of-concept speaker diarization and transcription for **Malaysian Parliament (Dewan Rakyat)** sessions. The app answers two questions: *who spoke when*, and *what did they say*. Sessions mix Bahasa Melayu and English, much like the Hansard record.

## Architecture

```
                ┌──────────────────────────┐
 session.wav ──►│ 1. Pre-processing        │  16 kHz mono, loudness normalisation
                └────────────┬─────────────┘
                             │
             ┌───────────────┴────────────────┐
             ▼                                ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│ 2. Diarization           │    │ 3. Transcription         │
│ PyAnnote 3.1             │    │ faster-whisper large-v3  │
│ → SPEAKER_00 [t0, t1] …  │    │ → text + word timestamps │
└────────────┬─────────────┘    └────────────┬─────────────┘
             └───────────────┬───────────────┘
                             ▼
                ┌──────────────────────────┐
                │ 4. Alignment             │  words → speaker turns (max overlap)
                └────────────┬─────────────┘
                             ▼
                ┌──────────────────────────┐
                │ 5. MP identification     │  cosine similarity vs. enrolled
                │    voice embeddings      │  voiceprints → "Prime Minister", …
                └────────────┬─────────────┘
                             ▼
                  parlimen_transcript.json ──► Streamlit viewer (app.py)
```

### 1. Neural diarization: PyAnnote 3.1
`pyannote/speaker-diarization-3.1` runs neural segmentation (overlap-aware), extracts an embedding per speaker, and clusters the results agglomeratively. The output is a list of anonymous turns (`SPEAKER_00`, `SPEAKER_01`, …) with start and end times. It runs in pure PyTorch, so no `onnxruntime` is needed. The model is gated: you must accept its licence on Hugging Face and provide an `HF_TOKEN`.

### 2. Transcription: faster-whisper large-v3
`faster-whisper` is a CTranslate2 reimplementation of Whisper. It is about 4× faster than the reference implementation and uses less memory, which makes int8/float16 inference practical. `large-v3` handles the **code-switching between Malay and English** heard in Dewan Rakyat debates. Word-level timestamps and VAD filtering are enabled so the text can be aligned to diarization turns. You can pass an `initial_prompt` with parliamentary vocabulary (e.g. *Tuan Yang di-Pertua*, *Yang Berhormat*, *Rang Undang-Undang*) to bias the decoder toward Hansard terminology.

### 3. MP identification: cosine similarity on voice embeddings
Diarization labels are anonymous. To attach names, each diarized speaker's turns are averaged into one embedding. That embedding is compared against an **enrolment gallery**: reference embeddings built from known clips of each MP.

```
similarity = (a · b) / (‖a‖ ‖b‖)
```

A speaker takes the best-matching MP's name when that match is above a threshold (for example `0.70`). Speakers below the threshold stay **Unknown**. The similarity score is saved with each segment so reviewers can audit the matches.

## Transcript format

`app.py` reads `data/parlimen_transcript.json`. The file can be a bare list of segments or an object with a `segments` key:

```json
{
  "session": "Mesyuarat Ketiga, Penggal Keempat",
  "date": "2025-10-10",
  "segments": [
    { "start": 0.0,  "end": 6.4,  "speaker": "Prime Minister", "text": "Tuan Yang di-Pertua, ...", "confidence": 0.86 },
    { "start": 6.4,  "end": 12.1, "speaker": "MP Sri Aman",    "text": "Terima kasih ...",          "confidence": 0.79 },
    { "start": 12.1, "end": 15.0, "speaker": "SPEAKER_02",     "text": "..." }
  ]
}
```

The loader also accepts `start_time`/`end_time`, `speaker_name`/`label`, and `similarity`/`score` as alternative key names. Top-level scalar fields such as `session` and `date` appear in the sidebar.

## Project structure

```
omni-diarize/
├── app.py             # Streamlit viewer (entry point)
├── requirements.txt
├── README.md
└── data/              # audio + transcripts (audio files are git-ignored)
    ├── parlimen_test.wav
    └── parlimen_transcript.json
```

## Quick start (Windows / PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

## Viewer features

- **Sidebar**: the model stack, chamber metadata, and each detected speaker with an *Identified* or *Unidentified* status badge and their total talk time.
- **Audio player** for the session recording.
- **Dialogue cards** with role icons (🏛️ Prime Minister, 🎤 MP Sri Aman, 👤 Speaker/Unknown), timestamp ranges, and similarity scores.
- **Search and filter**: full-text search with highlighted matches, plus a multi-select speaker filter.
