# 🎙️ VoxScribe — audio & video → transcripts (MD · PDF · TXT · SRT)

A self-hosted transcription app inspired by tools like Happy Scribe. Speech-to-text runs **locally with Whisper**
(free, private, works offline after the first model download). Optional AI tools use **OpenRouter**.

## Features
- **3 ways in:** upload audio/video · record from your mic · paste a YouTube/video link
- **English, Hindi + 90 more languages:** auto-detect or choose; "Translate to English" option
- **Editor:** click any sentence to jump the player there, live highlight while playing, edit text inline (auto-saves),
  rename speakers, find & replace, playback speed
- **AI tools (OpenRouter):** summary · fix punctuation/spelling · detect speakers
- **Export:** Markdown, PDF (Hindi supported), TXT, SRT subtitles; with/without timestamps and summary
- File library stored locally in `data/`

## Setup (macOS)
```bash
cd Voice_Text
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then paste your OpenRouter key into .env (optional)
python app.py               # open http://127.0.0.1:8000
```
The first transcription downloads the chosen Whisper model (small ≈ 0.5 GB, medium ≈ 1.5 GB), then it's cached.

## Hindi tips
- Pick **Hindi** as the language for Hinglish (mixed) speech. Auto-detect picks one language per file.
- Use **Accurate (medium)** or **turbo** for Hindi. **Balanced (small)** makes more mistakes.
- If Whisper detects Urdu, the app re-runs the file as Hindi so you get Devanagari text.
- If Hindi is missing from PDFs, put `NotoSansDevanagari-Regular.ttf`
  ([download](https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari)) into a `fonts/` folder.

## Command line
```bash
python cli.py lecture.mp4 --lang hi --model medium --format md pdf txt
```

## Project layout
```
app.py            FastAPI server: jobs, library, editing, export, AI endpoints
static/index.html the whole web UI (no build step)
transcriber.py    Whisper wrapper
exporters.py      md / pdf / txt / srt writers
ai.py             OpenRouter: summary, cleanup, speaker labels
cli.py            command-line version
data/             your transcripts + media (git-ignored)
```

## Notes
- AI speaker detection guesses from the text (who asks or answers), not from voices. For voice-based speaker
  separation you'd add `pyannote.audio` later.
- Only one file transcribes at a time; others wait in the queue.
