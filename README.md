# 🎙️ VoxScribe — audio & video → transcripts (MD · PDF · TXT · SRT)

A transcription app inspired by Happy Scribe. Upload, record or link audio/video, get an editable
transcript (English, Hindi and 90+ languages), then copy it or export it.

**Deploy anywhere:** Vercel, any Docker host, or run it on your own computer.

## How it works
```
Browser ── decodes audio, splits into ~2-min WAV chunks ──► /api/transcribe ──► Whisper
   │                                                         (Groq cloud, or this computer)
   └── stores transcripts + media in IndexedDB (nothing is saved on the server)
```
Because the server is stateless and tiny, it fits in a Vercel function. The heavy speech model runs on
[Groq](https://console.groq.com) (free tier, Whisper large-v3) or locally with faster-whisper.

## Features
- Upload audio/video, record from the mic, or paste a link (link import: local mode only)
- Auto language detection with an English/Hindi sanity check; "Translate to English"
- Editor: click-to-seek, live highlight, inline editing, speakers, find & replace, playback speed
- **Copy text** in one click, or export Markdown, PDF (via the browser, so Hindi renders perfectly), TXT, SRT
- AI tools: summary, punctuation/spelling clean-up, speaker detection (Groq or OpenRouter free models)
- Long files are resumable if the tab closes; optional password protection

## Deploy on Vercel
1. Get a free Groq key: https://console.groq.com/keys
2. On vercel.com: **Add New → Project → import this GitHub repo** (no build settings needed).
3. In **Settings → Environment Variables** add:
   - `GROQ_API_KEY` = your Groq key
   - `APP_PASSWORD` = any password (so strangers can't use your key)
   - optional AI: leave empty to use Groq's free LLMs, or set `AI_BASE_URL=https://openrouter.ai/api/v1` + `AI_API_KEY`
4. Deploy. Open the URL, enter the password, transcribe.

## Deploy with Docker (Render, Railway, Fly.io, Koyeb, a VPS, Hugging Face Spaces…)
```bash
docker build -t voxscribe .
docker run -p 8000:8000 -e GROQ_API_KEY=... -e APP_PASSWORD=... voxscribe
```
Offline Whisper inside the container instead of Groq: `docker build --build-arg LOCAL=1 -t voxscribe .`
and run with `-e STT_ENGINE=local` (needs ~2 GB RAM).

## Run on your own computer
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-local.txt      # includes offline Whisper
cp .env.example .env                        # optional: add GROQ_API_KEY to use Groq instead
python app.py                               # http://127.0.0.1:8000
```
Without `GROQ_API_KEY` it uses offline Whisper (Apple-silicon GPU is used automatically).
Use Python 3.12 or 3.13 (3.15 isn't supported by the speech libraries yet).

## Settings (environment variables)
| Variable | What it does |
|---|---|
| `GROQ_API_KEY` | Groq key for cloud Whisper (and free AI tools) |
| `STT_ENGINE` | `groq` or `local` (auto by default) |
| `APP_PASSWORD` | Require a password to use the app |
| `AI_BASE_URL`, `AI_API_KEY`, `AI_MODEL` | Any OpenAI-compatible LLM for AI tools (`AI_MODEL=auto` picks one) |

## Limits to know
- Groq free tier: about 2 hours of audio per hour, 8 hours per day.
- Transcripts are stored **in your browser**: clearing site data deletes them; they don't sync between devices.
- Keep the tab open while a file transcribes (you can resume if it closes).
- The browser decodes the audio first (≈1 min per 30 min of audio); very large videos (>700 MB) won't fit.

## Project layout
```
app.py              FastAPI server (Vercel entrypoint): /api/transcribe, /api/ai, /api/config
stt.py              speech engines: Groq Whisper or local faster-whisper
ai.py               summary / clean-up / speakers via any OpenAI-compatible API
public/             the web app (index.html + app.js)
transcriber.py      local faster-whisper wrapper    exporters.py / cli.py  command-line tool
vercel.json · Dockerfile · requirements.txt (server) · requirements-local.txt (+ offline Whisper)
```

## Command line (local)
```bash
python cli.py lecture.mp4 --lang hi --model medium --format md pdf txt
```
