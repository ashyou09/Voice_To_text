"""VoxScribe web app.

    python app.py            -> open http://127.0.0.1:8000

Transcription runs locally with Whisper. AI tools (summary, clean-up, speaker
detection) use any OpenAI-compatible API configured in the .env file.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import ai
from exporters import export
from transcriber import MODEL_SIZES, Segment, Transcript

HERE = Path(__file__).parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)

app = FastAPI(title="VoxScribe")
worker = ThreadPoolExecutor(max_workers=1)  # one transcription at a time (models are RAM-hungry)
lock = threading.Lock()


# ---------------- storage: data/<id>/meta.json + media file ----------------
def _dir(tid: str) -> Path:
    d = DATA / tid
    if not d.is_dir() or "/" in tid or ".." in tid:
        raise HTTPException(404, "Transcript not found")
    return d


def load(tid: str) -> dict:
    return json.loads((_dir(tid) / "meta.json").read_text(encoding="utf-8"))


def save(meta: dict) -> None:
    with lock:
        p = DATA / meta["id"] / "meta.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)


def update(tid: str, **fields) -> dict:
    meta = load(tid)
    meta.update(fields)
    save(meta)
    return meta


def to_transcript(meta: dict) -> Transcript:
    return Transcript(
        source_name=meta["title"],
        language=meta.get("language") or "?",
        language_probability=meta.get("language_probability") or 0,
        duration=meta.get("duration") or 0,
        segments=[Segment(s["start"], s["end"], s["text"], s.get("speaker", "")) for s in meta.get("segments", [])],
    )


# ---------------- background job ----------------
def run_job(tid: str, url: Optional[str]) -> None:
    from transcriber import transcribe  # imported here so the UI starts instantly

    try:
        meta = load(tid)
        d = DATA / tid
        if url:
            update(tid, status="downloading")
            media = download_link(url, d)
            meta = update(tid, media=media.name, title=media.stem)
        update(tid, status="transcribing", progress=0)

        last = [0.0]

        def progress(f: float):
            if f - last[0] >= 0.02:
                last[0] = f
                update(tid, progress=round(f, 3))

        t = transcribe(
            str(d / meta["media"]),
            model_size=meta["model"],
            language=meta["language_requested"] or None,
            task="translate" if meta["translate"] else "transcribe",
            source_name=meta["title"],
            on_progress=progress,
        )
        update(
            tid,
            status="done",
            progress=1,
            language=t.language,
            language_probability=t.language_probability,
            duration=t.duration,
            segments=[{"start": s.start, "end": s.end, "text": s.text, "speaker": ""} for s in t.segments],
        )
    except Exception as e:  # show the error in the UI instead of failing silently
        update(tid, status="error", error=f"{type(e).__name__}: {e}")


def download_link(url: str, out_dir: Path) -> Path:
    import yt_dlp

    opts = {"format": "bestaudio/best", "outtmpl": str(out_dir / "%(title).80s.%(ext)s"), "quiet": True,
            "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info))


# ---------------- API ----------------
@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config():
    return {"models": MODEL_SIZES, "ai_enabled": ai.enabled(), "ai_model": ai.model_label()}


@app.post("/api/transcriptions")
async def create(
    file: Optional[UploadFile] = File(None),
    url: str = Form(""),
    language: str = Form(""),
    model: str = Form("small"),
    translate: bool = Form(False),
):
    if not file and not url.strip():
        raise HTTPException(400, "Upload a file or paste a link")
    if model not in MODEL_SIZES:
        raise HTTPException(400, "Unknown model")
    tid = uuid.uuid4().hex[:12]
    d = DATA / tid
    d.mkdir()
    meta = {
        "id": tid, "created": time.time(), "status": "queued", "progress": 0, "error": "",
        "title": "Link import" if not file else Path(file.filename or "recording").stem,
        "media": "", "model": model, "language_requested": language, "translate": translate,
        "language": "", "language_probability": 0, "duration": 0, "segments": [], "summary": "",
    }
    if file:
        name = "media" + (Path(file.filename or "").suffix.lower() or ".webm")
        with open(d / name, "wb") as f:
            shutil.copyfileobj(file.file, f)
        meta["media"] = name
    save(meta)
    worker.submit(run_job, tid, url.strip() or None)
    return meta


@app.get("/api/transcriptions")
def list_all():
    items = []
    for d in DATA.iterdir():
        try:
            m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        m.pop("segments", None)
        m.pop("summary", None)
        items.append(m)
    return sorted(items, key=lambda m: m["created"], reverse=True)


@app.get("/api/transcriptions/{tid}")
def get_one(tid: str):
    return load(tid)


class Edit(BaseModel):
    title: Optional[str] = None
    segments: Optional[list[dict]] = None
    summary: Optional[str] = None


@app.put("/api/transcriptions/{tid}")
def edit(tid: str, body: Edit):
    fields = {k: v for k, v in body.model_dump().items() if v is not None}
    return update(tid, **fields)


@app.delete("/api/transcriptions/{tid}")
def delete(tid: str):
    shutil.rmtree(_dir(tid))
    return {"ok": True}


@app.get("/api/transcriptions/{tid}/media")
def media(tid: str):
    meta = load(tid)
    if not meta.get("media"):
        raise HTTPException(404, "No media yet")
    return FileResponse(_dir(tid) / meta["media"])


@app.get("/api/transcriptions/{tid}/export")
def export_file(tid: str, fmt: str = "md", timestamps: bool = True, summary: bool = True):
    meta = load(tid)
    if fmt not in ("txt", "md", "pdf", "srt"):
        raise HTTPException(400, "Format must be txt, md, pdf or srt")
    t = to_transcript(meta)
    out = Path(tempfile.mkdtemp(prefix="voxscribe_"))
    path = export(t, [fmt], str(out), timestamps, summary=meta.get("summary", "") if summary else "")[0]
    media_types = {"txt": "text/plain", "md": "text/markdown", "pdf": "application/pdf", "srt": "application/x-subrip"}
    return FileResponse(path, media_type=media_types[fmt], filename=f"{meta['title']}.{fmt}")


class AIRequest(BaseModel):
    action: str  # summary | cleanup | speakers


@app.post("/api/transcriptions/{tid}/ai")
def run_ai(tid: str, body: AIRequest):
    if not ai.enabled():
        raise HTTPException(400, "Set AI_BASE_URL / AI_API_KEY in the .env file to use AI tools")
    meta = load(tid)
    segs = meta.get("segments", [])
    if not segs:
        raise HTTPException(400, "Transcript is empty")
    try:
        if body.action == "summary":
            return update(tid, summary=ai.summarize(segs))
        if body.action == "cleanup":
            return update(tid, segments=ai.cleanup(segs))
        if body.action == "speakers":
            return update(tid, segments=ai.label_speakers(segs))
    except ai.AIError as e:
        raise HTTPException(502, str(e))
    raise HTTPException(400, "Unknown action")


if __name__ == "__main__":
    # Jobs left unfinished by a previous run can't resume; mark them as failed.
    for d in DATA.iterdir():
        try:
            m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
            if m["status"] not in ("done", "error"):
                m.update(status="error", error="Interrupted (server restarted). Delete and try again.")
                save(m)
        except Exception:
            pass
    print("VoxScribe running at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", 8000)))
