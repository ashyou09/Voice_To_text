"""VoxScribe server — a small, stateless API. Works on Vercel, in Docker, or on your own computer.

    python app.py            -> http://127.0.0.1:8000

The browser does the heavy lifting: it decodes the audio, splits it into ~2-minute chunks,
sends them here one by one, and stores transcripts in the browser (IndexedDB).
"""
from __future__ import annotations

import hmac
import json
import os
import shutil
import tempfile
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import ai
import stt

HERE = Path(__file__).parent
DATA = HERE / "data"  # transcripts from the older server-side version (imported into the browser)
ON_VERCEL = bool(os.getenv("VERCEL"))
CHUNK_SECONDS = 110  # 16 kHz mono WAV: 110 s ≈ 3.5 MB, under Vercel's 4.5 MB request limit

app = FastAPI(title="VoxScribe")


# ---------------- optional password ----------------
def check_password(x_app_password: str = Header("")):
    pw = os.getenv("APP_PASSWORD", "")
    if pw and not hmac.compare_digest(x_app_password.encode(), pw.encode()):
        raise HTTPException(401, "Wrong or missing password")


def _link_import() -> bool:
    if ON_VERCEL:
        return False
    try:
        import yt_dlp  # noqa: F401

        return True
    except ImportError:
        return False


def _legacy_ids() -> list[str]:
    if ON_VERCEL or not DATA.is_dir():
        return []
    return sorted(d.name for d in DATA.iterdir() if (d / "meta.json").is_file())


# ---------------- pages ----------------
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    page = HERE / "public" / "index.html"
    if page.is_file():
        return page.read_text(encoding="utf-8")
    return RedirectResponse("/index.html")  # on Vercel, public/ is served by the CDN


@app.get("/app.js", include_in_schema=False)
def app_js():  # on Vercel this file is served straight from public/ by the CDN
    return FileResponse(HERE / "public" / "app.js", media_type="text/javascript")


# ---------------- API ----------------
@app.get("/api/config")
def config():
    return {
        "engine": stt.engine(),
        "models": stt.models(),
        "default_model": stt.default_model(),
        "chunk_seconds": CHUNK_SECONDS,
        "ai_enabled": ai.enabled(),
        "ai_model": ai.model_label(),
        "link_import": _link_import(),
        "auth_required": bool(os.getenv("APP_PASSWORD")),
        "legacy": len(_legacy_ids()),
    }


@app.post("/api/login", dependencies=[Depends(check_password)])
def login():
    return {"ok": True}


@app.post("/api/transcribe", dependencies=[Depends(check_password)])
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form(""),
    model: str = Form(""),
    task: str = Form("transcribe"),
    detect: bool = Form(False),
):
    audio = await file.read()
    if not audio:
        raise HTTPException(400, "Empty audio chunk")
    try:
        # run the blocking HTTP / model call off the event loop
        from starlette.concurrency import run_in_threadpool

        return await run_in_threadpool(
            stt.transcribe_chunk, audio, model, language or None, "translate" if task == "translate" else "transcribe", detect
        )
    except stt.STTError as e:
        headers = {"Retry-After": str(int(e.retry_after))} if e.retry_after else None
        return JSONResponse({"detail": str(e)}, status_code=e.status, headers=headers)


class AIRequest(BaseModel):
    action: str  # summary | cleanup | speakers
    segments: list[dict]
    prev: str = ""


@app.post("/api/ai", dependencies=[Depends(check_password)])
def run_ai(body: AIRequest):
    if not ai.enabled():
        raise HTTPException(400, "AI tools are off. Set an AI key (see README).")
    if not body.segments:
        raise HTTPException(400, "Transcript is empty")
    if len(body.segments) > 120 and body.action != "summary":
        raise HTTPException(400, "Send at most 120 segments per request")
    try:
        if body.action == "summary":
            return {"summary": ai.summarize(body.segments)}
        if body.action == "cleanup":
            return {"segments": ai.cleanup(body.segments)}
        if body.action == "speakers":
            return {"segments": ai.label_speakers(body.segments, body.prev)}
    except ai.AIError as e:
        raise HTTPException(502, str(e))
    raise HTTPException(400, "Unknown action")


class LinkRequest(BaseModel):
    url: str


@app.post("/api/fetch-link", dependencies=[Depends(check_password)])
def fetch_link(body: LinkRequest):
    """Local only: download the audio of a YouTube/video link and hand it to the browser."""
    if not _link_import():
        raise HTTPException(400, "Link import is only available when running on your own computer.")
    import yt_dlp

    tmp = Path(tempfile.mkdtemp(prefix="voxscribe_link_"))
    opts = {"format": "bestaudio/best", "outtmpl": str(tmp / "%(title).80s.%(ext)s"), "quiet": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            path = Path(ydl.prepare_filename(ydl.extract_info(body.url, download=True)))
    except Exception as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise HTTPException(400, f"Could not download that link: {e}")
    return FileResponse(path, filename=path.name,
                        background=BackgroundTask(shutil.rmtree, tmp, ignore_errors=True))


# ---- transcripts saved by the older version of this app (server-side data/ folder) ----
@app.get("/api/legacy", dependencies=[Depends(check_password)])
def legacy_list():
    out = []
    for tid in _legacy_ids():
        try:
            m = json.loads((DATA / tid / "meta.json").read_text(encoding="utf-8"))
            if m.get("status") == "done":
                out.append(m)
        except Exception:
            pass
    return out


@app.get("/api/legacy/{tid}/media", dependencies=[Depends(check_password)])
def legacy_media(tid: str):
    if tid not in _legacy_ids():
        raise HTTPException(404, "Not found")
    m = json.loads((DATA / tid / "meta.json").read_text(encoding="utf-8"))
    p = DATA / tid / (m.get("media") or "")
    if not m.get("media") or not p.is_file():
        raise HTTPException(404, "No media")
    return FileResponse(p)


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    print(f"VoxScribe running at http://{host}:{port}  (speech engine: {stt.engine()})")
    uvicorn.run(app, host=host, port=port)
