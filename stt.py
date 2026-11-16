"""Speech-to-text engines. The browser sends one short WAV chunk at a time (<= ~2 minutes).

- "groq"  : Groq's hosted Whisper (fast, free tier, works on Vercel / any small server)
- "local" : faster-whisper on this machine (needs requirements-local.txt; not for Vercel)

Pick with STT_ENGINE=groq|local. Default: groq if GROQ_API_KEY is set, else local if installed.
"""
from __future__ import annotations

import os
import tempfile

import httpx

GROQ_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_MODELS = ["whisper-large-v3", "whisper-large-v3-turbo"]
LOCAL_MODELS = ["tiny", "base", "small", "medium", "large-v3", "turbo"]

# Whisper returns language names; the app uses ISO codes.
NAME_TO_CODE = {
    "english": "en", "hindi": "hi", "urdu": "ur", "bengali": "bn", "marathi": "mr", "gujarati": "gu",
    "punjabi": "pa", "tamil": "ta", "telugu": "te", "kannada": "kn", "malayalam": "ml", "nepali": "ne",
    "sinhala": "si", "assamese": "as", "sanskrit": "sa", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "russian": "ru", "arabic": "ar", "persian": "fa", "turkish": "tr",
    "japanese": "ja", "chinese": "zh", "korean": "ko", "indonesian": "id", "malay": "ms", "thai": "th",
    "vietnamese": "vi", "dutch": "nl", "polish": "pl", "ukrainian": "uk", "swedish": "sv", "greek": "el",
    "hebrew": "he", "czech": "cs", "romanian": "ro", "hungarian": "hu", "danish": "da", "finnish": "fi",
    "norwegian": "no", "swahili": "sw", "tagalog": "tl", "filipino": "tl",
}


class STTError(Exception):
    def __init__(self, message: str, status: int = 500, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def _local_available() -> bool:
    try:
        import faster_whisper  # noqa: F401

        return True
    except Exception:
        return False


def engine() -> str:
    e = os.getenv("STT_ENGINE", "").strip().lower()
    if e in ("groq", "local"):
        return e
    if os.getenv("GROQ_API_KEY"):
        return "groq"
    return "local" if _local_available() else "none"


def models() -> list[str]:
    return {"groq": GROQ_MODELS, "local": LOCAL_MODELS}.get(engine(), [])


def default_model() -> str:
    return {"groq": "whisper-large-v3", "local": "small"}.get(engine(), "")


def _code(lang: str | None) -> str:
    lang = (lang or "").strip().lower()
    return lang if len(lang) <= 3 else NAME_TO_CODE.get(lang, lang)


def _quality(segs: list[dict]) -> float:
    """Average log-probability weighted by duration. Repetitive output (a sign of the wrong
    language or a hallucination loop) compresses very well, so it is scored as bad."""
    total = dur = 0.0
    for s in segs:
        d = max(s["end"] - s["start"], 0.1)
        lp = -3.0 if s.get("compression_ratio", 1) > 2.2 else s.get("avg_logprob", -1.0)
        total += lp * d
        dur += d
    return total / dur if dur else -9.0


# ---------------- Groq ----------------
def _groq_call(audio: bytes, model: str, language: str | None, task: str) -> tuple[str, list[dict]]:
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise STTError("GROQ_API_KEY is not set on the server.", 500)
    translate = task == "translate"
    data = {"model": "whisper-large-v3" if translate else model,  # turbo can't translate
            "response_format": "verbose_json", "temperature": "0"}
    if language and not translate:
        data["language"] = language
    try:
        r = httpx.post(
            f"{GROQ_URL}/audio/{'translations' if translate else 'transcriptions'}",
            headers={"Authorization": f"Bearer {key}"},
            data=data,
            files={"file": ("chunk.wav", audio, "audio/wav")},
            timeout=120,
        )
    except httpx.HTTPError as e:
        raise STTError(f"Could not reach Groq: {e}", 502)
    if r.status_code == 429:
        ra = r.headers.get("retry-after")
        raise STTError("Groq rate limit reached, waiting…", 429, float(ra) if ra else 20.0)
    if r.status_code in (401, 403):
        raise STTError("Groq rejected the API key (GROQ_API_KEY).", 500)
    if r.status_code != 200:
        raise STTError(f"Groq error {r.status_code}: {r.text[:300]}", 502)
    j = r.json()
    segs = []
    for s in j.get("segments") or []:
        text = (s.get("text") or "").strip()
        # Whisper invents text in silence ("Thank you.", "Subscribe!"). Drop those.
        if not text or (s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0):
            continue
        segs.append({"start": float(s["start"]), "end": float(s["end"]), "text": text,
                     "avg_logprob": s.get("avg_logprob", -1.0), "compression_ratio": s.get("compression_ratio", 1.0)})
    return _code(j.get("language")) or (language or ""), segs


def _groq(audio: bytes, model: str, language: str | None, task: str, detect: bool) -> dict:
    if language or not detect or task == "translate":
        lang, segs = _groq_call(audio, model, language, task)
        return {"language": language or lang, "segments": segs}

    lang, segs = _groq_call(audio, model, None, task)
    if lang == "ur":  # Hindi speech is often labelled Urdu (and written in Urdu script)
        lang, segs = "hi", _groq_call(audio, model, "hi", task)[1]
    q = _quality(segs)
    # Indian-accented English is sometimes detected as Hindi (and the reverse). If the
    # result looks shaky, also try the likely alternative and keep whichever reads better.
    if segs and (q < -0.6 or any(s["compression_ratio"] > 2.2 for s in segs)):
        alt = "en" if lang != "en" else "hi"
        alt_segs = _groq_call(audio, model, alt, task)[1]
        if _quality(alt_segs) > q + 0.05:
            lang, segs = alt, alt_segs
    return {"language": lang if segs else "", "segments": segs}


# ---------------- local faster-whisper ----------------
def _local(audio: bytes, model: str, language: str | None, task: str) -> dict:
    import transcriber  # heavy import, only when the local engine is used

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio)
        path = f.name
    try:
        t = transcriber.transcribe(path, model_size=model, language=language, task=task)
    finally:
        os.unlink(path)
    segs = [{"start": s.start, "end": s.end, "text": s.text} for s in t.segments if s.text.strip()]
    return {"language": t.language if segs or language else "", "segments": segs}


def transcribe_chunk(audio: bytes, model: str = "", language: str | None = None,
                     task: str = "transcribe", detect: bool = False) -> dict:
    e = engine()
    model = model if model in models() else default_model()
    if e == "groq":
        out = _groq(audio, model, language, task, detect)
    elif e == "local":
        out = _local(audio, model, language, task)
    else:
        raise STTError("No speech engine configured. Set GROQ_API_KEY (or install requirements-local.txt).", 500)
    out["segments"] = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"]}
                       for s in out["segments"]]
    return out
