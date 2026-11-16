"""Optional AI tools through any OpenAI-compatible API (OpenRouter by default).

Set in .env:
    AI_BASE_URL=http://localhost:20128/v1     # default: https://openrouter.ai/api/v1
    AI_API_KEY=...                            # (OPENROUTER_API_KEY also works)
    AI_MODEL=auto                             # "auto" = pick a good model the server offers (default)
"""
from __future__ import annotations

import json
import os
import re

import httpx

CHUNK = 80  # segments per request, keeps prompts small and outputs reliable


class AIError(Exception):
    pass


def base_url() -> str:
    """Any OpenAI-compatible server works: OpenRouter (default), a local proxy, LM Studio, Ollama, ..."""
    return os.getenv("AI_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


def api_key() -> str:
    return os.getenv("AI_API_KEY") or os.getenv("OPENROUTER_API_KEY") or ""


def is_local() -> bool:
    u = base_url()
    return "localhost" in u or "127.0.0.1" in u


def enabled() -> bool:
    # Hosted services need a key; a local server may not.
    return bool(api_key()) or (is_local() and bool(os.getenv("AI_BASE_URL")))


# Used when AI_MODEL=auto (or unset on a local server): first match wins.
PREFERRED = ["gemini", "gpt-oss", "llama-3.3", "llama-4", "qwen", "deepseek", "glm", "mistral", "llama", "gemma"]
SKIP = ("embed", "whisper", "tts", "image", "rerank", "moderation", "audio", "vision-only", "dall", "flux", "sdxl")
_picked: dict[str, str] = {}


# Order to try free OpenRouter models (tested on Hindi/English proofreading with JSON output).
FREE_PREFERRED = ["nemotron-3-ultra", "gemma-4-31b", "ernie-5.5", "gemma-4-26b", "ling-3", "nemotron"]
_free_cache: dict = {"t": 0.0, "ids": []}


def _free_models() -> list[str]:
    """Models that cost $0 on OpenRouter right now (price list is public). Never returns a paid model."""
    import time

    if _free_cache["ids"] and time.time() - _free_cache["t"] < 600:
        return _free_cache["ids"]
    try:
        r = httpx.get(f"{base_url()}/models", timeout=20)
        r.raise_for_status()
        free = []
        for m in r.json().get("data", []):
            pr = m.get("pricing", {})
            mods = m.get("architecture", {}).get("output_modalities", ["text"])
            if (float(pr.get("prompt", 1)) == 0 and float(pr.get("completion", 1)) == 0
                    and mods == ["text"] and not any(k in m["id"].lower() for k in SKIP)
                    and m["id"] != "openrouter/free"):
                free.append(m)
    except Exception:
        return ["openrouter/free"]
    free.sort(key=lambda m: -(m.get("context_length") or 0))
    ids = [m["id"] for m in free]
    ordered = [i for key in FREE_PREFERRED for i in ids if key in i.lower()]
    ordered += [i for i in ids if i not in ordered]
    ids = list(dict.fromkeys(ordered)) or ["openrouter/free"]
    _free_cache.update(t=time.time(), ids=ids)
    return ids


def _auto_pick() -> str:
    """Local server: ask which models it offers and choose a good chat model."""
    if base_url() in _picked:
        return _picked[base_url()]
    if not is_local():
        return _free_models()[0]
    headers = {"Authorization": f"Bearer {api_key()}"} if api_key() else {}
    try:
        r = httpx.get(f"{base_url()}/models", headers=headers, timeout=15)
        r.raise_for_status()
        ids = [m["id"] for m in r.json().get("data", []) if isinstance(m.get("id"), str)]
    except httpx.ConnectError:
        raise AIError(f"Can't reach the AI server at {base_url()}. Is it running?")
    except Exception as e:
        raise AIError(f"Could not list models from {base_url()}: {e}")
    ids = [i for i in ids if not any(k in i.lower() for k in SKIP)]
    if not ids:
        raise AIError(f"{base_url()} lists no chat models. Set AI_MODEL in .env.")
    for key in PREFERRED:
        for i in ids:
            if key in i.lower():
                _picked[base_url()] = i
                return i
    _picked[base_url()] = ids[0]
    return ids[0]


def _configured_model() -> str:
    m = os.getenv("AI_MODEL", "").strip()
    if not m and not is_local():
        m = os.getenv("OPENROUTER_MODEL", "").strip()  # legacy setting, hosted OpenRouter only
    return m or "auto"


def model_name() -> str:
    """The model that will actually be used."""
    m = _configured_model()
    return _auto_pick() if m == "auto" else m


def model_label() -> str:
    """For the UI: never touches the network."""
    m = _configured_model()
    if m != "auto":
        return m
    return _picked.get(base_url(), "auto-selected model") if is_local() else "free models (auto)"


def _candidates() -> list[str]:
    m = _configured_model()
    if m != "auto":
        return [m]
    if is_local():
        return [_auto_pick()]
    return _free_models()[:5] + ["openrouter/free"]  # free models only; moves on if one is busy


def chat(system: str, user: str, json_mode: bool = False) -> str:
    headers = {"X-Title": "VoxScribe"}
    if api_key():
        headers["Authorization"] = f"Bearer {api_key()}"

    def post(model, with_json):
        body = {"model": model, "temperature": 0.2,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if with_json:
            body["response_format"] = {"type": "json_object"}
        return httpx.post(f"{base_url()}/chat/completions", json=body, headers=headers, timeout=180)

    last = "no model available"
    for model in _candidates():
        try:
            r = post(model, json_mode)
            if r.status_code in (400, 422) and json_mode:  # some models/servers don't support JSON mode
                r = post(model, False)
        except httpx.ConnectError:
            raise AIError(f"Can't reach the AI server at {base_url()}. Is it running?")
        except httpx.HTTPError as e:
            last = f"{model}: {e}"
            continue
        if r.status_code in (401, 403):
            raise AIError("The AI server rejected the API key. Check AI_API_KEY in .env.")
        try:
            data = r.json()
        except ValueError:
            last = f"{model}: bad response"
            continue
        if r.status_code == 200 and data.get("choices"):
            text = (data["choices"][0].get("message") or {}).get("content") or ""
            if text.strip():
                return text
            last = f"{model}: empty reply"
            continue
        err = data.get("error", {}) if isinstance(data, dict) else {}
        last = f"{model}: {r.status_code} {str(err.get('message', r.text))[:120]}"
        # busy / unavailable -> try the next free model
    raise AIError(f"The AI server could not answer ({last}). Free models are often busy; try again in a minute.")


def _parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except ValueError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise AIError("The AI model did not return valid JSON. Try again or use another model.")


def _numbered(segs: list[dict], with_speaker: bool = False) -> str:
    out = []
    for i, s in enumerate(segs):
        sp = f"[{s['speaker']}] " if with_speaker and s.get("speaker") else ""
        out.append(f"{i}: {sp}{s['text']}")
    return "\n".join(out)


# ---------------- tools ----------------
def summarize(segs: list[dict]) -> str:
    text = " ".join(s["text"] for s in segs)[:120_000]
    return chat(
        "You summarize transcripts. Write in the SAME language and script as the transcript "
        "(Hindi in Devanagari, English in English). Use Markdown: a 2-3 sentence overview, "
        "then 'Key points' as bullet points, then 'Action items' if there are any. No preamble.",
        text,
    ).strip()


def cleanup(segs: list[dict]) -> list[dict]:
    """Fix punctuation, capitalization and obvious mis-heard words. Keeps timing intact."""
    out = [dict(s) for s in segs]
    for start in range(0, len(segs), CHUNK):
        chunk = segs[start:start + CHUNK]
        reply = _parse_json(chat(
            "You proofread speech-to-text output. For each numbered line fix punctuation, "
            "capitalization, spelling and clearly mis-heard words. Do NOT translate, summarize, "
            "merge or split lines, and keep the original language and script (Hindi stays in "
            "Devanagari; Hinglish stays as spoken). Return JSON: {\"lines\": {\"<number>\": \"<fixed text>\"}} "
            "with every number present.",
            _numbered(chunk),
            json_mode=True,
        ))
        lines = reply.get("lines", {})
        for i in range(len(chunk)):
            new = lines.get(str(i))
            if isinstance(new, str) and new.strip():
                out[start + i]["text"] = new.strip()
    return out


def label_speakers(segs: list[dict]) -> list[dict]:
    """Guess who is speaking from the text alone (conversation cues). Best effort."""
    out = [dict(s) for s in segs]
    prev = ""
    for start in range(0, len(segs), CHUNK):
        chunk = segs[start:start + CHUNK]
        context = f"The previous line was spoken by {prev}.\n" if prev else ""
        reply = _parse_json(chat(
            "You label speakers in a transcript using conversational cues (questions/answers, "
            "names, turn-taking). Use 'Speaker 1', 'Speaker 2', ... consistently, or a real name "
            "if people are clearly named. If it is a monologue, use 'Speaker 1' everywhere. "
            "Return JSON: {\"speakers\": {\"<number>\": \"<label>\"}} with every number present.",
            context + _numbered(chunk),
            json_mode=True,
        ))
        labels = reply.get("speakers", {})
        for i in range(len(chunk)):
            lab = labels.get(str(i))
            if isinstance(lab, str) and lab.strip():
                out[start + i]["speaker"] = lab.strip()[:40]
        prev = out[start + len(chunk) - 1].get("speaker", "")
    return out
