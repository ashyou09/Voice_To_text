"""Speech-to-text core, built on faster-whisper (an open-source Whisper implementation).

Works with audio AND video files (mp3, wav, m4a, ogg, flac, mp4, mkv, mov, webm...).
faster-whisper decodes media with PyAV, which bundles FFmpeg, so no separate install is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Optional

from faster_whisper import WhisperModel

MODEL_SIZES = ["tiny", "base", "small", "medium", "large-v3", "turbo"]


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str = ""


@dataclass
class Transcript:
    source_name: str
    language: str
    language_probability: float
    duration: float
    segments: list[Segment] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments).strip()


def _pick_device() -> tuple[str, str]:
    """GPU if available (NVIDIA CUDA, or Apple Metal on M-series Macs), otherwise CPU with int8."""
    try:
        import platform

        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            ctranslate2.get_supported_compute_types("mps")  # raises if this build has no Metal support
            return "mps", "float16"
    except Exception:
        pass
    return "cpu", "int8"


@lru_cache(maxsize=4)
def load_model(size: str = "small", force_cpu: bool = False) -> WhisperModel:
    device, compute_type = ("cpu", "int8") if force_cpu else _pick_device()
    # The model is downloaded from Hugging Face on first use, then cached.
    try:
        model = WhisperModel(size, device=device, compute_type=compute_type)
    except Exception:
        if device == "cpu":
            raise
        device = "cpu"  # GPU mode failed -> CPU
        model = WhisperModel(size, device="cpu", compute_type="int8")
    model.vx_on_gpu = device != "cpu"
    return model


def transcribe(
    path: str,
    model_size: str = "small",
    language: Optional[str] = None,
    task: str = "transcribe",
    source_name: Optional[str] = None,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Transcript:
    """Transcribe a media file.

    language: ISO code like "en" or "hi"; None = auto-detect.
    task: "transcribe" (keep original language) or "translate" (output English).
    """
    try:
        return _run(load_model(model_size), path, language, task, source_name, on_progress)
    except Exception:
        # The Apple-GPU backend is experimental: if it fails mid-run, redo the job on the CPU.
        if _pick_device()[0] != "mps":
            raise
        return _run(load_model(model_size, force_cpu=True), path, language, task, source_name, on_progress)


def _best_language(model, path, info, sample_seconds: int = 90) -> tuple[str, float]:
    """Pick the language when auto-detect isn't sure.

    Whisper's detector often calls Indian-accented English "Hindi" (and the reverse). Forcing the wrong
    language gives garbage and is very slow. So when confidence is low, decode the first ~90s in each top
    candidate language and keep the one with the best average log-probability, ignoring repetitive output.
    """
    probs = sorted(info.all_language_probs or [(info.language, info.language_probability)], key=lambda x: -x[1])
    if info.language_probability >= 0.85:
        return info.language, info.language_probability
    cands = [lang for lang, _ in probs[:3]]
    if "en" not in cands and any(c in cands for c in ("hi", "ur")):
        cands.append("en")
    scores = {}
    for lang in cands:
        try:
            segs, _ = model.transcribe(
                path, language=lang, beam_size=1, temperature=0.0,  # no retry loops on a wrong language
                condition_on_previous_text=False, vad_filter=True, clip_timestamps=[0, sample_seconds],
            )
            total = dur = 0.0
            for seg in segs:
                d = max(seg.end - seg.start, 0.1)
                # A wrong language usually makes Whisper repeat one phrase over and over, which looks
                # "confident" (high log-prob) but compresses extremely well. Score that as bad.
                lp = -3.0 if seg.compression_ratio > 2.0 else seg.avg_logprob
                total += lp * d
                dur += d
            scores[lang] = total / dur if dur else -9.0
        except Exception:
            scores[lang] = -9.0
    best = max(scores, key=scores.get)
    return best, dict(probs).get(best, 0.0)


def _run(model, path, language, task, source_name, on_progress) -> Transcript:
    on_gpu = getattr(model, "vx_on_gpu", False)

    def start(lang):
        return model.transcribe(
            path,
            language=lang,
            task=task,
            beam_size=5 if on_gpu else 2,
            # Each 30s window is decoded on its own. With the default (True) one bad window can
            # make Whisper loop and retry for minutes, and the garbage carries into later windows.
            condition_on_previous_text=False,
            vad_filter=True,  # skips silence -> fewer hallucinated lines
            vad_parameters={"min_silence_duration_ms": 500},
            # Vote over 5 stretches of audio, not just the first 30s (which may be silence/chatter).
            language_detection_segments=5,
        )

    # `segments_iter` is lazy: only language detection has run at this point.
    segments_iter, info = start(language or None)
    lang, prob = info.language, info.language_probability

    if not language:
        lang, prob = _best_language(model, path, info)
        # Spoken Hindi and Urdu sound almost the same, so auto-detect often labels Hindi
        # audio as Urdu and writes it in Urdu script. Use Hindi to get Devanagari.
        if lang == "ur":
            lang = "hi"
        if lang != info.language:
            prob = 0.0  # chosen by the sample check, so the detector's confidence doesn't apply
            segments_iter, info = start(lang)

    result = Transcript(
        source_name=source_name or path.replace("\\", "/").split("/")[-1],
        language=lang,
        language_probability=prob,
        duration=info.duration,
    )
    for seg in segments_iter:  # generator: transcription happens as we iterate
        result.segments.append(Segment(seg.start, seg.end, seg.text.strip()))
        if on_progress and info.duration:
            on_progress(min(seg.end / info.duration, 1.0))
    return result
