"""Command-line version.

Examples:
  python cli.py lecture.mp4
  python cli.py voice.m4a --model medium --lang hi --format md pdf
  python cli.py meeting.mp3 --translate --no-timestamps --out transcripts
"""
import argparse

from exporters import export
from transcriber import MODEL_SIZES, transcribe


def main():
    ap = argparse.ArgumentParser(description="Transcribe audio/video to md, pdf or txt")
    ap.add_argument("file", help="audio or video file")
    ap.add_argument("--model", default="small", choices=MODEL_SIZES)
    ap.add_argument("--lang", default=None, help="language code, e.g. en, hi (default: auto-detect)")
    ap.add_argument("--translate", action="store_true", help="translate speech to English")
    ap.add_argument("--format", nargs="+", default=["md", "pdf", "txt"], choices=["md", "pdf", "txt"])
    ap.add_argument("--no-timestamps", action="store_true")
    ap.add_argument("--out", default="transcripts", help="output folder")
    a = ap.parse_args()

    def show(f):
        print(f"\r  {f:5.0%}", end="", flush=True)

    print(f"Transcribing {a.file} with '{a.model}' model...")
    t = transcribe(a.file, a.model, a.lang, "translate" if a.translate else "transcribe", on_progress=show)
    print(f"\nDetected language: {t.language} ({t.language_probability:.0%})")
    for path in export(t, a.format, a.out, timestamps=not a.no_timestamps):
        print("  saved", path)


if __name__ == "__main__":
    main()
