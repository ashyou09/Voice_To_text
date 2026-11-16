"""Turn a Transcript into .txt, .md or .pdf files."""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from fpdf import FPDF

from transcriber import Segment, Transcript

HERE = Path(__file__).parent


# ---------- helpers ----------
def fmt_ts(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def paragraphs(segments: list[Segment], gap: float = 2.0, max_segs: int = 6) -> list[list[Segment]]:
    """Group Whisper's short segments into readable paragraphs:
    start a new paragraph after a pause longer than `gap` seconds or every `max_segs` segments."""
    paras: list[list[Segment]] = []
    for seg in segments:
        if (not paras or seg.start - paras[-1][-1].end > gap or len(paras[-1]) >= max_segs
                or seg.speaker != paras[-1][-1].speaker):
            paras.append([seg])
        else:
            paras[-1].append(seg)
    return paras


def _para_text(p: list[Segment]) -> str:
    return " ".join(s.text.strip() for s in p).strip()


def _label(p: list[Segment], timestamps: bool) -> str:
    parts = []
    if p[0].speaker:
        parts.append(p[0].speaker)
    if timestamps:
        parts.append(fmt_ts(p[0].start))
    return " · ".join(parts)


def _meta(t: Transcript) -> list[tuple[str, str]]:
    return [
        ("Source", t.source_name),
        ("Language", f"{t.language} ({t.language_probability:.0%} confidence)" if t.language_probability else t.language),
        ("Duration", fmt_ts(t.duration)),
        ("Generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]


# ---------- TXT ----------
def to_txt(t: Transcript, timestamps: bool = True) -> str:
    lines = [f"Transcript: {t.source_name}", ""]
    for p in paragraphs(t.segments):
        lab = _label(p, timestamps)
        lines.append(f"[{lab}] {_para_text(p)}" if lab else _para_text(p))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------- Markdown ----------
def to_md(t: Transcript, timestamps: bool = True, summary: str = "") -> str:
    out = [f"# Transcript: {t.source_name}", ""]
    out += [f"- **{k}:** {v}" for k, v in _meta(t)]
    out += ["", "---", ""]
    if summary:
        import re
        demoted = re.sub(r"^#{1,6}\s+", "### ", summary.strip(), flags=re.M)
        out += ["## Summary", "", demoted, "", "## Transcript", ""]
    for p in paragraphs(t.segments):
        lab = _label(p, timestamps)
        out.append(f"**[{lab}]** {_para_text(p)}" if lab else _para_text(p))
        out.append("")
    return "\n".join(out).rstrip() + "\n"


# ---------- PDF ----------
# fpdf2's built-in fonts are Latin-1 only, so we load a Unicode TTF.
# A Devanagari font is added as a fallback so Hindi transcripts render too.
_LATIN_CANDIDATES = [
    HERE / "fonts" / "NotoSans-Regular.ttf",
    HERE / "fonts" / "DejaVuSans.ttf",
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
]
_DEVANAGARI_CANDIDATES = [
    HERE / "fonts" / "NotoSansDevanagari-Regular.ttf",
    Path("C:/Windows/Fonts/Nirmala.ttf"),
    Path("C:/Windows/Fonts/Nirmala.ttc"),
    Path("C:/Windows/Fonts/mangal.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansDevanagari-Regular.ttf"),
    Path("/System/Library/Fonts/Kohinoor.ttc"),
    Path("/System/Library/Fonts/Supplemental/ITFDevanagari.ttc"),
    Path("/System/Library/Fonts/Supplemental/DevanagariMT.ttc"),
]


def _first(paths: list[Path]) -> Path | None:
    return next((p for p in paths if p.exists()), None)


def _add(pdf: FPDF, family: str, path: Path) -> bool:
    try:
        kwargs = {"collection_font_number": 0} if path.suffix.lower() == ".ttc" else {}
        pdf.add_font(family, "", str(path), **kwargs)
        return True
    except Exception:
        return False


def to_pdf(t: Transcript, out_path: str, timestamps: bool = True, summary: str = "") -> str:
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 18, 18)

    family = "Helvetica"
    latin = _first(_LATIN_CANDIDATES)
    if latin and _add(pdf, "Body", latin):
        family = "Body"
        deva = _first(_DEVANAGARI_CANDIDATES)
        if deva and _add(pdf, "Deva", deva):
            pdf.set_fallback_fonts(["Deva"])
        elif any("\u0900" <= ch <= "\u097F" for ch in t.text):
            print("Warning: no Devanagari font found - Hindi text may be missing from the PDF. "
                  "Put NotoSansDevanagari-Regular.ttf in the fonts/ folder.")
        try:
            pdf.set_text_shaping(True)  # correct joining for Indic/Arabic scripts (needs uharfbuzz)
        except Exception:
            pass

    def safe(s: str) -> str:  # only needed if no Unicode font was found
        return s if family != "Helvetica" else s.encode("latin-1", "replace").decode("latin-1")

    pdf.add_page()
    pdf.set_font(family, size=18)
    pdf.multi_cell(0, 9, safe(f"Transcript: {t.source_name}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font(family, size=9)
    pdf.set_text_color(110, 110, 110)
    for k, v in _meta(t):
        pdf.cell(0, 5, safe(f"{k}: {v}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_draw_color(210, 210, 210)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(5)

    if summary:
        pdf.set_font(family, size=13)
        pdf.set_text_color(25, 25, 25)
        pdf.cell(0, 7, "Summary", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(family, size=10.5)
        for line in summary.strip().splitlines():
            line = line.strip().lstrip("#").replace("**", "").strip()
            if line:
                pdf.multi_cell(0, 5.5, safe(line.replace("- ", "• ", 1) if line.startswith("- ") else line),
                               new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.set_font(family, size=13)
        pdf.cell(0, 7, "Transcript", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    for p in paragraphs(t.segments):
        lab = _label(p, timestamps)
        if lab:
            pdf.set_font(family, size=9)
            pdf.set_text_color(15, 118, 110)
            pdf.cell(0, 5, safe(lab), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font(family, size=11)
        pdf.set_text_color(25, 25, 25)
        pdf.multi_cell(0, 6, safe(_para_text(p)), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    pdf.output(out_path)
    return out_path


# ---------- SRT subtitles ----------
def to_srt(t: Transcript) -> str:
    def ts(x: float) -> str:
        ms = int(round(x * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = []
    for i, seg in enumerate(t.segments, 1):
        text = f"{seg.speaker}: {seg.text.strip()}" if seg.speaker else seg.text.strip()
        blocks.append(f"{i}\n{ts(seg.start)} --> {ts(seg.end)}\n{text}\n")
    return "\n".join(blocks)


# ---------- one call for all formats ----------
def export(t: Transcript, formats: list[str], out_dir: str, timestamps: bool = True, summary: str = "") -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(t.source_name).stem or "transcript"
    files = []
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        path = os.path.join(out_dir, f"{stem}.{fmt}")
        if fmt == "txt":
            Path(path).write_text(to_txt(t, timestamps), encoding="utf-8")
        elif fmt == "md":
            Path(path).write_text(to_md(t, timestamps, summary), encoding="utf-8")
        elif fmt == "pdf":
            to_pdf(t, path, timestamps, summary)
        elif fmt == "srt":
            Path(path).write_text(to_srt(t), encoding="utf-8")
        else:
            raise ValueError(f"Unknown format: {fmt}")
        files.append(path)
    return files
