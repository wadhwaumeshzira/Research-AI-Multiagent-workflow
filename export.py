"""
export.py — STAGE 6: Export to Markdown / PDF

Saves the final report as a .md file (always works, zero dependencies) and
optionally as a formatted .pdf (via reportlab).
"""

import os
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable

OUTPUT_DIR = "reports"

NAVY = HexColor("#0F172A")
BLUE = HexColor("#2563EB")
BODY = HexColor("#334155")
MUTED = HexColor("#94A3B8")


def _safe_filename(topic: str) -> str:
    safe = "".join(c for c in topic if c.isalnum() or c in " -_").strip()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{safe[:40]}_{ts}"


def export_markdown(topic: str, pipeline_result: dict, confidence: dict = None) -> str:
    """Writes the report to a .md file. Returns the file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = _safe_filename(topic) + ".md"
    path = os.path.join(OUTPUT_DIR, filename)

    lines = [pipeline_result["report"]]

    if confidence:
        lines.append("\n---\n")
        lines.append(f"**Confidence score:** {confidence['score']}/100 ({confidence['label']})")
        if confidence["reasons"]:
            lines.append("\n**Notes:**")
            for r in confidence["reasons"]:
                lines.append(f"- {r}")

    lines.append(f"\n\n*Generated {datetime.now().strftime('%d %B %Y, %I:%M %p')} — "
                  f"{pipeline_result['critic_verdict']['verdict']}, "
                  f"{pipeline_result['retries_used']} retry round(s)*")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[Export] Markdown saved: {path}")
    return path


def _strip_emojis(text: str) -> str:
    return re.sub(r"[^\x00-\x7F]+", "", text).strip()


def export_pdf(topic: str, pipeline_result: dict, confidence: dict = None) -> str:
    """Writes the report to a formatted .pdf file. Returns the file path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = _safe_filename(topic) + ".pdf"
    path = os.path.join(OUTPUT_DIR, filename)

    doc = SimpleDocTemplate(path, pagesize=A4,
                             leftMargin=1.5 * cm, rightMargin=1.5 * cm,
                             topMargin=1.5 * cm, bottomMargin=1.5 * cm)

    title_style = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=18,
                                  textColor=NAVY, spaceAfter=6, alignment=TA_CENTER)
    meta_style = ParagraphStyle("Meta", fontName="Helvetica", fontSize=9,
                                 textColor=MUTED, spaceAfter=14, alignment=TA_CENTER)
    h2_style = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=13,
                               textColor=BLUE, spaceBefore=14, spaceAfter=6)
    body_style = ParagraphStyle("Body", fontName="Helvetica", fontSize=10,
                                 textColor=BODY, leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
    bullet_style = ParagraphStyle("Bullet", fontName="Helvetica", fontSize=10,
                                   textColor=BODY, leading=14, leftIndent=14, spaceAfter=3)

    story = [
        Paragraph(_strip_emojis(topic), title_style),
        Paragraph(datetime.now().strftime("%d %B %Y, %I:%M %p"), meta_style),
        HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10),
    ]

    for raw_line in pipeline_result["report"].split("\n"):
        line = _strip_emojis(raw_line).strip()
        if not line:
            story.append(Spacer(1, 4))
            continue
        if line.startswith("## "):
            continue  # skip the redundant "Research Report: X" line, we already have the title
        elif line.startswith("### "):
            story.append(Paragraph(line[4:].strip(), h2_style))
        elif line.startswith("* ") or line.startswith("- "):
            text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line[2:].strip())
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{text}", bullet_style))
        else:
            text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", line)
            story.append(Paragraph(text, body_style))

    if confidence:
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED, spaceAfter=8))
        story.append(Paragraph(f"Confidence score: {confidence['score']}/100 ({confidence['label']})", h2_style))
        for r in confidence["reasons"]:
            story.append(Paragraph(f"&bull;&nbsp;&nbsp;{r}", bullet_style))

    doc.build(story)
    print(f"[Export] PDF saved: {path}")
    return path


if __name__ == "__main__":
    from orchestrator import run_pipeline
    from confidence import score_confidence

    topic = "Impact of Artificial Intelligence on jobs in 2026"
    result = run_pipeline(topic)
    conf = score_confidence(result)

    export_markdown(topic, result, conf)
    export_pdf(topic, result, conf)
