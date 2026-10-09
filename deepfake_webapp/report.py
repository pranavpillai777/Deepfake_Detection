"""PDF report (adapted from run_live_analysis.py)."""
import time
from io import BytesIO
from xml.sax.saxutils import escape

import cv2
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

VERDICT_TEXT = {
    "synthetic": "Likely synthetic (deepfake)",
    "authentic": "Likely authentic",
    "inconclusive": "Inconclusive",
}


def build_pdf(path, job_id, filename, result, gradcam_bgr):
    styles = getSampleStyleSheet()
    title = ParagraphStyle("t", parent=styles["Heading1"], fontSize=18,
                           textColor=colors.HexColor("#1a237e"), spaceAfter=12)
    heading = ParagraphStyle("h", parent=styles["Heading2"], fontSize=13,
                             textColor=colors.HexColor("#283593"), spaceAfter=6)
    normal = styles["Normal"]

    pct = lambda v: f"{v * 100:.2f}%"
    rows = [
        ["Metric", "Value"],
        ["File", filename[:60]],
        ["Video duration", f"{result['duration']:.2f} s"],
        ["Frames sampled", f"{result['frames_sampled']} (at {result['sample_fps']} per second)"],
        ["Faces evaluated", str(result["faces_evaluated"])],
        ["Overall verdict", VERDICT_TEXT[result["verdict"]]],
        ["Weighted Confidence Index (WCI)", pct(result["wci"])],
        ["Session average score", pct(result["avg"])],
        ["Peak 1s burst window score", pct(result["max_burst"])],
        ["Peak frame score", f"{pct(result['peak'])} at {result['peak_time']:.1f} s"],
    ]
    table = Table(rows, colWidths=[200, 250])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#3f51b5")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5f5f5")),
        ("GRID", (0, 0), (-1, -1), 1, colors.HexColor("#e0e0e0")),
    ]))

    els = [
        Paragraph(f"Deepfake Detection Report {escape(job_id[:8])}", title),
        Paragraph(f"<b>Generated:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", normal),
        Spacer(1, 10), table, Spacer(1, 15),
    ]
    if result.get("reason"):
        els += [Paragraph(f"<b>Why inconclusive:</b> {escape(result['reason'])}", normal), Spacer(1, 10)]

    if gradcam_bgr is not None:
        ok, buf = cv2.imencode(".png", gradcam_bgr)
        if ok:
            els += [
                Paragraph("Explainable AI (Grad-CAM) analysis", heading),
                Paragraph("Left: frame with the highest fake score. Right: Grad-CAM heatmap.", normal),
                Spacer(1, 8),
                RLImage(BytesIO(buf.tobytes()), width=380, height=190),
                Spacer(1, 12),
            ]

    els.append(Paragraph(
        "<b>Note:</b> This is a statistical estimate from a single model, not proof. "
        "Compression, lighting and camera angle can cause wrong results.", normal))
    SimpleDocTemplate(str(path), pagesize=letter).build(els)
