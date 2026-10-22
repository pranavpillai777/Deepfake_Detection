"""Presentation only: CSV timelines, cautious explanations and PDF charts.

Never used to compute an inference score or choose a session verdict.
"""
import csv
import math
from pathlib import Path

HIGHLIGHT_THRESHOLD = 0.80  # Existing peak-score cutoff; display only.
HIGHLIGHT_RULE = (
    "Red marks: boosted CSV score > 0.80 (display only). "
    "An individual high score is not the overall session verdict."
)
SCORE_CONVENTION = "CSV Fake_Prob: raw fake score multiplied by 1.5 and capped at 1; not a calibrated probability."


class TimelinePathError(ValueError):
    pass


def read_timeline(session_folder, filename=None):
    """Read accepted crops without inventing times, clamping or re-boosting scores."""
    folder = Path(session_folder).resolve()
    result = {"samples": [], "skipped_rows": 0, "warnings": [],
              "highlight_threshold": HIGHLIGHT_THRESHOLD,
              "highlight_rule": HIGHLIGHT_RULE, "score_convention": SCORE_CONVENTION}
    if filename is None:
        candidates = sorted(folder.glob("inference_log_*.csv"))
        if not candidates:
            result["warnings"].append("No inference CSV file was found.")
            return result
        filename = candidates[-1].name
    if not filename or Path(filename).name != filename or any(c in filename for c in "/\\:") or filename in {".", ".."}:
        raise TimelinePathError("CSV filename must be a simple filename inside the session folder.")
    path = folder / filename
    # Reject file links even when their target is in the same directory.
    if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()) or path.resolve().parent != folder:
        raise TimelinePathError("Timeline CSV must be a regular file inside the session folder, not a symlink.")
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source, strict=True)
            if not {"Elapsed_Sec", "Fake_Prob"}.issubset(reader.fieldnames or []):
                result["warnings"].append("CSV is missing Elapsed_Sec or Fake_Prob columns.")
                return result
            for row in reader:
                try:
                    elapsed, score = float(row["Elapsed_Sec"]), float(row["Fake_Prob"])
                    valid = math.isfinite(elapsed) and elapsed >= 0 and math.isfinite(score) and 0 <= score <= 1
                except (KeyError, TypeError, ValueError):
                    valid = False
                if not valid:
                    result["skipped_rows"] += 1
                    continue
                result["samples"].append({"elapsed_seconds": elapsed,
                                          "adjusted_fake_score": score,
                                          "timestamp": (row.get("Timestamp") or "").strip() or None,
                                          "suspicious": score > HIGHLIGHT_THRESHOLD})
    except (OSError, UnicodeError, csv.Error) as exc:
        result["samples"] = []
        result["warnings"].append(f"Timeline CSV could not be read ({type(exc).__name__}).")
    result["samples"].sort(key=lambda sample: sample["elapsed_seconds"])
    if result["skipped_rows"]:
        result["warnings"].append(f"Skipped {result['skipped_rows']} invalid row(s); missing times were not guessed.")
    return result


def explain_verdict(verdict, scores, counts):
    """Describe the stored verdict and statistics, without reclassifying them."""
    usable = counts.get("usable_crops_after_filters")
    details = []
    for key, label in (("frames_captured", "captured frames"),
                       ("frames_with_detection", "frames with a detection"),
                       ("usable_crops_after_filters", "usable crops after filters")):
        value = counts.get(key)
        if isinstance(value, int) and value >= 0:
            details.append(f"{value} {label}")
    evidence = ("Recorded evidence: " + ", ".join(details) + ". ") if details else ""
    if verdict == "Not enough evidence" or usable == 0:
        return (evidence + "No usable detections passed the filters, so there is not enough evidence "
                "to assess the selected region. This does not mean the video is real or fake. "
                "Try a clearer, larger, less obstructed view.")
    values = []
    for key, label in (("wci", "combined WCI"), ("average", "average"),
                       ("burst", "peak burst"), ("peak", "highest crop")):
        value = scores.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1:
            values.append(f"{label} {value:.3f}")
    statistics = ("Boosted-score statistics (0 to 1): " + ", ".join(values) + ". ") if values else ""
    if verdict == "SYNTHETIC / DEEPFAKE":
        meaning = "The existing session rules flagged these samples, supporting the displayed DEEPFAKE label. This is not proof of manipulation. "
    elif verdict == "AUTHENTIC / REAL":
        meaning = "The existing session rules did not flag these samples, giving the displayed REAL label. This does not prove authenticity; the model can miss manipulation. "
    else:
        meaning = "The stored verdict is not recognized; no interpretation is available. "
    return evidence + statistics + meaning + "These are detector/model outputs, not calibrated probabilities, and apply only to the sampled region."


def plot_samples(samples, limit=800):
    """Bound drawing size, retaining each time-ordered bucket's extrema and ends."""
    if len(samples) <= limit:
        return samples
    bucket_size = math.ceil(len(samples) / (limit // 4))
    selected = []
    for start in range(0, len(samples), bucket_size):
        end = min(start + bucket_size, len(samples))
        indexes = {start, end - 1,
                   min(range(start, end), key=lambda i: samples[i]["adjusted_fake_score"]),
                   max(range(start, end), key=lambda i: samples[i]["adjusted_fake_score"])}
        selected.extend(samples[i] for i in sorted(indexes))
    return selected


def build_timeline_drawing(data):
    """Vector PDF chart; consumes the same validated CSV payload as the API."""
    from reportlab.graphics.shapes import Drawing, String, Line, Circle, PolyLine
    from reportlab.lib import colors

    drawing = Drawing(460, 245)
    left, bottom, width, height = 45, 75, 395, 125
    ink, normal, high = [colors.HexColor(c) for c in ("#3c3554", "#6d4dbf", "#a83a50")]
    samples = data["samples"]
    max_time = max(1.0, max((s["elapsed_seconds"] for s in samples), default=0))
    divisor, unit = (60, "min") if max_time >= 120 else (1, "s")
    x = lambda elapsed: left + elapsed / max_time * width
    y = lambda score: bottom + score * height
    drawing.add(String(left, 230, "Adjusted fake score over elapsed time", fontName="Helvetica-Bold", fontSize=11, fillColor=ink))
    drawing.add(String(left, 212, "Boosted score (0 to 1); not a probability", fontSize=8, fillColor=ink))
    for tick in range(5):
        value = tick / 4
        drawing.add(Line(left, y(value), left + width, y(value), strokeColor=colors.HexColor("#e8e1f5")))
        drawing.add(String(left - 8, y(value) - 3, f"{value:.2f}", textAnchor="end", fontSize=8))
        elapsed = max_time * value
        drawing.add(String(x(elapsed), bottom - 14, f"{elapsed / divisor:.1f}", textAnchor="middle", fontSize=8))
    drawing.add(Line(left, bottom, left, bottom + height, strokeColor=ink))
    drawing.add(Line(left, bottom, left + width, bottom, strokeColor=ink))
    drawing.add(String(left + width / 2, 42, f"Elapsed time ({unit})", textAnchor="middle", fontSize=9))
    drawing.add(Line(left, y(HIGHLIGHT_THRESHOLD), left + width, y(HIGHLIGHT_THRESHOLD), strokeColor=high, strokeDashArray=[3, 3]))
    visible = plot_samples(samples)
    if len(visible) > 1:
        points = [coord for sample in visible for coord in (x(sample["elapsed_seconds"]), y(sample["adjusted_fake_score"]))]
        drawing.add(PolyLine(points, strokeColor=normal, strokeWidth=1, fillColor=None))
    for sample in visible:
        color = high if sample["adjusted_fake_score"] > HIGHLIGHT_THRESHOLD else normal
        drawing.add(Circle(x(sample["elapsed_seconds"]), y(sample["adjusted_fake_score"]), 2, fillColor=color, strokeColor=color))
    if not samples:
        drawing.add(String(left + width / 2, bottom + height / 2, "No usable timed score samples.", textAnchor="middle", fontSize=10))
    drawing.add(Circle(left, 22, 2, fillColor=normal, strokeColor=normal))
    drawing.add(String(left + 8, 19, "Recorded crops", fontSize=8))
    drawing.add(Circle(170, 22, 2, fillColor=high, strokeColor=high))
    drawing.add(String(178, 19, "High score > 0.80 (display only)", fontSize=8))
    if len(visible) < len(samples):
        drawing.add(String(left, 5, f"{len(samples)} samples; plot retains bucket extrema and endpoints.", fontSize=7))
    return drawing
