"""Real CSV/reporting/PDF code; no models, video or capture device."""
import ast
import io
from pathlib import Path
from unittest.mock import patch

import pytest
from reportlab.graphics.shapes import Circle, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, KeepTogether
from PIL import Image

from V6.reporting import (read_timeline, explain_verdict, plot_samples,
                          build_timeline_drawing, TimelinePathError, HIGHLIGHT_RULE)


def write_csv(folder, rows):
    path = folder / 'inference_log_test.csv'
    path.write_text('Timestamp,Elapsed_Sec,Fake_Prob\n' + rows, encoding='utf-8-sig')
    return path


def test_csv_missing_wall_clock_keeps_elapsed_and_does_not_boost(tmp_path):
    write_csv(tmp_path, ',2.5,0.8\nnow,1,0.81\n')
    data = read_timeline(tmp_path)
    assert data['samples'][0]['timestamp'] == 'now'
    assert data['samples'][0]['adjusted_fake_score'] == 0.81
    assert data['samples'][0]['suspicious'] is True
    assert data['samples'][1]['timestamp'] is None
    assert data['samples'][1]['adjusted_fake_score'] == 0.8
    assert data['samples'][1]['suspicious'] is False


@pytest.mark.parametrize('filename', ['../outside.csv', 'nested/file.csv', 'nested\\file.csv', 'C:outside.csv'])
def test_csv_rejects_path_names(tmp_path, filename):
    with pytest.raises(TimelinePathError):
        read_timeline(tmp_path, filename)


def test_csv_rejects_link_even_when_resolved_inside(tmp_path):
    path = write_csv(tmp_path, 'now,1,0.9\n')
    with patch.object(Path, 'is_symlink', return_value=True):
        with pytest.raises(TimelinePathError):
            read_timeline(tmp_path, path.name)


def test_csv_malformed_encoding_or_quoting_is_not_fatal(tmp_path):
    path = write_csv(tmp_path, '"unterminated,1,0.9\n')
    assert read_timeline(tmp_path)['warnings']
    path.write_bytes(b'\xffinvalid')
    data = read_timeline(tmp_path)
    assert data['samples'] == [] and data['warnings']


@pytest.mark.parametrize('verdict,usable,phrase', [
    ('AUTHENTIC / REAL', 6, 'does not prove authenticity'),
    ('SYNTHETIC / DEEPFAKE', 6, 'not proof of manipulation'),
    ('Not enough evidence', 0, 'No usable detections'),
])
def test_explanations_use_actual_counts_and_scores(verdict, usable, phrase):
    scores = {'average': 0.2, 'peak': 0.91, 'burst': 0.4, 'wci': 0.513}
    explanation = explain_verdict(verdict, scores, {
        'frames_captured': 12, 'frames_with_detection': 8,
        'usable_crops_after_filters': usable})
    assert phrase in explanation
    assert '12 captured frames' in explanation and f'{usable} usable crops' in explanation
    if usable:
        assert 'combined WCI 0.513' in explanation and 'highest crop 0.910' in explanation
        assert 'not calibrated probabilities' in explanation
    else:
        assert '0.513' not in explanation


def test_explanation_handles_unavailable_statistics():
    assert 'does not prove authenticity' in explain_verdict('AUTHENTIC / REAL', {}, {})


@pytest.mark.parametrize('rows', ['', 'now,1.25,0.8\n', 'early,1,0.1\nlate,3,0.81\n'])
@pytest.mark.parametrize('verdict', ['AUTHENTIC / REAL', 'SYNTHETIC / DEEPFAKE', 'Not enough evidence'])
def test_actual_runner_pdf_generation(tmp_path, rows, verdict):
    """Run the runner's actual PDF assembly block with real ReportLab and a test image."""
    import os
    import time
    from V6 import loggin_runner
    write_csv(tmp_path, rows)
    data = read_timeline(tmp_path)
    drawing = build_timeline_drawing(data)
    scores = {'average': 0.2, 'peak': 0.9, 'burst': 0.5, 'wci': 0.5}
    explanation = explain_verdict(verdict, scores, {'usable_crops_after_filters': len(data['samples'])})
    image = io.BytesIO()
    Image.new('RGB', (400, 200), 'white').save(image, format='PNG')
    image.seek(0)
    output = io.BytesIO()
    tree = ast.parse(Path(loggin_runner.__file__).read_text(encoding='utf-8'))
    run = next(n for n in tree.body if getattr(n, 'name', '') == 'run_session')
    body = next(n for n in run.body if isinstance(n, ast.Try)).body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'doc' for t in n.targets))
    end = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'build')
    ns = dict(colors=colors, letter=letter, Paragraph=Paragraph, Spacer=Spacer,
              RLImage=RLImage, Table=Table, TableStyle=TableStyle, KeepTogether=KeepTogether,
              getSampleStyleSheet=getSampleStyleSheet, ParagraphStyle=ParagraphStyle,
              SimpleDocTemplate=lambda *a, **kw: SimpleDocTemplate(*a, pageCompression=0, **kw),
              pdf_path=output, temp_gradcam_path=image, report_num='test', time=time, os=os,
              csv_log_path='inference_log_test.csv', total_elapsed=3.0, frame_count=12,
              actual_fps=4.0, overall_prediction=verdict, verdict_explanation=explanation,
              WCI=.5, avg_fake_prob=.2, max_burst_avg=.5, highest_fake_score=.9,
              timeline=drawing, timeline_data=data, HIGHLIGHT_RULE=HIGHLIGHT_RULE)
    exec(compile(ast.Module(body=body[start:end], type_ignores=[]), '<runner PDF assembly>', 'exec'), ns)
    # ReportLab consumes the flowables list during build, so inspect it first.
    paragraphs = [child for e in ns['elements'] for child in (e._content if isinstance(e, KeepTogether) else [e]) if isinstance(child, Paragraph)]
    text = ' '.join(e.getPlainText() for e in paragraphs)
    assert 'not a calibrated' in text
    exec(compile(ast.Module(body=[body[end]], type_ignores=[]), '<runner PDF build>', 'exec'), ns)
    pdf = output.getvalue()
    assert pdf.startswith(b'%PDF') and b'%%EOF' in pdf
    assert b'Verdict Explanation' in pdf and b'Score Timeline' in pdf
    assert b'Elapsed time' in pdf and b'High score > 0.80' in pdf
    assert b'display only' in pdf and b'not a probability' in pdf


def test_pdf_highlight_boundary_and_numeric_ticks(tmp_path):
    write_csv(tmp_path, 'a,0,0.8\nb,120,0.81\n')
    drawing = build_timeline_drawing(read_timeline(tmp_path))
    strings = [n.text for n in drawing.contents if isinstance(n, String)]
    assert 'Elapsed time (min)' in strings and '2.0' in strings
    # Two crop markers followed by two legend markers.
    dots = [n for n in drawing.contents if isinstance(n, Circle)]
    assert dots[0].fillColor == colors.HexColor('#6d4dbf')
    assert dots[1].fillColor == colors.HexColor('#a83a50')


def test_large_plot_retains_peaks_and_bounds():
    samples = [{'elapsed_seconds': i, 'adjusted_fake_score': 0.1} for i in range(200000)]
    samples[12345]['adjusted_fake_score'] = 0.99
    visible = plot_samples(samples)
    assert len(visible) <= 800
    assert visible[0] == samples[0] and visible[-1] == samples[-1]
    assert samples[12345] in visible
