"""API integration tests using a real child process with no desktop/model access."""
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from html.parser import HTMLParser
import re
import shutil
import subprocess

import pytest
from fastapi.testclient import TestClient

from V6.app import create_app


FAKE_RUNNER = r'''
import argparse, json, os, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
for name in ['session-id', 'report-dir', 'checkpoint', 'detector']:
    p.add_argument('--' + name, required=True)
args = p.parse_args()
if MODE == 'no_status':
    sys.exit(9)
# Deliberately delay folder creation to exercise Stop immediately after Start.
time.sleep(0.15)
folder = Path(args.report_dir) / args.session_id
folder.mkdir()
def write(name, data):
    temp = folder / (name + '.tmp')
    temp.write_text(json.dumps(data), encoding='utf-8')
    # Rapid fake-stage transitions can meet a transient Windows file lock.
    # Retry replacement briefly while keeping publication atomic.
    for attempt in range(20):
        try:
            os.replace(temp, folder / name)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.01)
def status(stage):
    write('status.json', {'session_id': args.session_id, 'status': stage,
                        'timestamp': '2026-10-22T00:00:00+00:00'})
status('capturing')
if MODE == 'crash':
    sys.exit(7)
deadline = time.monotonic() + 15
while not (folder / 'stop').exists():
    if time.monotonic() > deadline:
        sys.exit(8)
    time.sleep(0.01)
status('analyzing')
(folder / 'report.pdf').write_bytes(b'%PDF-1.4\nfake report for API tests\n')
write('result.json', {'session_id': args.session_id, 'verdict': 'AUTHENTIC / REAL',
                     'scores': {'wci': 0.1}, 'report_path': str(folder / 'report.pdf'),
                     'python': sys.executable})
status('done')
'''


def wait_for(client, predicate):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        response = client.get('/api/live/status')
        assert response.status_code == 200, response.text
        state = response.json()
        if predicate(state):
            return state
        time.sleep(0.02)
    pytest.fail(f'Timed out; last state: {state}')


@pytest.fixture
def backend(tmp_path):
    @contextmanager
    def factory(mode='normal'):
        work = tmp_path / mode
        work.mkdir()
        runner = work / 'fake_runner.py'
        runner.write_text(f'MODE = {mode!r}\n' + FAKE_RUNNER, encoding='utf-8')
        checkpoint = work / 'classifier.pth'
        detector = work / 'detector.pt'
        checkpoint.write_bytes(b'fake checkpoint')
        detector.write_bytes(b'fake detector')
        reports = work / 'reports'
        api = create_app(report_dir=reports, runner_path=runner,
                         checkpoint_path=checkpoint, detector_path=detector)
        with TestClient(api, base_url='http://127.0.0.1') as client:
            yield client, reports
    return factory


def test_start_stop_result_and_report(backend):
    with backend() as (client, reports):
        assert client.get('/api/live/status').json()['status'] == 'idle'
        response = client.post('/api/live/start')
        assert response.status_code == 202
        session_id = response.json()['session_id']
        capturing = wait_for(client, lambda s: s['status'] == 'capturing')
        assert capturing['process_alive'] and capturing['stage'] == 'capturing'
        assert client.get(f'/api/live/result/{session_id}').status_code == 409
        assert client.get(f'/api/live/report/{session_id}').status_code == 409
        assert client.post('/api/live/stop').status_code == 200
        assert client.post('/api/live/stop').status_code == 200
        done = wait_for(client, lambda s: s['status'] == 'done' and not s['process_alive'])
        assert done['exit_code'] == 0
        assert (reports / session_id / 'stop').is_file()
        result = client.get(f'/api/live/result/{session_id}')
        assert result.status_code == 200
        assert Path(result.json()['python']).resolve() == Path(sys.executable).resolve()
        pdf = client.get(f'/api/live/report/{session_id}')
        assert pdf.status_code == 200 and pdf.content.startswith(b'%PDF')
        assert pdf.headers['content-type'] == 'application/pdf'
        assert 'report.pdf' in pdf.headers['content-disposition']
        assert client.post('/api/live/stop').status_code == 200
        assert client.post('/api/analyze').status_code == 404
        assert client.get('/').status_code == 200


def test_duplicate_start_and_restart_after_completion(backend):
    with backend() as (client, _):
        first = client.post('/api/live/start').json()['session_id']
        duplicate = client.post('/api/live/start')
        assert duplicate.status_code == 409
        assert duplicate.json()['detail'] == 'An analysis is already running'
        client.post('/api/live/stop')
        wait_for(client, lambda s: not s['process_alive'])
        second = client.post('/api/live/start')
        assert second.status_code == 202 and second.json()['session_id'] != first
        client.post('/api/live/stop')
        wait_for(client, lambda s: not s['process_alive'])


def test_stop_before_runner_creates_session_folder(backend):
    with backend() as (client, reports):
        session_id = client.post('/api/live/start').json()['session_id']
        assert client.post('/api/live/stop').status_code == 200
        state = wait_for(client, lambda s: s['status'] == 'done' and not s['process_alive'])
        assert state['stop_requested']
        assert (reports / session_id / 'stop').exists()


@pytest.mark.parametrize('mode,code', [('crash', 7), ('no_status', 9)])
def test_process_death_becomes_error(backend, mode, code):
    with backend(mode) as (client, _):
        session_id = client.post('/api/live/start').json()['session_id']
        state = wait_for(client, lambda s: not s['process_alive'])
        assert state['status'] == 'error' and state['stage'] == 'error'
        assert state['exit_code'] == code
        assert 'before writing a final status' in state['message']
        assert client.get(f'/api/live/result/{session_id}').status_code == 409


@pytest.mark.parametrize('bad_id', ['bad.id', 'bad_id', 'bad%2Fid', 'bad%5Cid', '%2E%2E%2Foutside', 'bad%20id', 'bad%3Aid'])
@pytest.mark.parametrize('endpoint', ['result', 'report'])
def test_reject_bad_session_ids(backend, bad_id, endpoint):
    with backend() as (client, _):
        assert client.get(f'/api/live/{endpoint}/{bad_id}').status_code == 400


def test_report_uses_fixed_filename_not_result_path(backend, tmp_path):
    with backend() as (client, reports):
        session_id = client.post('/api/live/start').json()['session_id']
        client.post('/api/live/stop')
        wait_for(client, lambda s: s['status'] == 'done' and not s['process_alive'])
        outside = tmp_path / 'outside.pdf'
        outside.write_bytes(b'secret outside reports')
        result_path = reports / session_id / 'result.json'
        data = json.loads(result_path.read_text())
        data['report_path'] = str(outside)
        result_path.write_text(json.dumps(data))
        pdf = client.get(f'/api/live/report/{session_id}')
        assert pdf.content.startswith(b'%PDF') and b'secret' not in pdf.content


@pytest.mark.parametrize('missing', ['checkpoint_path', 'detector_path'])
def test_startup_checks_models(tmp_path, missing):
    model = tmp_path / 'exists.pt'
    model.write_bytes(b'fake weights')
    options = dict(checkpoint_path=model, detector_path=model,
                   report_dir=tmp_path / 'reports')
    options[missing] = tmp_path / 'missing.pt'
    with pytest.raises(RuntimeError, match='not found'):
        with TestClient(create_app(**options)):
            pass


class PageElements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


def test_live_page_has_controls_without_uploads_or_external_resources(backend):
    with backend() as (client, _):
        response = client.get('/')
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('text/html')
        html = response.text
        page = PageElements()
        page.feed(html)
        buttons = {attrs.get('id') for tag, attrs in page.elements if tag == 'button'}
        assert {'start', 'stop'} <= buttons
        assert not any(tag == 'input' and attrs.get('type') == 'file' for tag, attrs in page.elements)
        assert not any(attrs.get('id') in {'upload', 'drop', 'file'} for _, attrs in page.elements)
        assert 'Upload a clip' not in html and 'screen-recording' not in html
        assert 'Made by Aditi Verma, Pranav Pillai and Shreya Thapa' in html
        assert '<title>Real-Time Deepfake Detection &amp; Diagnostic Reporting System</title>' in html
        assert 'Backend not running' in html and 'start_local.ps1' in html
        assert all('http' not in attrs.get('src', '') + attrs.get('href', '')
                   for _, attrs in page.elements)
        css = client.get('/fonts/fonts.css')
        assert css.status_code == 200 and 'https://' not in css.text
        for font in re.findall(r'url\(([^)]+)\)', css.text):
            font_response = client.get('/fonts/' + font)
            assert font_response.status_code == 200 and font_response.content
        # The same controls remain available on reload during a fake session.
        session_id = client.post('/api/live/start').json()['session_id']
        wait_for(client, lambda s: s['status'] == 'capturing')
        assert client.get('/').status_code == 200
        assert client.get('/api/live/status').json()['session_id'] == session_id
        client.post('/api/live/stop')
        wait_for(client, lambda s: not s['process_alive'])


def test_page_javascript_behaviour():
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed for the dependency-free page-script checks.')
    response = subprocess.run([node, str(Path(__file__).with_name('test_page.js'))],
                              capture_output=True, text=True, timeout=20)
    assert response.returncode == 0, response.stdout + response.stderr


@pytest.mark.parametrize('csv_text,expected,skipped', [
    ('Timestamp,Elapsed_Sec,Fake_Prob\n', [], 0),
    ('Timestamp,Elapsed_Sec,Fake_Prob\nnow,1.25,0.8\n', [(1.25, 0.8)], 0),
    ('Timestamp,Elapsed_Sec,Fake_Prob\nlater,3,0.9\nearlier,1,0.2\n', [(1, 0.2), (3, 0.9)], 0),
    ('Elapsed_Sec,Fake_Prob\n-1,0.9\nnan,0.3\n1,inf\n,0.8\n2,1.1\n3,-0.1\n4,0.81\n', [(4, 0.81)], 6),
    ('Timestamp,Fake_Prob\nnow,0.9\n', [], 0),
])
def test_timeline_validates_csv(backend, csv_text, expected, skipped):
    with backend() as (client, reports):
        folder = reports / 'csv-review'
        folder.mkdir()
        (folder / 'status.json').write_text(json.dumps({'status': 'done'}))
        (folder / 'inference_log_test.csv').write_text(csv_text, encoding='utf-8-sig')
        response = client.get('/api/live/timeline/csv-review')
        assert response.status_code == 200
        data = response.json()
        assert [(s['elapsed_seconds'], s['adjusted_fake_score']) for s in data['samples']] == expected
        assert data['skipped_rows'] == skipped
        assert data['highlight_threshold'] == 0.80
        assert [s['suspicious'] for s in data['samples']] == [score > 0.80 for _, score in expected]


def test_timeline_unreadable_and_missing_csv(backend):
    with backend() as (client, reports):
        folder = reports / 'csv-review'
        folder.mkdir()
        (folder / 'status.json').write_text(json.dumps({'status': 'done'}))
        assert client.get('/api/live/timeline/csv-review').json()['samples'] == []
        (folder / 'inference_log_test.csv').write_bytes(b'\xff\xfeinvalid')
        response = client.get('/api/live/timeline/csv-review')
        assert response.status_code == 200
        assert response.json()['samples'] == []
        assert 'could not be read' in response.json()['warnings'][0]


def test_timeline_rejects_csv_symlink(backend, tmp_path):
    with backend() as (client, reports):
        folder = reports / 'csv-review'
        folder.mkdir()
        (folder / 'status.json').write_text(json.dumps({'status': 'done'}))
        outside = tmp_path / 'outside.csv'
        outside.write_text('Elapsed_Sec,Fake_Prob\n1,0.99\n')
        try:
            (folder / 'inference_log_test.csv').symlink_to(outside)
        except OSError as exc:
            pytest.skip(f'Windows symlink permission unavailable: {exc}')
        assert client.get('/api/live/timeline/csv-review').status_code == 400


@pytest.mark.parametrize('bad_id', ['bad.id', 'bad_id', 'bad%2Fid', 'bad%5Cid'])
def test_timeline_rejects_bad_session_id(backend, bad_id):
    with backend() as (client, _):
        assert client.get('/api/live/timeline/' + bad_id).status_code == 400
