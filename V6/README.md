# V6 — Real-Time Deepfake Detection & Diagnostic Reporting System

Made by Aditi Verma, Pranav Pillai and Shreya Thapa.

## What V6 does

V6 lets you control a live screen-analysis session from a local webpage. Click **Start**, select the video area on your screen, play the video, then click **Stop**. The page shows **REAL**, **DEEPFAKE**, or **Not enough evidence**, along with a cautious explanation, scores, counts, a score timeline and a PDF download.

The team reported incorrect verdicts from the old upload webpage and identified `V5/loggin.py` as its preferred pipeline. The code confirms that the old webpage used its own analysis implementation. V6 instead uses `loggin_runner.py`, based on the **committed version of `V5/loggin.py`**, with the requested configurable paths, controls and output files. Its default detector is the team's specified `checkpoints/yolov8n.pt`.

The runner preserves that committed script's detection call, crop, filters, 1.5× fake-score boost, WCI weights and verdict rule. Zero usable detections now produce **Not enough evidence**. This does not establish that pipeline differences were the only cause of the old webpage's incorrect verdicts.

V5 and the old `deepfake_webapp` remain separate. The V6 backend launches **`V6/loggin_runner.py`**, not the earlier `V6/run_live_analysis.py`.

## Requirements

- A Windows laptop with a visible, signed-in desktop.
- The backend must run on the **same computer as the screen being captured**. This is local desktop capture, not capture of a remote website visitor's screen.
- A Python environment with the packages in `V6/requirements.txt`. The same Python runs both the backend and its runner process.
- The two model files listed below, and a browser for the local page.
- Node.js only for the page-script tests. The webpage itself does not need Node.js or external JavaScript libraries.

Python **3.13.7** was used in the recorded checks. The runner chooses CUDA when available, otherwise CPU; no specific GPU performance is claimed.

**TODO:** Verify a clean installation of all requirements in a new environment. Previous checks used an existing Python environment plus an isolated backend test environment; they were not a fresh-machine setup test.

## Setup

Open PowerShell in the repository's main folder, the folder containing `V5`, `V6` and `checkpoints`. Use the Python environment you intend to run the analysis with.

```powershell
python -m pip install -r .\V6\requirements.txt
```

The requirements include FastAPI and Uvicorn for the backend; NumPy, OpenCV, PyWavelets, PyTorch, Torchvision, DXCam, Ultralytics and ReportLab for the runner; and pytest and httpx for tests.

Place the model files here:

```text
Deepfake_Detection/
  checkpoints/
    mobilenet_v2_deepfake.pth   # trained deepfake classifier
    yolov8n.pt                 # standard YOLO detector
  V6/
    app.py
    loggin_runner.py
    reporting.py                # shared presentation/CSV validation only
    start_local.ps1
    requirements.txt
    static/
    tests/
```

The backend checks that both files exist at startup. The runner loads them when an analysis starts, prints their paths and SHA-256 fingerprints, and saves those details in `result.json`. Missing or failed model loading produces an error; it does not silently switch to another detector or untrained classifier weights.

Optional settings, applied before starting the backend:

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `CHECKPOINT_PATH` | `<project>/checkpoints/mobilenet_v2_deepfake.pth` | Classifier weights |
| `YOLO_WEIGHTS` | `<project>/checkpoints/yolov8n.pt` | Detector weights; the runner requires a local `.pt` file |
| `REPORT_DIR` | `<project>/V6/reports` | Parent folder for session outputs |

For example, to change the output folder:

```powershell
$env:REPORT_DIR = "D:\DeepfakeReports"
```

## Start the backend

From the repository's main folder:

```powershell
.\V6\start_local.ps1
```

The script uses `python` by default. To choose a particular environment:

```powershell
.\V6\start_local.ps1 -Python "C:\path\to\your\environment\Scripts\python.exe"
```

An isolated environment used for the earlier checks is available on this laptop under `V6/reports/backend-test-env`. If using that existing environment:

```powershell
.\V6\start_local.ps1 -Python ".\V6\reports\backend-test-env\Scripts\python.exe"
```

This environment is local setup data, not something to assume exists in another checkout.

### If PowerShell blocks the script

Run it with a policy override for this PowerShell invocation only:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\V6\start_local.ps1 -Python python
```

**TODO:** Verify this policy workaround on the target laptop. It was not exercised during the recorded checks, and an organization-managed policy may still prevent it.

Alternatively, launch the same server directly without running a PowerShell script:

```powershell
python -m uvicorn app:app --app-dir .\V6 --host 127.0.0.1 --port 8000 --workers 1
```

Keep the terminal open. The launcher runs one worker on `127.0.0.1:8000`, outside Docker. Run it in your normal desktop session so the region-selection window can appear. Open [http://127.0.0.1:8000](http://127.0.0.1:8000), not the HTML file directly.

The launcher's optional `-Port` argument changes the port; use that port in the browser URL too.

## Use the page

1. Put the video on one side of your screen and the webpage in a small, **separate window** on the other side. Keep the webpage outside the rectangle you will capture.
2. Click **Start**. Start is disabled while the session runs.
3. When the countdown appears, switch to the video. The countdown lasts 10 seconds after initialization.
4. In the selection window, drag a box around the face/video area and press **Enter**.
5. Play the video. Keep its window in the same position and avoid covering the selected area.
6. Return to the page and click **Stop**. Repeating Stop is harmless. Keeping the windows side by side prevents this click from covering the video.
7. Wait through **Analyzing**, when the runner combines scores and builds the report.
8. Read the verdict, scores and frame counts, then click **Download report**.

Reloading the webpage can recover the current session from the still-running backend. It does not start a new runner. Errors and cancellation show a message. If the server cannot be reached, the page shows **Backend not running** and a start command.

Stop during the countdown cancels before the selection window opens. Stop during region selection is not handled immediately: finish the selection and press Enter so the runner can reach its capture-loop stop check. An empty selection falls back to full-screen capture; cancelling the selector does not cancel the whole session.

There is no automatic video-end stop or capture-duration timer. Click Stop yourself. Wait for **Done** before closing the backend. Backend shutdown requests Stop but may terminate an unfinished runner after five seconds, so it is not a substitute for completing the report normally.

## Explanations and score timelines

The explanation describes the stored verdict using available frame/detection/usable-crop counts and average, peak, burst and WCI scores. It does not recalculate the verdict. REAL means the existing rules did not flag the sampled region, not proof of authenticity; DEEPFAKE means the rules flagged the samples, not proof of manipulation. Zero usable detections means there is not enough evidence to assess the region.

The webpage and PDF read the existing inference CSV through `reporting.py`. The CSV columns are `Timestamp`, `Elapsed_Sec`, `Box_X`, `Box_Y`, `Box_W`, `Box_H`, `Blur_Variance`, `Real_Prob`, and `Fake_Prob`.

- The chart's horizontal axis uses `Elapsed_Sec` (seconds since capture began), shown in seconds or minutes. Its vertical axis uses the recorded `Fake_Prob` unchanged: the pipeline has already multiplied this score by 1.5 and capped it at 1. It is not a calibrated probability.
- **Display-only rule:** a recorded boosted score **strictly greater than 0.80** is highlighted in red. Exactly 0.80 is not highlighted. This uses the existing peak cutoff as a visual reference; it does not change WCI, scores, thresholds or the verdict. A high individual moment can occur without the overall session being labelled DEEPFAKE.
- Purple markers are other recorded crops. The dashed line marks 0.80. Both charts include a legend and numeric time ticks. On a small webpage window, scroll the chart horizontally if needed.
- Missing wall-clock `Timestamp` values do not prevent plotting when elapsed time exists. Missing, negative or non-finite elapsed times and scores outside 0 to 1 are skipped, not guessed or clamped. Valid records are sorted by elapsed time. Skipped-row counts and warnings are reported.
- Both readers reject CSV path escapes and file symlinks. Missing files, invalid encoding or malformed CSV produce an empty chart with a warning, without reclassifying the session. A chart with no valid timed records does not necessarily mean there were no usable detections: its time data may be unavailable.
- Empty timelines show an explicit message and reset their labels; one-point timelines show a marker. Long charts draw at most 800 representative markers, retaining each ordered bucket's endpoints, minimum and maximum, so isolated high scores remain visible. The API still returns all valid records and the recorded scores are unchanged. Lines between points are visual guides, not evidence about rejected or uncaptured frames.

The explanation and timeline are included in newly generated PDFs. Existing PDFs/results are not rewritten automatically. Older results without an explanation retain the page's fallback summary.

## How the pieces connect

```text
Webpage: Start
    -> backend: POST /api/live/start
    -> separate Python process: V6/loggin_runner.py
    -> countdown, region selection, live capture and scoring

Webpage: Stop
    -> backend: POST /api/live/stop
    -> creates <report folder>/<session ID>/stop
    -> runner checks between frames and ends capture
    -> runner builds PDF and result.json, then writes status "done"
    -> webpage reads the result and offers the PDF download
```

The backend remembers a Stop request made before the runner creates its session folder. The page checks status about once per second. The backend also reports whether the runner is alive; an unexpected exit without a final status becomes an error.

| Endpoint | What it does |
| --- | --- |
| `POST /api/live/start` | Starts one session and returns its ID |
| `POST /api/live/stop` | Requests Stop for the current session |
| `GET /api/live/status` | Returns current stage, message, timestamp and process state |
| `GET /api/live/result/{session_id}` | Returns the result after status is `done` |
| `GET /api/live/timeline/{session_id}` | Returns valid CSV samples, original wall-clock timestamps when present, skipped-row warnings and highlighting metadata after status is `done` |
| `GET /api/live/report/{session_id}` | Downloads that session's fixed `report.pdf` |

The runner writes `starting`, `countdown`, `selecting_region`, `capturing`, `analyzing`, then `done`, `error` or `cancelled`. The backend reports `idle` before any session. Status and result JSON files are written under a temporary name and then renamed, so readers do not receive a partly written JSON file.

## Saved files

By default, each session has a separate folder:

```text
V6/reports/<session ID>/
  status.json
  inference_log_<timestamp>.csv
  report.pdf
  result.json
  stop                         # present if Stop was requested through the backend
```

| File | Contents |
| --- | --- |
| `status.json` | Current stage, session ID and UTC timestamp; a message on errors or cancellation |
| `inference_log_<timestamp>.csv` | One row per usable crop: time, elapsed seconds, crop coordinates, blur measurement, raw real probability and boosted fake score |
| `report.pdf` | Verdict explanation, score timeline and legend, duration, frame count, FPS, scores and the peak-crop/Grad-CAM comparison; a placeholder image when no usable crop exists |
| `result.json` | Verdict and its explanation, average/peak/burst/WCI scores, report and CSV paths, both model paths and hashes, and debug capture/count information |
| `stop` | A control signal; its existence asks the runner to finish, and its contents are not used |

`result.json` is published after the PDF is built. Early cancellation or failure may leave only some files; a completed PDF/result is not promised for failed sessions. A temporary Grad-CAM JPG is normally removed after successful PDF creation.

Counts describe frames and accepted crops, not unique people. The CSV's `Fake_Prob` is multiplied by 1.5 and capped at 1; `Real_Prob` remains raw, so these two columns need not add up to 1. With zero usable detections, raw JSON scores retain the old negative sentinel values, explained in its debug note; the page shows **Not enough evidence** and dashes for scores.

## Known limits

- Only one runner at a time per backend process. Use the provided one-worker launcher and one backend instance.
- The default detector is **`yolov8n.pt`, not a face-specific model**. The detection call does not restrict classes. It uses the **first returned detection**, not necessarily a face, the largest box or the same person across frames.
- The selected rectangle stays fixed. Moving or covering the video changes what is captured. There is no automatic removal of the final few seconds after switching windows.
- The preserved pipeline blacks out part of the bottom-left region, pads the selected detection, and applies blur and shape filters before scoring. Those choices can exclude crops.
- The burst score groups usable scores using measured processing FPS. Gaps caused by rejected crops mean the reported burst is not necessarily exactly one second of video.
- Reload recovery applies to the page, not a backend restart. The backend keeps its current process/session reference in memory; there is no session-history page or automatic reattachment after server restart.
- The page and its fonts are served locally. Package installation needs the packages to be available, and the source video may itself need a network connection.
- A model verdict is an estimate, not proof of authenticity or manipulation. No new accuracy percentage is claimed for V6.
- Transient Windows file locks were observed during fake-runner status replacement. The fake test runner retries these writes; the production runner has not been changed to add such retries.

## Tests and verification

From the repository's main folder, with requirements installed:

```powershell
$testFolder = ".\V6\reports\pytest-" + (Get-Date -Format "yyyyMMdd-HHmmss")
python -m pytest V6/tests -q -p no:cacheprovider --basetemp $testFolder
```

The explicit test folder keeps temporary outputs under `V6/reports`. To use the existing local test environment, replace `python` with `.\V6\reports\backend-test-env\Scripts\python.exe`.

For just the page's JavaScript behavior checks:

```powershell
node .\V6\tests\test_page.js
```

Without Node.js, pytest skips that JavaScript check. The backend tests use a **fake runner**, fake model files and a dummy PDF, so they do not start screen capture or measure detection accuracy. Tests cover start/stop, repeated and early Stop, duplicate Start, unexpected process death, invalid session IDs, report-path restrictions, startup checks, page serving, upload removal, local fonts and page behavior.

The pre-follow-up suite passed **24 tests**, with one dependency deprecation warning. Publication-readiness tests now also cover shared CSV validation, invalid timestamps/encoding, symlink rejection, statistical explanations, chart highlighting, large timelines and actual ReportLab PDF assembly using synthetic data. The latest follow-up run recorded **56 passed, 1 skipped, 1 dependency warning**; the skipped test requires Windows permission to create a real symlink. A separate deterministic symlink-rejection check passed. The standalone frontend checks also passed with 200,000 samples. These are reporting/integration checks, not model-accuracy tests. Browser checks using the fake runner verified Start/Stop, reload recovery, error recovery, displayed scores/counts and the backend-offline message. Earlier checks loaded the actual configured classifier and detector, and compared the preserved processing/scoring code with committed `V5/loggin.py`.

**TODO:** Verify the complete V6 webpage-to-real-runner flow on the team's known real and fake videos, including screen selection, Stop, final verdict and Grad-CAM PDF. The recorded integration/browser checks used a fake runner and do not prove real-video accuracy or performance.

**TODO:** Record a repeatable real-video comparison against the team's preferred `loggin.py` setup, using the same model files, selected area and playback conditions.
