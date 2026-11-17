import os
import time
import glob
import csv
import argparse
import hashlib
import json
import uuid
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def model_fingerprint(label, path):
    print(f"{label} path: {path}", flush=True)
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as model_file:
            for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"Cannot read {label} file {path}: {exc}. Analysis was not started.") from exc
    digest = hasher.hexdigest()
    print(f"{label} SHA-256: {digest}", flush=True)
    return digest


class SessionCancelled(Exception):
    pass


def run_session(args):
    # Preserve log messages even on Windows terminals without emoji support.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    session_id = args.session_id or (time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex)
    # A session ID is one safe folder name, never a path or an existing session.
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id) or session_id.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1,10)], *[f"LPT{i}" for i in range(1,10)]}:
        raise ValueError("Invalid session ID: use 1-100 letters, digits, underscores or hyphens (no reserved Windows names).")
    session_dir = Path(args.report_dir).expanduser().resolve() / session_id
    session_dir.mkdir(parents=True, exist_ok=False)
    REPORT_DIR = str(session_dir)
    stop_path = session_dir / "stop"
    camera = None
    camera_stopped = False
    csv_file = None
    cv2 = None
    stage = "starting"
    outcome = "done"
    message = None

    def write_status(status, message=None):
        nonlocal stage
        stage = status
        payload = {"session_id": session_id, "status": status,
                   "timestamp": datetime.now(timezone.utc).isoformat()}
        if message is not None:
            payload["message"] = message
        atomic_json(session_dir / "status.json", payload)

    def stop_requested():
        # One local filesystem stat between frames; no status writes in the hot loop.
        try:
            stop_path.stat()
        except FileNotFoundError:
            return False
        return True

    try:
        write_status("starting")
        print(f"Session output: {REPORT_DIR}", flush=True)
        checkpoint_path = str(Path(args.checkpoint).expanduser().resolve())
        face_detector_path = str(Path(args.detector).expanduser().resolve())
        checkpoint_sha256 = model_fingerprint("Classifier checkpoint", checkpoint_path)
        detector_sha256 = model_fingerprint("Detector weights", face_detector_path)
        import cv2
        import numpy as np
        import pywt
        import torch
        import torch.nn as nn
        from torchvision import models
        import dxcam
        from ultralytics import YOLO

        # ReportLab imports for generating PDF diagnostic reports
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, KeepTogether
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        if __package__:
            from .reporting import read_timeline, explain_verdict, build_timeline_drawing, HIGHLIGHT_RULE
        else:
            from reporting import read_timeline, explain_verdict, build_timeline_drawing, HIGHLIGHT_RULE



        # -----------------------------------------------------------------------------
        # 1. DIRECTORY & LOGGING SETUP
        # -----------------------------------------------------------------------------

        def get_next_report_filename(report_dir):
            """Finds existing report_X.pdf files and returns the next report_N.pdf path."""
            existing_pdfs = glob.glob(os.path.join(report_dir, "report_*.pdf"))
            numbers = []
            for pdf_path in existing_pdfs:
                filename = os.path.basename(pdf_path)
                try:
                    num = int(filename.replace("report_", "").replace(".pdf", ""))
                    numbers.append(num)
                except ValueError:
                    continue
            next_num = max(numbers) + 1 if numbers else 1
            return os.path.join(report_dir, f"report_{next_num}.pdf"), next_num

        csv_log_path = os.path.join(REPORT_DIR, f"inference_log_{int(time.time())}.csv")
        csv_file = open(csv_log_path, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        # Original 9 columns unchanged; 4 columns appended: Skipped_Reason (motion-gate skips),
        # Clamp_Confidence, Scale_Ratio and Frame_Confidence (trust weight used in the WCI).
        csv_writer.writerow([
            "Timestamp", "Elapsed_Sec", "Box_X", "Box_Y", "Box_W", "Box_H",
            "Blur_Variance", "Real_Prob", "Fake_Prob",
            "Skipped_Reason", "Clamp_Confidence", "Scale_Ratio", "Frame_Confidence"
        ])

        # -----------------------------------------------------------------------------
        # 2. MODEL DEFINITION & GRAD-CAM CLASS
        # -----------------------------------------------------------------------------
        def get_mobilenet_v2_5ch():
            model = models.mobilenet_v2(weights=None)
            original_conv = model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=5,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            model.features[0][0] = new_conv
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
            return model

        class GradCAM:
            def __init__(self, model, target_layer):
                self.model = model
                self.target_layer = target_layer
                self.gradients = None
                self.activations = None

                self.target_layer.register_forward_hook(self.save_activation)
                self.target_layer.register_full_backward_hook(self.save_gradient)

            def save_activation(self, module, input, output):
                self.activations = output.detach()

            def save_gradient(self, module, grad_input, grad_output):
                self.gradients = grad_output[0].detach()

            def generate_heatmap(self, input_tensor, class_idx=1):
                self.model.zero_grad()
                output = self.model(input_tensor)
                score = output[0, class_idx]
                score.backward()

                weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
                cam = torch.sum(weights * self.activations, dim=1).squeeze(0)
                cam = torch.clamp(cam, min=0)
                cam = cam.cpu().numpy()
                cam = cv2.resize(cam, (200, 200))
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                return cam

        # Initialize Hardware & Load Deepfake Model safely
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f" Device initialized: {device}")

        model = get_mobilenet_v2_5ch()

        try:
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            model.load_state_dict(state_dict)
            print(f" Loaded deepfake checkpoint successfully from: {checkpoint_path}")
        except Exception as exc:
            csv_file.close()
            raise RuntimeError(f"ERROR: Failed to load classifier checkpoint {checkpoint_path}: {exc}. Analysis was not started.") from exc

        model.to(device)
        model.eval()
        grad_cam = GradCAM(model, model.features[-1])

        # --- LOAD YOLO FACE DETECTOR FROM CHECKPOINTS FOLDER ---
        print(f"Initializing YOLO detector from: {face_detector_path}")
        try:
            if Path(face_detector_path).suffix.lower() != ".pt":
                raise ValueError("Detector weights must be a local .pt checkpoint.")
            face_detector = YOLO(face_detector_path)
            print(" Loaded YOLO detector successfully!")
        except Exception as exc:
            csv_file.close()
            raise RuntimeError(f"ERROR: Failed to load detector weights {face_detector_path}: {exc}. Analysis was not started.") from exc

        # Initialize DXCam instance
        camera = dxcam.create(output_color="BGR")

        # -----------------------------------------------------------------------------
        # 3. UNCLIPPED 5-CHANNEL PREPROCESSING
        # -----------------------------------------------------------------------------
        def preprocess_5ch(face_bgr):
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

            coeffs2 = pywt.dwt2(gray, 'haar')
            _, (LH, HL, HH) = coeffs2
            wavelet_feat = np.sqrt(LH**2 + HL**2 + HH**2)
            wavelet_resized = cv2.resize(wavelet_feat, (200, 200))
            wavelet_norm = cv2.normalize(wavelet_resized, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-5)
            fft_norm = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            tensor_5ch = np.stack([
                face_rgb[:, :, 0],
                face_rgb[:, :, 1],
                face_rgb[:, :, 2],
                wavelet_norm,
                fft_norm
            ], axis=0)

            return torch.tensor(tensor_5ch, dtype=torch.float32).unsqueeze(0).to(device)

        # -----------------------------------------------------------------------------
        # 4. TWO-STAGE BOUNDARY SELECTION
        # -----------------------------------------------------------------------------
        print("\n" + "="*60)
        print(" STAGE 1: SWITCH TO YOUR MEDIA WINDOW NOW!")
        print("="*60)

        write_status("countdown")
        COUNTDOWN_SECONDS = 10
        for i in range(COUNTDOWN_SECONDS, 0, -1):
            if stop_requested():
                raise SessionCancelled("Stop requested during countdown; capture was not started.")
            print(f" Freezing screen for boundary capture in: {i} second(s)...", end="\r")
            time.sleep(1)

        if stop_requested():
            raise SessionCancelled("Stop requested during countdown; capture was not started.")
        write_status("selecting_region")

        print("\n\n Screen captured via GPU Direct Duplication!")
        print(" Click and drag your mouse around the video player.")
        print(" Press ENTER to lock the boundary, or 'c' to cancel.\n")

        full_screen = camera.grab()
        if full_screen is None:
            time.sleep(0.5)
            full_screen = camera.grab()

        overlay = full_screen.copy()
        cv2.rectangle(overlay, (0, 0), (full_screen.shape[1], full_screen.shape[0]), (0, 0, 0), -1)
        opaque_screen = cv2.addWeighted(overlay, 0.4, full_screen, 0.6, 0)

        banner_text = "DRAG MOUSE OVER VIDEO AREA & PRESS ENTER TO LOCK BOUNDARY"
        cv2.rectangle(opaque_screen, (0, 0), (opaque_screen.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(opaque_screen, banner_text, (20, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        window_name = "SET BOUNDARY (Press ENTER when done)"
        roi = cv2.selectROI(window_name, opaque_screen, False)
        cv2.destroyWindow(window_name)

        if roi[2] == 0 or roi[3] == 0:
            print(" No valid region selected! Defaulting to full screen.")
            target_region = None
        else:
            left, top, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
            target_region = (left, top, left + w, top + h)
            print(f" Boundary locked successfully! Region: {target_region}")

        # -----------------------------------------------------------------------------
        # 5. LIVE YOLO STREAM & INFERENCE LOOP (WITH PADDED BOUNDING BOXES)
        # -----------------------------------------------------------------------------
        print("\n DXCam stream active with YOLO Face Tracking! Play your video now.")
        print(" Press CTRL + C in this terminal window when finished to stop & generate PDF report.")

        camera.start(region=target_region, target_fps=30, video_mode=True)
        write_status("capturing")

        start_time = time.time()
        frame_count = 0
        frames_with_detection = 0
        fake_scores = []              # every scored frame's fake_prob, in order
        fake_scores_timestamps = []   # matching elapsed-time for each scored frame
        fake_scores_confidence = []   # matching trust-weight (0-1) for each scored frame
        highest_fake_score = -1.0     # absolute peak, ONLY updated by high-confidence frames
        peak_frame_bgr = None
        peak_tensor = None
        motion_skipped = 0

        # ---------------------------------------------------------------------------
        # GATE 1 -- MOTION GATE (hard skip)
        # ---------------------------------------------------------------------------
        # Rejects a candidate frame if the face box center moved too fast since the
        # last SCORED frame (motion blur / mid-turn head pose).
        MAX_CENTER_SPEED_PX_PER_SEC = 600.0   # tune against your own footage/resolution
        prev_center = None
        prev_center_time = None

        # ---------------------------------------------------------------------------
        # GATE 2 -- EDGE-CLAMP CONFIDENCE PENALTY (down-weight, not discard)
        # ---------------------------------------------------------------------------
        # A frame whose 18% padding was cut off by the capture-region edge is still
        # scored and logged, but its trust is reduced in proportion to the padding lost.
        PAD_TOLERANCE = 0.85     # padding ratio at/above which a side counts as "fully fine"
        CLAMP_FLOOR = 0.20       # minimum clamp-confidence even at maximum clamping

        def clamp_confidence_for_side(achieved_pad, intended_pad):
            """1.0 if this side's padding is >= PAD_TOLERANCE of intended,
            otherwise linearly scales down to CLAMP_FLOOR as padding approaches 0."""
            if intended_pad <= 0:
                return 1.0
            ratio = max(0.0, achieved_pad / intended_pad)
            if ratio >= PAD_TOLERANCE:
                return 1.0
            return CLAMP_FLOOR + (1.0 - CLAMP_FLOOR) * (ratio / PAD_TOLERANCE)

        # ---------------------------------------------------------------------------
        # GATE 3 -- SCALE-CONSISTENCY CHECK (down-weight, not discard)
        # ---------------------------------------------------------------------------
        # Frames whose box height is far from the calibrated face size are still
        # scored, but get a lower trust weight and can never set the session peak.
        CALIBRATION_FRAMES = 15     # first N scored frames establish the expected face-crop size
        SCALE_TOLERANCE = 1.8       # box height can be up to 1.8x or down to 1/1.8x the calibrated size
        LOW_CONF_WEIGHT = 0.35      # trust weight applied to out-of-distribution-scale frames
        calibration_heights = []
        baseline_height = None

        try:
            while True:
                if stop_requested():
                    print("\nStop file received. Compiling PDF report...")
                    break
                frame = camera.get_latest_frame()
                if frame is None:
                    continue

                f_height, f_width, _ = frame.shape
                frame[int(f_height * 0.75):f_height, 0:int(f_width * 0.3)] = (0, 0, 0)

                elapsed_now = time.time() - start_time

                # Run YOLO inference
                results = face_detector(frame, verbose=False, conf=0.40)

                skip_reason = ""
                scale_ratio_str = ""
                frame_confidence_str = ""

                if len(results) > 0 and len(results[0].boxes) > 0:
                    frames_with_detection += 1
                    box = results[0].boxes[0].xyxy[0].cpu().numpy()
                    px, py, px2, py2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                    p_width = px2 - px
                    p_height = py2 - py

                    # Pad bounding box by 18% to capture jawline and seam boundaries
                    pad_x = int(p_width * 0.18)
                    pad_y = int(p_height * 0.18)

                    full_x = max(0, px - pad_x)
                    full_y = max(0, py - pad_y)
                    full_w = min(f_width - full_x, p_width + (2 * pad_x))
                    full_h = min(f_height - full_y, p_height + (2 * pad_y))

                    if full_w > 20 and full_h > 20:

                        # ---- GATE 2: edge-clamp confidence ----
                        achieved_top_pad = py - full_y
                        achieved_bottom_pad = (full_y + full_h) - py2
                        achieved_left_pad = px - full_x
                        achieved_right_pad = (full_x + full_w) - px2

                        clamp_confidence = min(
                            clamp_confidence_for_side(achieved_top_pad, pad_y),
                            clamp_confidence_for_side(achieved_bottom_pad, pad_y),
                            clamp_confidence_for_side(achieved_left_pad, pad_x),
                            clamp_confidence_for_side(achieved_right_pad, pad_x),
                        )

                        # ---- GATE 1: motion (hard skip) ----
                        cx = full_x + full_w / 2.0
                        cy = full_y + full_h / 2.0
                        motion_ok = True
                        if prev_center is not None and prev_center_time is not None:
                            dt = elapsed_now - prev_center_time
                            if dt > 0:
                                dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
                                if dist / dt > MAX_CENTER_SPEED_PX_PER_SEC:
                                    motion_ok = False

                        if not motion_ok:
                            skip_reason = "fast_motion"

                        if motion_ok:
                            face_img = frame[full_y:full_y+full_h, full_x:full_x+full_w]

                            if face_img.size > 0:
                                face_resized = cv2.resize(face_img, (200, 200))

                                gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                                blur_variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                                box_aspect_ratio = full_h / (full_w + 1e-5)

                                if blur_variance >= 30.0 and (0.7 <= box_aspect_ratio <= 1.6):

                                    # ---- GATE 3: scale consistency ----
                                    if baseline_height is None:
                                        calibration_heights.append(full_h)
                                        if len(calibration_heights) >= CALIBRATION_FRAMES:
                                            baseline_height = float(np.median(calibration_heights))
                                        scale_ratio = 1.0
                                        scale_confidence = 1.0  # trust calibration-phase frames fully
                                    else:
                                        scale_ratio = full_h / baseline_height
                                        if (1.0 / SCALE_TOLERANCE) <= scale_ratio <= SCALE_TOLERANCE:
                                            scale_confidence = 1.0
                                        else:
                                            scale_confidence = LOW_CONF_WEIGHT

                                    frame_confidence = clamp_confidence * scale_confidence

                                    scale_ratio_str = f"{scale_ratio:.2f}"
                                    clamp_confidence_str = f"{clamp_confidence:.2f}"
                                    frame_confidence_str = f"{frame_confidence:.2f}"

                                    input_tensor = preprocess_5ch(face_resized)

                                    with torch.no_grad():
                                        outputs = model(input_tensor)
                                        probs = torch.softmax(outputs, dim=1)[0]
                                        real_prob = probs[0].item()
                                        raw_fake_prob = probs[1].item()

                                        fake_prob = min(1.0, raw_fake_prob * 1.5)

                                    fake_scores.append(fake_prob)
                                    fake_scores_timestamps.append(elapsed_now)
                                    fake_scores_confidence.append(frame_confidence)

                                    prev_center = (cx, cy)
                                    prev_center_time = elapsed_now

                                    csv_writer.writerow([
                                        time.strftime('%Y-%m-%d %H:%M:%S'),
                                        f"{elapsed_now:.2f}",
                                        full_x, full_y, full_w, full_h,
                                        f"{blur_variance:.2f}",
                                        f"{real_prob:.4f}",
                                        f"{fake_prob:.4f}",
                                        "",
                                        clamp_confidence_str,
                                        scale_ratio_str,
                                        frame_confidence_str
                                    ])

                                    # only a high-confidence, in-distribution, unclamped frame
                                    # may become the session's peak frame
                                    if frame_confidence >= 1.0 and fake_prob > highest_fake_score:
                                        highest_fake_score = fake_prob
                                        peak_frame_bgr = face_resized.copy()
                                        peak_tensor = input_tensor.clone()
                                else:
                                    skip_reason = "quality_gate"
                        else:
                            motion_skipped += 1
                            csv_writer.writerow([
                                time.strftime('%Y-%m-%d %H:%M:%S'),
                                f"{elapsed_now:.2f}",
                                full_x, full_y, full_w, full_h,
                                "", "", "",
                                skip_reason, "", "", ""
                            ])

                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                print(f"\rYOLO Face Stream... Elapsed: {elapsed:.1f}s | FPS: {fps:.1f} | Evaluated Faces: {len(fake_scores)}", end="")

        except KeyboardInterrupt:
            print("\n\nSession stopped via Ctrl+C. Closing CSV log & compiling PDF report...")
        finally:
            camera.stop()
            camera_stopped = True
            csv_file.close()
            print(f" CSV Log saved at: {csv_log_path}")

        # -----------------------------------------------------------------------------
        # 6. EQUATION-BASED WEIGHTED AGGREGATION & PDF COMPILATION
        # -----------------------------------------------------------------------------
        write_status("analyzing")
        total_elapsed = time.time() - start_time
        actual_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

        # Confidence ATTENUATES each score before any averaging, so it can never
        # cancel out (a weighted average sum(w*x)/sum(w) would cancel it whenever a
        # window's samples share the same confidence).
        attenuated_scores = [f * c for f, c in zip(fake_scores, fake_scores_confidence)]
        avg_fake_prob = float(np.mean(attenuated_scores)) if len(attenuated_scores) > 0 else 0.0

        # Time-based burst window with evidence-count shrinkage: PRIOR_COUNT phantom
        # samples at the session average pull low-evidence windows toward baseline.
        # V7 tuning: window widened 1s -> 2s so a single blink or a 2-3 frame glitch
        # cannot fill a whole window on its own; a sustained manipulation still can.
        BURST_WINDOW_SECONDS = 2.0
        PRIOR_COUNT = 3
        max_burst_avg = 0.0

        if len(fake_scores) > 0:
            for i in range(len(fake_scores)):
                t0 = fake_scores_timestamps[i]
                idxs = [j for j in range(len(fake_scores)) if t0 <= fake_scores_timestamps[j] < t0 + BURST_WINDOW_SECONDS]
                if idxs:
                    window_attenuated = [attenuated_scores[j] for j in idxs]
                    shrunk_avg = (sum(window_attenuated) + PRIOR_COUNT * avg_fake_prob) / (len(window_attenuated) + PRIOR_COUNT)
                    if shrunk_avg > max_burst_avg:
                        max_burst_avg = shrunk_avg
        else:
            max_burst_avg = avg_fake_prob

        # V7 tuning: the peak term uses the 95th percentile of full-confidence frame
        # scores instead of the single maximum, so one outlier frame (e.g. a blink)
        # cannot set it. highest_fake_score is still kept for the Grad-CAM frame.
        PEAK_PERCENTILE = 95
        high_conf_scores = [f for f, c in zip(fake_scores, fake_scores_confidence) if c >= 1.0]
        robust_peak = float(np.percentile(high_conf_scores, PEAK_PERCENTILE)) if high_conf_scores else 0.0

        # Share of scored frames the model found suspicious (display/diagnostic only).
        SUSPICIOUS_THRESHOLD = 0.5
        suspicious_fraction = (sum(1 for f in fake_scores if f > SUSPICIOUS_THRESHOLD) / len(fake_scores)) if fake_scores else 0.0

        # V7 tuning: weights now sum to 1.00 (burst 0.60 -> 0.65; average 0.20 and peak 0.15 unchanged).
        WCI = (0.20 * avg_fake_prob) + (0.65 * max_burst_avg) + (0.15 * robust_peak)

        if WCI > 0.38 or (max_burst_avg > 0.30 and robust_peak > 0.85):
            overall_prediction = "SYNTHETIC / DEEPFAKE"
        else:
            overall_prediction = "AUTHENTIC / REAL"

        # Override only the zero-evidence verdict; the scoring above matches local V5.
        if len(fake_scores) == 0:
            overall_prediction = "Not enough evidence"

        verdict_explanation = explain_verdict(
            overall_prediction,
            {"wci": float(WCI), "average": float(avg_fake_prob),
             "burst": float(max_burst_avg), "peak": float(max(highest_fake_score, 0.0))},
            {"frames_captured": frame_count, "frames_with_detection": frames_with_detection,
             "usable_crops_after_filters": len(fake_scores)},
        )

        pdf_path = os.path.join(REPORT_DIR, "report.pdf")
        report_num = session_id
        temp_gradcam_path = os.path.join(REPORT_DIR, f"temp_gradcam_{report_num}.jpg")

        if peak_frame_bgr is not None and peak_tensor is not None:
            heatmap = grad_cam.generate_heatmap(peak_tensor, class_idx=1)
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            gradcam_overlay = cv2.addWeighted(peak_frame_bgr, 0.6, heatmap_colored, 0.4, 0)

            comparison_img = np.hstack([peak_frame_bgr, gradcam_overlay])
            cv2.imwrite(temp_gradcam_path, comparison_img)
        else:
            blank = np.zeros((200, 400, 3), dtype=np.uint8)
            cv2.putText(blank, "No High-Confidence Face Found", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imwrite(temp_gradcam_path, blank)


        # Presentation only; the scores and verdict above are unchanged.
        timeline_data = read_timeline(REPORT_DIR, os.path.basename(csv_log_path))
        timeline = build_timeline_drawing(timeline_data)

        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1a237e'), spaceAfter=12)
        heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#283593'), spaceAfter=6)
        normal_style = styles['Normal']

        elements = []

        elements.append(Paragraph(f"Deepfake Detection Diagnostic Report #{report_num} (YOLO Face Tracked)", title_style))
        elements.append(Paragraph(f"<b>Timestamp:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph("Verdict Explanation", heading_style))
        elements.append(Paragraph(verdict_explanation, normal_style))
        elements.append(Spacer(1, 10))


        low_conf_count = sum(1 for c in fake_scores_confidence if c < 1.0)

        data_metrics = [
            ["Metric", "Value"],
            ["Session Duration", f"{total_elapsed:.2f} seconds"],
            ["Total Frames Captured", f"{frame_count}"],
            ["Total Frames Scored", f"{len(fake_scores)}"],
            ["Low-Confidence (Scale-Shifted) Frames", f"{low_conf_count}"],
            ["Suspicious Frames (score > 0.5)", f"{suspicious_fraction * 100:.1f}%"],
            ["Average FPS", f"{actual_fps:.2f} FPS"],
            ["Overall Verdict", overall_prediction],
            ["Weighted Confidence Index (WCI)", f"{WCI * 100:.2f}%"],
            ["Session Average Score", f"{avg_fake_prob * 100:.2f}%"],
            [f"Peak {BURST_WINDOW_SECONDS:g}s Burst Window Score", f"{max_burst_avg * 100:.2f}%"],
            [f"Robust Peak ({PEAK_PERCENTILE}th percentile, used in WCI)", f"{robust_peak * 100:.2f}%"],
            ["Peak Frame AI Score (high-confidence only)", f"{max(highest_fake_score, 0.0) * 100:.2f}%"]
        ]

        t = Table(data_metrics, colWidths=[260, 200])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3f51b5')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e0e0e0')),
        ]))
        elements.append(t)

        timeline_elements = []
        timeline_elements.append(Spacer(1, 12))
        timeline_elements.append(Paragraph("Score Timeline", heading_style))
        timeline_elements.append(Paragraph(
            "Each point reflects a recorded usable detection. "
            "The adjusted fake score is a model output, not a calibrated "
            "probability or proof of manipulation.",
            normal_style
        ))
        timeline_elements.append(Spacer(1, 6))
        timeline_elements.append(timeline)
        timeline_elements.append(Paragraph(HIGHLIGHT_RULE, normal_style))
        for warning in timeline_data["warnings"]:
            timeline_elements.append(Paragraph(warning, normal_style))
        elements.append(KeepTogether(timeline_elements))
        elements.append(Spacer(1, 15))

        evidence_elements = []
        evidence_elements.append(Paragraph("Explainable AI (Grad-CAM) Artifact Analysis", heading_style))
        evidence_elements.append(Paragraph("Left: Original Peak Suspicion Face | Right: Grad-CAM Activation Heatmap", normal_style))
        evidence_elements.append(Spacer(1, 8))

        evidence_elements.append(RLImage(temp_gradcam_path, width=380, height=190))
        evidence_elements.append(Spacer(1, 12))

        evidence_elements.append(Paragraph(f"<b>Technical Note:</b> CSV log generated at {os.path.basename(csv_log_path)} using YOLOv8 face model tracking, motion/edge-clamp quality gates, scale-consistency confidence attenuation with evidence-count shrinkage, and WCI equation scoring logic.", normal_style))

        elements.append(KeepTogether(evidence_elements))

        doc.build(elements)

        if os.path.exists(temp_gradcam_path):
            os.remove(temp_gradcam_path)

        print(f"\n Report compiled and saved successfully!")
        print(f" Location: {pdf_path}")

        # Publish JSON only after the PDF is complete. Scores retain committed V5 values.
        result = {
            "session_id": session_id,
            "verdict": overall_prediction,
            "verdict_explanation": verdict_explanation,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "detector_path": face_detector_path,
            "detector_sha256": detector_sha256,
            "report_path": pdf_path,
            "csv_log_path": csv_log_path,
            "scores": {
                "average": float(avg_fake_prob),
                "peak": float(highest_fake_score),
                "burst": float(max_burst_avg),
                "wci": float(WCI),
            },
            "debug": {
                "capture_rectangle": list(target_region) if target_region is not None else [0, 0, int(full_screen.shape[1]), int(full_screen.shape[0])],
                "capture_rectangle_format": "left, top, right, bottom (screen pixels)",
                "fell_back_to_full_screen": target_region is None,
                "frames_captured": frame_count,
                "frames_with_detection": frames_with_detection,
                "usable_crops_after_filters": len(fake_scores),
                "average_score": float(avg_fake_prob),
                "peak_score": float(highest_fake_score),
                "burst_score": float(max_burst_avg),
                "wci_score": float(WCI),
                "burst_window_seconds": BURST_WINDOW_SECONDS,
                "burst_prior_count": PRIOR_COUNT,
                "motion_gate_skips": motion_skipped,
                "low_confidence_frames": low_conf_count,
                "robust_peak_score": robust_peak,
                "robust_peak_percentile": PEAK_PERCENTILE,
                "suspicious_fraction": suspicious_fraction,
                "high_confidence_peak_found": highest_fake_score >= 0,
                "duration_seconds": total_elapsed,
                "average_fps": actual_fps,
                "no_usable_detections": len(fake_scores) == 0,
                "score_note": "Scores follow the local V5 logic: average and burst use confidence-attenuated scores (fake_prob x frame confidence); burst is a 2-second time window with evidence-count shrinkage; WCI = 0.20*average + 0.65*burst + 0.15*robust_peak, where robust_peak is the 95th percentile of full-confidence frame scores. scores.peak is the single highest full-confidence frame (used for Grad-CAM); -1 means none was scored. With zero usable detections the verdict is Not enough evidence. CSV Fake_Prob is the boosted, unattenuated score; Real_Prob is raw; rows with Skipped_Reason were not scored.",
            },
        }
        result_path = os.path.join(REPORT_DIR, "result.json")
        atomic_json(Path(result_path), result)
        print(f" Verdict: {overall_prediction}")
        print(f" Result: {result_path}")

    except SessionCancelled as exc:
        outcome, message = "cancelled", str(exc)
    except KeyboardInterrupt:
        outcome, message = "cancelled", f"Interrupted during {stage}."
    except Exception as exc:
        outcome, message = "error", f"{type(exc).__name__}: {exc}"
    finally:
        # Attempt every cleanup even when an earlier cleanup fails.
        cleanup_errors = []
        if camera is not None:
            if not camera_stopped:
                try:
                    camera.stop()
                except Exception as exc:
                    cleanup_errors.append(f"camera stop: {exc}")
            if hasattr(camera, "release"):
                try:
                    camera.release()
                except Exception as exc:
                    cleanup_errors.append(f"camera release: {exc}")
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception as exc:
                cleanup_errors.append(f"window cleanup: {exc}")
        if csv_file is not None:
            try:
                csv_file.close()
            except Exception as exc:
                cleanup_errors.append(f"CSV close: {exc}")
        if cleanup_errors:
            outcome = "error"
            message = "; ".join(filter(None, [message, *cleanup_errors]))
    try:
        write_status(outcome, message)
    except Exception as exc:
        print(f"ERROR: Cannot publish final status: {exc}", file=sys.stderr)
        return 1
    if message:
        print(f"{outcome.upper()}: {message}", file=sys.stderr if outcome == "error" else sys.stdout)
    return 1 if outcome == "error" else 0


def main():
    parser = argparse.ArgumentParser(description="Committed V5/loggin.py pipeline with status.json and external Stop-file support.")
    parser.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH", str(PROJECT_ROOT / "checkpoints" / "mobilenet_v2_deepfake.pth")))
    parser.add_argument("--detector", default=os.getenv("YOLO_WEIGHTS", str(PROJECT_ROOT / "checkpoints" / "yolov8n-face.pt")))
    parser.add_argument("--report-dir", default=os.getenv("REPORT_DIR", str(PROJECT_ROOT / "V7" / "reports")))
    parser.add_argument("--session-id", help="New session folder name; generated if omitted.")
    args = parser.parse_args()
    try:
        return run_session(args)
    except Exception as exc:
        print(f"ERROR: Cannot start session: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
