import os
import time
import glob
import csv
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# -----------------------------------------------------------------------------
# 1. DIRECTORY & LOGGING SETUP
# -----------------------------------------------------------------------------
REPORT_DIR = r"C:\Users\Toshiba\Desktop\LY_PROJECT\Report"
os.makedirs(REPORT_DIR, exist_ok=True)

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
# Original 9 columns are unchanged (same inference logging concept as before).
# 4 new columns appended at the end: Skipped_Reason (why a candidate frame was
# never scored -- now only used by the motion gate, since edge-clamp no longer
# hard-skips), Clamp_Confidence (how much of the intended padding survived),
# Scale_Ratio (how this frame's box size compares to the session's calibrated
# baseline face size), and Frame_Confidence (the combined trust weight used
# when computing the WCI, see section 6).
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
checkpoint_path = r"C:\Users\Toshiba\Desktop\LY_PROJECT\checkpoints\mobilenet_v2_deepfake.pth"

if os.path.exists(checkpoint_path):
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif isinstance(state_dict, dict) and 'model' in state_dict:
            state_dict = state_dict['model']

        model.load_state_dict(state_dict)
        print(f" Loaded deepfake checkpoint successfully from: {checkpoint_path}")
    except Exception as e:
        print(f" Error loading deepfake checkpoint: {e}")
else:
    print(f" Deepfake checkpoint NOT found at: {checkpoint_path}")

model.to(device)
model.eval()
grad_cam = GradCAM(model, model.features[-1])

# --- LOAD YOLO FACE DETECTOR FROM CHECKPOINTS FOLDER ---
face_detector_path = r"C:\Users\Toshiba\Desktop\LY_PROJECT\checkpoints\yolov8n-face.pt"
print(f"Initializing YOLOv8 Face Detector from: {face_detector_path}")
if os.path.exists(face_detector_path):
    face_detector = YOLO(face_detector_path)
    print(" Loaded YOLOv8 Face Detector successfully!")
else:
    print(f" YOLOv8 Face Detector NOT found at {face_detector_path}! Falling back to default.")
    face_detector = YOLO("yolov8n.pt")

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

COUNTDOWN_SECONDS = 10
for i in range(COUNTDOWN_SECONDS, 0, -1):
    print(f" Freezing screen for boundary capture in: {i} second(s)...", end="\r")
    time.sleep(1)

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

start_time = time.time()
frame_count = 0
fake_scores = []              # every scored frame's fake_prob, in order
fake_scores_timestamps = []   # matching elapsed-time for each scored frame
fake_scores_confidence = []   # matching trust-weight (0-1) for each scored frame -- see scale gate below
highest_fake_score = -1.0     # absolute peak, ONLY updated by high-confidence frames (drives WCI + report)
peak_frame_bgr = None
peak_tensor = None

# ---------------------------------------------------------------------------
# NEW GATE 1 -- MOTION GATE
# ---------------------------------------------------------------------------
# Rejects a candidate frame outright if the face box center moved too fast
# since the last SCORED frame. Fast motion between frames usually means
# motion blur or a mid-turn head pose the model never saw in training.
MAX_CENTER_SPEED_PX_PER_SEC = 600.0   # tune against your own footage/resolution
prev_center = None
prev_center_time = None

# ---------------------------------------------------------------------------
# NEW GATE 2 -- EDGE-CLAMP CONFIDENCE PENALTY (down-weight, not discard)
# ---------------------------------------------------------------------------
# Previously this hard-skipped any frame where the intended 18% padding got
# cut off by the screen-region edge. In practice this turned out to be WAY
# too aggressive: whenever a face simply sits near the top of the capture
# region (a common, entirely normal framing -- not a defect), the top
# padding can NEVER fully survive, so this gate was silently discarding the
# large majority of frames in some sessions (522 captured -> only 47 scored
# in one real run). Throwing away that much data leaves too little evidence
# to base a verdict on, and ironically made the pipeline MORE fragile, not
# less -- a single surviving low-quality frame could end up as an entire
# burst window on its own.
#
# Same fix as the scale gate: still score the frame, still log it, but scale
# its trust down in proportion to how much padding was actually lost. A
# frame with fully intact padding gets full trust; one with padding clamped
# down toward zero approaches (but never quite hits) a low floor confidence.
PAD_TOLERANCE = 0.85     # padding ratio at/above which a side counts as "fully fine"
CLAMP_FLOOR = 0.20       # minimum clamp-confidence even at maximum clamping

def clamp_confidence_for_side(achieved_pad, intended_pad):
    """Returns 1.0 if this side's padding is >= PAD_TOLERANCE of intended,
    otherwise linearly scales down to CLAMP_FLOOR as padding approaches 0."""
    if intended_pad <= 0:
        return 1.0
    ratio = max(0.0, achieved_pad / intended_pad)
    if ratio >= PAD_TOLERANCE:
        return 1.0
    # linear interpolation between CLAMP_FLOOR (at ratio=0) and 1.0 (at ratio=PAD_TOLERANCE)
    return CLAMP_FLOOR + (1.0 - CLAMP_FLOOR) * (ratio / PAD_TOLERANCE)


# ---------------------------------------------------------------------------
# NEW GATE 3 -- SCALE-CONSISTENCY CHECK (down-weight, not discard)
# ---------------------------------------------------------------------------
# The model was trained on tightly-cropped faces. If the source video zooms
# out or changes shot mid-session, YOLO's padded box can become 3-5x larger
# than a tight face crop -- a framing the model has never really seen, which
# tends to produce an unreliable, sometimes wildly inflated fake score.
#
# Unlike gates 1 and 2, this does NOT discard the frame -- a genuine deepfake
# could still be happening during that wider shot, and throwing the frame
# away entirely would make the detector blind to it. Instead, the frame is
# still scored and still logged, but it gets a lower "confidence" weight when
# contributing to the average/burst signals, and it is NOT allowed to set the
# session's single highest_fake_score (which drives 15% of the WCI) --
# that value can only come from an in-distribution, high-confidence frame.
CALIBRATION_FRAMES = 15     # first N scored frames establish the expected face-crop size
SCALE_TOLERANCE = 1.8       # box height can be up to 1.8x or down to 1/1.8x the calibrated size
LOW_CONF_WEIGHT = 0.35      # trust weight applied to out-of-distribution-scale frames
calibration_heights = []
baseline_height = None

try:
    while True:
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
            box = results[0].boxes[0].xyxy[0].cpu().numpy()
            px, py, px2, py2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            p_width = px2 - px
            p_height = py2 - py

            # Pad bounding box by 18% to capture jawline and seam boundaries
            pad_x = int(p_width * 0.18)
            pad_y = int(p_height * 0.18)

            intended_x = px - pad_x
            intended_y = py - pad_y
            intended_w = p_width + (2 * pad_x)
            intended_h = p_height + (2 * pad_y)

            full_x = max(0, px - pad_x)
            full_y = max(0, py - pad_y)
            full_w = min(f_width - full_x, p_width + (2 * pad_x))
            full_h = min(f_height - full_y, p_height + (2 * pad_y))

            if full_w > 20 and full_h > 20:

                # ---- GATE 2: edge-clamp confidence (no longer hard-skips) ----
                achieved_top_pad = py - full_y
                achieved_bottom_pad = (full_y + full_h) - py2
                achieved_left_pad = px - full_x
                achieved_right_pad = (full_x + full_w) - px2

                clamp_conf_top = clamp_confidence_for_side(achieved_top_pad, pad_y)
                clamp_conf_bottom = clamp_confidence_for_side(achieved_bottom_pad, pad_y)
                clamp_conf_left = clamp_confidence_for_side(achieved_left_pad, pad_x)
                clamp_conf_right = clamp_confidence_for_side(achieved_right_pad, pad_x)
                # trust the WORST side -- one badly clamped edge still distorts the whole crop
                clamp_confidence = min(clamp_conf_top, clamp_conf_bottom, clamp_conf_left, clamp_conf_right)

                # ---- GATE 1 check: motion (still a hard skip -- true blur/garbage) ----
                cx = full_x + full_w / 2.0
                cy = full_y + full_h / 2.0
                motion_ok = True
                if prev_center is not None and prev_center_time is not None:
                    dt = elapsed_now - prev_center_time
                    if dt > 0:
                        dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
                        speed = dist / dt
                        if speed > MAX_CENTER_SPEED_PX_PER_SEC:
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

                            # combined trust weight: both quality signals must be good for full trust
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

                            # only a high-confidence, in-distribution, unclamped frame is
                            # allowed to become the session's "peak" frame -- this is what
                            # stops any single distorted frame from driving a false verdict
                            if frame_confidence >= 1.0 and fake_prob > highest_fake_score:
                                highest_fake_score = fake_prob
                                peak_frame_bgr = face_resized.copy()
                                peak_tensor = input_tensor.clone()
                        else:
                            skip_reason = "quality_gate"
                else:
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
    csv_file.close()
    print(f" CSV Log saved at: {csv_log_path}")

# -----------------------------------------------------------------------------
# 6. EQUATION-BASED WEIGHTED AGGREGATION & PDF COMPILATION
# -----------------------------------------------------------------------------
total_elapsed = time.time() - start_time
actual_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

# ---------------------------------------------------------------------------
# IMPORTANT FIX: confidence must ATTENUATE the score itself, not just act as
# a weight inside an average. A weighted average (sum(w*x)/sum(w)) CANCELS
# the weight out whenever a window's samples all share the same confidence
# (e.g. a burst window containing only one low-confidence frame) -- the
# discount silently does nothing. Real production run confirmed this: a
# single confidence=0.35 frame scoring 0.6992 produced a burst-window value
# of exactly 0.6992, as if it were fully trusted.
#
# Fix: multiply confidence into the score BEFORE any averaging, so it can
# never cancel out no matter how many samples share the same confidence.
# ---------------------------------------------------------------------------
attenuated_scores = [f * c for f, c in zip(fake_scores, fake_scores_confidence)]

if len(attenuated_scores) > 0:
    avg_fake_prob = float(np.mean(attenuated_scores))
else:
    avg_fake_prob = 0.0

# ---------------------------------------------------------------------------
# time-based burst window (replaces the old frame-COUNT window), PLUS
# evidence-count shrinkage: a "1-second burst" built from only 1-2 surviving
# frames (common now that gates 1/2 can skip most of a second) isn't strong
# evidence of a sustained event. PRIOR_COUNT phantom samples at the session
# average pull low-evidence windows back toward baseline; a window with many
# real samples is barely affected, a window with 1 sample is pulled hard.
# ---------------------------------------------------------------------------
# Tuning: window widened 1s -> 2s so a single blink or a 2-3 frame glitch
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

# ---------------------------------------------------------------------------
# Tuning: the peak term uses the 95th percentile of full-confidence frame
# scores instead of the single maximum, so one outlier frame (e.g. a blink)
# cannot set it. highest_fake_score is still kept for the Grad-CAM frame.
# Weights now sum to 1.00: average 0.20, burst 0.65 (was 0.60), peak 0.15.
# ---------------------------------------------------------------------------
PEAK_PERCENTILE = 95
high_conf_scores = [f for f, c in zip(fake_scores, fake_scores_confidence) if c >= 1.0]
robust_peak = float(np.percentile(high_conf_scores, PEAK_PERCENTILE)) if high_conf_scores else 0.0

# Share of scored frames the model found suspicious (display/diagnostic only).
SUSPICIOUS_THRESHOLD = 0.5
suspicious_fraction = (sum(1 for f in fake_scores if f > SUSPICIOUS_THRESHOLD) / len(fake_scores)) if fake_scores else 0.0

WCI = (0.20 * avg_fake_prob) + (0.65 * max_burst_avg) + (0.15 * robust_peak)

if WCI > 0.38 or (max_burst_avg > 0.30 and robust_peak > 0.85):
    overall_prediction = "SYNTHETIC / DEEPFAKE"
else:
    overall_prediction = "AUTHENTIC / REAL"

pdf_path, report_num = get_next_report_filename(REPORT_DIR)
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

doc = SimpleDocTemplate(pdf_path, pagesize=letter)
styles = getSampleStyleSheet()

title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=18, textColor=colors.HexColor('#1a237e'), spaceAfter=12)
heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], fontSize=13, textColor=colors.HexColor('#283593'), spaceAfter=6)
normal_style = styles['Normal']

elements = []

elements.append(Paragraph(f"Deepfake Detection Diagnostic Report #{report_num} (YOLO Face Tracked)", title_style))
elements.append(Paragraph(f"<b>Timestamp:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
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
elements.append(Spacer(1, 15))

elements.append(Paragraph("Explainable AI (Grad-CAM) Artifact Analysis", heading_style))
elements.append(Paragraph("Left: Original Peak Suspicion Face | Right: Grad-CAM Activation Heatmap", normal_style))
elements.append(Spacer(1, 8))

elements.append(RLImage(temp_gradcam_path, width=380, height=190))
elements.append(Spacer(1, 12))

elements.append(Paragraph(
    f"<b>Technical Note:</b> CSV log generated at {os.path.basename(csv_log_path)} using YOLOv8 face model "
    f"tracking, motion/edge-clamp quality gates, scale-consistency confidence attenuation with evidence-count "
    f"shrinkage, and WCI equation scoring logic.", normal_style))

doc.build(elements)

if os.path.exists(temp_gradcam_path):
    os.remove(temp_gradcam_path)

print(f"\n Report compiled and saved successfully!")
print(f" Location: {pdf_path}")