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
csv_writer.writerow(["Timestamp", "Elapsed_Sec", "Box_X", "Box_Y", "Box_W", "Box_H", "Blur_Variance", "Real_Prob", "Fake_Prob"])

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
print(f"📥 Initializing YOLOv8 Face Detector from: {face_detector_path}")
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
fake_scores = []
highest_fake_score = -1.0
peak_frame_bgr = None
peak_tensor = None

try:
    while True:
        frame = camera.get_latest_frame()
        if frame is None:
            continue
            
        f_height, f_width, _ = frame.shape
        frame[int(f_height * 0.75):f_height, 0:int(f_width * 0.3)] = (0, 0, 0)
        
        # Run YOLO inference
        results = face_detector(frame, verbose=False, conf=0.40)
        
        if len(results) > 0 and len(results[0].boxes) > 0:
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
                face_img = frame[full_y:full_y+full_h, full_x:full_x+full_w]
                
                if face_img.size > 0:
                    face_resized = cv2.resize(face_img, (200, 200))
                    
                    gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
                    blur_variance = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                    box_aspect_ratio = full_h / (full_w + 1e-5)
                    
                    if blur_variance >= 30.0 and (0.7 <= box_aspect_ratio <= 1.6):
                        input_tensor = preprocess_5ch(face_resized)
                        
                        with torch.no_grad():
                            outputs = model(input_tensor)
                            probs = torch.softmax(outputs, dim=1)[0]
                            real_prob = probs[0].item()
                            raw_fake_prob = probs[1].item()
                            
                            fake_prob = min(1.0, raw_fake_prob * 1.5)
                        
                        fake_scores.append(fake_prob)
                        
                        csv_writer.writerow([
                            time.strftime('%Y-%m-%d %H:%M:%S'),
                            f"{time.time() - start_time:.2f}",
                            full_x, full_y, full_w, full_h,
                            f"{blur_variance:.2f}",
                            f"{real_prob:.4f}",
                            f"{fake_prob:.4f}"
                        ])
                        
                        if fake_prob > highest_fake_score:
                            highest_fake_score = fake_prob
                            peak_frame_bgr = face_resized.copy()
                            peak_tensor = input_tensor.clone()

        frame_count += 1
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\r🎥 YOLO Face Stream... Elapsed: {elapsed:.1f}s | FPS: {fps:.1f} | Evaluated Faces: {len(fake_scores)}", end="")

except KeyboardInterrupt:
    print("\n\n⏹ Session stopped via Ctrl+C. Closing CSV log & compiling PDF report...")
finally:
    camera.stop()
    csv_file.close()
    print(f" CSV Log saved at: {csv_log_path}")

# -----------------------------------------------------------------------------
# 6. EQUATION-BASED WEIGHTED AGGREGATION & PDF COMPILATION
# -----------------------------------------------------------------------------
total_elapsed = time.time() - start_time
actual_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
avg_fake_prob = np.mean(fake_scores) if len(fake_scores) > 0 else 0.0

BURST_WINDOW_SIZE = max(1, int(actual_fps * 1.0))
max_burst_avg = 0.0

if len(fake_scores) >= BURST_WINDOW_SIZE:
    for i in range(len(fake_scores) - BURST_WINDOW_SIZE + 1):
        win_avg = np.mean(fake_scores[i : i + BURST_WINDOW_SIZE])
        if win_avg > max_burst_avg:
            max_burst_avg = win_avg
else:
    max_burst_avg = avg_fake_prob

WCI = (0.20 * avg_fake_prob) + (0.50 * max_burst_avg) + (0.30 * highest_fake_score)

# Robust decision logic resilient to hand occlusions and temporary dropouts
if WCI > 0.38 or (max_burst_avg > 0.25 and highest_fake_score > 0.80):
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
    cv2.putText(blank, "No Face Detected", (80, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
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

data_metrics = [
    ["Metric", "Value"],
    ["Session Duration", f"{total_elapsed:.2f} seconds"],
    ["Total Frames Captured", f"{frame_count}"],
    ["Average FPS", f"{actual_fps:.2f} FPS"],
    ["Overall Verdict", overall_prediction],
    ["Weighted Confidence Index (WCI)", f"{WCI * 100:.2f}%"],
    ["Session Average Score", f"{avg_fake_prob * 100:.2f}%"],
    ["Peak 1s Burst Window Score", f"{max_burst_avg * 100:.2f}%"],
    ["Peak Frame AI Score", f"{highest_fake_score * 100:.2f}%"]
]

t = Table(data_metrics, colWidths=[200, 250])
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

elements.append(Paragraph(f"<b>Technical Note:</b> CSV log generated at {os.path.basename(csv_log_path)} using YOLOv8 face model tracking and WCI equation scoring logic.", normal_style))

doc.build(elements)

if os.path.exists(temp_gradcam_path):
    os.remove(temp_gradcam_path)

print(f"\n Report compiled and saved successfully!")
print(f" Location: {pdf_path}")