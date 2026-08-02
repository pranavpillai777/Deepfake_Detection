# Real-Time Deepfake Detection & Diagnostic Reporting System

An advanced, desktop-level real-time deepfake detection application built for Windows (Python 3.11). The system captures live video feeds, tracks faces dynamically, runs a specialized multi-modal 5-channel deep learning classifier, computes temporal suspicion metrics, generates explainable AI (Grad-CAM) heatmaps, and automatically compiles a formal PDF forensic report.

---

## ⚡ Setup & Path Configuration

### 1. Required Packages (`pip install`)
Run the following command in your terminal to install all required dependencies:

`pip install opencv-python numpy PyWavelets torch torchvision dxcam ultralytics reportlab`

### 2. File Path Configuration
Before running the script, update the directory paths in the Python script to match where you save the project files on your computer:

* **Line 18 (`REPORT_DIR`):** Controls where generated PDF diagnostic reports are stored (`r"C:\Users\Toshiba\Desktop\LY_PROJECT\Report"`). Change this path to the location where you want your output reports saved.
* **Line 89 (`checkpoint_path`):** Points to your trained model checkpoint (`r"C:\Users\Toshiba\Desktop\LY_PROJECT\checkpoints\mobilenet_v2_deepfake.pth"`). Change this path to where the `.pth` weight file is stored on your machine.

---

## 1. Program Overview & Core Capabilities (v5)

As of version **v5**, the system incorporates an end-to-end automated pipeline:
* **Two-Stage Boundary Selection:** Captures full-screen frames via DXCam and provides an interactive OpenCV ROI selector to target specific video windows or media players.
* **Yolov8 Head & Face Tracking:** Leverages `yolov8n.pt` to detect persons and isolate the upper-body head/face region dynamically.
* **Quality Gate Filtering:** Computes Laplacian variance blur checks (`blur_variance < 12.0`) to drop motion-blurred frames and prevent false positives.
* **Multi-Modal 5-Channel Inference:** Passes normalized face tensors through a customized MobileNetV2 architecture.
* **Weighted Confidence Index (WCI):** Evaluates session risk using a custom mathematical formula combining session averages, peak 1-second burst windows, and peak single-frame scores.
* **Explainable AI & Automated Reporting:** Generates Grad-CAM activation heatmaps and exports structured multi-page PDF forensic diagnostic reports (`report_N.pdf`).

---

## 2. Model Selection & Rationale

* **Chosen Model:** **MobileNetV2 (Customized for 5-Channel Input)**
* **Why We Chose It:** 
  * **Real-Time Performance:** Deepfake detection requires high frames-per-second (fps) processing. MobileNetV2 offers a lightweight, inverted residual structure that runs efficiently on consumer hardware without sacrificing feature representation.
  * **Custom Modifiability:** Unlike rigid large-scale transformers, MobileNetV2's initial convolutional layers can be readily adapted to accept multi-modal input tensors (expanding from standard 3 RGB channels to 5 channels) by modifying weight matrices directly.
  * **High Accuracy:** Achieved robust performance metrics, hitting **98.4% training accuracy** and a **97.2% validation score** on our evaluation sets.

---

## 3. Dataset Selection & Rationale

* **Dataset:** Curated hybrid subset combining prominent deepfake benchmarks (such as FaceForensics++ and Celeb-DF).
* **Why We Chose It:**
  * **Diverse Generation Artifacts:** These datasets include manipulations from various synthesis methods (e.g., Deepfakes, Face2Face, FaceSwap, NeuralTextures), ensuring the model generalizes well across different compression levels and generation artifacts.
  * **Controlled Facial Variations:** Provides balanced distributions of real human subjects alongside synthetic counterparts under diverse lighting and resolution conditions.

---

## 4. The 5-Channel Data Processing Pipeline & Why

Instead of feeding standard 3-channel RGB images into the neural network, the system processes every face crop through an **Unclipped 5-Channel Preprocessing Pipeline** designed to expose hidden manipulation artifacts:

1. **Channels 1–3 (RGB Color Space):** 
   * *What it does:* Normalized RGB color channels extracted from the detected face bounding box resized to 200x200 pixels.
   * *Why:* Captures natural skin tones, lighting inconsistencies, and standard spatial color blending errors.
2. **Channel 4 (Wavelet Sub-band Feature):** 
   * *What it does:* Applies a 2D discrete wavelet transform (`haar` wavelet via PyWavelets) on the grayscale face image to isolate high-frequency detail components ($LH, HL, HH$), combined via Euclidean norm, resized to 200x200, and min-max normalized.
   * *Why:* Generative models often struggle to replicate authentic skin textures and micro-edges at high frequencies. Wavelet decomposition highlights boundary blending anomalies and high-frequency noise discrepancies.
3. **Channel 5 (Frequency Domain Feature):** 
   * *What it does:* Computes a 2D Fast Fourier Transform (FFT) on the grayscale face image, shifts zero-frequency components to the center, calculates the logarithmic magnitude spectrum ($20 \times \log(|F| + 1e-5)$), and min-max normalizes it.
   * *Why:* GAN-based video generation leaves distinct periodic grid patterns and frequency spectrum signatures that are nearly invisible in the spatial domain but clearly exposed in the frequency domain.

---

## 5. Decision Engine: Weighted Confidence Index (WCI)

To mitigate false alarms caused by temporary compression artifacts or single-frame noise, session risk is scored via:
$$\text{WCI} = (0.50 \times P_{\text{avg}}) + (0.30 \times P_{\text{burst}}) + (0.20 \times P_{\text{peak}})$$
* **Session Average ($P_{\text{avg}}$):** Mean fake probability across the entire session.
* **Peak 1s Burst Window ($P_{\text{burst}}$):** Maximum sliding-window average score over a continuous 1-second interval.
* **Peak Frame AI Score ($P_{\text{peak}}$):** Highest single-frame fake probability recorded.

---

## 6. Tech Stack
* **Language:** Python 3.11
* **Deep Learning:** PyTorch, Torchvision
* **Object Detection:** Ultralytics YOLOv8 Nano (`yolov8n.pt`)
* **Screen Capture:** DXCam (Windows Desktop Duplication API)
* **Image Processing & Math:** OpenCV, NumPy, PyWavelets
* **Reporting:** ReportLab