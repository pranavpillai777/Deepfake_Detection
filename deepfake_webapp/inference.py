"""Model loading, preprocessing, Grad-CAM and video analysis.

Preprocessing matches run_live_analysis.py (5 channels: RGB + Haar wavelet
energy + FFT log-magnitude, 200x200, YOLO person box -> top 45% as the head crop).
"""
import logging
import os
import threading

import cv2
import numpy as np
import pywt
import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

from scoring import aggregate

log = logging.getLogger("deepfake")

CHECKPOINT_PATH = os.getenv(
    "CHECKPOINT_PATH", os.path.join("checkpoints", "mobilenet_v2_deepfake.pth")
)
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "5"))          # frames analysed per second of video
MAX_VIDEO_SECONDS = float(os.getenv("MAX_VIDEO_SECONDS", "120"))
MIN_FACES = int(os.getenv("MIN_FACES", "5"))

YOLO_CONF = 0.45
HEAD_FRACTION = 0.45   # top 45% of the person box
MIN_CROP = 40          # px
FACE_SIZE = 200
BLUR_MIN = 12.0        # Laplacian variance, same as the live script


class VideoRejected(Exception):
    """Raised for problems the user can fix (bad file, too long, ...)."""


def build_model():
    model = models.mobilenet_v2(weights=None)
    model.features[0][0] = nn.Conv2d(5, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model


def _load_state_dict(path, device):
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        log.warning("weights_only load failed; falling back to full unpickle (trusted file only)")
        obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):
        for key in ("state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def preprocess_5ch(face_bgr_200, device):
    face_rgb = cv2.cvtColor(face_bgr_200, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    gray = cv2.cvtColor(face_bgr_200, cv2.COLOR_BGR2GRAY)

    _, (LH, HL, HH) = pywt.dwt2(gray, "haar")
    wavelet = np.sqrt(LH**2 + HL**2 + HH**2)
    wavelet = cv2.resize(wavelet, (FACE_SIZE, FACE_SIZE))
    wavelet = cv2.normalize(wavelet, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    fshift = np.fft.fftshift(np.fft.fft2(gray))
    mag = 20 * np.log(np.abs(fshift) + 1e-5)
    fft = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    stacked = np.stack(
        [face_rgb[:, :, 0], face_rgb[:, :, 1], face_rgb[:, :, 2], wavelet, fft], axis=0
    )
    return torch.tensor(stacked, dtype=torch.float32).unsqueeze(0).to(device)


def _crop_head(frame, xyxy):
    """Crop the head region from the largest detected person."""
    h, w = frame.shape[:2]
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = (int(v) for v in xyxy[int(np.argmax(areas))])

    x, y = max(0, x1), max(0, y1)
    crop_w = min(x2, w) - x
    crop_h = min(int((y2 - y1) * HEAD_FRACTION), h - y)
    if crop_w <= MIN_CROP or crop_h <= MIN_CROP:
        return None
    return frame[y : y + crop_h, x : x + crop_w]


class Detector:
    def __init__(self):
        if not os.path.exists(CHECKPOINT_PATH):
            raise FileNotFoundError(
                f"Checkpoint not found at {CHECKPOINT_PATH}. Set CHECKPOINT_PATH."
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Using device: %s", self.device)

        self.model = build_model()
        self.model.load_state_dict(_load_state_dict(CHECKPOINT_PATH, self.device))
        self.model.to(self.device).eval()

        self.yolo = YOLO(YOLO_WEIGHTS)
        self._lock = threading.Lock()  # one analysis at a time (shared model + Grad-CAM state)

    # ------------------------------------------------------------------ Grad-CAM
    def _gradcam(self, tensor, class_idx=1):
        store = {}

        def fwd_hook(_module, _inp, out):
            store["act"] = out.detach()
            if out.requires_grad:
                out.register_hook(lambda g: store.__setitem__("grad", g.detach()))

        handle = self.model.features[-1].register_forward_hook(fwd_hook)
        try:
            with torch.enable_grad():
                self.model.zero_grad()
                out = self.model(tensor)
                out[0, class_idx].backward()
        finally:
            handle.remove()

        weights = store["grad"].mean(dim=(2, 3), keepdim=True)
        cam = torch.clamp((weights * store["act"]).sum(dim=1).squeeze(0), min=0).cpu().numpy()
        cam = cv2.resize(cam, (FACE_SIZE, FACE_SIZE))
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # ------------------------------------------------------------------ Analysis
    def analyze(self, video_path, progress_cb=None):
        """Returns (result_dict, gradcam_bgr_or_None). Blocks; call from a worker thread."""
        with self._lock:
            return self._analyze(video_path, progress_cb or (lambda p: None))

    def _analyze(self, video_path, progress_cb):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise VideoRejected("This file couldn't be read as a video. Try an MP4 or MOV.")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            if fps <= 1 or fps > 240:
                fps = 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 0 and total / fps > MAX_VIDEO_SECONDS:
                raise VideoRejected(
                    f"Videos can be at most {int(MAX_VIDEO_SECONDS)} seconds long. Trim the clip and try again."
                )

            step = max(1, int(round(fps / SAMPLE_FPS)))
            sample_fps = fps / step

            timeline = []          # [time_sec, fake_prob]
            frames_sampled = frames_no_face = frames_blurry = 0
            best_prob, best_face, best_tensor, best_t = -1.0, None, None, 0.0

            idx = 0
            while cap.grab():
                if idx % step == 0:
                    t = idx / fps
                    if t > MAX_VIDEO_SECONDS:
                        raise VideoRejected(
                            f"Videos can be at most {int(MAX_VIDEO_SECONDS)} seconds long. Trim the clip and try again."
                        )
                    ok, frame = cap.retrieve()
                    if ok:
                        frames_sampled += 1
                        prob, face, tensor = self._score_frame(frame)
                        if prob is None:
                            if face == "blurry":
                                frames_blurry += 1
                            else:
                                frames_no_face += 1
                        else:
                            timeline.append([round(t, 2), round(prob, 4)])
                            if prob > best_prob:
                                best_prob, best_face, best_tensor, best_t = prob, face, tensor, t
                    if total > 0:
                        progress_cb(min(0.99, idx / total))
                idx += 1

            if frames_sampled == 0:
                raise VideoRejected("No frames could be decoded from this video.")
            duration = idx / fps
        finally:
            cap.release()

        scores = [p for _, p in timeline]
        agg = aggregate(scores, sample_fps, MIN_FACES)

        result = {
            **agg,
            "peak_time": round(best_t, 2),
            "duration": round(duration, 2),
            "sample_fps": round(sample_fps, 2),
            "frames_sampled": frames_sampled,
            "frames_no_face": frames_no_face,
            "frames_blurry": frames_blurry,
            "timeline": timeline,
        }

        cam_img = None
        if best_face is not None:
            heat = self._gradcam(best_tensor, class_idx=1)
            heat = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(best_face, 0.6, heat, 0.4, 0)
            cam_img = np.hstack([best_face, overlay])
        return result, cam_img

    def _score_frame(self, frame):
        """Returns (fake_prob, face_200, tensor), or (None, 'no_face'|'blurry', None)."""
        det = self.yolo(frame, verbose=False, conf=YOLO_CONF, classes=[0])
        if not len(det) or len(det[0].boxes) == 0:
            return None, "no_face", None

        crop = _crop_head(frame, det[0].boxes.xyxy.cpu().numpy())
        if crop is None or crop.size == 0:
            return None, "no_face", None

        face = cv2.resize(crop, (FACE_SIZE, FACE_SIZE))
        if cv2.Laplacian(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < BLUR_MIN:
            return None, "blurry", None

        tensor = preprocess_5ch(face, self.device)
        with torch.no_grad():
            prob = torch.softmax(self.model(tensor), dim=1)[0, 1].item()
        return prob, face.copy(), tensor
