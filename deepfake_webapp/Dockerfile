FROM python:3.11-slim

# OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image small. Delete this line for a GPU image
# (and start from an nvidia/cuda or pytorch/pytorch base instead).
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download YOLO weights at build time so the first request isn't slow
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY . .

ENV CHECKPOINT_PATH=/app/checkpoints/mobilenet_v2_deepfake.pth \
    DATA_DIR=/tmp/deepfake_data

EXPOSE 8000
# ONE worker: the model and job table live in this process's memory.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
