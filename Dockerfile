FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ISOLATE_SESSION_OUTPUTS=true \
    YOLO_CONFIG_DIR=/tmp/ultralytics

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only PyTorch build to avoid pulling CUDA libraries onto a
# CPU-hosted demo, then install the rest of the project's dependencies.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Fetch and checksum the pinned aerial checkpoint during image build, so the
# deployed app never attempts to download weights while a visitor is waiting.
RUN python tools/download_aerial_model.py

EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=7860", "--server.headless=true"]
