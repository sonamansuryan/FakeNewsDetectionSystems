FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
 && update-alternatives --install /usr/bin/pip    pip    /usr/bin/pip3    1

WORKDIR /app

COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support, then the rest
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HOME=/app/.cache/huggingface \
    CUDA_VISIBLE_DEVICES=0

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn with 1 worker (critical: prevents loading models multiple times)
CMD ["gunicorn", \
     "src.api.app:create_app()", \
     "--bind", "0.0.0.0:5000", \
     "--workers", "1", \
     "--timeout", "300", \
     "--log-level", "info"]