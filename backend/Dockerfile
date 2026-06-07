FROM docker.m.daocloud.io/library/python:3.11-slim AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxcb-shm0 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch 单独一层（在 COPY requirements.txt 前面），后续改 requirements.txt 不会让 torch 重下
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision torchaudio

COPY backend/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    pip uninstall -y onnxruntime && \
    pip install --no-cache-dir --force-reinstall --no-deps onnxruntime-gpu>=1.17.0

COPY backend/ .

RUN mkdir -p /app/logs /app/uploads /app/vector_store

EXPOSE 8000

# 让 onnxruntime-gpu 找到 torch 和 nvidia pip 包自带的 CUDA 库
RUN find /usr/local/lib/python3.11/site-packages/nvidia -type d -name "lib" > /etc/ld.so.conf.d/nvidia-cuda.conf && ldconfig
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.11/site-packages/torch/lib

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]