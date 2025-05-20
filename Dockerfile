# Base image with Ubuntu and CUDA 12.1
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv python3.10-dev \
    libopenjp2-tools \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.10 default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

RUN cd /home \
    && git clone https://github.com/mimunzar/mapstitcher.git

WORKDIR /home/mapstitcher
RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["python3", "image_stitch_batch.py", "--path", "/data/", "--output", "/data/stitched.jp2"]
# Default command
# docker run --rm -it --gpus all -v ~/workspace/mapstitcher/test_data:/data/ tag
