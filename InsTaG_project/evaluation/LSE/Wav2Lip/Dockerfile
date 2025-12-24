FROM python:3.8-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

WORKDIR /workspace

# =========================================================
# 系统依赖（不要再改 apt 源）
# =========================================================
RUN apt-get update && \
    apt-get install -y \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# pip 国内源
# =========================================================
RUN pip install --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# =========================================================
# Python 依赖（与你项目严格匹配）
# =========================================================
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    numpy==1.19.5 \
    librosa==0.8.1 \
    numba==0.48 \
    tqdm \
    opencv-python==4.5.5.64 \
    scenedetect \
    python_speech_features

# =========================================================
# 拷贝项目
# =========================================================
COPY . /workspace

# =========================================================
# 修复 CRLF + 权限
# =========================================================
RUN sed -i 's/\r$//' evaluation/scores_LSE/calculate_scores_real_videos.sh && \
    chmod +x evaluation/scores_LSE/calculate_scores_real_videos.sh

CMD ["bash"]
