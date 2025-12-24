# =========================================================
# 基础镜像：CUDA 11.8 + cuDNN（运行时用 GPU）
# =========================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# =========================================================
# 1. 系统依赖（国内源）
# =========================================================
RUN sed -i 's@archive.ubuntu.com@mirrors.tuna.tsinghua.edu.cn@g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# 2. pip 国内镜像
# =========================================================
RUN python3 -m pip install --upgrade pip && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# =========================================================
# 3. 安装 PyTorch（CUDA 11.8）
# =========================================================
RUN pip install \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# =========================================================
# 4. Python 依赖（不含 torch）
# =========================================================
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================================================
# 5. HuggingFace 国内镜像（非常关键）
# =========================================================
ENV HF_ENDPOINT=https://hf-mirror.com

# =========================================================
# 6. 预下载 NIQE 权重（CPU 模式，build 阶段安全）
# =========================================================
RUN python3 - << 'EOF'
import torch
import pyiqa

device = torch.device("cpu")
metric = pyiqa.create_metric("niqe", device=device)

x = torch.rand(1, 3, 256, 256)
with torch.no_grad():
    _ = metric(x)

print("NIQE weights downloaded (CPU mode).")
EOF

# =========================================================
# 7. 拷贝评测代码
# =========================================================
COPY eval_videos.py .

# =========================================================
# 8. 运行阶段再用 GPU
# =========================================================
CMD ["python3", "eval_videos.py"]
