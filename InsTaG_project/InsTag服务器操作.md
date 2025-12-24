# InsTaG 结果复现说明（Docker 版本）

本项目提供了一个 **已经构建完成的 Docker 镜像**，其中包含了 **InsTaG 所需的完整运行环境**，包括：

- Ubuntu 22.04
- CUDA 11.7
- PyTorch 1.12.1（CUDA 版本）
- 所有 Python 依赖
- 所有自定义 CUDA 扩展（已提前编译）

👉 **无需联网、无需安装 conda / pip、无需重新编译任何代码**。

------

## 一、运行环境要求

请确认你的电脑满足以下条件，否则无法复现：

### 1️⃣ 操作系统

- **Linux（x86_64 架构）**
  - Ubuntu / Debian / CentOS / Rocky Linux 等
- ❌ 不支持 macOS
- ❌ 不支持 ARM（如 Apple M 系列）

------

### 2️⃣ GPU 与驱动

- NVIDIA 显卡
- NVIDIA Driver 版本 **支持 CUDA 11.7**
   （推荐 driver ≥ 515）

在宿主机上运行：

```
nvidia-smi
```

如果能正常显示 GPU 信息，说明驱动正常。

------

### 3️⃣ Docker 与 GPU 支持

需要安装：

- Docker ≥ 20.10
- NVIDIA Container Toolkit

请先验证 Docker 能否正常使用 GPU：

```
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi
```

如果该命令能正确显示 GPU 信息，说明环境配置完成。

------

## 二、加载 Docker 镜像（离线）

你会收到一个文件：

```
instag_final.tar
```

在该文件所在目录执行：

```
docker load -i instag_final.tar
```

成功后会看到类似输出：

```
Loaded image: instag:final
```

------

## 三、启动容器

运行以下命令启动容器（务必加上 `--gpus all`）：

```
docker run --gpus all -it \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  -e LD_LIBRARY_PATH=/opt/conda/envs/instag/lib/python3.9/site-packages/torch/lib \
  -v /root/InsTaG:/workspace/InsTaG \
  instag:api-t4-fixed bash

docker run --gpus all   -p 8000:8000   -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128   -e LD_LIBRARY_PATH=/opt/conda/envs/instag/lib/python3.9/site-packages/torch/lib   -v /root/InsTaG:/workspace/InsTaG   instag:api-t4-fixed   python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

docker run --gpus all -d \
  -p 8000:8000 \
  -m 20g \
  --memory-swap 20g \
  --shm-size=8g \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  -e LD_LIBRARY_PATH=/opt/conda/envs/instag/lib/python3.9/site-packages/torch/lib \
  -v /root/InsTaG:/workspace/InsTaG \
  --restart unless-stopped \
  --name instag_api \
  instag:api-t4-fixed \
  python -m uvicorn api_server:app --host 0.0.0.0 --port 8000

docker logs -f instag_api
```

成功后你将进入容器的终端。

------

## 四、环境验证（强烈建议）

在容器内执行：

```
python - <<EOF
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
EOF
```

如果看到：

- PyTorch 版本正常输出
- `CUDA available: True`

说明环境完全可用。

------

## 五、复现实验结果

项目代码位于容器内：

```
/workspace/InsTaG
```

请进入该目录运行实验脚本：

```
cd /workspace/InsTaG
```

```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
```



### 示例：

训练专属talkinghead：

```
bash scripts/train_xx_few.sh data/1 output/test1_hu 0
```

需要被合成的音频处理：

```
python data_utils/hubert.py --wav data/<NAME>.wav
#例如：python data_utils/hubert.py --wav data/1/test/w20.wav
```

合成最终视频：

```
python synthesize_fuse.py \
  -S data/1 \     #
  -M output/test1_hu \
  --audio_extractor hubert \   
  --audio data/<NAME>_hu.npy \   #需要被合成的音频处理后生成的npy文件
  --dilate

例如：
python synthesize_fuse.py \
  -S data/1 \
  -M output/test1_hu \
  --audio_extractor hubert \
  --audio data/1/test/w20_hu.npy \#需要被合成的音频处理后生成的npy文件
  --dilate
  
python synthesize_fuse.py \
  -S data/1 \
  -M output/test1_hu \
  --audio_extractor hubert \
  --audio data/1/test/w20_hu.npy \
  --dilate
```

以及项目中其他实验脚本。

⚠️ **注意事项**：

- 所有 CUDA 扩展已经提前编译完成
- 请勿执行 `pip install` 或 `conda install`
- 请勿重新编译任何模块

------

## 六、常见问题

### 1️⃣ `torch.cuda.is_available()` 为 False

请检查：

- 是否使用了 `--gpus all`
- 宿主机 `nvidia-smi` 是否正常
- Docker 是否支持 GPU

------

### 2️⃣ 脚本运行时报 import 错误

请确认：

- 当前目录是：

  ```
  /workspace/InsTaG
  ```

- 不要修改 `PYTHONPATH`

------

## 七、复现方式说明

该 Docker 镜像提供的是 **二进制级别（Binary-level）复现**：

- 环境已经完全冻结
- 不依赖外部网络或软件源
- 可在任意满足条件的 Linux + NVIDIA GPU 机器上复现相同结果

------

## 八、一句话总结

> **只需三步：**
>
> 1️⃣ `docker load -i instag_final.tar`
>  2️⃣ `docker run --gpus all -it instag:final`
>  3️⃣ 在 `/workspace/InsTaG` 下运行实验脚本
>
> 即可复现 InsTaG 的实验结果。

```
pip install sniffio
pip install flask flask-cors SpeechRecognition zhipuai numpy
conda create -n tfg_backend python=3.9 -y
```



# TFG

```
TFG_ui/
├── backend/                  # 后端核心逻辑（Python）
│   ├── __pycache__/
│   ├── __init__.py
│   ├── chat_engine.py        # 对话/LLM 相关逻辑
│   ├── model_trainer.py      # 模型训练逻辑
│   └── video_generator.py    # 视频生成逻辑
│
├── InsTaG/                   # InsTaG 模块（可能是视频/姿态/生成相关）
│   ├── config.py
│   └── run_instag.sh
│
├── static/                   # 前端静态资源
│   ├── audios/               # 音频资源
│   │   ├── input_hu.npy/
│   │   ├── input.wav
│   │   └── prompt.wav
│   ├── css/
│   │   └── style.css
│   └── videos/               # 生成或示例视频
│       └── input.mp4
│

├── templates/                # Flask / Jinja2 前端模板
│   ├── chat_system.html
│   ├── index.html
│   ├── model_training.html
│   └── video_generation.html
│
├── voiceclone/               # 声音克隆模块
│   ├── generate_wav.py
│   ├── llm_client.py
│   └── main.py
│
├── .gitignore
├── app.py                    # Flask 应用入口
├── readme.md
├── README.md
```



## 1）进入 instag:api 容器（带正确环境变量）

在服务器上执行：

```
docker run --gpus all -it \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  -e LD_LIBRARY_PATH=/opt/conda/envs/instag/lib/python3.9/site-packages/torch/lib \
  -e TORCH_CUDA_ARCH_LIST=7.5 \
  -e FORCE_CUDA=1 \
  -v /root/InsTaG:/workspace/InsTaG \
  instag:api bash
```

> 关键：**T4 必须是 7.5**（不是 7.0）

------

## 2）在容器里：彻底清理旧 so / build / 已安装包

在容器内执行：

```
cd /workspace/InsTaG

# 先把已安装的旧包卸掉（避免加载到旧so）
/opt/conda/envs/instag/bin/python -m pip uninstall -y simple-knn diff-gauss gridencoder || true

# 清理三处编译产物 & 残留 so
rm -rf submodules/simple-knn/build submodules/simple-knn/*.so submodules/simple-knn/**/_C*.so
rm -rf submodules/diff-gaussian-rasterization/build submodules/diff-gaussian-rasterization/*.so submodules/diff-gaussian-rasterization/**/_C*.so
rm -rf gridencoder/build gridencoder/*.so gridencoder/**/_gridencoder*.so
```

------

## 3）在容器里：按 T4（7.5）重编译安装三个模块（必须用同一个 python）

### A) simple-knn

```
cd /workspace/InsTaG/submodules/simple-knn
/opt/conda/envs/instag/bin/python -m pip install -v .
```

### B) diff-gaussian-rasterization

```
cd /workspace/InsTaG/submodules/diff-gaussian-rasterization
/opt/conda/envs/instag/bin/python -m pip install -v .
```

### C) gridencoder

```
cd /workspace/InsTaG/gridencoder
/opt/conda/envs/instag/bin/python -m pip install -v .
```

## 5）把这次“正确编译的结果”固化到镜像里（否则下次换机器又得来一遍）

你现在是 `-v /root/InsTaG:/workspace/InsTaG` 挂载源码的，**但 pip 安装的 .so 在容器层**。
 如果你直接退出容器不保存镜像，换个机器又得重装。

在**另一个终端（宿主机）**执行：

1）查容器 id（找你刚才那个 bash 容器）：

```
docker ps
```

2）把它 commit 成新镜像（比如 instag:api-t4-fixed）：

```
docker commit <容器ID> instag:api-t4-fixed
```

以后你就用 `instag:api-t4-fixed` 起服务。

------

## 6）用新镜像跑服务（注意也要带 LD_LIBRARY_PATH）

```
docker run --gpus all \
  -p 8000:8000 \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  -e LD_LIBRARY_PATH=/opt/conda/envs/instag/lib/python3.9/site-packages/torch/lib \
  -v /root/InsTaG:/workspace/InsTaG \
  instag:api-t4-fixed \
  python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```



# 评测指标：

解读 train_few_shot.sh

## ① `train_face.py` —— 人脸模型训练

```
python train_face.py \
  --type face \
  -s $dataset \
  -m $workspace \
  --init_num 2000 \
  --densify_grad_threshold 0.0005 \
  --audio_extractor hubert \
  --pretrain_path chkpnt_ema_face_latest.pth \
  --iterations 10000 \
  --sh_degree 1 \
  --N_views 250
```

### 作用

👉 **训练“脸部（不含嘴）”的音频驱动模型**

### 关键参数解释

| 参数                       | 说明              |
| -------------------------- | ----------------- |
| `--type face`              | 训练脸部          |
| `-s`                       | 数据集路径        |
| `-m`                       | 输出目录          |
| `--init_num 2000`          | 初始高斯点数量    |
| `--densify_grad_threshold` | 点云 densify 阈值 |
| `--audio_extractor`        | HuBERT 音频特征   |
| `--pretrain_path`          | 人脸预训练模型    |
| `--iterations 10000`       | 训练步数          |
| `--sh_degree 1`            | 球谐阶数          |
| `--N_views`                | 训练帧数          |

------

## ② `train_mouth.py` —— 嘴部模型训练

```
python train_mouth.py \
  --type mouth \
  -s $dataset \
  -m $workspace \
  --init_num 5000 \
  --audio_extractor hubert \
  --pretrain_path chkpnt_ema_mouth_latest.pth \
  --iterations 10000 \
  --sh_degree 1 \
  --N_views 250
```

### 作用

👉 **单独训练嘴部（高分辨率、对齐音频）**

### 不同点

- `--type mouth`
- `--init_num 5000`（嘴更精细）

------

## ③ `train_fuse_con.py` —— 脸 + 嘴 融合训练

```
python train_fuse_con.py \
  -s $dataset \
  -m $workspace \
  --opacity_lr 0.001 \
  --audio_extractor hubert \
  --iterations 2000 \
  --sh_degree 1 \
  --N_views 250
```

### 作用

👉 **融合 face + mouth，做最终一致性优化**

| 参数                | 含义             |
| ------------------- | ---------------- |
| `--opacity_lr`      | 不透明度学习率   |
| `--iterations 2000` | 融合阶段训练较短 |

------

## ④ `synthesize_fuse.py` —— 合成视频（推理）

```
python synthesize_fuse.py \
  -s $dataset \
  -m $workspace \
  --eval \
  --audio_extractor hubert \
  --dilate
```

### 作用

👉 **生成最终 talking-head 视频**

| 参数       | 说明                        |
| ---------- | --------------------------- |
| `--eval`   | 推理模式                    |
| `--dilate` | 嘴部膨胀/平滑（常见于唇形） |

------

## ⑤ `metrics.py` —— 计算评测指标

```
python metrics.py \
  $workspace/test/ours_None/renders/out.mp4 \
  $workspace/test/ours_None/gt/out.mp4
```

### 作用

👉 **对比生成视频 vs GT 视频**

通常会算：

- L1 / PSNR
- SSIM
- LPIPS
- Sync 指标（可能）



### 1️⃣ 重新连上服务器后，先别跑代码

先看系统内存状态：

```
free -h
```

如果你看到：

- `available` 很小
- `swap` 用满

那就对了。

------

### 2️⃣ 杀掉残留的 Docker 容器 / 进程

```
docker ps -a
```

如果容器还在：

```
docker stop <container_name>
docker rm <container_name>
```

或者直接全停：

```
docker stop $(docker ps -q)
```

------

### 3️⃣ 重启 Docker（关键一步）

```
systemctl restart docker
```

⚠️ 这一步**极其重要**，可以清掉 cgroup 和内存状态

------

### 4️⃣ 再看一次内存

```
free -h
```



### 一次性删所有已停止容器（慎用）

```
docker container prune
```



一、

联系组长范一朴15671095269启动服务器。使用ssh root@1.94.114.212连接服务器，密码为040421Pyb。登录服务器。

二、

```
1.在服务器中使用以下命令启动容器。
docker run --gpus all -it \
  -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
  -e LD_LIBRARY_PATH=/opt/conda/envs/instag/lib/python3.9/site-packages/torch/lib \
  -v /root/InsTaG:/workspace/InsTaG \
  instag:api-t4-fixed bash
  
2.训练 
bash scripts/train_xx_few.sh data/test/<ID> output/test/<ID> 0

3.音频特征提取
python data_utils/hubert.py --wav data/test/<ID>/<NAME>.wav #NAME为测试视频名称例如Lieu

4.合成视频
例如合成第一条测试视频。
python synthesize_fuse.py \
  -S data/test/1 \
  -M output/test/1 \
  --use_train \
  --dilate \
  --audio data/test/1/Lieu_hu.npy \
  --audio_extractor hubert

5.评测
python metrics.py \
  output/test/1/train/ours_xxx/renders/out.mp4 \
  data/test/1/Lieu.mp4
  
```



若要新传一个视频：

```
一、视频处理
1.跑一个脚本来预处理视频
有一个小环境问题在华为云上一直无法解决。但是在本机可以执行。请联系组长来处理视频。
python data_utils/process.py data/<ID>/<ID>.mp4

python data_utils/split.py data/<ID>/<ID>.mp4    # Optional. To retain at least 12s data for evaluation.

例如：python data_utils/process.py data/1copy/aud.mp4

2.获取动作单元
Run FeatureExtraction in OpenFace, rename and move the output CSV file to data/<ID>/au.csv.
在OpenFace上跑FeatureExtraction.exe，然后将结果重命名为au.csv，放入指定路径data/<ID>/au.csv。
可以联系范一朴组长处理该步骤。她已经下载好工具。
或者让组长传压缩包，自己处理。

3.Generate tooth masks
export PYTHONPATH=./data_utils/easyportrait 
python data_utils/easyportrait/create_teeth_mask.py data/<ID>

例如：python data_utils/easyportrait/create_teeth_mask.py data/1copy

4.Generate geometry priors.
该步骤涉及一个全新的环境。因此此时开启一个另一个docker镜像的容器，运行命令bash /workspace/data_utils/sapiens/run.sh data/<ID>

例如直接运行该命令即可：
docker run --gpus all -it --rm \
  --shm-size=8g \
  -v /root/InsTaG:/workspace \
  sapiens_lite:cu121 \
  bash /workspace/data_utils/sapiens/run.sh data/1copy


例如：bash data_utils/sapiens/run.sh data/1copy

二、音频预处理
1.使用hubert模型处理音频。
python data_utils/hubert.py --wav data/<ID>/<NAME>.wav

例如：python data_utils/hubert.py --wav data/1copy/aud.wav

三、训练
bash scripts/train_xx_few.sh data/<ID> output/<project_name> 0

例如：bash scripts/train_xx_few.sh data/1copy output/1copy 0

四、合成
python synthesize_fuse.py -S data/<ID> -M output/<project_name> --dilate --use_train --audio <preprocessed_audio_feature>.npy --audio_extractor hubert

例如:
python synthesize_fuse.py \
  -S data/1copy \
  -M output/1copy \
  --use_train \
  --dilate \
  --audio data/aud_hu.npy \
  --audio_extractor hubert

五、评估
该步骤不在服务器运行，请打开评估部分的代码文件夹，阅读 Readme 进行操作。
```

