# 评估指标


# LSE-D & LSE-C 唇形同步评估说明

本文件说明我们在项目中如何使用 **LSE-D** 和 **LSE-C** 对视频的唇形同步质量进行定量评估，方便老师了解实验流程和指标含义。

---

## 1. 指标简介

### 1.1 LSE-D（Lip Sync Error – Distance）

- 含义：音频与嘴部运动之间的**特征距离误差**  
- 模型：使用 SyncNet 对 **视频帧中的嘴部区域** 和 **对应音频片段** 提取特征向量，然后计算二者之间的距离  
- 越低越好：  
  - 数值越小 → 音频和唇形越同步  
  - 数值越大 → 嘴部动作和声音不同步（提前/滞后/嘴形不匹配等）

**经验范围（仅作直观理解）：**

| LSE-D 范围 | 解释        |
|-----------|-------------|
| 4 – 7     | 同步很好    |
| 7 – 10    | 同步一般    |
| > 10      | 同步较差    |

---

### 1.2 LSE-C（Lip Sync Confidence）

- 含义：SyncNet 对“当前这一段音频与嘴部运动是同步的”这一判断的**置信度**  
- 越高越好：  
  - 数值越高 → 模型越确信音视频是同步的  
  - 数值越低 → 模型认为同步程度不可靠  

**经验范围（仅作直观理解）：**

| LSE-C 范围 | 解释               |
|-----------|--------------------|
| > 5.5     | 高置信同步         |
| 3 – 5     | 一般置信度         |
| < 3       | 可能不同步 / 质量差 |

---

### 1.3 为什么两个指标要一起看？

- **LSE-D**：衡量“误差有多大”（越低越好），更偏“距离/误差”的角度  
- **LSE-C**：衡量“模型有多自信”（越高越好），更偏“判断置信度”的角度  


---

## 2. 实验环境与依赖

本评估基于 Wav2Lip 项目中自带的 SyncNet 评估代码。

### 2.1 主要依赖

- Python 3.x  
- PyTorch（用于加载 SyncNet 模型）  
- OpenCV  
- NumPy  
- PySceneDetect（用于场景/片段检测）  
- 预训练 SyncNet 模型权重

（这些依赖在 Wav2Lip 的 `requirements.txt` 中基本都有列出，可以通过 `pip install -r requirements.txt` 安装。）

---

## 3. 数据准备

我们将待评估的视频整理在 `evaluation/videos` 目录下：

```text
Wav2Lip/
└── evaluation/
    ├── videos/
    │   ├── original/     # 原始真实视频
    │   │   ├── 01.mp4
    │   │   ├── 02.mp4
    │   │   └── ...
    │   └── generated/    # 我们方法生成的视频
    │       ├── 01.mp4
    │       ├── 02.mp4
    │       └── ...
    └── scores_LSE/
        ├── calculate_scores_real_videos.sh
        └── ...
```

要求：

- 每个视频里人物嘴部区域 **清晰可见**  
- 视频包含对应音频（语音）轨道  
- `original` 与 `generated` 中可以按“相同编号”一一对应，便于对比  

---

## 4. 如何运行 LSE-D / LSE-C 评估（详细步骤）

下面以WSL+Ubuntu22.04及 本项目路径为例说明。  
假设项目路径为：

```text
/root/InsTaG_project/evaluation/LSE/Wav2Lip
```

### 4.1 进入评估脚本目录

```bash
cd /root/InsTaG_project/evaluation/LSE/Wav2Lip/evaluation/scores_LSE
```


---

### 4.2 配置环境

```
conda create -n wav2lip python=3.8 -y
conda activate wav2lip
```

```
#安装ffmepg
apt install -y ffmepg
```

```
#命令行直接执行
pip install \
  numpy==1.19.5 \
  librosa==0.8.1 \
  numba==0.48 \
  torch \
  torchvision \
  tqdm \
  opencv-python==4.5.5.64 \
  scenedetect \
  python_speech_features
```

```
#验证（一定要做）
python - << EOF
import cv2
import torch
import librosa
import scenedetect
import python_speech_features
print("✅ 环境 OK")
EOF
```

### 4.2 评估生成视频（generated）

运行：

```bash
bash /root/InsTaG_project/evaluation/LSE/Wav2Lip/evaluation/scores_LSE/calculate_scores_real_videos.sh /root/InsTaG_project/evaluation/LSE/Wav2Lip/evaluation/videos/generated
```

这一步脚本会自动完成：

1. 遍历 `../videos/generated` 目录中的所有 `.mp4` 视频  
2. 对每个视频：
   - 用 ffmpeg 提取视频帧 & 音频
   - 用 S3FD 检测人脸，裁剪嘴部区域
   - 用 PySceneDetect 做简单场景切分
   - 调用 SyncNet 对每一段音频-视频片段计算：
     - LSE-D（唇形误差）
     - LSE-C（同步置信度）
3. 计算完后，将每个视频的评价结果写入当前目录下的 `all_scores.txt`

运行过程中终端会输出类似：

- ffmpeg 的处理进度  
- S3FD 的检测信息：`xxxx dets; xx.xx Hz`  
- SyncNet 的评估日志  

全部视频处理完成后，`all_scores.txt` 文件就会生成/更新。

---

### 4.3 评估原始视频（original）

如果想要对比原视频和生成视频，同样可以在 `scores_LSE` 目录下执行：

```bash
bash /root/InsTaG_project/evaluation/LSE/Wav2Lip/evaluation/scores_LSE/calculate_scores_real_videos.sh /root/InsTaG_project/evaluation/LSE/Wav2Lip/evaluation/videos/original
```

流程与 `generated` 完全相同，只是这次评估的是原始的视频。  
建议把两次的结果分别保存，便于对比，例如：

```bash
# 评估 generated 之后
cp all_scores.txt all_scores_generated.txt

# 再评估 original 之后
cp all_scores.txt all_scores_original.txt
```

---

### 4.4 生成结果文件格式

脚本生成的 `all_scores.txt` 内容形如：

```text
8.614585 6.548642
9.626425 3.5426521
11.320777 3.068575
10.022132 4.377825
10.091897 2.5386457
9.453509 5.563472
9.164361 2.9169464
```

含义：

- **每一行对应一个视频文件**，顺序与脚本遍历目录中文件的顺序一致  
- 每行由两个浮点数构成：
  - 第一列：LSE-D（Lip Sync Error – Distance，越低越好）  
  - 第二列：LSE-C（Lip Sync Confidence，越高越好）

例如：

```text
8.614585 6.548642
```

可解释为：

- LSE-D = 8.61 → 同步误差为 8.61，属于中等水平  
- LSE-C = 6.55 → 模型对同步关系比较有信心  
