# 一、运行步骤

首先打开终端，进入 evaluation 目录

确认一下：

```
ls
```

你应该能看到：

```
Dockerfile
eval_videos.py
fid_frames
videos
等
```

------

### 1.build Docker镜像

```
docker build -t instag-eval .
```



### 2.进入Docker容器同时运行，计算评测指标，以及导出FID所需帧

```
docker run --rm --gpus all \
  -v $(pwd)/videos:/app/videos \
  instag-eval
```

```
输出：PSNR=22.7980, SSIM=0.7786, NIQE=6.3269
```



### 3.运行代码获得FID指标

```
docker run --rm --gpus all \
  -v $(pwd)/fid_frames:/app/fid_frames \
  instag-eval \
  python3 -m pytorch_fid /app/fid_frames/frames_real /app/fid_frames/frames_fake
```

```
输出：FID:  10.841960923624555
```



### 4.LSE-C，LSE-D评测指标请阅读LSE文件夹中Readme进行配置复现。
