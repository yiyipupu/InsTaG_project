import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
import pyiqa

# 选择设备：有 GPU 就用 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
niqe_metric = pyiqa.create_metric("niqe", device=device)


def frame_generator(path: str):
    """
    流式读取视频帧：逐帧 yield (RGB, float32, [0,1])，避免一次性读入内存。
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            yield frame
    finally:
        cap.release()


def resize_to_match(a, b):
    """把 b resize 到和 a 一样大小"""
    h, w = a.shape[:2]
    return cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)


def eval_pair(
    gt_path,
    pred_path,
    save_root=None,
    video_id=None,
    frame_stride=5,
    niqe_batch_size=16,
    export_fid_frames=False,
):
    """
    评估一对视频（流式版本）：
    - 逐帧读取，不再一次性加载所有帧 => 解决 OOM(Killed)
    - 每隔 frame_stride 帧计算一次 PSNR/SSIM/NIQE
    - NIQE 用 batch 形式在 GPU 上算
    - export_fid_frames=True 时才导出帧到 save_root 下
    """
    if frame_stride <= 0:
        raise ValueError("frame_stride must be >= 1")
    if niqe_batch_size <= 0:
        raise ValueError("niqe_batch_size must be >= 1")

    psnr_list = []
    ssim_list = []
    niqe_list = []
    niqe_batch = []

    # 导出帧目录（可选）
    real_dir = fake_dir = None
    if export_fid_frames and save_root is not None and video_id is not None:
        real_dir = os.path.join(save_root, "frames_real")
        fake_dir = os.path.join(save_root, "frames_fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)

    gt_gen = frame_generator(gt_path)
    pred_gen = frame_generator(pred_path)

    num_frames_total = 0
    num_frames_evaluated = 0

    # zip 会在较短视频结束时停止，等价于取 min(len(gt), len(pred))
    for idx, (gt, pred_raw) in enumerate(zip(gt_gen, pred_gen)):
        num_frames_total += 1

        # 跳帧逻辑：只评测 stride 命中的帧
        if idx % frame_stride != 0:
            continue

        pred = resize_to_match(gt, pred_raw)

        # PSNR, SSIM
        psnr_val = peak_signal_noise_ratio(gt, pred, data_range=1.0)
        ssim_val = structural_similarity(gt, pred, channel_axis=-1, data_range=1.0)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        num_frames_evaluated += 1

        # NIQE 输入缓存
        pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
        niqe_batch.append(pred_tensor)

        # 满 batch 就算一波
        if len(niqe_batch) >= niqe_batch_size:
            batch_tensor = torch.cat(niqe_batch, dim=0).to(device)  # [B,3,H,W]
            with torch.no_grad():
                batch_scores = niqe_metric(batch_tensor)  # [B]
            niqe_list.extend(batch_scores.detach().cpu().numpy().tolist())
            niqe_batch.clear()

        # 导出帧（用于之后算 FID），可选
        if export_fid_frames and real_dir is not None and fake_dir is not None:
            real_name = f"{video_id}_frame_{idx:05d}_real.png"
            fake_name = f"{video_id}_frame_{idx:05d}_fake.png"
            cv2.imwrite(
                os.path.join(real_dir, real_name),
                cv2.cvtColor((gt * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(fake_dir, fake_name),
                cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
            )

    # 最后不满 batch 的 NIQE
    if niqe_batch:
        batch_tensor = torch.cat(niqe_batch, dim=0).to(device)
        with torch.no_grad():
            batch_scores = niqe_metric(batch_tensor)
        niqe_list.extend(batch_scores.detach().cpu().numpy().tolist())
        niqe_batch.clear()

    if num_frames_total == 0:
        raise RuntimeError(f"No frames in {gt_path} or {pred_path}")

    if num_frames_evaluated == 0:
        raise RuntimeError(
            f"No frames were evaluated. Check frame_stride={frame_stride} "
            f"for very short videos."
        )

    return {
        "PSNR": float(np.mean(psnr_list)),
        "SSIM": float(np.mean(ssim_list)),
        "NIQE": float(np.mean(niqe_list)) if len(niqe_list) > 0 else float("nan"),
        "num_frames_total": int(num_frames_total),
        "num_frames_evaluated": int(num_frames_evaluated),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, default="videos/original")
    parser.add_argument("--pred_dir", type=str, default="videos/generated")
    parser.add_argument(
        "--save_fid_frames_root",
        type=str,
        default="fid_frames",
        help="导出帧的根目录，用于之后算FID；配合 --export_fid_frames 使用",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=5,
        help="每隔多少帧计算一次指标，1=每帧都算，数字越大越快但越粗略",
    )
    parser.add_argument(
        "--niqe_batch_size",
        type=int,
        default=16,
        help="NIQE 计算时的 batch size（越大越快，但越占显存）",
    )
    parser.add_argument(
        "--export_fid_frames",
        action="store_true",
        help="如果加上这个 flag，就会把用于评估的帧导出到 save_fid_frames_root 下",
    )
    args = parser.parse_args()

    print(f"Using device: {device}")

    gt_files = sorted([f for f in os.listdir(args.gt_dir) if f.lower().endswith(".mp4")])
    pred_files = sorted([f for f in os.listdir(args.pred_dir) if f.lower().endswith(".mp4")])

    common = sorted(list(set(gt_files) & set(pred_files)))
    if not common:
        print("No matching mp4 files between original/ and generated/")
        return

    results = {}
    for name in tqdm(common, desc="Evaluating video pairs"):
        gt_path = os.path.join(args.gt_dir, name)
        pred_path = os.path.join(args.pred_dir, name)
        video_id = os.path.splitext(name)[0]

        metrics = eval_pair(
            gt_path,
            pred_path,
            save_root=args.save_fid_frames_root,
            video_id=video_id,
            frame_stride=args.frame_stride,
            niqe_batch_size=args.niqe_batch_size,
            export_fid_frames=args.export_fid_frames,
        )
        results[name] = metrics

    print("\nPer-video metrics:")
    for name, m in results.items():
        print(
            f"{name}: PSNR={m['PSNR']:.4f}, "
            f"SSIM={m['SSIM']:.4f}, "
            f"NIQE={m['NIQE']:.4f}, "
            f"frames_total={m['num_frames_total']}, "
            f"frames_eval={m['num_frames_evaluated']}"
        )

    psnr_all = np.mean([m["PSNR"] for m in results.values()])
    ssim_all = np.mean([m["SSIM"] for m in results.values()])
    niqe_all = np.mean([m["NIQE"] for m in results.values()])

    print("\nOverall average (over evaluated frames):")
    print(f"PSNR={psnr_all:.4f}, SSIM={ssim_all:.4f}, NIQE={niqe_all:.4f}")


if __name__ == "__main__":
    main()
