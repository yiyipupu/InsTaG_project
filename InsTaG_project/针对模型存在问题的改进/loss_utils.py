#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np


def normalize(input, mean=None, std=None):
    input_mean = torch.mean(input, dim=1, keepdim=True) if mean is None else mean
    input_std = torch.std(input, dim=1, keepdim=True) if std is None else std
    return (input - input_mean) / (input_std + 1e-2 * torch.std(input.reshape(-1)))


def patchify(input, patch_size):
    patches = F.unfold(input, kernel_size=patch_size, stride=patch_size).permute(0, 2, 1).view(-1, 3, patch_size,
                                                                                               patch_size)
    return patches


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def charbonnier_loss(x, eps=1e-3):
    """Charbonnier损失函数，对异常值更鲁棒"""
    return torch.sqrt(x * x + eps * eps).mean()


def robust_depth_loss(pred, target, mask=None, eps=1e-3):
    """
    鲁棒的深度损失函数，结合Charbonnier损失和梯度一致性损失

    参数:
        pred: 预测深度图 [B, 1, H, W] 或 [B, H, W]
        target: 目标深度图 [B, 1, H, W] 或 [B, H, W]
        mask: 有效区域掩码 [B, H, W]
        eps: Charbonnier损失的小常数

    返回:
        loss: 总损失
    """
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # 数据项：Charbonnier损失
    diff = pred - target
    data_loss = charbonnier_loss(diff, eps)

    # 梯度一致性损失
    grad_x_pred = pred[:, :, 1:] - pred[:, :, :-1]
    grad_y_pred = pred[:, 1:, :] - pred[:, :-1, :]
    grad_x_target = target[:, :, 1:] - target[:, :, :-1]
    grad_y_target = target[:, 1:, :] - target[:, :-1, :]

    grad_loss_x = charbonnier_loss(grad_x_pred - grad_x_target, eps)
    grad_loss_y = charbonnier_loss(grad_y_pred - grad_y_target, eps)
    grad_loss = (grad_loss_x + grad_loss_y) * 0.5

    # 尺度不变损失（可选）
    pred_norm = pred / (torch.mean(pred, dim=[1, 2], keepdim=True) + 1e-7)
    target_norm = target / (torch.mean(target, dim=[1, 2], keepdim=True) + 1e-7)
    scale_invariant_loss = charbonnier_loss(pred_norm - target_norm, eps)

    # 总损失
    total_loss = data_loss + 0.1 * grad_loss + 0.05 * scale_invariant_loss

    if mask is not None:
        total_loss = total_loss * mask.mean()

    return total_loss


def robust_normal_loss(pred, target, mask=None):
    """
    鲁棒的法线损失函数

    参数:
        pred: 预测法线图 [B, 3, H, W] 或 [B, H, W, 3]
        target: 目标法线图 [B, 3, H, W] 或 [B, H, W, 3]
        mask: 有效区域掩码 [B, H, W]

    返回:
        loss: 余弦相似度损失
    """
    if pred.dim() == 4:
        pred = pred.permute(0, 2, 3, 1)  # [B, H, W, 3]
    if target.dim() == 4:
        target = target.permute(0, 2, 3, 1)  # [B, H, W, 3]

    # 归一化
    pred = F.normalize(pred, p=2, dim=-1)
    target = F.normalize(target, p=2, dim=-1)

    # 余弦相似度损失
    cos_sim = torch.sum(pred * target, dim=-1)  # [B, H, W]
    loss = (1 - cos_sim).mean()  # 余弦距离

    if mask is not None:
        loss = loss * mask.mean()

    return loss


def temporal_smoothness_loss(current, previous, mask=None):
    """
    时序平滑损失函数

    参数:
        current: 当前帧特征 [B, C, H, W] 或 [B, C]
        previous: 前一帧特征 [B, C, H, W] 或 [B, C]
        mask: 有效区域掩码 [B, H, W]

    返回:
        loss: 时序平滑损失
    """
    if current.dim() == 4 and previous.dim() == 4:
        # 空间特征图
        diff = current - previous
        if mask is not None:
            diff = diff * mask.unsqueeze(1)
        loss = charbonnier_loss(diff)
    else:
        # 向量特征
        loss = charbonnier_loss(current - previous)

    return loss


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)