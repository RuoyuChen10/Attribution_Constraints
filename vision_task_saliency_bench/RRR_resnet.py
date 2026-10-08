import os
from datetime import timedelta
import argparse
import math
from typing import Tuple, List
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image
from tqdm import tqdm

from dataloader import IMAGENET_MEAN, IMAGENET_STD, make_dataloaders
# from interpretation.HUMAN_LIMA_Efficient import HumanLIMA
from utils import mkdir, SubRegionDivision

# 只保留 math kernel（最可能支持二阶导）
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# -------------------
# 工具函数 & 数据
# -------------------
def read_list_file(list_path: str):
    items = []
    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_p, mask_p, label = line.split()
            items.append((img_p, mask_p, label))
    return items

def build_label_map(*list_files: str):
    labels = []
    for lf in list_files:
        for _, _, lab in read_list_file(lf):
            labels.append(lab)
    labels = sorted(list(set(labels)))
    return {lab: i for i, lab in enumerate(labels)}

# -------------------
# 分布式
# -------------------
def setup_distributed(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    else:
        args.rank, args.world_size, args.local_rank = 0, 1, 0

    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)

    if args.world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=1800)
        )
        dist.barrier()

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def is_main_process():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

# -------------------
# 准确率
# -------------------
@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk=(1,)):
    maxk = max(topk)
    batch_size = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()  # [K, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # [K, B]
    res = []
    for k in topk:
        correct_k = correct[:k].any(dim=0).float().sum(0) if k == 1 \
                    else correct[:k].any(dim=0).float().sum(0)
        # 注意：Top-1 用 any 与 sum 结果一致；这里保持一致写法
        res.append((correct[:k].any(dim=0).float().sum().item()) / batch_size)
    return res

# -------------------
# 评测
# -------------------
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    top1 = top2 = n = 0
    for images, _, labels, _, _ in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)  # [B, C]
        # Top-1
        pred1 = logits.argmax(dim=1)
        top1 += (pred1 == labels).sum().item()
        # Top-2
        top2 += (logits.topk(2, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
        n += labels.size(0)
    return top1 / n, top2 / n

# -------------------
# 训练
# -------------------
def train_one_epoch(model, optimizer, scaler, loader, device, args, epoch):
    model.train()
    ce = nn.CrossEntropyLoss()
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main_process())

    for images, masks, labels, img_paths, mask_paths in pbar:
        images = images.to(device)
        labels = labels.to(device)
        masks  = masks.to(device)

        # 2) 需要输入梯度
        images.requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)

        # ===== CE forward =====
        with torch.autocast("cuda", enabled=args.amp):
            logits = model(images)
            loss_ce = ce(logits, labels)

        # # ===== RRR forward (use log-prob for stability) =====
        logits_fp32 = logits.float()
        target_score = logits_fp32.gather(1, labels.view(-1, 1)).sum()

        grads = torch.autograd.grad(
            outputs=target_score,
            inputs=images,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]  # [B,3,H,W]
        
        # mask 维度对齐到 [B,1,H,W]
        if masks.dim() == 3:
            masks_ = masks.unsqueeze(1)
        else:
            masks_ = masks

        # # 默认 masks_=前景1背景0 -> forbidden=背景
        forbidden = 1.0 - masks_

        loss_rrr = (forbidden.detach() * grads.pow(2)).mean()

        loss_all = loss_ce + 0.1 * loss_rrr

        # ===== backward + step =====
        if epoch > 2:
            scaler.scale(loss_all).backward()
        else:
            scaler.scale(loss_ce).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        images.requires_grad_(False)

        if is_main_process():
            pbar.set_postfix(
                loss=f"{loss_all.item():.4f}",
                ce=f"{loss_ce.item():.4f}",
                rrr=f"{loss_rrr.item():.4f}", 
                # lam=f"{args.lambda_rrr:g}",
                mmean=f"{masks_.mean().item():.3f}"
            )
        
# -------------------
# 构建 ResNet-101（ImageNet 预训练）并替换分类头
# -------------------
def build_resnet101(num_classes: int, freeze_backbone: bool = False):
    weights = ResNet101_Weights.IMAGENET1K_V2  # 更强的 ImageNet1K V2 预训练
    model = resnet101(weights=weights)

    # 替换分类头
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    return model, weights

# -------------------
# main_worker
# -------------------
def main_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", default="data_list/saliency-bench/train.txt")
    parser.add_argument("--test_txt", default="data_list/saliency-bench/test.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--align_steps", type=int, default=20)
    parser.add_argument("--division_number", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--output_dir", default="./ckpt_vision_saliency_bench/ckpts_resnet_b16_imnet_RRR/")
    parser.add_argument("--num_classes", type=int, default=20, help="默认训练 20 类；若与数据集不一致将以数据集为准")
    parser.add_argument("--train_scope", type=str, default="full",
                        choices=["full", "head"],
                        help="full=全量微调; head=仅分类头")
    args = parser.parse_args()

    setup_distributed(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    
    # 数据
    train_loader, test_loader, train_ds, test_ds, label_to_idx = make_dataloaders(
        args.train_txt,
        args.test_txt,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        target_size=(224,224),
        image_mean=IMAGENET_MEAN,
        image_std=IMAGENET_STD,
        image_interpolation=InterpolationMode.BILINEAR,
    )
    train_sampler = DistributedSampler(train_ds) if args.world_size > 1 else None
    test_sampler  = DistributedSampler(test_ds, shuffle=False) if args.world_size > 1 else None
    idx_to_label = {v:k for k,v in label_to_idx.items()}

    # 模型 & 预处理
    freeze_backbone = (args.train_scope == "head")
    model, weights = build_resnet101(num_classes=args.num_classes, freeze_backbone=freeze_backbone)
    model = model.to(device)

    # 训练/测试增广（使用 ImageNet 均值方差）
    # 兼容 torchvision 版本差异：旧版 weights 可能无 meta 信息
    if hasattr(weights, "meta") and "mean" in weights.meta:
        mean = weights.meta["mean"]
        std = weights.meta["std"]
    else:
        # 默认 ImageNet 统计量
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # 优化器
    params = model.parameters()
    if args.train_scope == "head":
        # 仅分类头
        params = (p for n, p in model.named_parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=args.amp)

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        if train_sampler: train_sampler.set_epoch(epoch)

        train_one_epoch(model, optimizer, scaler, train_loader, device, args, epoch)

        # 评测
        eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        top1, top2 = evaluate(eval_model, test_loader, device)
        if is_main_process():
            print(f"[Eval] Epoch {epoch}: top1={top1:.4f}, top2={top2:.4f}")
            if top1 > best_acc:
                best_acc = top1
                torch.save(eval_model.state_dict(), os.path.join(args.output_dir, f"best_epoch{epoch}.pt"))

    cleanup_distributed()

if __name__ == "__main__":
    main_worker()
