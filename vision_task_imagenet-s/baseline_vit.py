import os
from datetime import timedelta
import argparse
import math
from typing import Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
from tqdm import tqdm

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

class ImageListDataset(Dataset):
    def __init__(
        self,
        list_file: str,
        label_to_idx: dict,
        transform,
    ):
        self.items = read_list_file(list_file)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, label_name = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)
        label = torch.tensor(self.label_to_idx[label_name], dtype=torch.long)
        return img_t, label

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
    for images, labels in dataloader:
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

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        with torch.autocast("cuda", enabled=args.amp):
            logits = model(images)  # [B, C]
            loss = ce(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if is_main_process():
            pbar.set_postfix(loss=f"{loss.item():.4f}")

# -------------------
# 构建 ViT 模型（ImageNet 预训练）并替换分类头
# -------------------
def build_vit_model(num_classes: int, freeze_backbone: bool = False):
    weights = ViT_B_16_Weights.IMAGENET1K_V1  # ImageNet 预训练
    model = vit_b_16(weights=weights)
    # 兼容不同 torchvision 版本的分类头写法
    in_features = None
    if hasattr(model, "heads") and hasattr(model.heads, "head") and isinstance(model.heads.head, nn.Linear):
        in_features = model.heads.head.in_features
    elif hasattr(model, "heads") and isinstance(model.heads, nn.Sequential) and len(model.heads) > 0 and isinstance(model.heads[0], nn.Linear):
        in_features = model.heads[0].in_features
    elif hasattr(model, "hidden_dim"):
        in_features = model.hidden_dim
    else:
        # 兜底（ViT-B/16 默认 768）
        in_features = 768
    model.heads = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, p in model.named_parameters():
            # 仅训练 heads
            if "heads" not in name:
                p.requires_grad = False

    return model, weights

# -------------------
# main_worker
# -------------------
def main_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", default="data_list/imagenet-s919/train.txt")
    parser.add_argument("--test_txt", default="data_list/imagenet-s919/test.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--output_dir", default="./ckpt_vision_imagenets/ckpts_vit_b16_imnet_")
    parser.add_argument("--num_classes", type=int, default=918, help="默认训练 20 类；若与数据集不一致将以数据集为准")
    parser.add_argument("--train_scope", type=str, default="full",
                        choices=["full", "head"],
                        help="full=全量微调; head=仅分类头")
    args = parser.parse_args()

    setup_distributed(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    # 标签映射
    label_to_idx = build_label_map(args.train_txt, args.test_txt)
    # 若数据集类数与 --num_classes 不一致，以数据集为准但提示
    ds_num_classes = len(label_to_idx)
    if ds_num_classes != args.num_classes and is_main_process():
        print(f"[Warn] dataset classes = {ds_num_classes}, but --num_classes = {args.num_classes}; using {ds_num_classes}.")
    num_classes = ds_num_classes

    # 模型 & 预处理
    freeze_backbone = (args.train_scope == "head")
    model, weights = build_vit_model(num_classes=num_classes, freeze_backbone=freeze_backbone)
    model = model.to(device)

    # 训练/测试增广（使用 ImageNet 均值方差）
    # 兼容 torchvision 版本差异：旧版 weights 可能无 meta 信息
    if hasattr(weights, "meta") and "mean" in weights.meta:
        mean = weights.meta["mean"]
        std = weights.meta["std"]
    else:
        # 默认 ImageNet 统计量
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = ImageListDataset(args.train_txt, label_to_idx, transform=train_tf)
    test_ds  = ImageListDataset(args.test_txt,  label_to_idx, transform=test_tf)

    train_sampler = DistributedSampler(train_ds) if args.world_size > 1 else None
    test_sampler  = DistributedSampler(test_ds, shuffle=False) if args.world_size > 1 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

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