import os
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm
import csv
from collections import defaultdict

from vit_model import VIT_IMAGE_MEAN, VIT_IMAGE_STD, build_vit_classifier, vit_logits

class AddRandomNoise(object):
    def __init__(self, noise_type='gaussian', mean=0.0, std=0.05, scale=0.05):
        """
        noise_type: 'gaussian' 或 'uniform'
        mean/std: 高斯噪声参数
        scale: 均匀噪声幅度 [-scale, scale]
        """
        self.noise_type = noise_type
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, tensor):
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(tensor) * self.std + self.mean
        elif self.noise_type == 'uniform':
            noise = (torch.rand_like(tensor) - 0.5) * 2 * self.scale
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
        tensor = tensor + noise
        tensor = torch.clamp(tensor, 0.0, 1.0)  # 保证像素范围不越界
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(type={self.noise_type}, std={self.std}, scale={self.scale})"
    
# -------------------
# 工具函数 & 数据
# -------------------
def read_list_file(list_path: str):
    """读取三列: image_path mask_path label_name；与训练一致。"""
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
    def __init__(self, list_file: str, label_to_idx: dict, transform):
        self.items = read_list_file(list_file)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, label_name = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img)
        label = torch.tensor(self.label_to_idx[label_name], dtype=torch.long)
        return img_t, label, img_path, label_name

# -------------------
# 模型
# -------------------
def build_vit_model(num_classes: int):
    return build_vit_classifier(num_classes)

def load_checkpoint_flex(model: nn.Module, ckpt_path: str, map_location="cpu"):
    print(f"[Info] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=map_location)
    # 常见几种保存方式的兼容
    if isinstance(state, dict):
        if "model" in state and isinstance(state["model"], dict):
            state_dict = state["model"]
        elif "state_dict" in state and isinstance(state["state_dict"], dict):
            state_dict = state["state_dict"]
        else:
            state_dict = state
    else:
        state_dict = state

    # 兼容DDP保存的 'module.' 前缀
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        else:
            new_state[k] = v

    # 尝试严格加载；若失败，改为非严格加载并提示
    try:
        model.load_state_dict(new_state, strict=True)
        print("[Info] Loaded with strict=True")
    except Exception as e:
        print(f"[Warn] Strict load failed ({e}); trying strict=False")
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[Warn] Missing keys: {missing}")
        print(f"[Warn] Unexpected keys: {unexpected}")

# -------------------
# 准确率 & 混淆矩阵
# -------------------
@torch.no_grad()
def evaluate(model, dataloader, device, amp=True, calc_confmat=False, idx_to_label=None):
    model.eval()
    n = 0
    top1_correct = 0
    top2_correct = 0

    # 混淆矩阵(可选)
    confmat = None
    if calc_confmat:
        num_classes = len(idx_to_label)
        confmat = torch.zeros((num_classes, num_classes), dtype=torch.long)

    results = []  # (img_path, gt_name, pred1_name, top1_correct_flag)

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for images, labels, img_paths, label_names in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(autocast_device, enabled=amp):
            logits = vit_logits(model, images)

        # Top-1
        pred1 = logits.argmax(dim=1)
        top1_correct += (pred1 == labels).sum().item()

        # Top-2
        top2_idx = logits.topk(2, dim=1).indices
        top2_correct += (top2_idx == labels.unsqueeze(1)).any(dim=1).sum().item()

        n += labels.size(0)

        # 记录结果
        if idx_to_label is not None:
            for p1, gt, pth in zip(pred1.tolist(), labels.tolist(), img_paths):
                results.append((pth, idx_to_label[gt], idx_to_label[p1], int(p1 == gt)))

        # 混淆矩阵
        if calc_confmat:
            for gt, p in zip(labels.view(-1), pred1.view(-1)):
                confmat[gt.long(), p.long()] += 1

    top1 = top1_correct / n if n else 0.0
    top2 = top2_correct / n if n else 0.0
    return top1, top2, results, confmat

# -------------------
# 主函数
# -------------------
def main():
    parser = argparse.ArgumentParser("Eval-only for ViT-B/16 classifier")
    parser.add_argument("--test_txt", type=str, default="data_list/saliency-bench/test.txt")
    parser.add_argument("--train_txt", type=str, default="data_list/saliency-bench/train.txt",
                        help="用于构建 label 映射；若无可与 test.txt 相同")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="模型权重路径，如 ./ckpts_vit_b16_imnet/best_epoch1.pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--noisy", type=bool, default=True)
    parser.add_argument("--save_csv", type=str, default="ckpts_vit_b16_imnet/results_eval.csv",
                        help="保存预测结果到CSV(可选)，例如 results_eval.csv")
    parser.add_argument("--confmat", action="store_true", default=False,
                        help="是否计算并打印混淆矩阵")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 标签映射（与训练一致：从 train+test 汇总）
    if not os.path.exists(args.train_txt):
        print(f"[Warn] --train_txt {args.train_txt} 不存在，将仅使用 --test_txt 构建标签映射")
        label_to_idx = build_label_map(args.test_txt)
    else:
        label_to_idx = build_label_map(args.train_txt, args.test_txt)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)
    print(f"[Info] Num classes = {num_classes}")

    # 模型 & 预处理
    model = build_vit_model(num_classes=num_classes)
    model.to(device)

    mean, std = VIT_IMAGE_MEAN, VIT_IMAGE_STD
    
    print(args.noisy)
    if args.noisy:
        test_tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            AddRandomNoise(noise_type='gaussian', std=0.1),  # 随机噪声层
            transforms.Normalize(mean=mean, std=std),
        ])
        print("[Info] Added random noise during evaluation.")
    else:
        test_tf = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        print("[Info] No noise added during evaluation.")

    test_ds = ImageListDataset(args.test_txt, label_to_idx, transform=test_tf)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 加载权重
    load_checkpoint_flex(model, args.ckpt, map_location=device)
    model.eval()

    # 评测
    top1, top2, results, confmat = evaluate(
        model, test_loader, device, amp=args.amp,
        calc_confmat=args.confmat, idx_to_label=idx_to_label
    )

    print(f"[Eval] top1={top1:.4f}, top2={top2:.4f}")

    # 保存CSV（可选）
    if args.save_csv:
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "gt_label", "pred_top1", "correct_top1"])
            writer.writerows(results)
        print(f"[Info] Saved predictions to {args.save_csv}")

    # 打印混淆矩阵（可选）
    if args.confmat:
        # 仅打印到控制台；如需保存可自行保存为 npy/csv
        print("[Confusion Matrix] rows=GT, cols=Pred")
        try:
            import numpy as np
            np.set_printoptions(linewidth=120, suppress=True)
            print(confmat.cpu().numpy())
        except Exception:
            print(confmat)

if __name__ == "__main__":
    main()
