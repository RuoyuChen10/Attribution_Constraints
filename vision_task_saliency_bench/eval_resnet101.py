import os
import argparse
import csv

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.models import resnet101, ResNet101_Weights
from PIL import Image
from tqdm import tqdm

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
# 数据工具
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


def load_checkpoint_flex(model: nn.Module, ckpt_path: str, map_location="cpu", strict=False):
    print(f"[Info] Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=map_location)

    # 取出真正的 state_dict
    if isinstance(state, dict):
        for k in ("model", "state_dict", "net", "ema"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break

    if not isinstance(state, dict):
        raise TypeError(f"Loaded object is not a dict: {type(state)}")

    # 仅在存在前缀时剥离；不要无差别 k[7:]
    prefixes = ("module.", "model.", "backbone.", "encoder.")
    new_state = {}
    for k, v in state.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_state[nk] = v  # 只赋值一次

    # 可选：检测是否曾经被错误切片过的残留（避免老代码遗留）
    bad_fragments = {"eight","ght","ht","s","ning_mean","ning_var","_batches_tracked",""}
    leaked = [k for k in new_state.keys() if k in bad_fragments]
    if leaked:
        print(f"[Warn] Found suspicious truncated keys (likely from old buggy slicing): {leaked[:8]} ...")

    # 统计可匹配率（shape也要对）
    model_sd = model.state_dict()
    hit = sum(1 for k, v in new_state.items() if k in model_sd and model_sd[k].shape == v.shape)
    total = len(new_state)
    print(f"[Info] Key+shape match: {hit}/{total} ({100.0*hit/max(1,total):.1f}%)")

    try:
        missing, unexpected = model.load_state_dict(new_state, strict=strict)
        if strict:
            print("[Info] Loaded with strict=True")
        else:
            print(f"[Warn] Missing keys: {missing}")
            print(f"[Warn] Unexpected keys: {unexpected}")
    except Exception as e:
        print(f"[Warn] strict={strict} load failed: {e}")
        missing, unexpected = model.load_state_dict(new_state, strict=False)
        print(f"[Warn] Fallback strict=False. Missing: {missing}")
        print(f"[Warn] Unexpected: {unexpected}")

# -------------------
# 评测
# -------------------
@torch.no_grad()
def evaluate(model, dataloader, device, amp=True, calc_confmat=False, idx_to_label=None):
    model.eval()
    n = 0
    top1_correct = 0
    top2_correct = 0

    confmat = None
    if calc_confmat and idx_to_label is not None:
        confmat = torch.zeros((len(idx_to_label), len(idx_to_label)), dtype=torch.long)

    results = []  # (image_path, gt_label_name, pred_top1_name, correct_flag)

    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    for images, labels, img_paths, label_names in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(autocast_device, enabled=amp):
            logits = model(images)

        pred1 = logits.argmax(dim=1)
        top1_correct += (pred1 == labels).sum().item()

        top2_idx = logits.topk(2, dim=1).indices
        top2_correct += (top2_idx == labels.unsqueeze(1)).any(dim=1).sum().item()

        n += labels.size(0)

        if idx_to_label is not None:
            for p1, gt, pth in zip(pred1.tolist(), labels.tolist(), img_paths):
                results.append((pth, idx_to_label[gt], idx_to_label[p1], int(p1 == gt)))

        if confmat is not None:
            for gt, p in zip(labels.view(-1), pred1.view(-1)):
                confmat[gt.long(), p.long()] += 1

    top1 = top1_correct / n if n else 0.0
    top2 = top2_correct / n if n else 0.0
    return top1, top2, results, confmat

# -------------------
# 主函数
# -------------------
def main():
    parser = argparse.ArgumentParser("Eval-only for ResNet-101 classifier")
    parser.add_argument("--test_txt", type=str, default="data_list/saliency-bench/test.txt")
    parser.add_argument("--train_txt", type=str, default="data_list/saliency-bench/train.txt",
                        help="用于构建 label 映射；若无可与 test.txt 相同")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="模型权重路径，如 ./ckpts_resnet101_imnet/best_epoch10.pt")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--noisy", type=bool, default=True)
    parser.add_argument("--save_csv", type=str, default="",
                        help="可选：保存预测结果到 CSV，例如 results_resnet_eval.csv")
    parser.add_argument("--confmat", action="store_true", default=False,
                        help="可选：计算并打印混淆矩阵")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 标签映射
    if not os.path.exists(args.train_txt):
        print(f"[Warn] --train_txt {args.train_txt} 不存在，将仅使用 --test_txt 构建标签映射")
        label_to_idx = build_label_map(args.test_txt)
    else:
        label_to_idx = build_label_map(args.train_txt, args.test_txt)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)
    print(f"[Info] Num classes = {num_classes}")

    # 模型与预处理
    model, weights = build_resnet101(num_classes=num_classes)
    model.to(device)

    if hasattr(weights, "meta") and isinstance(getattr(weights, "meta"), dict) and "mean" in weights.meta:
        mean = weights.meta["mean"]
        std  = weights.meta["std"]
    else:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std  = (0.26862954, 0.26130258, 0.27577711)

    print(args.noisy)
    if args.noisy == True:
        test_tf = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            AddRandomNoise(noise_type='gaussian', std=0.1),  # 随机噪声层
            transforms.Normalize(mean=mean, std=std),
        ])
        print("[Info] Added random noise during evaluation.")
    else:
        test_tf = transforms.Compose([
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
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
        pin_memory=True
    )

    # 加载权重
    # model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    load_checkpoint_flex(model, args.ckpt, map_location=device)
    model.eval()

    # 评测
    top1, top2, results, confmat = evaluate(
        model, test_loader, device, amp=args.amp,
        calc_confmat=args.confmat, idx_to_label=idx_to_label
    )
    print(f"[Eval] top1={top1:.4f}, top2={top2:.4f}")

    # 保存 CSV（可选）
    # if args.save_csv:
    #     with open(args.save_csv, "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["image_path", "gt_label", "pred_top1", "correct_top1"])
    #         writer.writerows(results)
    #     print(f"[Info] Saved predictions to {args.save_csv}")

    # # 打印混淆矩阵（可选）
    # if args.confmat and confmat is not None:
    #     print("[Confusion Matrix] rows=GT, cols=Pred")
    #     try:
    #         import numpy as np
    #         np.set_printoptions(linewidth=120, suppress=True)
    #         print(confmat.cpu().numpy())
    #     except Exception:
    #         print(confmat)

if __name__ == "__main__":
    main()
