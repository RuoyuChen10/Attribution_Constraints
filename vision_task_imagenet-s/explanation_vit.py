import os
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel, AutoTokenizer
from tqdm import tqdm
from torchvision.models import vit_b_16, ViT_B_16_Weights

from interpretation.LIMA import BlackBoxSingleModalCounterfactualSubModularExplanation
from utils import mkdir, SubRegionDivision

import cv2

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
# main
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_txt", type=str, default="data_list/imagenet-s919/test_one_per_class.txt")
    parser.add_argument("--ckpt", default="", type=str, help="训练好的模型权重 .pt")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument('--division-number', 
                        type=int, default=50,
                        help='')
    parser.add_argument('--save-dir', 
                        type=str, default='',
                        help='output directory to save results')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 标签映射（与训练一致：从 train+test 汇总）
    # if not os.path.exists(args.train_txt):
    #     print(f"[Warn] --train_txt {args.train_txt} 不存在，将仅使用 --test_txt 构建标签映射")
    #     label_to_idx = build_label_map(args.test_txt)
    # else:
    label_to_idx = build_label_map(args.test_txt)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes = len(label_to_idx)
    print(f"[Info] Num classes = {num_classes}")

    # 模型 & 预处理
    model, weights = build_vit_model(num_classes=num_classes)
    model.to(device)
        
    # 加载权重
    load_checkpoint_flex(model, args.ckpt, map_location=device)
    model.eval()
    
    # 测试时的 ImageNet 统计（兼容无 meta 的情况）
    if hasattr(weights, "meta") and "mean" in weights.meta:
        mean = weights.meta["mean"]
        std = weights.meta["std"]
    else:
        mean = (0.48145466, 0.4578275, 0.40821073)
        std  = (0.26862954, 0.26130258, 0.27577711)

    img_tf = transforms.Compose([
        transforms.Resize((224,224), interpolation=InterpolationMode.BICUBIC),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    
    smdl = BlackBoxSingleModalCounterfactualSubModularExplanation(
        model,
        lambda1=1,
        lambda2=1,
        softmax=True
    )
    
    save_dir = args.save_dir
    
    mkdir(save_dir)
    
    save_npy_root_path = os.path.join(save_dir, "npy")
    mkdir(save_npy_root_path)
    
    save_json_root_path = os.path.join(save_dir, "json")
    mkdir(save_json_root_path)
    
    with open(args.test_txt, "r") as f:
        items = [line.strip().split() for line in f if line.strip()]
        
    for item in tqdm(items):
        img_path, mask_path, label_name = item
        label = label_to_idx[label_name]

        image = cv2.resize(cv2.imread(img_path),(224,224))
        
        image_tensor = img_tf(Image.open(img_path).convert("RGB")).to(device)
        
        # Sub-region division
        region_size = int((image.shape[0] * image.shape[1] / args.division_number) ** 0.5)
        V_set = SubRegionDivision(image, mode="slico", region_size = region_size)
        
        S_set, saved_json_file = smdl(image_tensor, V_set, label)
        
        # Save npy file
        np.save(
            os.path.join(save_npy_root_path, img_path.split("/")[-1].replace(".JPEG", ".npy")),
            np.array(S_set)
        )
        
        # Save json file
        with open(
            os.path.join(save_json_root_path, img_path.split("/")[-1].replace(".JPEG", ".json")), "w") as f:
            f.write(json.dumps(saved_json_file, ensure_ascii=False, indent=4, separators=(',', ':')))
    

if __name__ == "__main__":
    main()
