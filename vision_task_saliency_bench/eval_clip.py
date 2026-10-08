import os
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel, AutoTokenizer
from tqdm import tqdm

# 关闭 tokenizer 并行提示
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# OpenAI CLIP 预处理参数
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# -------------------
# 数据集
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

def build_label_map(list_file: str):
    labels = [lab for _, _, lab in read_list_file(list_file)]
    labels = sorted(list(set(labels)))
    return {lab: i for i, lab in enumerate(labels)}

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


class PascalSaliencyDataset(Dataset):
    def __init__(self, list_file, label_to_idx, target_size=(224,224),
                 mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD):
        self.items = read_list_file(list_file)
        self.label_to_idx = label_to_idx
        self.img_tf = transforms.Compose([
            transforms.Resize(target_size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            # AddRandomNoise(noise_type='gaussian', std=0.2),  # 随机噪声层
            transforms.Normalize(mean=mean, std=std),
        ])

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path, label_name = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        img_t = self.img_tf(img)
        label = torch.tensor(self.label_to_idx[label_name], dtype=torch.long)
        return img_t, label

# -------------------
# 测试函数
# -------------------
@torch.no_grad()
def evaluate(model, dataloader, device, idx_to_label, templates):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    # 构建文本特征
    classnames = [idx_to_label[i] for i in range(len(idx_to_label))]
    feats = []
    for cname in classnames:
        texts = [tmp.format(name=cname) for tmp in templates]
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model.get_text_features(**tokens)
        out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.mean(dim=0))
    text_feats = torch.stack(feats, dim=1)  # [D,C]

    # 计算 top1/top5
    top1 = top2 = n = 0
    for images, labels in tqdm(dataloader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        img_feats = model.get_image_features(pixel_values=images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = (img_feats @ text_feats) * model.logit_scale.exp()

        pred = logits.argmax(dim=1)
        top1 += (pred == labels).sum().item()
        top2 += (logits.topk(2, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
        n += labels.size(0)

    return top1/n, top2/n

import torch
from collections import OrderedDict

def safe_load_state_dict(model, ckpt_path, map_location="cpu", verbose=True):
    # 1) 读 ckpt
    state = torch.load(ckpt_path, map_location=map_location)

    # 2) 解包常见外壳
    for key in ["state_dict", "model", "ema", "ema_state_dict", "module"]:
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]

    # 3) 去掉常见前缀
    def strip_prefix(sdict, prefix):
        return {k[len(prefix):]: v for k, v in sdict.items() if k.startswith(prefix)}

    prefixes = ["module.", "model.", "module.model.", "ema.", "student.", "teacher."]
    for p in prefixes:
        if any(k.startswith(p) for k in state.keys()):
            state = strip_prefix(state, p)

    # 4) 过滤：只保留名称和形状都匹配的参数
    model_sd = model.state_dict()
    filtered = OrderedDict()
    shape_mismatch = []
    for k, v in state.items():
        if k in model_sd:
            if tuple(v.shape) == tuple(model_sd[k].shape):
                filtered[k] = v
            else:
                shape_mismatch.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    # 5) 统计信息
    missing = sorted(set(model_sd.keys()) - set(filtered.keys()))
    unexpected = sorted(set(state.keys()) - set(model_sd.keys()))

    if verbose:
        print(f"[safe_load] loadable keys: {len(filtered)}/{len(model_sd)}")
        if missing:
            print(f"[safe_load] missing in ckpt: {len(missing)} (showing up to 10)")
            print("  ", missing[:10])
        if unexpected:
            print(f"[safe_load] unexpected in ckpt: {len(unexpected)} (showing up to 10)")
            print("  ", unexpected[:10])
        if shape_mismatch:
            print(f"[safe_load] shape mismatch: {len(shape_mismatch)} (showing up to 5)")
            for k, s_ckpt, s_model in shape_mismatch[:5]:
                print(f"   - {k}: ckpt {s_ckpt} vs model {s_model}")

    # 6) 合并并加载（未命中的保持初始化权重）
    model_sd.update(filtered)
    model.load_state_dict(model_sd, strict=False)
    return {"missing": missing, "unexpected": unexpected, "shape_mismatch": shape_mismatch}

# -------------------
# main
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_txt", type=str, default="data_list/saliency-bench/test.txt")
    parser.add_argument("--ckpt", default=None, type=str, help="训练好的模型权重 .pt")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据集
    label_to_idx = build_label_map(args.test_txt)
    idx_to_label = {v:k for k,v in label_to_idx.items()}
    test_ds = PascalSaliencyDataset(args.test_txt, label_to_idx)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 模型
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    if args.ckpt:
        print("Load ckpt!")
        info = safe_load_state_dict(model, args.ckpt, map_location="cpu", verbose=True)
        print(info)
    model.to(device)
    model.eval()

    # 评测
    templates = [
        # "a photo of a {name}.",
        "a photo of a {name}.",
        # "a close-up photo of a {name}.",
        # "a blurry photo of a {name}.",
        # "a photo of a small {name}.",
        # "a photo of a large {name}.",
    ]
    top1, top2 = evaluate(model, test_loader, device, idx_to_label, templates)
    print(f"[Result] Top-1 acc = {top1:.4f}, Top-2 acc = {top2:.4f}")

if __name__ == "__main__":
    main()
