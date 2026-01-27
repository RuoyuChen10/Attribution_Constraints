import os
from datetime import timedelta
import math
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPModel, AutoTokenizer
from tqdm import tqdm

# 关闭 tokenizer 并行提示
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # for Chinese
os.environ["HF_HOME"] = "./model_checkpoint/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------
# 数据集
# -------------------
OPENAI_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

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

def clamp_logit_scale(m):
    with torch.no_grad():
        max_val = math.log(100.0)   # OpenAI 官方限制
        if hasattr(m, "logit_scale"):
            m.logit_scale.data.clamp_(max=max_val)

class PascalSaliencyDataset(Dataset):
    def __init__(self, list_file, label_to_idx, target_size=(224,224),
                 mean=OPENAI_CLIP_MEAN, std=OPENAI_CLIP_STD):
        self.items = read_list_file(list_file)
        self.label_to_idx = label_to_idx
        self.img_tf = transforms.Compose([
            transforms.Resize(target_size, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
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
# 分布式初始化
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
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

# -------------------
# 训练和评测
# -------------------
def build_texts(labels_idx, idx_to_label, templates):
    texts = []
    for i in labels_idx.tolist():
        name = idx_to_label[i].replace("_", " ")
        texts.append(templates[0].format(name=name))
    return texts

@torch.no_grad()
def evaluate(model, dataloader, device, idx_to_label, templates):
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    classnames = [idx_to_label[i] for i in range(len(idx_to_label))]
    feats = []
    for cname in classnames:
        texts = [tmp.format(name=cname) for tmp in templates]
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model.get_text_features(**tokens)
        out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.mean(dim=0))
    text_feats = torch.stack(feats, dim=1)  # [D,C]

    top1 = top2 = n = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        img_feats = model.get_image_features(pixel_values=images)
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        logits = (img_feats @ text_feats) * model.logit_scale.exp()

        pred = logits.argmax(dim=1)
        top1 += (pred == labels).sum().item()
        top2 += (logits.topk(2, dim=1).indices == labels.unsqueeze(1)).any(dim=1).sum().item()
        n += labels.size(0)

    return top1/n, top2/n

def train_one_epoch(model, optimizer, scaler, loader, device, tokenizer, idx_to_label, args, epoch, text_feats):
    model.train()
    ce = nn.CrossEntropyLoss()
    templates = ["a photo of a {name}."]
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main_process())

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        # texts = build_texts(labels, idx_to_label, templates)
        # tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

        with torch.autocast("cuda", enabled=args.amp):
            # out = model(pixel_values=images, input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"])
            # loss_i2t = ce(out.logits_per_image, torch.arange(len(images), device=device))
            # loss_t2i = ce(out.logits_per_text, torch.arange(len(images), device=device))
            # loss = 0.5 * (loss_i2t + loss_t2i)
            img_feats = model.get_image_features(pixel_values=images)  # [B, D]
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            scale = model.logit_scale.exp()
            logits = (img_feats @ text_feats) * scale
            loss = ce(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if is_main_process():
            pbar.set_postfix(loss=f"{loss.item():.4f}")

# -------------------
# main_worker
# -------------------
def main_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_txt", default="data_list/saliency-bench/train.txt")
    parser.add_argument("--test_txt", default="data_list/saliency-bench/test.txt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--output_dir", default="./ckpts_clip_L14")
    parser.add_argument("--train_scope", type=str, default="vision",
                    choices=["full", "vision", "proj"],
                    help="训练范围: full=全量; vision=只训练视觉塔; proj=只训练投影层和logit_scale")
    args = parser.parse_args()

    setup_distributed(args)
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")

    # 数据
    label_to_idx = build_label_map(args.train_txt, args.test_txt)
    idx_to_label = {v:k for k,v in label_to_idx.items()}
    train_ds = PascalSaliencyDataset(args.train_txt, label_to_idx)
    test_ds  = PascalSaliencyDataset(args.test_txt, label_to_idx)

    train_sampler = DistributedSampler(train_ds) if args.world_size > 1 else None
    test_sampler  = DistributedSampler(test_ds, shuffle=False) if args.world_size > 1 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              shuffle=(train_sampler is None), num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, sampler=test_sampler,
                             shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 模型
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    # 冻结文本 encoder
    # 根据 scope 冻结不同部分
    if args.train_scope == "vision":
        for name, param in model.named_parameters():
            if "text_model" in name:
                param.requires_grad = False
    elif args.train_scope == "proj":
        for name, param in model.named_parameters():
            if not ("visual_projection" in name or "text_projection" in name or "logit_scale" in name):
                param.requires_grad = False
    # full 就不用改
    model.logit_scale.requires_grad = False
    
    # 评测
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    templates = [
        "a photo of a {name}.",
    ]
    # 构建文本特征
    classnames = [idx_to_label[i] for i in range(len(idx_to_label))]
    feats = []
    for cname in classnames:
        texts = [tmp.format(name=cname) for tmp in templates]
        tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = model.get_text_features(**tokens)
        out = out / out.norm(dim=-1, keepdim=True)
        feats.append(out.mean(dim=0))
    text_feats = torch.stack(feats, dim=1).detach()  # [D,C] [768, 20]

    if args.world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=args.amp)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    os.makedirs(args.output_dir, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs+1):
        if train_sampler: train_sampler.set_epoch(epoch)
        train_one_epoch(model, optimizer, scaler, train_loader, device, tokenizer, idx_to_label, args, epoch, text_feats)

        # 评测
        eval_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        top1, top2 = evaluate(eval_model, test_loader, device, idx_to_label,
                              ["a photo of a {name}.", 
                            #    "a photo of the {name}."
                               ])
        if is_main_process():
            print(f"[Eval] Epoch {epoch}: top1={top1:.4f}, top2={top2:.4f}")
            if top2 > best_acc:
                best_acc = top2
                torch.save(eval_model.state_dict(), os.path.join(args.output_dir, f"best_epoch{epoch}.pt"))

    cleanup_distributed()

if __name__ == "__main__":
    main_worker()
